/// Loop Detection Index (LDI)
///
/// Identifies circular reasoning patterns: the agent revisits the same state,
/// tool, or conclusion without making progress.
///
/// Formula: LDI = max_cycle_length / total_steps
/// Target: 0 (no loops). Warning above 0.1.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::{StepFlag, Trace, TraceStep};

/// A detected loop in the trace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DetectedLoop {
    /// Step IDs involved in the loop.
    pub step_ids: Vec<u32>,
    /// Length of the loop.
    pub length: usize,
    /// Whether this loop is based on repeated state hashes (state loop)
    /// or repeated tool calls (tool loop).
    pub loop_type: LoopType,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopType {
    #[default]
    StateHash,
    ToolRepeat,
    CycleDetect,
    /// Same command *template* (argument literals abstracted away) repeated for
    /// different arguments — e.g. running the same shell command per file.
    ParametricRepeat,
}

/// Result of the LDI metric computation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LdiResult {
    /// LDI value (0.0 = no loops). Lower is better.
    pub score: f64,
    pub loops: Vec<DetectedLoop>,
    pub max_cycle_length: usize,
    pub total_steps: usize,
    pub pass: bool,
    /// Warning threshold.
    pub warning_threshold: f64,
}

impl LdiResult {
    /// Normalised score for TAS composite (0.0–1.0, higher = better).
    pub fn normalised(&self) -> f64 {
        (1.0 - self.score).clamp(0.0, 1.0)
    }
}

const WARNING_THRESHOLD: f64 = 0.1;

/// Compute the LDI metric for a trace using two complementary methods:
/// 1. State-hash repeated detection (same tool + params seen twice).
/// 2. Sequence-level repeat detection (same N-step pattern seen twice).
pub fn compute(trace: &Trace) -> LdiResult {
    let steps = &trace.steps;
    let total = steps.len();

    let mut loops: Vec<DetectedLoop> = Vec::new();

    // Method 1: State hash repetition (tool-call level).
    //
    // A repeat *after an intervening state change* (an edit, a write, a
    // booking) is not a loop: re-running the identical check against a
    // changed world is how agents verify their changes. The chain restarts
    // at the verification run.
    let mut state_seen: HashMap<String, (u32, usize)> = HashMap::new(); // hash -> (first_id, last_idx)
    let mut tool_loop_groups: HashMap<u32, Vec<u32>> = HashMap::new();

    for (idx, step) in steps.iter().enumerate() {
        if step.tool_name.is_some() {
            let hash = step.state_hash();
            match state_seen.get_mut(&hash) {
                Some((first_id, last_idx)) => {
                    let mutated_between =
                        steps[*last_idx + 1..idx].iter().any(|s| s.is_mutating());
                    if mutated_between && step.tool_success != Some(false) {
                        // Verification re-run: world changed since the last
                        // occurrence — fresh chain, not a loop iteration.
                        *first_id = step.id;
                    } else {
                        tool_loop_groups.entry(*first_id).or_default().push(step.id);
                    }
                    *last_idx = idx;
                }
                None => {
                    state_seen.insert(hash, (step.id, idx));
                }
            }
        }
    }

    for (first_id, repeat_ids) in &tool_loop_groups {
        let mut ids = vec![*first_id];
        ids.extend(repeat_ids);
        ids.sort();
        let len = ids.len();
        loops.push(DetectedLoop {
            step_ids: ids,
            length: len,
            loop_type: LoopType::StateHash,
        });
    }

    // Method 1b: Parametric command loops — same command *template* (argument
    // literals abstracted away) repeated for different arguments. This is the
    // dominant loop shape for tool-using agents (e.g. running the same shell
    // command once per file), which Method 1 misses because the concrete
    // params differ each time. Scoped to command-style tools so structured
    // tools (e.g. a flight search called for two routes) are never affected.
    {
        let mut skeleton_groups: HashMap<String, Vec<u32>> = HashMap::new();
        for step in steps {
            if let Some(skel) = command_skeleton(step) {
                skeleton_groups.entry(skel).or_default().push(step.id);
            }
        }
        let id_to_idx: HashMap<u32, usize> =
            steps.iter().enumerate().map(|(i, s)| (s.id, i)).collect();
        let mut grouped: Vec<(String, Vec<u32>)> = skeleton_groups.into_iter().collect();
        grouped.sort_by(|a, b| a.1.first().cmp(&b.1.first()));
        for (_skel, mut ids) in grouped {
            ids.sort();
            // Split the occurrence chain wherever a mutating step intervenes:
            // a test→edit→test→edit→test cycle is the agent *verifying* each
            // change, not looping. Only unbroken segments count.
            let mut segments: Vec<Vec<u32>> = Vec::new();
            let mut cur = vec![ids[0]];
            for w in ids.windows(2) {
                let (a, b) = (id_to_idx[&w[0]], id_to_idx[&w[1]]);
                if steps[a + 1..b].iter().any(|s| s.is_mutating()) {
                    segments.push(std::mem::take(&mut cur));
                    cur = vec![w[1]];
                } else {
                    cur.push(w[1]);
                }
            }
            segments.push(cur);
            for seg in segments {
                // Require a clear repeat (>=3 identical templates) before
                // calling it a loop, so an incidental pair is not penalised.
                if seg.len() < 3 {
                    continue;
                }
                let already_reported = loops
                    .iter()
                    .any(|l| l.step_ids.iter().any(|id| seg.contains(id)));
                if already_reported {
                    continue;
                }
                let len = seg.len();
                loops.push(DetectedLoop {
                    step_ids: seg,
                    length: len,
                    loop_type: LoopType::ParametricRepeat,
                });
            }
        }
    }

    // Method 2: Consecutive sub-sequence repeat detection.
    // Look for patterns of length 2–5 that repeat consecutively.
    for window in 2..=5usize {
        if steps.len() < window * 2 {
            break;
        }
        let mut i = 0;
        while i + window * 2 <= steps.len() {
            let pattern: Vec<String> = steps[i..i + window]
                .iter()
                .map(|s| s.state_hash())
                .collect();
            let next: Vec<String> = steps[i + window..i + window * 2]
                .iter()
                .map(|s| s.state_hash())
                .collect();

            if pattern == next {
                let loop_ids: Vec<u32> = steps[i..i + window * 2]
                    .iter()
                    .map(|s| s.id)
                    .collect();
                // Avoid duplicate loop reports overlapping with state-hash loops.
                let already_reported = loops.iter().any(|l| {
                    l.step_ids.iter().any(|id| loop_ids.contains(id))
                });
                if !already_reported {
                    let len = loop_ids.len();
                    loops.push(DetectedLoop {
                        step_ids: loop_ids,
                        length: len,
                        loop_type: LoopType::CycleDetect,
                    });
                }
                i += window;
            } else {
                i += 1;
            }
        }
    }

    // Sort by first step_id for deterministic canonical_bytes() across process runs.
    loops.sort_by_key(|l| l.step_ids.first().copied().unwrap_or(u32::MAX));

    let max_cycle_length = loops.iter().map(|l| l.length).max().unwrap_or(0);
    let score = if total == 0 {
        0.0
    } else {
        max_cycle_length as f64 / total as f64
    };

    LdiResult {
        score: (score * 1000.0).round() / 1000.0,
        loops,
        max_cycle_length,
        total_steps: total,
        pass: score <= WARNING_THRESHOLD,
        warning_threshold: WARNING_THRESHOLD,
    }
}

/// Tool names whose calls carry a free-text command/query worth skeletonizing.
const COMMAND_TOOLS: &[&str] = &[
    "bash", "sh", "shell", "zsh", "terminal", "console", "sql", "mysql",
    "psql", "python", "python3", "execute", "run", "exec", "code", "cmd",
];

/// Parameter keys that hold a command/query string.
const COMMAND_KEYS: &[&str] = &[
    "command", "query", "cmd", "code", "script", "sql", "shell", "statement",
];

/// Extract the command/query text from a command-style tool call, if any.
///
/// Returns `None` for structured tools (e.g. a flight search with origin /
/// destination params) so parametric-loop detection never touches them.
fn command_text(step: &TraceStep) -> Option<String> {
    if let Some(serde_json::Value::Object(map)) = &step.tool_params {
        for key in COMMAND_KEYS {
            if let Some(serde_json::Value::String(s)) = map.get(*key) {
                let t = s.trim();
                if !t.is_empty() {
                    return Some(t.to_string());
                }
            }
        }
    }
    let is_cmd_tool = step
        .tool_name
        .as_deref()
        .map(|t| COMMAND_TOOLS.contains(&t.to_ascii_lowercase().as_str()))
        .unwrap_or(false);
    if is_cmd_tool {
        let t = step.content.trim();
        if !t.is_empty() {
            return Some(t.to_string());
        }
    }
    None
}

/// Structural punctuation kept verbatim in a command skeleton.
fn is_structural_token(t: &str) -> bool {
    matches!(
        t,
        "|" | "||" | "&&" | ">" | ">>" | "<" | ";" | "2>" | "2>&1" | "&"
    )
}

/// Normalise one command token: argument literals (quoted strings, paths,
/// globs, numbers) become placeholders; command words and flags are preserved.
fn skeleton_token(tok: &str) -> String {
    let t = tok.trim();
    if t.is_empty() {
        return String::new();
    }
    if is_structural_token(t) {
        return t.to_string();
    }
    if t.contains('"') || t.contains('\'') || t.contains('`') {
        return "STR".to_string();
    }
    if t.contains('*') || t.contains('?') {
        return "GLOB".to_string();
    }
    if t.contains('/') || t.starts_with('~') || t.starts_with("./") {
        return "PATH".to_string();
    }
    let is_num = t.chars().any(|c| c.is_ascii_digit())
        && t.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | ',' | '-' | '+'));
    if is_num {
        return "NUM".to_string();
    }
    t.to_ascii_lowercase()
}

/// Build a structural skeleton for a command-style step, or `None` if the step
/// is not command-style or the skeleton has no real command word left.
fn command_skeleton(step: &TraceStep) -> Option<String> {
    let cmd = command_text(step)?;
    let skel = cmd
        .split_whitespace()
        .map(skeleton_token)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let has_command_word = skel
        .split_whitespace()
        .any(|w| !matches!(w, "STR" | "GLOB" | "PATH" | "NUM") && !is_structural_token(w));
    if has_command_word {
        Some(skel)
    } else {
        None
    }
}

/// Apply LDI flags to trace steps.
pub fn annotate_steps(steps: &mut [TraceStep], result: &LdiResult) {
    if result.loops.is_empty() {
        return;
    }
    for detected_loop in &result.loops {
        let ids = &detected_loop.step_ids;
        let cycle_str = ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("→");

        for (pos, &step_id) in ids.iter().enumerate() {
            if let Some(step) = steps.iter_mut().find(|s| s.id == step_id) {
                if pos == 0 {
                    step.flags.push(StepFlag::LoopStart);
                    step.flag_details.push(format!("cycle: {}", cycle_str));
                } else {
                    step.flags.push(StepFlag::Loop);
                    if step.tool_name.is_some() {
                        step.flag_details.push(format!(
                            "re-fetching data already retrieved at step {}",
                            ids[0]
                        ));
                    } else {
                        step.flag_details.push("redundant re-evaluation".into());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, TraceStep};
    use std::collections::HashMap;

    fn tool_step(id: u32, tool: &str) -> TraceStep {
        TraceStep {
            id,
            step_type: StepType::ToolCall,
            content: format!("call {tool}"),
            tokens: 100,
            tool_name: Some(tool.to_string()),
            tool_params: Some(serde_json::json!({"k": "v"})),
            tool_success: Some(true),
            tool_error: None,
            agent_id: None,
            input_context: None,
            output: None,
            flags: vec![],
            flag_details: vec![],
        }
    }

    fn reason_step(id: u32) -> TraceStep {
        TraceStep {
            id,
            step_type: StepType::Reasoning,
            content: format!("reasoning {id}"),
            tokens: 100,
            tool_name: None,
            tool_params: None,
            tool_success: None,
            tool_error: None,
            agent_id: None,
            input_context: None,
            output: None,
            flags: vec![],
            flag_details: vec![],
        }
    }

    #[test]
    fn test_no_loops() {
        let trace = Trace {
            trace_id: "t1".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                reason_step(1),
                tool_step(2, "get_order"),
                tool_step(3, "check_refund"),
                tool_step(4, "process_refund"),
                reason_step(5),
            ],
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let result = compute(&trace);
        assert_eq!(result.max_cycle_length, 0);
        assert!(result.pass);
    }

    /// A command-style tool call with a free-text command param.
    fn bash_step(id: u32, command: &str) -> TraceStep {
        TraceStep {
            id,
            step_type: StepType::ToolCall,
            content: format!("Act: bash {command}"),
            tokens: 100,
            tool_name: Some("bash".into()),
            tool_params: Some(serde_json::json!({ "command": command })),
            tool_success: Some(true),
            tool_error: None,
            agent_id: None,
            input_context: None,
            output: None,
            flags: vec![],
            flag_details: vec![],
        }
    }

    #[test]
    fn test_detects_parametric_command_loop() {
        // Same command template run once per file — differs only by the path
        // argument. Method 1 (exact state hash) cannot see this; the parametric
        // detector must. Mirrors the real AgentInstruct os_6 trajectory.
        let trace = Trace {
            trace_id: "param".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                bash_step(1, "grep -rl \"Linux\" ~/ --include \"*.txt\""),
                bash_step(2, "grep -o \"Linux\" /root/love_linux.txt | wc -l"),
                bash_step(3, "grep -o \"Linux\" /root/example1.txt | wc -l"),
                bash_step(4, "grep -o \"Linux\" /root/favor.txt | wc -l"),
                bash_step(5, "grep -o \"Linux\" /root/course.txt | wc -l"),
                reason_step(6),
            ],
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let result = compute(&trace);
        assert!(
            result
                .loops
                .iter()
                .any(|l| matches!(l.loop_type, LoopType::ParametricRepeat) && l.length == 4),
            "expected a parametric loop of length 4, got {:?}",
            result.loops
        );
        assert!(result.score > 0.0, "LDI must fire on a parametric loop");
    }

    #[test]
    fn test_parametric_ignores_distinct_pipelines() {
        // Progressive pipeline building (different structure each step) is NOT a
        // loop — guards against false positives on refinement sequences.
        let trace = Trace {
            trace_id: "refine".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                bash_step(1, "ls *.conf"),
                bash_step(2, "cat *.conf | grep -v \"^$\""),
                bash_step(3, "cat *.conf | grep -v \"^$\" | awk '{print $1}'"),
                bash_step(4, "cat *.conf | grep -v \"^$\" | sort | uniq -c"),
                reason_step(5),
            ],
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let result = compute(&trace);
        assert!(
            !result
                .loops
                .iter()
                .any(|l| matches!(l.loop_type, LoopType::ParametricRepeat)),
            "distinct pipelines must not be flagged as a parametric loop: {:?}",
            result.loops
        );
    }

    #[test]
    fn test_parametric_skips_structured_tools() {
        // Two flight searches with the same tool but different structured params
        // must NOT be treated as a parametric loop (no free-text command).
        let mut s1 = tool_step(1, "search_flight");
        s1.tool_params = Some(serde_json::json!({"origin": "JFK", "destination": "LAX"}));
        let mut s2 = tool_step(2, "search_flight");
        s2.tool_params = Some(serde_json::json!({"origin": "JFK", "destination": "SEA"}));
        let mut s3 = tool_step(3, "search_flight");
        s3.tool_params = Some(serde_json::json!({"origin": "JFK", "destination": "ORD"}));
        let trace = Trace {
            trace_id: "flights".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![s1, s2, s3, reason_step(4), reason_step(5)],
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let result = compute(&trace);
        assert!(
            !result
                .loops
                .iter()
                .any(|l| matches!(l.loop_type, LoopType::ParametricRepeat)),
            "structured-tool calls must not be parametric loops: {:?}",
            result.loops
        );
    }

    #[test]
    fn test_detects_repeated_tool() {
        let trace = Trace {
            trace_id: "t2".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                reason_step(1),
                tool_step(2, "get_order"),
                reason_step(3),
                reason_step(4),
                tool_step(5, "get_order"), // repeated
            ],
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let result = compute(&trace);
        assert!(!result.loops.is_empty());
    }
}
