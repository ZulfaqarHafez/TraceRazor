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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedLoop {
    /// Step IDs involved in the loop.
    pub step_ids: Vec<u32>,
    /// Length of the loop.
    pub length: usize,
    /// Whether this loop is based on repeated state hashes (state loop)
    /// or repeated tool calls (tool loop).
    pub loop_type: LoopType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopType {
    StateHash,
    ToolRepeat,
    CycleDetect,
}

/// Result of the LDI metric computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        (1.0 - self.score).max(0.0).min(1.0)
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
    let mut state_seen: HashMap<String, u32> = HashMap::new();
    let mut tool_loop_groups: HashMap<u32, Vec<u32>> = HashMap::new();

    for step in steps {
        if step.tool_name.is_some() {
            let hash = step.state_hash();
            if let Some(&first_id) = state_seen.get(&hash) {
                tool_loop_groups
                    .entry(first_id)
                    .or_default()
                    .push(step.id);
            } else {
                state_seen.insert(hash, step.id);
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

/// Apply LDI flags to trace steps.
pub fn annotate_steps(steps: &mut Vec<TraceStep>, result: &LdiResult) {
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
