//! Action/Claim Grounding Fidelity (AGF) — provenance diagnostic.
//!
//! Measures how much of what the agent *did* and *concluded* is traceable to
//! evidence inside the trace, with deterministic literal matching — no model,
//! no network, reproducible byte-for-byte:
//!
//! * **Action grounding** — the share of tool-call argument literals (paths,
//!   quoted strings, file names, numbers) that appear somewhere in the prior
//!   context: the task statement, earlier agent turns, or earlier
//!   observations. A fabricated path or identifier shows up as ungrounded.
//! * **Claim grounding** — the share of literals in the final answer that
//!   appear in *environment-provided* text (task, observations, inputs).
//!   Derived values (e.g. arithmetic the agent did in its head) are counted
//!   as ungrounded on purpose: they are exactly the claims an auditor must
//!   verify by hand.
//!
//! Grounding-fidelity metrics for agent trajectories follow the
//! trajectory-hallucination line of work (arXiv:2605.24219, 2601.06818,
//! 2601.05214); the deterministic-literal formulation here is chosen so the
//! score itself is auditable.
//!
//! Reported as a diagnostic next to TAS and **not** folded into the composite
//! (uncalibrated weights distort realised metric influence; see the paper's
//! metric-evaluation section). Target for the pass flag: ≥ 0.70.
use serde::{Deserialize, Serialize};

use crate::types::{StepType, Trace};

pub const TARGET: f64 = 0.70;

/// Minimum literal length considered evidence-worthy. Shorter strings (and
/// 1-digit numbers) match everywhere by accident and only add noise.
const MIN_LITERAL_LEN: usize = 2;
const MIN_STRING_LITERAL_LEN: usize = 4;

/// One literal that could not be traced to prior evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UngroundedLiteral {
    pub step_id: u32,
    /// The literal, truncated to 80 chars for report hygiene.
    pub literal: String,
    /// `"action_param"` or `"claim"`.
    pub kind: String,
}

/// Aggregate AGF result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgfResult {
    /// Combined grounding score 0.0–1.0 (mean of the available components;
    /// 1.0 when the trace has neither checkable params nor claims).
    pub score: f64,
    /// Share of tool-call argument literals found in prior context.
    pub action_grounding: Option<f64>,
    /// Share of final-answer literals found in environment-provided text.
    pub claim_grounding: Option<f64>,
    /// Total literals checked across both components.
    pub checked_literals: usize,
    /// Every literal that failed the evidence check.
    pub ungrounded: Vec<UngroundedLiteral>,
    pub pass: bool,
    pub target: f64,
}

impl AgfResult {
    pub fn normalised(&self) -> f64 {
        self.score.clamp(0.0, 1.0)
    }
}

/// A quoted span that is regex/awk/shell syntax rather than a task string
/// (e.g. '[ =]', '{print $1}', '\\.conf$', '((25[0-5]...'). These are the
/// agent's own pattern language — flagging them as "ungrounded claims" is
/// pure reviewer noise.
fn quoted_span_is_syntax(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return true;
    }
    if t.starts_with(['[', '{', '^', '(', '$']) || t.contains('\\') {
        return true;
    }
    // Glob patterns ("*.jpeg") are search syntax, not claims about the world.
    if t.contains('*') || t.contains('?') {
        return true;
    }
    if t.contains("$1") || t.contains("$2") || t.contains("[:") || t.contains("{print") {
        return true;
    }
    // Mostly-punctuation spans are pattern syntax.
    let non_alnum = t
        .chars()
        .filter(|c| !c.is_alphanumeric() && *c != ' ')
        .count();
    non_alnum * 5 > t.chars().count() * 2 // > 40% punctuation
}

/// Extract checkable literals from a free-text command/query string:
/// quoted spans, path/file-like tokens, and multi-digit numbers. Shell/regex/
/// awk/glob *syntax* is excluded — patterns are how the agent searches, not
/// claims about the world.
fn text_literals(text: &str) -> Vec<String> {
    text_literals_inner(text, true)
}

/// `single_quotes`: treat `'` as a quote delimiter (true for code/commands,
/// false for prose, where apostrophes are contractions — "doesn't" must not
/// open a phantom span).
fn text_literals_inner(text: &str, single_quotes: bool) -> Vec<String> {
    let mut lits = Vec::new();
    let mut unquoted = String::new();
    let mut in_quote: Option<char> = None;
    let mut span = String::new();
    for c in text.chars() {
        match in_quote {
            Some(q) if c == q => {
                if span.trim().len() >= MIN_STRING_LITERAL_LEN && !quoted_span_is_syntax(&span) {
                    lits.push(span.clone());
                }
                span.clear();
                in_quote = None;
            }
            Some(_) => span.push(c),
            None if c == '"' || (single_quotes && c == '\'') => in_quote = Some(c),
            None => unquoted.push(c),
        }
    }
    if span.trim().len() >= MIN_STRING_LITERAL_LEN && !quoted_span_is_syntax(&span) {
        lits.push(span);
    }
    for tok in unquoted.split_whitespace() {
        // Trim sentence punctuation and markdown emphasis so prose like
        // "lines." or "**Cabin**" is not mistaken for a file-like literal;
        // inner dots (conf.yaml) survive.
        let t = tok.trim_matches(|c: char| ",;().:!?*`'\"".contains(c));
        if t.starts_with('-') || t.len() < MIN_LITERAL_LEN {
            continue;
        }
        // Shell/regex/glob syntax is the agent's pattern language, not a
        // claim: variables ($line), classes ([:digit:]), awk blocks, home
        // shorthand (~/), and glob patterns (*.conf) are all skipped.
        if t.starts_with(['$', '[', '{', '~']) || t.contains('*') || t.contains('?') {
            continue;
        }
        let pathish = t.contains('/') || (t.contains('.') && t.len() > 3);
        let numeric = t.chars().all(|c| c.is_ascii_digit());
        if pathish || numeric {
            lits.push(t.to_string());
        }
    }
    lits
}

/// Recursively pull literals out of a tool-params JSON value.
fn param_literals(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::String(s) => out.extend(text_literals(s)),
        serde_json::Value::Number(n) => {
            let s = n.to_string();
            if s.len() >= MIN_LITERAL_LEN {
                out.push(s);
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| param_literals(x, out)),
        serde_json::Value::Object(o) => o.values().for_each(|x| param_literals(x, out)),
        _ => {}
    }
}

/// Literals in a final answer worth evidencing: quoted spans and numbers.
fn claim_literals(text: &str) -> Vec<String> {
    let mut lits = text_literals_inner(text, false);
    // Also catch bare multi-digit numbers embedded in prose ("it is 172.").
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() {
            cur.push(c);
        } else {
            if cur.len() >= MIN_LITERAL_LEN {
                lits.push(cur.clone());
            }
            cur.clear();
        }
    }
    if cur.len() >= MIN_LITERAL_LEN {
        lits.push(cur);
    }
    lits.sort();
    lits.dedup();
    lits
}

/// Compute AGF for a trace. Deterministic; no similarity backend needed.
pub fn compute(trace: &Trace) -> AgfResult {
    // The task statement is evidence for both components.
    let task = trace
        .metadata
        .get("task")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();

    // Environment-provided text (for claim grounding): task + observations.
    let mut env_text = task.clone();
    // Full prior text (for action grounding): task + everything seen so far.
    let mut prior_text = task;

    let mut action_checked = 0usize;
    let mut action_grounded = 0usize;
    let mut claim_checked = 0usize;
    let mut claim_grounded = 0usize;
    let mut ungrounded = Vec::new();

    // Identify the final answer step: the last reasoning-typed step.
    let final_step_id = trace
        .steps
        .iter()
        .rev()
        .find(|s| s.step_type == StepType::Reasoning)
        .map(|s| s.id);

    for step in &trace.steps {
        // 0) This step's input_context arrived *before* the agent produced
        //    the step — it is evidence for the step's own params and claims
        //    (e.g. a zip code the user just stated).
        if let Some(ic) = &step.input_context {
            let low = ic.to_lowercase();
            prior_text.push('\n');
            prior_text.push_str(&low);
            env_text.push('\n');
            env_text.push_str(&low);
        }

        // 1) Action grounding: check this step's params against PRIOR text.
        //    Content-creation tools (edit/write/create/insert/...) are
        //    excluded: their params are the agent's *new artifact* — code it
        //    is writing — not references whose existence can be checked.
        let creates_content = step
            .tool_name
            .as_deref()
            .map(|n| {
                let n = n.to_lowercase();
                ["edit", "write", "create", "insert", "str_replace", "append"]
                    .iter()
                    .any(|p| n.contains(p))
            })
            .unwrap_or(false);
        if step.step_type == StepType::ToolCall && !creates_content {
            if let Some(params) = &step.tool_params {
                let mut lits = Vec::new();
                param_literals(params, &mut lits);
                lits.sort();
                lits.dedup();
                for lit in lits {
                    action_checked += 1;
                    if prior_text.contains(&lit.to_lowercase()) {
                        action_grounded += 1;
                    } else {
                        ungrounded.push(UngroundedLiteral {
                            step_id: step.id,
                            literal: lit.chars().take(80).collect(),
                            kind: "action_param".into(),
                        });
                    }
                }
            }
        }

        // 2) Claim grounding for the final answer, against ENVIRONMENT text
        //    accumulated so far (everything before this step).
        if Some(step.id) == final_step_id {
            for lit in claim_literals(&step.content) {
                claim_checked += 1;
                if env_text.contains(&lit.to_lowercase()) {
                    claim_grounded += 1;
                } else {
                    ungrounded.push(UngroundedLiteral {
                        step_id: step.id,
                        literal: lit.chars().take(80).collect(),
                        kind: "claim".into(),
                    });
                }
            }
        }

        // 3) Accumulate this step into the evidence pools for later steps.
        let mut own = String::new();
        own.push('\n');
        own.push_str(&step.content.to_lowercase());
        if let Some(out) = &step.output {
            own.push('\n');
            own.push_str(&out.to_lowercase());
            env_text.push('\n');
            env_text.push_str(&out.to_lowercase());
        }
        prior_text.push_str(&own);
    }

    let action = if action_checked > 0 {
        Some(action_grounded as f64 / action_checked as f64)
    } else {
        None
    };
    let claim = if claim_checked > 0 {
        Some(claim_grounded as f64 / claim_checked as f64)
    } else {
        None
    };
    let score = match (action, claim) {
        (Some(a), Some(c)) => (a + c) / 2.0,
        (Some(a), None) => a,
        (None, Some(c)) => c,
        (None, None) => 1.0,
    };
    let score = (score * 1000.0).round() / 1000.0;

    AgfResult {
        score,
        action_grounding: action.map(|v| (v * 1000.0).round() / 1000.0),
        claim_grounding: claim.map(|v| (v * 1000.0).round() / 1000.0),
        checked_literals: action_checked + claim_checked,
        ungrounded,
        pass: score >= TARGET,
        target: TARGET,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, Trace, TraceStep};
    use std::collections::HashMap;

    fn step(id: u32, ty: StepType, content: &str) -> TraceStep {
        TraceStep {
            id,
            step_type: ty,
            content: content.into(),
            tokens: 50,
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

    fn trace(steps: Vec<TraceStep>, task: Option<&str>) -> Trace {
        let mut metadata = HashMap::new();
        if let Some(t) = task {
            metadata.insert("task".to_string(), serde_json::json!(t));
        }
        Trace {
            trace_id: "agf-test".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            total_tokens: steps.iter().map(|s| s.tokens).sum(),
            steps,
            task_value_score: 1.0,
            metadata,
        }
    }

    #[test]
    fn grounded_params_and_claims_score_high() {
        // Path comes from the observation; answer number comes from the
        // observation — everything is evidence-backed.
        let mut find = step(1, StepType::ToolCall, "list the log files first");
        find.tool_name = Some("bash".into());
        find.tool_params = Some(serde_json::json!({"command": "ls /var/log"}));
        find.output = Some("/var/log/syslog\n/var/log/auth.log".into());

        let mut count = step(2, StepType::ToolCall, "count lines in the syslog");
        count.tool_name = Some("bash".into());
        count.tool_params = Some(serde_json::json!({"command": "wc -l /var/log/syslog"}));
        count.output = Some("4242 /var/log/syslog".into());

        let answer = step(3, StepType::Reasoning, "The syslog has 4242 lines.");

        let t = trace(
            vec![find, count, answer],
            Some("Count the lines in /var/log/syslog"),
        );
        let r = compute(&t);
        assert_eq!(r.action_grounding, Some(1.0), "{:?}", r.ungrounded);
        assert_eq!(r.claim_grounding, Some(1.0), "{:?}", r.ungrounded);
        assert!(r.pass);
    }

    #[test]
    fn fabricated_path_is_flagged_ungrounded() {
        // The agent invents a path never seen in task or observations.
        let mut cat = step(1, StepType::ToolCall, "read the secret config now");
        cat.tool_name = Some("bash".into());
        cat.tool_params = Some(serde_json::json!({"command": "cat /opt/imaginary/conf.yaml"}));
        cat.output = Some("nope".into());
        let t = trace(
            vec![cat, step(2, StepType::Reasoning, "Done.")],
            Some("Inspect the configuration"),
        );
        let r = compute(&t);
        assert_eq!(r.action_grounding, Some(0.0));
        assert!(r
            .ungrounded
            .iter()
            .any(|u| u.kind == "action_param" && u.literal.contains("/opt/imaginary")));
    }

    #[test]
    fn derived_numeric_claim_is_ungrounded_by_design() {
        // 95+25+52=172 computed by the agent: 172 never appears in any
        // observation, so the claim needs manual verification → ungrounded.
        let mut du = step(1, StepType::ToolCall, "get the sizes of the pdf files");
        du.tool_name = Some("bash".into());
        du.tool_params = Some(serde_json::json!({"command": "du -b /docs"}));
        du.output = Some("95 a.pdf\n25 b.pdf\n52 c.pdf".into());
        let answer = step(2, StepType::Reasoning, "Total size is 172 bytes.");
        let t = trace(vec![du, answer], Some("Total bytes of pdfs in /docs"));
        let r = compute(&t);
        assert_eq!(r.claim_grounding, Some(0.0));
        assert!(r
            .ungrounded
            .iter()
            .any(|u| u.kind == "claim" && u.literal == "172"));
    }

    #[test]
    fn no_params_no_claims_is_neutral() {
        let t = trace(
            vec![step(1, StepType::Reasoning, "pure thought, no facts")],
            None,
        );
        let r = compute(&t);
        assert_eq!(r.score, 1.0);
        assert!(r.pass);
        assert_eq!(r.checked_literals, 0);
    }

    #[test]
    fn deterministic_across_runs() {
        let mut s = step(1, StepType::ToolCall, "look in two places");
        s.tool_name = Some("bash".into());
        s.tool_params = Some(serde_json::json!({
            "command": "grep -r 'needle' /haystack /other/place"
        }));
        let t = trace(
            vec![s, step(2, StepType::Reasoning, "found 37 matches")],
            None,
        );
        let a = compute(&t);
        let b = compute(&t);
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }
}
