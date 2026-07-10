//! Experimental, diagnostic context-accumulation features.
//!
//! Motivation: calibrating the 13 composite metrics against real recoverable
//! token waste (tau-bench before/after pairs) gave a negative cross-validated
//! R^2 — the metrics measure within-trace reasoning redundancy, but the dominant
//! recoverable cost is *context accumulation*: tool observations that are
//! verbose, redundant, or stale (see AgentDiet, arXiv 2509.23586). These
//! candidate features measure that, from per-step fields the parsers already
//! populate (`output`, `input_context`, `tool_params`, `tokens`).
//!
//! They are emitted on `TraceReport.features` next to the score, but are **not**
//! part of the TAS composite. The calibration tool can test their predictiveness
//! on real data; a feature is promoted to a weighted metric only if it
//! demonstrably helps. Values are raw (not direction-normalised) scalars in a
//! documented range; calibration learns the appropriate sign.

use std::collections::BTreeMap;
use std::io::Write;

use flate2::write::GzEncoder;
use flate2::Compression;

use crate::types::{StepType, Trace, TraceStep};

/// Below this many bytes, gzip header overhead dominates and the ratio is noisy.
const MIN_GZIP_BYTES: usize = 128;
/// Minimum observation length (chars) considered for stale-retention matching.
const MIN_OBS_MATCH_LEN: usize = 40;
/// A tool observation is "stale" if it is still present in the context of a step
/// at least this many positions later.
const STALE_LOOKBACK: usize = 2;
/// Minimum fraction of steps that must carry `input_context` for the
/// context-dependent features to be emitted at all.
const CTX_COVERAGE_MIN: f64 = 0.5;

fn is_tool(s: &TraceStep) -> bool {
    s.step_type == StepType::ToolCall || s.tool_name.is_some()
}

/// Observation text for a tool step: the tool result (`output`) when present,
/// else the step content (some parsers put the result there).
fn obs_text(s: &TraceStep) -> &str {
    s.output.as_deref().unwrap_or(s.content.as_str())
}

/// Rough token estimate for text without its own token count (~1.3 tokens/word),
/// matching the estimator used by the calibration message connector.
fn approx_tokens(text: &str) -> f64 {
    text.split_whitespace().count() as f64 * 1.3
}

/// gzip compressibility of `text` in [0,1] (`1 - compressed/original`); 0.0 for
/// text below the reliable-size floor.
fn gzip_compressibility(text: &str) -> f64 {
    let bytes = text.as_bytes();
    if bytes.len() < MIN_GZIP_BYTES {
        return 0.0;
    }
    let mut enc = GzEncoder::new(Vec::with_capacity(bytes.len()), Compression::default());
    if enc.write_all(bytes).is_err() {
        return 0.0;
    }
    let Ok(compressed) = enc.finish() else {
        return 0.0;
    };
    (1.0 - compressed.len() as f64 / bytes.len() as f64).clamp(0.0, 1.0)
}

/// Compute the candidate feature map for a trace. Keys are stable snake_case
/// strings; values are in \[0,1\]. Context-dependent features (`stale_*`,
/// `context_growth_*`) are omitted entirely when `input_context` coverage is too
/// low for them to be meaningful, so calibration can drop ragged samples.
pub fn compute(trace: &Trace) -> BTreeMap<String, f64> {
    let mut f = BTreeMap::new();
    let steps = &trace.steps;
    let n = steps.len();
    if n == 0 {
        return f;
    }

    let total_tokens: f64 = steps.iter().map(|s| s.tokens as f64).sum::<f64>().max(1.0);
    let tool_steps: Vec<&TraceStep> = steps.iter().filter(|s| is_tool(s)).collect();

    // 1. Observation token share: fraction of tokens spent on tool I/O.
    let obs_tokens: f64 = tool_steps.iter().map(|s| s.tokens as f64).sum();
    f.insert(
        "obs_token_share".into(),
        (obs_tokens / total_tokens).clamp(0.0, 1.0),
    );

    // 2. Observation compressibility: how redundant the tool outputs are.
    let joined: String = tool_steps
        .iter()
        .map(|s| obs_text(s))
        .collect::<Vec<_>>()
        .join("\n");
    f.insert(
        "obs_gzip_compressibility".into(),
        gzip_compressibility(&joined),
    );

    // 5. Redundant tool-call rate: identical (type, tool, params) re-issued.
    if !tool_steps.is_empty() {
        let mut seen = std::collections::HashMap::<String, u32>::new();
        let mut repeats = 0u32;
        for s in &tool_steps {
            let c = seen.entry(s.state_hash()).or_insert(0);
            if *c > 0 {
                repeats += 1;
            }
            *c += 1;
        }
        f.insert(
            "redundant_tool_call_rate".into(),
            (repeats as f64 / tool_steps.len() as f64).clamp(0.0, 1.0),
        );

        // 6. Repeated-observation rate (token-weighted): identical tool outputs
        // re-entering the trace.
        let mut obs_seen = std::collections::HashMap::<String, u32>::new();
        let mut dup_tokens = 0f64;
        for s in &tool_steps {
            let key = obs_text(s).trim().to_string();
            if key.is_empty() {
                continue;
            }
            let c = obs_seen.entry(key).or_insert(0);
            if *c > 0 {
                dup_tokens += s.tokens as f64;
            }
            *c += 1;
        }
        f.insert(
            "repeated_obs_rate".into(),
            (dup_tokens / total_tokens).clamp(0.0, 1.0),
        );
    }

    // ── Path / length structure (the cross-run token delta is driven mostly by
    // how long the trajectory is and how it is shaped). All token-agnostic. ──
    // Trajectory length, soft-capped so it stays in [0,1].
    f.insert("step_count_norm".into(), (n as f64 / 60.0).clamp(0.0, 1.0));
    // Mean tokens per step (verbose steps), soft-capped.
    f.insert(
        "mean_step_tokens_norm".into(),
        ((total_tokens / n as f64) / 500.0).clamp(0.0, 1.0),
    );
    // Longest run of consecutive reasoning steps (rambling without acting).
    let mut max_run = 0usize;
    let mut run = 0usize;
    for s in steps {
        if is_tool(s) {
            run = 0;
        } else {
            run += 1;
            max_run = max_run.max(run);
        }
    }
    f.insert(
        "reasoning_run_max".into(),
        (max_run as f64 / n as f64).clamp(0.0, 1.0),
    );
    // Fraction of steps whose (type, tool, params) state repeats an earlier one
    // (path revisiting / churn), token-agnostic and over all step types.
    let mut seen_state = std::collections::HashSet::<String>::new();
    let mut revisits = 0usize;
    for s in steps {
        if !seen_state.insert(s.state_hash()) {
            revisits += 1;
        }
    }
    f.insert(
        "revisit_rate".into(),
        (revisits as f64 / n as f64).clamp(0.0, 1.0),
    );
    // Tool diversity: distinct tools / tool calls (low = repetitive tool use).
    if !tool_steps.is_empty() {
        let uniq: std::collections::HashSet<&str> = tool_steps
            .iter()
            .filter_map(|s| s.tool_name.as_deref())
            .collect();
        f.insert(
            "tool_diversity".into(),
            (uniq.len() as f64 / tool_steps.len() as f64).clamp(0.0, 1.0),
        );
    }

    // Context-dependent features: only when input_context is broadly populated.
    let ctx_cov = steps.iter().filter(|s| s.input_context.is_some()).count() as f64 / n as f64;
    if ctx_cov >= CTX_COVERAGE_MIN {
        // 3. Stale observation retention: tokens of tool observations that still
        // appear in the context of a step >= STALE_LOOKBACK positions later.
        let mut stale_tokens = 0f64;
        for (j, s) in steps.iter().enumerate() {
            if !is_tool(s) {
                continue;
            }
            let o = obs_text(s).trim();
            if o.len() < MIN_OBS_MATCH_LEN {
                continue;
            }
            let retained = steps.iter().skip(j + STALE_LOOKBACK + 1).any(|later| {
                later
                    .input_context
                    .as_deref()
                    .is_some_and(|ic| ic.contains(o))
            });
            if retained {
                stale_tokens += s.tokens as f64;
            }
        }
        f.insert(
            "stale_obs_retention".into(),
            (stale_tokens / total_tokens).clamp(0.0, 1.0),
        );

        // 4. Context growth: fraction of the final context that is growth over
        // the run (0 = flat context, ->1 = context balloons).
        let ctx_first = steps
            .iter()
            .find_map(|s| s.input_context.as_deref().map(approx_tokens))
            .unwrap_or(0.0);
        let ctx_last = steps
            .iter()
            .rev()
            .find_map(|s| s.input_context.as_deref().map(approx_tokens))
            .unwrap_or(0.0);
        let growth = if ctx_last > 0.0 {
            ((ctx_last - ctx_first) / ctx_last).clamp(0.0, 1.0)
        } else {
            0.0
        };
        f.insert("context_growth".into(), growth);
    }

    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, Trace, TraceStep};
    use std::collections::HashMap;

    fn step(id: u32, st: StepType, content: &str, tokens: u32) -> TraceStep {
        let tool = st == StepType::ToolCall;
        TraceStep {
            id,
            step_type: st,
            content: content.to_string(),
            tokens,
            tool_name: if tool { Some("t".into()) } else { None },
            tool_params: None,
            tool_success: if tool { Some(true) } else { None },
            tool_error: None,
            agent_id: None,
            input_context: None,
            output: None,
            flags: vec![],
            flag_details: vec![],
        }
    }

    fn trace(steps: Vec<TraceStep>) -> Trace {
        Trace {
            trace_id: "t".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps,
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn obs_share_zero_for_all_reasoning() {
        let t = trace(vec![
            step(1, StepType::Reasoning, "think", 100),
            step(2, StepType::Reasoning, "think more", 100),
        ]);
        assert_eq!(compute(&t)["obs_token_share"], 0.0);
    }

    #[test]
    fn obs_share_half_for_even_split() {
        let t = trace(vec![
            step(1, StepType::Reasoning, "think", 100),
            step(2, StepType::ToolCall, "call", 100),
        ]);
        assert!((compute(&t)["obs_token_share"] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn redundant_call_rate_detects_repeat() {
        let mut a = step(1, StepType::ToolCall, "call", 50);
        let mut b = step(2, StepType::ToolCall, "call", 50);
        a.tool_name = Some("get".into());
        b.tool_name = Some("get".into());
        let t = trace(vec![a, b]);
        assert!(compute(&t)["redundant_tool_call_rate"] > 0.0);
    }

    #[test]
    fn repeated_obs_rate_detects_identical_output() {
        let mut a = step(1, StepType::ToolCall, "x", 50);
        let mut b = step(2, StepType::ToolCall, "x", 50);
        a.output = Some("identical result payload".into());
        b.output = Some("identical result payload".into());
        let t = trace(vec![a, b]);
        assert!(compute(&t)["repeated_obs_rate"] > 0.0);
    }

    #[test]
    fn context_features_absent_without_input_context() {
        let t = trace(vec![
            step(1, StepType::Reasoning, "a", 100),
            step(2, StepType::ToolCall, "b", 100),
        ]);
        let f = compute(&t);
        assert!(!f.contains_key("stale_obs_retention"));
        assert!(!f.contains_key("context_growth"));
    }

    #[test]
    fn all_features_finite_in_range() {
        let t = trace(vec![
            step(1, StepType::Reasoning, "a", 100),
            step(2, StepType::ToolCall, "b", 100),
            step(3, StepType::ToolCall, "b", 100),
        ]);
        for (k, v) in compute(&t) {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "{k}={v} out of range"
            );
        }
    }
}
