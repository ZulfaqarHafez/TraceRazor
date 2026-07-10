/// Tool Call Accuracy (TCA)
///
/// Measures how often the agent selects the correct tool on the first attempt
/// versus requiring retries or fallbacks.
///
/// Formula: TCA = (successful_first_attempt_calls / total_tool_calls) * 100
/// Detection: Find tool_call → error/failure → retry sequences.
/// Target: > 85%. Below 60% indicates poor tool descriptions.
use serde::{Deserialize, Serialize};

use crate::types::{StepFlag, StepType, Trace, TraceStep};

/// A detected tool misfire (failed first attempt requiring retry).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMisfire {
    /// The step ID of the failed tool call.
    pub failed_step: u32,
    /// The step ID of the retry (if detected).
    pub retry_step: Option<u32>,
    pub tool_name: String,
    /// Error message from the failed call.
    pub error: Option<String>,
    /// Estimated wasted tokens (failed call + retry).
    pub wasted_tokens: u32,
}

/// Result of the TCA metric computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcaResult {
    /// TCA as a percentage (0–100). Higher is better.
    pub score: f64,
    pub misfires: Vec<ToolMisfire>,
    pub successful_first_attempts: usize,
    pub total_tool_calls: usize,
    pub pass: bool,
    pub target: f64,
}

impl TcaResult {
    /// Normalised score for TAS composite (0.0–1.0, higher = better).
    pub fn normalised(&self) -> f64 {
        self.score / 100.0
    }
}

const TARGET_PERCENT: f64 = 85.0;

/// How many subsequent tool calls to scan for a same-tool retry. Allows a
/// diagnostic call or two between the failure and the corrected attempt
/// without attributing an unrelated tool as "the retry".
const RETRY_LOOKAHEAD: usize = 3;

/// Compute the TCA metric for a trace.
///
/// Detection algorithm:
/// 1. Every tool_call with `tool_success == false` or `tool_error` set is a
///    misfire (it failed on first attempt — score is charged regardless of
///    what happens next).
/// 2. The retry is the next call **to the same tool** within the next
///    `RETRY_LOOKAHEAD` tool calls. A follow-up call to a *different* tool
///    is a pivot, not a retry — it is neither flagged nor counted as wasted
///    tokens for this misfire.
/// 3. A tool call that has no success/error signal is optimistically treated
///    as success.
pub fn compute(trace: &Trace) -> TcaResult {
    let steps = &trace.steps;
    let total_tool_calls = steps
        .iter()
        .filter(|s| s.step_type == StepType::ToolCall)
        .count();

    let mut misfires: Vec<ToolMisfire> = Vec::new();

    let tool_steps: Vec<&TraceStep> = steps
        .iter()
        .filter(|s| s.step_type == StepType::ToolCall)
        .collect();

    for (i, step) in tool_steps.iter().enumerate() {
        let is_failed = step.tool_success == Some(false) || step.tool_error.is_some();
        if !is_failed {
            continue;
        }

        // The retry must target the same tool; anything else is a pivot.
        let retry = tool_steps[i + 1..]
            .iter()
            .take(RETRY_LOOKAHEAD)
            .find(|r| r.tool_name == step.tool_name)
            .copied();
        let retry_id = retry.map(|r| r.id);
        let wasted_tokens = step.tokens + retry.map(|r| r.tokens).unwrap_or(0);

        misfires.push(ToolMisfire {
            failed_step: step.id,
            retry_step: retry_id,
            tool_name: step.tool_name.clone().unwrap_or_default(),
            error: step.tool_error.clone(),
            wasted_tokens,
        });
    }

    let misfire_count = misfires.len();
    // Each misfire costs one tool call slot (the failed one).
    let successful_first_attempts = total_tool_calls.saturating_sub(misfire_count);

    let score = if total_tool_calls == 0 {
        100.0
    } else {
        (successful_first_attempts as f64 / total_tool_calls as f64) * 100.0
    };

    TcaResult {
        score: (score * 10.0).round() / 10.0,
        misfires,
        successful_first_attempts,
        total_tool_calls,
        pass: score >= TARGET_PERCENT,
        target: TARGET_PERCENT,
    }
}

/// Apply TCA flags to trace steps.
pub fn annotate_steps(steps: &mut [TraceStep], result: &TcaResult) {
    for misfire in &result.misfires {
        if let Some(step) = steps.iter_mut().find(|s| s.id == misfire.failed_step) {
            step.flags.push(StepFlag::Misfire);
            step.flag_details.push(match misfire.retry_step {
                Some(retry_id) => format!(
                    "wrong params for {}, retried at step {retry_id}",
                    misfire.tool_name
                ),
                None => format!(
                    "{} failed; no same-tool retry (agent pivoted or abandoned)",
                    misfire.tool_name
                ),
            });
        }
        if let Some(retry_id) = misfire.retry_step {
            if let Some(step) = steps.iter_mut().find(|s| s.id == retry_id) {
                step.flags.push(StepFlag::Retry);
                step.flag_details
                    .push(format!("correction of step {}", misfire.failed_step));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, TraceStep};
    use std::collections::HashMap;

    fn tool_step(id: u32, tool: &str, success: bool) -> TraceStep {
        TraceStep {
            id,
            step_type: StepType::ToolCall,
            content: format!("call {tool}"),
            tokens: 300,
            tool_name: Some(tool.to_string()),
            tool_params: Some(serde_json::json!({})),
            tool_success: Some(success),
            tool_error: if success {
                None
            } else {
                Some("missing required param".into())
            },
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
            content: "reasoning".into(),
            tokens: 200,
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
    fn test_perfect_tca() {
        let trace = Trace {
            trace_id: "t1".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                reason_step(1),
                tool_step(2, "get_order", true),
                tool_step(3, "check_refund", true),
                tool_step(4, "process_refund", true),
                reason_step(5),
            ],
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let result = compute(&trace);
        assert_eq!(result.score, 100.0);
        assert!(result.pass);
    }

    #[test]
    fn test_misfire_detected() {
        let trace = Trace {
            trace_id: "t2".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                reason_step(1),
                tool_step(2, "check_refund", false), // misfire
                tool_step(3, "check_refund", true),  // retry
                reason_step(4),
                reason_step(5),
            ],
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let result = compute(&trace);
        assert_eq!(result.misfires.len(), 1);
        assert_eq!(result.misfires[0].failed_step, 2);
        assert_eq!(result.misfires[0].retry_step, Some(3));
        assert!(!result.pass);
    }

    #[test]
    fn test_pivot_to_different_tool_is_not_a_retry() {
        // A failed search followed by a *different* tool is a pivot: the
        // failure is still a misfire, but the pivot step is neither the
        // retry nor charged as wasted tokens, and must not be flagged Retry.
        let trace = Trace {
            trace_id: "t3".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                reason_step(1),
                tool_step(2, "search_products", false), // misfire
                tool_step(3, "get_order", true),        // pivot, not retry
                reason_step(4),
                reason_step(5),
            ],
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let mut steps = trace.steps.clone();
        let result = compute(&trace);
        assert_eq!(result.misfires.len(), 1);
        assert_eq!(
            result.misfires[0].retry_step, None,
            "pivot misattributed as retry"
        );
        assert_eq!(
            result.misfires[0].wasted_tokens, 300,
            "pivot tokens must not be charged to the misfire"
        );
        annotate_steps(&mut steps, &result);
        assert!(
            !steps[2].flags.contains(&StepFlag::Retry),
            "pivot step must not carry the Retry flag"
        );
    }

    #[test]
    fn test_same_tool_retry_found_across_an_intervening_call() {
        // fail search -> diagnostic get_config -> retry search: the retry is
        // the same-tool call, not the intervening diagnostic.
        let trace = Trace {
            trace_id: "t4".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                tool_step(1, "search_products", false),
                tool_step(2, "get_config", true),
                tool_step(3, "search_products", true),
                reason_step(4),
                reason_step(5),
            ],
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let result = compute(&trace);
        assert_eq!(result.misfires.len(), 1);
        assert_eq!(result.misfires[0].retry_step, Some(3));
    }
}
