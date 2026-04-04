/// Decision Branch Optimality (DBO)
///
/// At each decision point in the trace (conditional branching, tool selection,
/// strategy choice), evaluates whether the agent chose the most token-efficient
/// path that still leads to a correct outcome.
///
/// Formula: DBO = optimal_branch_selections / total_branch_points
/// Score of 1.0 means every decision was optimal.
/// Target: > 0.70. Improves over time as historical trace data accumulates.
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::types::Trace;

/// A decision point identified and evaluated by the LLM judge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchDecision {
    pub step_id: u32,
    /// Whether the LLM judge determined this was the optimal branch.
    pub was_optimal: bool,
    /// The judge's reasoning.
    pub reasoning: String,
}

/// Result of the DBO metric computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DboResult {
    /// DBO score (0.0–1.0). Higher is better.
    pub score: f64,
    pub optimal_selections: usize,
    pub total_branch_points: usize,
    pub decisions: Vec<BranchDecision>,
    pub pass: bool,
    pub target: f64,
}

impl DboResult {
    pub fn normalised(&self) -> f64 {
        self.score
    }
}

const TARGET: f64 = 0.70;

/// Compute the DBO metric using an LLM retrospective judge.
///
/// The judge is given a compact trace summary and asked to identify decision
/// points and evaluate whether each was optimal.
pub async fn compute<F, Fut>(trace: &Trace, llm_complete: F) -> Result<DboResult>
where
    F: Fn(String, String) -> Fut,
    Fut: std::future::Future<Output = Result<String>>,
{
    // Build a compact trace summary for the LLM.
    let trace_summary = build_summary(trace);

    let system = "\
You are an AI agent trace analyser specialising in decision branch optimality. \
Given a trace summary, identify each key decision point (tool selection, branching, \
strategy choice) and evaluate whether the agent took the most token-efficient path \
that still leads to a correct outcome. \
\
Respond in this exact JSON format (no markdown, raw JSON only): \
{\"branch_points\": [{\"step\": <id>, \"optimal\": true/false, \"reason\": \"<brief>\"}]}";

    let user = format!(
        "Agent: {}\nFramework: {}\nTrace:\n{}",
        trace.agent_name, trace.framework, trace_summary
    );

    let response = llm_complete(system.to_string(), user).await?;

    parse_dbo_response(&response, trace)
}

/// Summarise the trace compactly for the LLM judge.
fn build_summary(trace: &Trace) -> String {
    trace
        .steps
        .iter()
        .map(|s| {
            let tool = s
                .tool_name
                .as_deref()
                .map(|t| format!("[tool: {}]", t))
                .unwrap_or_default();
            let success = match s.tool_success {
                Some(true) => " ✓",
                Some(false) => " ✗",
                None => "",
            };
            format!(
                "Step {}: {} {} {}{}",
                s.id,
                s.step_type,
                tool,
                s.content.chars().take(80).collect::<String>(),
                success
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse the LLM's JSON response into DBO decisions.
fn parse_dbo_response(response: &str, trace: &Trace) -> Result<DboResult> {
    // Strip any markdown code fences the model might add.
    let cleaned = response
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    #[derive(serde::Deserialize)]
    struct LlmBranchPoints {
        branch_points: Vec<LlmBranch>,
    }

    #[derive(serde::Deserialize)]
    struct LlmBranch {
        step: u32,
        optimal: bool,
        reason: String,
    }

    let parsed: LlmBranchPoints = serde_json::from_str(cleaned)
        .map_err(|e| anyhow::anyhow!("Failed to parse DBO response: {} — raw: {}", e, cleaned))?;

    let decisions: Vec<BranchDecision> = parsed
        .branch_points
        .into_iter()
        .map(|b| BranchDecision {
            step_id: b.step,
            was_optimal: b.optimal,
            reasoning: b.reason,
        })
        .collect();

    let total = decisions.len();
    let optimal = decisions.iter().filter(|d| d.was_optimal).count();

    let score = if total == 0 {
        // No branch points identified — default to passing (no decisions = no bad decisions).
        1.0
    } else {
        optimal as f64 / total as f64
    };

    Ok(DboResult {
        score: (score * 1000.0).round() / 1000.0,
        optimal_selections: optimal,
        total_branch_points: total,
        decisions,
        pass: score >= TARGET,
        target: TARGET,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dbo_response_valid() {
        use crate::types::{StepType, Trace, TraceStep};
        use std::collections::HashMap;

        let trace = Trace {
            trace_id: "t1".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![TraceStep {
                id: 1,
                step_type: StepType::Reasoning,
                content: "parse".into(),
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
            }],
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };

        let json = r#"{"branch_points": [{"step": 1, "optimal": true, "reason": "correct tool"}, {"step": 3, "optimal": false, "reason": "could have skipped"}]}"#;
        let result = parse_dbo_response(json, &trace).unwrap();
        assert_eq!(result.total_branch_points, 2);
        assert_eq!(result.optimal_selections, 1);
        assert!((result.score - 0.5).abs() < 0.01);
        assert!(!result.pass);
    }
}
