/// Reasoning Depth Appropriateness (RDA)
///
/// Evaluates whether reasoning depth matches task complexity.
/// Detects both overthinking (too deep for simple tasks) and
/// underthinking (too shallow for complex tasks).
///
/// Formula: RDA = 1 - |actual_depth - expected_depth| / max(actual_depth, expected_depth)
/// Score of 1.0 means perfectly calibrated. Target: > 0.75.
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::types::Trace;

/// Complexity tier and its expected step range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskComplexity {
    /// 1–2 steps expected (e.g., simple lookup, single-tool task).
    Trivial,
    /// 3–5 steps expected (e.g., 2-tool workflow, basic reasoning).
    Moderate,
    /// 6–10 steps expected (e.g., multi-step research, conditional branching).
    Complex,
    /// 10+ steps expected (e.g., multi-agent coordination, iterative refinement).
    Expert,
}

impl TaskComplexity {
    /// Midpoint of the expected step range for this complexity tier.
    pub fn expected_steps(&self) -> f64 {
        match self {
            TaskComplexity::Trivial => 1.5,
            TaskComplexity::Moderate => 4.0,
            TaskComplexity::Complex => 8.0,
            TaskComplexity::Expert => 12.0,
        }
    }

    pub fn parse(s: &str) -> Self {
        let lower = s.to_lowercase();
        if lower.contains("trivial") || lower.contains("simple") {
            TaskComplexity::Trivial
        } else if lower.contains("moderate") || lower.contains("medium") {
            TaskComplexity::Moderate
        } else if lower.contains("complex") {
            TaskComplexity::Complex
        } else if lower.contains("expert") || lower.contains("advanced") {
            TaskComplexity::Expert
        } else {
            TaskComplexity::Moderate
        }
    }
}

/// Result of the RDA metric computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdaResult {
    /// RDA score (0.0–1.0). Higher is better.
    pub score: f64,
    pub classified_complexity: TaskComplexity,
    pub expected_steps: f64,
    pub actual_steps: usize,
    pub pass: bool,
    pub target: f64,
}

impl RdaResult {
    pub fn normalised(&self) -> f64 {
        self.score
    }
}

const TARGET: f64 = 0.75;

/// Compute the RDA metric using an LLM to classify task complexity.
///
/// `llm_complete(system, user)` is an async closure injected from
/// `tracerazor-semantic` to keep core independent of HTTP clients.
pub async fn compute<F, Fut>(trace: &Trace, llm_complete: F) -> Result<RdaResult>
where
    F: Fn(String, String) -> Fut,
    Fut: std::future::Future<Output = Result<String>>,
{
    let actual_steps = trace.steps.len();
    let task_description = trace
        .steps
        .first()
        .map(|s| s.content.as_str())
        .unwrap_or("unknown task");

    let system = "\
You are a task complexity classifier for AI agent traces. \
Given a task description, classify its complexity as exactly one of: \
trivial, moderate, complex, or expert. \
Respond with ONLY the single word classification. \
trivial = 1-2 steps needed, moderate = 3-5, complex = 6-10, expert = 10+.";

    let user = format!(
        "Task description from agent trace: \"{}\"\nAgent: {}\nClassify complexity:",
        task_description, trace.agent_name
    );

    let response = llm_complete(system.to_string(), user).await?;
    let complexity = TaskComplexity::parse(response.trim());
    let expected = complexity.expected_steps();

    let rda = if actual_steps == 0 {
        0.0
    } else {
        let diff = (actual_steps as f64 - expected).abs();
        let max_val = (actual_steps as f64).max(expected);
        (1.0 - diff / max_val).max(0.0)
    };

    Ok(RdaResult {
        score: (rda * 1000.0).round() / 1000.0,
        classified_complexity: complexity,
        expected_steps: expected,
        actual_steps,
        pass: rda >= TARGET,
        target: TARGET,
    })
}

#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_rda_trivial_task_over_reasoned() {
        // Simulate: task classified as trivial (1.5 steps), but 11 steps used.
        let expected = 1.5_f64;
        let actual = 11_usize;
        let diff = (actual as f64 - expected).abs();
        let max_val = (actual as f64).max(expected);
        let rda = (1.0 - diff / max_val).max(0.0);
        assert!(rda < 0.75, "Over-reasoned trivial task should fail RDA");
    }

    #[tokio::test]
    async fn test_rda_complex_task_well_calibrated() {
        // Complex task (expected 8 steps), used 9 steps.
        let expected = 8.0_f64;
        let actual = 9_usize;
        let diff = (actual as f64 - expected).abs();
        let max_val = (actual as f64).max(expected);
        let rda = (1.0 - diff / max_val).max(0.0);
        assert!(rda >= 0.75, "Well-calibrated complex task should pass RDA");
    }
}
