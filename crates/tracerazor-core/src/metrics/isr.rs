/// Information Sufficiency Rate (ISR)
///
/// Measures whether each reasoning step introduces new, task-relevant
/// information versus restating or paraphrasing prior context.
///
/// Formula: ISR = (steps_with_novel_info / total_steps) * 100
/// Detection: For each step, compute cosine distance from the union of
///            all prior steps using embeddings. Steps with < 10% novel
///            information content are flagged.
/// Target: > 80%. An ISR below 60% suggests the agent is verbose.
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::types::Trace;

/// A step identified as lacking novel information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowNoveltyStep {
    pub step_id: u32,
    /// Information gain (0.0 = entirely redundant, 1.0 = completely novel).
    pub novelty_score: f64,
}

/// Result of the ISR metric computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsrResult {
    /// ISR as a percentage (0–100). Higher is better.
    pub score: f64,
    pub steps_with_novel_info: usize,
    pub total_steps: usize,
    pub low_novelty_steps: Vec<LowNoveltyStep>,
    pub pass: bool,
    pub target: f64,
}

impl IsrResult {
    pub fn normalised(&self) -> f64 {
        self.score / 100.0
    }
}

const TARGET: f64 = 80.0;
/// Steps with less than this information gain are flagged.
const NOVELTY_THRESHOLD: f64 = 0.10;

/// Compute ISR using pre-computed embeddings.
///
/// `similarities` is a function that returns the cosine similarity between
/// step i's text and step j's text (injected from the semantic crate).
/// This keeps core independent of the HTTP client.
///
/// For each step i, information gain = 1 - max(similarity(i, j)) for all j < i.
/// High max similarity = low novelty.
pub fn compute_from_similarities(
    trace: &Trace,
    similarity_fn: impl Fn(&str, &str) -> f64,
) -> IsrResult {
    let steps = &trace.steps;
    let total = steps.len();

    let mut novel_count = 0usize;
    let mut low_novelty: Vec<LowNoveltyStep> = Vec::new();

    for i in 0..steps.len() {
        if i == 0 {
            // First step is always novel by definition.
            novel_count += 1;
            continue;
        }

        let curr_text = steps[i].semantic_content();

        // Compute max similarity to any prior step.
        let max_sim = steps[..i]
            .iter()
            .map(|prev| similarity_fn(&curr_text, &prev.semantic_content()))
            .fold(0.0_f64, f64::max);

        // Information gain = 1 - max_similarity
        let novelty = 1.0 - max_sim;

        if novelty >= NOVELTY_THRESHOLD {
            novel_count += 1;
        } else {
            low_novelty.push(LowNoveltyStep {
                step_id: steps[i].id,
                novelty_score: (novelty * 100.0).round() / 100.0,
            });
        }
    }

    let score = if total == 0 {
        100.0
    } else {
        (novel_count as f64 / total as f64) * 100.0
    };

    IsrResult {
        score: (score * 10.0).round() / 10.0,
        steps_with_novel_info: novel_count,
        total_steps: total,
        low_novelty_steps: low_novelty,
        pass: score >= TARGET,
        target: TARGET,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, Trace, TraceStep};
    use std::collections::HashMap;

    fn make_step(id: u32, content: &str) -> TraceStep {
        TraceStep {
            id,
            step_type: StepType::Reasoning,
            content: content.to_string(),
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
    fn test_isr_all_novel() {
        let trace = Trace {
            trace_id: "t1".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                make_step(1, "parse user request about refund"),
                make_step(2, "fetch order from database"),
                make_step(3, "check refund policy eligibility"),
                make_step(4, "process the refund transaction"),
                make_step(5, "send confirmation email to customer"),
            ],
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };

        // Steps have different content → novelty > 0.10 for all.
        let result = compute_from_similarities(&trace, |a, b| {
            let shared = a.split_whitespace()
                .filter(|w| b.contains(*w))
                .count();
            let total = a.split_whitespace().count().max(1);
            shared as f64 / total as f64
        });
        assert!(result.pass, "All-novel trace should pass ISR");
    }

    #[test]
    fn test_isr_repetitive() {
        let repeated = "parse the user request about refund order details";
        let trace = Trace {
            trace_id: "t2".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                make_step(1, repeated),
                make_step(2, "fetch order data"),
                make_step(3, repeated), // exact repeat
                make_step(4, "process refund"),
                make_step(5, repeated), // exact repeat
            ],
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let result = compute_from_similarities(&trace, |a, b| if a == b { 1.0 } else { 0.0 });
        // Steps 3 and 5 are exact repeats → ISR = 3/5 = 60%
        assert!(!result.pass, "Repetitive trace should fail ISR");
    }
}
