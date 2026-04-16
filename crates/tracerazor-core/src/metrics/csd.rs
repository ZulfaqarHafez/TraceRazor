/// Cross-Step Semantic Drift (M4)
///
/// Measures semantic continuity between consecutive reasoning steps.
/// High drift = agent is wandering. This is the complement to SRR (which flags
/// re-visiting the same ground — CSD flags leaving necessary ground without resolution).
///
/// For each consecutive pair of reasoning steps, compute cosine similarity.
/// Score = mean(similarities) ∈ [0,1], where 1.0 = perfect semantic continuity.
///
/// Requires Phase 2 embedding backend (similarity_fn).
use serde::{Deserialize, Serialize};

use crate::types::{Trace, StepType};

pub const TARGET: f64 = 0.60;
const HIGH_DRIFT_THRESHOLD: f64 = 0.30;

/// Per-step result for a single consecutive pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsdStepResult {
    /// Source step ID (step i).
    pub step_id_from: u32,
    /// Target step ID (step i+1).
    pub step_id_to: u32,
    /// Cosine similarity between consecutive steps, 0.0–1.0.
    /// Higher = more semantic continuity, lower = more drift.
    pub similarity: f64,
    /// True when similarity < HIGH_DRIFT_THRESHOLD (0.30).
    pub high_drift: bool,
}

/// Aggregate semantic drift result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsdResult {
    /// Mean consecutive-pair similarity, 0.0–1.0.
    /// Higher is better (lower drift = agent stays on topic).
    pub score: f64,
    /// Per-pair details.
    pub step_results: Vec<CsdStepResult>,
    /// Pairs flagged as high drift (similarity < 0.30).
    pub high_drift_pairs: Vec<(u32, u32)>,
    /// True when score >= TARGET (0.60).
    pub pass: bool,
    /// Target minimum score to pass.
    pub target: f64,
}

impl CsdResult {
    pub fn normalised(&self) -> f64 {
        self.score.clamp(0.0, 1.0)
    }
}

/// Compute cross-step semantic drift for a trace.
///
/// `similarity_fn` is the same BoW function used by the main analysis engine.
/// Passing `|_, _| 0.0` disables semantic comparisons (useful for fast estimates).
pub fn compute<F>(trace: &Trace, similarity_fn: F) -> CsdResult
where
    F: Fn(&str, &str) -> f64,
{
    // Filter to reasoning steps only (same as GAR).
    let reasoning_steps: Vec<_> = trace
        .steps
        .iter()
        .filter(|s| s.step_type == StepType::Reasoning)
        .collect();

    // Fewer than 2 reasoning steps → no pairs possible.
    if reasoning_steps.len() < 2 {
        return CsdResult {
            score: 1.0,
            step_results: vec![],
            high_drift_pairs: vec![],
            pass: true,
            target: TARGET,
        };
    }

    // Compute similarities for consecutive pairs.
    let mut step_results = Vec::new();
    let mut similarities = Vec::new();
    let mut high_drift_pairs = Vec::new();

    for i in 0..reasoning_steps.len() - 1 {
        let from_step = reasoning_steps[i];
        let to_step = reasoning_steps[i + 1];

        let sim = similarity_fn(&from_step.content, &to_step.content)
            .clamp(0.0, 1.0);

        let sim_rounded = (sim * 1000.0).round() / 1000.0;

        let is_high_drift = sim < HIGH_DRIFT_THRESHOLD;
        if is_high_drift {
            high_drift_pairs.push((from_step.id, to_step.id));
        }

        step_results.push(CsdStepResult {
            step_id_from: from_step.id,
            step_id_to: to_step.id,
            similarity: sim_rounded,
            high_drift: is_high_drift,
        });

        similarities.push(sim);
    }

    // Score = mean consecutive-pair similarity.
    let score = if similarities.is_empty() {
        1.0
    } else {
        let sum: f64 = similarities.iter().sum();
        sum / similarities.len() as f64
    };

    let score_rounded = (score * 1000.0).round() / 1000.0;

    CsdResult {
        score: score_rounded,
        step_results,
        high_drift_pairs,
        pass: score_rounded >= TARGET,
        target: TARGET,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, TraceStep};
    use std::collections::HashMap;

    fn step(id: u32, content: &str, step_type: StepType) -> TraceStep {
        TraceStep {
            id,
            step_type,
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

    fn make_trace(steps: Vec<TraceStep>) -> Trace {
        Trace {
            trace_id: "t1".into(),
            agent_name: "test".into(),
            framework: "raw".into(),
            steps,
            total_tokens: 1000,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    fn const_sim(v: f64) -> impl Fn(&str, &str) -> f64 {
        move |_, _| v
    }

    #[test]
    fn single_step_returns_perfect_score() {
        let trace = make_trace(vec![
            step(1, "step 1", StepType::Reasoning),
        ]);
        let result = compute(&trace, |_, _| 0.5);
        assert_eq!(result.score, 1.0);
        assert!(result.pass);
        assert!(result.step_results.is_empty());
    }

    #[test]
    fn two_identical_steps_score_one() {
        let trace = make_trace(vec![
            step(1, "content", StepType::Reasoning),
            step(2, "content", StepType::Reasoning),
        ]);
        let result = compute(&trace, |a, b| if a == b { 1.0 } else { 0.0 });
        assert_eq!(result.score, 1.0);
        assert!(result.pass);
        assert_eq!(result.step_results.len(), 1);
        assert_eq!(result.step_results[0].similarity, 1.0);
        assert!(!result.step_results[0].high_drift);
    }

    #[test]
    fn all_high_similarity_passes() {
        let trace = make_trace(vec![
            step(1, "step 1", StepType::Reasoning),
            step(2, "step 2", StepType::Reasoning),
            step(3, "step 3", StepType::Reasoning),
        ]);
        let result = compute(&trace, const_sim(0.8));
        assert_eq!(result.score, 0.8);
        assert!(result.pass); // 0.8 >= 0.60
        assert_eq!(result.step_results.len(), 2);
        assert!(!result.high_drift_pairs.is_empty() == false); // no high drift pairs
    }

    #[test]
    fn all_low_similarity_fails() {
        let trace = make_trace(vec![
            step(1, "step 1", StepType::Reasoning),
            step(2, "step 2", StepType::Reasoning),
        ]);
        let result = compute(&trace, const_sim(0.2));
        assert_eq!(result.score, 0.2);
        assert!(!result.pass); // 0.2 < 0.60
        assert_eq!(result.step_results.len(), 1);
        assert!(result.step_results[0].high_drift);
        assert_eq!(result.high_drift_pairs, vec![(1, 2)]);
    }

    #[test]
    fn mixed_pairs_averages_correctly() {
        let trace = make_trace(vec![
            step(1, "step1", StepType::Reasoning),
            step(2, "step2", StepType::Reasoning),
            step(3, "step3", StepType::Reasoning),
            step(4, "step4", StepType::Reasoning),
        ]);
        // Use a similarity function that gives different values based on content patterns
        let result = compute(&trace, |a, b| {
            match (a, b) {
                ("step1", "step2") => 0.4,
                ("step2", "step3") => 0.6,
                ("step3", "step4") => 0.8,
                _ => 0.0,
            }
        });
        // Mean = (0.4 + 0.6 + 0.8) / 3 = 0.6
        assert_eq!(result.score, 0.6);
        assert!(result.pass);
        assert_eq!(result.step_results.len(), 3);
    }

    #[test]
    fn high_drift_pairs_identified() {
        let trace = make_trace(vec![
            step(1, "reasoninga", StepType::Reasoning),
            step(2, "reasoningb", StepType::Reasoning),
            step(3, "reasoningc", StepType::Reasoning),
            step(4, "reasoningd", StepType::Reasoning),
        ]);
        // Pairs: (1,2) = 0.25 (drift), (2,3) = 0.5 (ok), (3,4) = 0.1 (drift)
        let result = compute(&trace, |a, b| {
            match (a, b) {
                ("reasoninga", "reasoningb") => 0.25,
                ("reasoningb", "reasoningc") => 0.5,
                ("reasoningc", "reasoningd") => 0.1,
                _ => 0.0,
            }
        });
        assert_eq!(
            result.high_drift_pairs,
            vec![(1, 2), (3, 4)]
        );
    }

    #[test]
    fn non_reasoning_steps_ignored() {
        let trace = make_trace(vec![
            step(1, "reasoning 1", StepType::Reasoning),
            step(2, "tool call", StepType::ToolCall),
            step(3, "reasoning 2", StepType::Reasoning),
        ]);
        let result = compute(&trace, |_, _| 0.7);
        // Only one reasoning pair: (1, 3)
        assert_eq!(result.step_results.len(), 1);
        assert_eq!(result.step_results[0].step_id_from, 1);
        assert_eq!(result.step_results[0].step_id_to, 3);
    }

    #[test]
    fn sim_clamped_to_unit_interval() {
        let trace = make_trace(vec![
            step(1, "a", StepType::Reasoning),
            step(2, "b", StepType::Reasoning),
        ]);
        let result = compute(&trace, |_, _| 1.5); // out-of-range
        assert_eq!(result.score, 1.0); // clamped
        assert_eq!(result.step_results[0].similarity, 1.0);
    }

    #[test]
    fn normalised_returns_score() {
        let trace = make_trace(vec![
            step(1, "a", StepType::Reasoning),
            step(2, "b", StepType::Reasoning),
        ]);
        let result = compute(&trace, const_sim(0.7));
        assert_eq!(result.normalised(), result.score);
        assert_eq!(result.normalised(), 0.7);
    }

    #[test]
    fn rounding_to_three_decimals() {
        let trace = make_trace(vec![
            step(1, "a", StepType::Reasoning),
            step(2, "b", StepType::Reasoning),
        ]);
        let result = compute(&trace, |_, _| 0.123456);
        assert_eq!(result.score, 0.123);
        assert_eq!(result.step_results[0].similarity, 0.123);
    }

    #[test]
    fn empty_trace_returns_perfect() {
        let trace = make_trace(vec![]);
        let result = compute(&trace, |_, _| 0.0);
        assert_eq!(result.score, 1.0);
        assert!(result.pass);
    }
}
