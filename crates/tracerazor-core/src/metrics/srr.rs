/// Step Redundancy Rate (SRR)
///
/// Measures the percentage of reasoning steps that are semantically redundant.
/// Uses cosine similarity on bag-of-words vectors; pairs above the threshold
/// are flagged. Three confidence tiers: High (≥0.95), Medium (0.85–0.94), Low (0.75–0.84).
///
/// Target: SRR < 15%. Traces above 30% are flagged critical.
use serde::{Deserialize, Serialize};

use crate::types::{Confidence, StepFlag, Trace, TraceStep};

/// A detected redundant step pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SrrRedundantPair {
    pub step_a: u32,
    pub step_b: u32,
    pub similarity: f64,
    pub confidence: Confidence,
}

/// Result of the SRR metric computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SrrResult {
    /// SRR as a percentage (0–100). Lower is better.
    pub score: f64,
    pub redundant_steps: Vec<SrrRedundantPair>,
    /// Number of redundant steps (step_b side of each pair).
    pub redundant_count: usize,
    pub total_steps: usize,
    pub pass: bool,
    /// Target: below this percentage.
    pub target: f64,
}

impl SrrResult {
    /// Normalised score for TAS composite (0.0–1.0, higher = better).
    pub fn normalised(&self) -> f64 {
        (1.0 - self.score / 100.0).max(0.0)
    }
}

/// Default similarity threshold for redundancy flagging.
///
/// NOTE: This threshold is calibrated for the Phase 1 bag-of-words similarity backend.
/// Phase 2 will use ONNX all-MiniLM-L6-v2 sentence embeddings, which produce
/// higher-fidelity similarity scores and use the PRD's 0.85 threshold.
/// BoW at 0.65 ≈ sentence-embedding at 0.85 for near-duplicate step detection.
pub const DEFAULT_THRESHOLD: f64 = 0.65;
/// High confidence threshold (BoW equivalent of sentence-embedding 0.95).
pub const HIGH_CONFIDENCE: f64 = 0.85;
/// Low confidence lower bound (shown in verbose mode only).
pub const LOW_CONFIDENCE: f64 = 0.55;
/// Target: SRR below this percentage.
pub const TARGET_PERCENT: f64 = 15.0;
/// Critical flag threshold.
pub const CRITICAL_PERCENT: f64 = 30.0;

/// Compute the SRR metric for a trace.
///
/// `similarity_fn` is a closure that takes two step text strings and returns
/// a cosine similarity score (0.0–1.0). This is injected so the metric crate
/// remains independent of the embedding backend.
pub fn compute<F>(trace: &Trace, similarity_fn: F, threshold: Option<f64>) -> SrrResult
where
    F: Fn(&str, &str) -> f64,
{
    let threshold = threshold.unwrap_or(DEFAULT_THRESHOLD);
    let steps = &trace.steps;
    let total = steps.len();

    let mut pairs: Vec<SrrRedundantPair> = Vec::new();
    let mut redundant_step_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();

    // Compare every step against all prior steps.
    for i in 1..steps.len() {
        let curr = &steps[i];
        let curr_text = curr.semantic_content();

        for prev in steps.iter().take(i) {
            let prev_text = prev.semantic_content();

            let sim = similarity_fn(&curr_text, &prev_text);

            // Only flag at or above the low confidence bound.
            if sim >= LOW_CONFIDENCE {
                let confidence = if sim >= HIGH_CONFIDENCE {
                    Confidence::High
                } else if sim >= threshold {
                    Confidence::Medium
                } else {
                    Confidence::Low
                };

                // Only record as a hard redundancy at/above the main threshold.
                if sim >= threshold {
                    redundant_step_ids.insert(curr.id);
                    pairs.push(SrrRedundantPair {
                        step_a: prev.id,
                        step_b: curr.id,
                        similarity: (sim * 100.0).round() / 100.0,
                        confidence,
                    });
                    // Only flag the most similar prior step.
                    break;
                }
            }
        }
    }

    // Deduplicate: count each step as redundant at most once.
    let redundant_count = redundant_step_ids.len();
    let score = if total == 0 {
        0.0
    } else {
        (redundant_count as f64 / total as f64) * 100.0
    };

    SrrResult {
        score: (score * 10.0).round() / 10.0,
        redundant_steps: pairs,
        redundant_count,
        total_steps: total,
        pass: score < TARGET_PERCENT,
        target: TARGET_PERCENT,
    }
}

/// Apply SRR flags to the trace steps (mutates the flag lists in place).
pub fn annotate_steps(steps: &mut [TraceStep], result: &SrrResult) {
    for pair in &result.redundant_steps {
        if let Some(step) = steps.iter_mut().find(|s| s.id == pair.step_b) {
            step.flags.push(StepFlag::Redundant);
            step.flag_details.push(format!(
                "{}: {:.0}% sim w/ step {}",
                pair.confidence,
                pair.similarity * 100.0,
                pair.step_a
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, TraceStep};

    fn make_trace(contents: &[&str]) -> Trace {
        use std::collections::HashMap;
        Trace {
            trace_id: "t1".into(),
            agent_name: "agent".into(),
            framework: "raw".into(),
            steps: contents
                .iter()
                .enumerate()
                .map(|(i, c)| TraceStep {
                    id: (i + 1) as u32,
                    step_type: StepType::Reasoning,
                    content: c.to_string(),
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
                })
                .collect(),
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    // Simple exact-match similarity for testing.
    fn exact_sim(a: &str, b: &str) -> f64 {
        if a == b { 1.0 } else { 0.0 }
    }

    #[test]
    fn test_srr_no_redundancy() {
        let trace = make_trace(&["step one content", "different step content", "another step"]);
        let result = compute(&trace, exact_sim, None);
        assert_eq!(result.redundant_count, 0);
        assert!(result.pass);
    }

    #[test]
    fn test_srr_detects_duplicate() {
        let text = "parse the user request about order details";
        let trace = make_trace(&[text, "fetch order from database", text]);
        let result = compute(&trace, exact_sim, None);
        assert_eq!(result.redundant_count, 1);
        assert_eq!(result.redundant_steps[0].step_a, 1);
        assert_eq!(result.redundant_steps[0].step_b, 3);
    }
}
