/// Sycophancy / Hedging Level (SHL)
///
/// Measures the fraction of sentences across all trace steps that are
/// sycophantic (start with a preamble pattern) or heavily hedged
/// (contain ≥ 2 independent hedge phrases).
///
/// Target: SHL < 0.20 (at most 20% of sentences may be sycophantic or hedged).
/// Weight in TAS composite: 5%.
use serde::{Deserialize, Serialize};

use super::verbosity_data::{HEDGE_PHRASES, PREAMBLE_PATTERNS};
use crate::types::Trace;

pub const TARGET: f64 = 0.20;

/// Aggregate SHL result across the full trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShlResult {
    /// Ratio of flagged sentences to total sentences (0.0–1.0). Lower is better.
    pub score: f64,
    pub flagged_sentences: usize,
    pub total_sentences: usize,
    pub pass: bool,
    pub target: f64,
}

impl ShlResult {
    /// Normalised score for TAS composite (0.0–1.0, higher = better).
    /// SHL is inverted: lower hedge ratio → higher normalised score.
    pub fn normalised(&self) -> f64 {
        (1.0 - self.score).clamp(0.0, 1.0)
    }
}

/// Compute SHL across all steps in the trace.
pub fn compute(trace: &Trace) -> ShlResult {
    let mut total_sentences: usize = 0;
    let mut flagged: usize = 0;

    for step in &trace.steps {
        let (step_total, step_flagged) = classify_sentences(&step.content);
        total_sentences += step_total;
        flagged += step_flagged;
    }

    let score = if total_sentences == 0 {
        0.0
    } else {
        (flagged as f64 / total_sentences as f64).clamp(0.0, 1.0)
    };

    ShlResult {
        score,
        flagged_sentences: flagged,
        total_sentences,
        pass: score < TARGET,
        target: TARGET,
    }
}

/// Split `text` into sentences and return (total, flagged) counts.
fn classify_sentences(text: &str) -> (usize, usize) {
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();

    let total = sentences.len();
    let flagged = sentences.iter().filter(|s| is_flagged(s)).count();
    (total, flagged)
}

/// A sentence is flagged when it:
/// - Contains ≥ 2 distinct hedge phrase matches, OR
/// - Starts with a preamble pattern (sycophantic opener).
fn is_flagged(sentence: &str) -> bool {
    let lower = sentence.to_lowercase();

    // Check for sycophantic preamble (starts with).
    let is_sycophantic = PREAMBLE_PATTERNS
        .iter()
        .any(|&pat| lower.trim_start().starts_with(pat));

    if is_sycophantic {
        return true;
    }

    // Check for multi-hedge (≥ 2 distinct hedge phrases in one sentence).
    let hedge_count = HEDGE_PHRASES
        .iter()
        .filter(|&&pat| lower.contains(pat))
        .count();

    hedge_count >= 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, Trace, TraceStep};
    use std::collections::HashMap;

    fn make_trace(contents: Vec<&str>) -> Trace {
        Trace {
            trace_id: "shl-test".into(),
            agent_name: "agent".into(),
            framework: "raw".into(),
            steps: contents
                .into_iter()
                .enumerate()
                .map(|(i, c)| TraceStep {
                    id: (i + 1) as u32,
                    step_type: StepType::Reasoning,
                    content: c.into(),
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
                })
                .collect(),
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_shl_zero_hedge_trace() {
        let trace = make_trace(vec![
            "Order ORD-9182 is eligible for refund.",
            "Refund processed. Transaction ID 7821.",
            "Customer notified via email.",
        ]);
        let result = compute(&trace);
        assert_eq!(
            result.score, 0.0,
            "zero-hedge trace should score 0.0, got {:.2}",
            result.score
        );
        assert!(result.pass);
    }

    #[test]
    fn test_shl_sycophantic_trace() {
        let trace = make_trace(vec![
            "Let me help you with this request.",
            "I'd be happy to process the refund.",
            "Certainly, I can help resolve this issue.",
        ]);
        let result = compute(&trace);
        assert!(
            result.score > 0.30,
            "sycophantic trace should score > 0.30, got {:.2}",
            result.score
        );
        assert!(!result.pass, "high-sycophancy trace should fail");
    }

    #[test]
    fn test_shl_multi_hedge_sentence() {
        // Two hedge phrases in one sentence → flagged.
        let trace = make_trace(vec![
            "This might possibly be the right approach.",
            "The order is confirmed and ready.",
        ]);
        let result = compute(&trace);
        assert!(
            result.flagged_sentences >= 1,
            "at least one sentence should be flagged"
        );
    }

    #[test]
    fn test_shl_single_hedge_not_flagged() {
        // Only one hedge phrase → not flagged.
        let trace = make_trace(vec!["This might be the correct order ID."]);
        let result = compute(&trace);
        assert_eq!(result.flagged_sentences, 0, "single hedge should not flag");
        assert!(result.pass);
    }

    #[test]
    fn test_shl_empty_content() {
        let trace = make_trace(vec!["", ""]);
        let result = compute(&trace);
        assert_eq!(result.score, 0.0);
        assert_eq!(result.total_sentences, 0);
    }
}