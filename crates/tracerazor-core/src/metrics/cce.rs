/// Context Carry-over Efficiency (CCE)
///
/// Measures how much context is unnecessarily duplicated across sequential
/// LLM calls within the same trace.
///
/// Formula: CCE = 1 - (duplicate_context_tokens / total_input_tokens)
/// Score of 1.0 means zero redundant context.
/// Target: > 0.6. Below 0.4 indicates severe context duplication.
use serde::{Deserialize, Serialize};

use crate::types::{StepFlag, Trace, TraceStep};

/// A step with detected context bloat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBloatStep {
    pub step_id: u32,
    /// Estimated percentage of input context that is duplicated.
    pub duplicate_pct: f64,
    /// Estimated duplicate token count.
    pub duplicate_tokens: u32,
}

/// Result of the CCE metric computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CceResult {
    /// CCE score (0.0–1.0). Higher is better.
    pub score: f64,
    pub total_input_tokens: u32,
    pub duplicate_tokens: u32,
    pub bloated_steps: Vec<ContextBloatStep>,
    pub pass: bool,
    pub target: f64,
}

impl CceResult {
    /// Normalised score for TAS composite (0.0–1.0, higher = better).
    pub fn normalised(&self) -> f64 {
        self.score
    }
}

const TARGET: f64 = 0.60;

/// Compute the CCE metric using n-gram overlap between consecutive step inputs.
///
/// For each step i (starting at step 2), we measure how much of its
/// `input_context` (or `content` as fallback) overlaps with the concatenation
/// of all prior step contents. High overlap = context bloat.
pub fn compute(trace: &Trace) -> CceResult {
    let steps = &trace.steps;

    // Collect the input text for each step.
    let texts: Vec<String> = steps
        .iter()
        .map(|s| {
            s.input_context
                .clone()
                .unwrap_or_else(|| s.content.clone())
        })
        .collect();

    let total_input_tokens: u32 = steps.iter().map(|s| s.tokens).sum();

    let mut duplicate_tokens: u32 = 0;
    let mut bloated_steps: Vec<ContextBloatStep> = Vec::new();

    // Build cumulative "seen" vocabulary from all prior steps.
    for i in 1..steps.len() {
        let prior_text: String = texts[..i].join(" ");
        let current_text = &texts[i];

        let overlap = ngram_overlap_ratio(current_text, &prior_text, 4);

        if overlap > 0.40 {
            let dup_tokens = (steps[i].tokens as f64 * overlap) as u32;
            duplicate_tokens += dup_tokens;
            bloated_steps.push(ContextBloatStep {
                step_id: steps[i].id,
                duplicate_pct: (overlap * 100.0).round(),
                duplicate_tokens: dup_tokens,
            });
        }
    }

    let score = if total_input_tokens == 0 {
        1.0
    } else {
        let ratio = 1.0 - (duplicate_tokens as f64 / total_input_tokens as f64);
        ratio.clamp(0.0, 1.0)
    };

    CceResult {
        score: (score * 1000.0).round() / 1000.0,
        total_input_tokens,
        duplicate_tokens,
        bloated_steps,
        pass: score >= TARGET,
        target: TARGET,
    }
}

/// Compute the n-gram overlap ratio between `text` and `reference`.
/// Returns the fraction of `text`'s n-grams that appear in `reference`.
fn ngram_overlap_ratio(text: &str, reference: &str, n: usize) -> f64 {
    let text_ngrams = extract_ngrams(text, n);
    if text_ngrams.is_empty() {
        return 0.0;
    }
    let ref_ngrams: std::collections::HashSet<_> = extract_ngrams(reference, n)
        .into_iter()
        .collect();

    let overlap = text_ngrams
        .iter()
        .filter(|ng| ref_ngrams.contains(*ng))
        .count();

    overlap as f64 / text_ngrams.len() as f64
}

/// Extract n-grams (as joined strings) from a text.
fn extract_ngrams(text: &str, n: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < n {
        return vec![words.join(" ")];
    }
    words
        .windows(n)
        .map(|w| w.join(" "))
        .collect()
}

/// Apply CCE flags to trace steps.
pub fn annotate_steps(steps: &mut [TraceStep], result: &CceResult) {
    for bloat in &result.bloated_steps {
        if let Some(step) = steps.iter_mut().find(|s| s.id == bloat.step_id) {
            step.flags.push(StepFlag::ContextBloat);
            step.flag_details.push(format!(
                "{:.0}% duplicated input context",
                bloat.duplicate_pct
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, TraceStep};
    use std::collections::HashMap;

    fn step_with_context(id: u32, content: &str, context: &str, tokens: u32) -> TraceStep {
        TraceStep {
            id,
            step_type: StepType::Reasoning,
            content: content.to_string(),
            tokens,
            tool_name: None,
            tool_params: None,
            tool_success: None,
            tool_error: None,
            agent_id: None,
            input_context: Some(context.to_string()),
            output: None,
            flags: vec![],
            flag_details: vec![],
        }
    }

    #[test]
    fn test_no_bloat() {
        let trace = Trace {
            trace_id: "t1".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                step_with_context(1, "parse request", "user wants a refund", 100),
                step_with_context(2, "fetch order", "order id ORD-123", 150),
                step_with_context(3, "check eligibility", "check the policy rules", 120),
                step_with_context(4, "process refund", "initiate the refund", 130),
                step_with_context(5, "confirm", "send confirmation email", 100),
            ],
            total_tokens: 600,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let result = compute(&trace);
        // Steps have different context, so overlap should be low.
        assert!(result.bloated_steps.is_empty() || result.score > 0.5);
    }

    #[test]
    fn test_bloat_detected() {
        let long_context = "user wants a refund for order ORD-9182 placed on 2024-01-15 amount 45 dollars item blue jacket";
        let trace = Trace {
            trace_id: "t2".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                step_with_context(1, "parse", long_context, 200),
                step_with_context(2, "fetch", long_context, 200),
                step_with_context(3, "check", long_context, 200),
                step_with_context(4, "process", long_context, 200),
                step_with_context(5, "confirm", long_context, 200),
            ],
            total_tokens: 1000,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let result = compute(&trace);
        // All steps carry the same context — bloat should be detected.
        assert!(!result.bloated_steps.is_empty());
        assert!(result.score < 0.8);
    }
}
