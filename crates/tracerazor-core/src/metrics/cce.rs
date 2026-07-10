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
        .map(|s| s.input_context.clone().unwrap_or_else(|| s.content.clone()))
        .collect();

    let total_input_tokens: u32 = steps.iter().map(|s| s.tokens).sum();

    let mut duplicate_tokens: u32 = 0;
    let mut bloated_steps: Vec<ContextBloatStep> = Vec::new();

    // Build the cumulative prior-n-gram set incrementally instead of
    // re-joining the whole prefix per step (O(n²·len) → O(n·len)). A tail of
    // the last n-1 words is carried so n-grams spanning step boundaries are
    // produced exactly as `texts[..i].join(" ")` would; output is identical.
    const N: usize = 4;
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut tail: Vec<String> = Vec::new();
    let mut cum_words: usize = 0;

    for i in 1..steps.len() {
        // Absorb texts[i-1] into the cumulative reference.
        {
            let mut window: Vec<&str> = tail.iter().map(String::as_str).collect();
            window.extend(texts[i - 1].split_whitespace());
            cum_words += texts[i - 1].split_whitespace().count();
            for w in window.windows(N) {
                seen.insert(w.join(" "));
            }
            let keep = window.len().saturating_sub(N - 1);
            tail = window[keep..].iter().map(|s| s.to_string()).collect();
        }

        let current_text = &texts[i];
        let cur_ngrams = extract_ngrams(current_text, N);
        let overlap = if cur_ngrams.is_empty() {
            0.0
        } else if cum_words < N {
            // Mirror extract_ngrams' short-reference behaviour: the whole
            // prior text (< N words) acts as one "short-gram".
            let ref_single = tail.join(" ");
            cur_ngrams.iter().filter(|g| **g == ref_single).count() as f64 / cur_ngrams.len() as f64
        } else {
            cur_ngrams.iter().filter(|g| seen.contains(*g)).count() as f64 / cur_ngrams.len() as f64
        };

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
/// Retained as the reference implementation the incremental path in
/// [`compute`] is equivalence-tested against.
#[cfg_attr(not(test), allow(dead_code))]
fn ngram_overlap_ratio(text: &str, reference: &str, n: usize) -> f64 {
    let text_ngrams = extract_ngrams(text, n);
    if text_ngrams.is_empty() {
        return 0.0;
    }
    let ref_ngrams: std::collections::HashSet<_> =
        extract_ngrams(reference, n).into_iter().collect();

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
    words.windows(n).map(|w| w.join(" ")).collect()
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

    #[test]
    fn incremental_overlap_matches_reference_implementation() {
        // The incremental cumulative-set path must reproduce the original
        // whole-prefix-join semantics exactly, including n-grams spanning
        // step boundaries and the short-reference (<4 words) edge case.
        let corpora: Vec<Vec<&str>> = vec![
            // boundary-spanning: "delta echo foxtrot golf" spans steps 1|2
            vec![
                "alpha bravo charlie delta echo",
                "foxtrot golf hotel india juliet",
                "delta echo foxtrot golf something else entirely here",
            ],
            // short reference + exact short repeat
            vec!["hello world", "hello world", "now a much longer step text"],
            // empties and tiny texts
            vec![
                "",
                "one",
                "one two three four five",
                "one two three four five",
            ],
        ];
        for texts_src in corpora {
            let trace = make_trace_with_contexts(&texts_src);
            let result = compute(&trace);

            // Reference recomputation, the original O(n²) way.
            let texts: Vec<String> = trace
                .steps
                .iter()
                .map(|s| s.input_context.clone().unwrap_or_else(|| s.content.clone()))
                .collect();
            let mut ref_dup: u32 = 0;
            for i in 1..trace.steps.len() {
                let prior: String = texts[..i].join(" ");
                let overlap = ngram_overlap_ratio(&texts[i], &prior, 4);
                if overlap > 0.40 {
                    ref_dup += (trace.steps[i].tokens as f64 * overlap) as u32;
                }
            }
            assert_eq!(
                result.duplicate_tokens, ref_dup,
                "incremental CCE diverged from reference on {texts_src:?}"
            );
        }
    }

    fn make_trace_with_contexts(contexts: &[&str]) -> Trace {
        let steps: Vec<TraceStep> = contexts
            .iter()
            .enumerate()
            .map(|(i, c)| TraceStep {
                id: (i + 1) as u32,
                step_type: StepType::Reasoning,
                content: format!("step {i}"),
                tokens: 100,
                tool_name: None,
                tool_params: None,
                tool_success: None,
                tool_error: None,
                agent_id: None,
                input_context: Some(c.to_string()),
                output: None,
                flags: vec![],
                flag_details: vec![],
            })
            .collect();
        Trace {
            trace_id: "cce-eq".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            total_tokens: steps.iter().map(|s| s.tokens).sum(),
            steps,
            task_value_score: 1.0,
            metadata: std::collections::HashMap::new(),
        }
    }
}
