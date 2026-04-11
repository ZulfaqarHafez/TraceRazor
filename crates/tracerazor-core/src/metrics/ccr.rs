/// Caveman Compression Ratio (CCR)
///
/// Measures how much of each step's text is linguistically compressible —
/// articles, filler adverbs, and preamble sentences that an LLM could
/// reconstruct from sequence context alone. Inspired by the Caveman
/// Compression approach (wilpel/caveman-compression), repurposed as a
/// diagnostic metric rather than an active rewriter.
///
/// CCR = 1 - (compressed_tokens / original_tokens), averaged across all steps.
/// Higher CCR = more compressible = more linguistic waste.
///
/// Target: CCR < 0.30 (at most 30% of tokens are compressible filler).
/// Weight in TAS composite: 4%.
/// Normalised for TAS: (1 - CCR) so higher composite = less waste.
use serde::{Deserialize, Serialize};

use super::verbosity_data::{FILLER_WORDS, HEDGE_PHRASES, PREAMBLE_PATTERNS};
use crate::types::Trace;

pub const TARGET: f64 = 0.30;

/// CCR result for a single step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CcrStepResult {
    pub step_id: u32,
    pub original_tokens: u32,
    pub compressed_tokens: u32,
    /// Fraction of tokens removed: (orig - comp) / orig.
    pub ratio: f64,
}

/// Aggregate CCR result across the full trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CcrResult {
    /// Aggregate ratio: total_removed / total_original (0.0–1.0). Lower is better.
    pub score: f64,
    pub step_results: Vec<CcrStepResult>,
    /// Estimated number of tokens that could be eliminated.
    pub total_cuttable_tokens: u32,
    pub pass: bool,
    pub target: f64,
}

impl CcrResult {
    /// Normalised score for TAS composite (0.0–1.0, higher = better).
    /// CCR is inverted: lower compression ratio → higher normalised score.
    pub fn normalised(&self) -> f64 {
        (1.0 - self.score).clamp(0.0, 1.0)
    }
}

/// Compute CCR for all steps in the trace.
pub fn compute(trace: &Trace) -> CcrResult {
    let mut step_results = Vec::with_capacity(trace.steps.len());
    let mut total_original: u32 = 0;
    let mut total_compressed: u32 = 0;

    for step in &trace.steps {
        let result = compute_step(step.id, &step.content);
        total_original += result.original_tokens;
        total_compressed += result.compressed_tokens;
        step_results.push(result);
    }

    let total_cuttable = total_original.saturating_sub(total_compressed);
    let score = if total_original == 0 {
        0.0
    } else {
        (total_cuttable as f64 / total_original as f64).clamp(0.0, 1.0)
    };

    CcrResult {
        score,
        step_results,
        total_cuttable_tokens: total_cuttable,
        pass: score < TARGET,
        target: TARGET,
    }
}

/// Apply the local Caveman compressor to a single step and return per-step CCR.
fn compute_step(step_id: u32, content: &str) -> CcrStepResult {
    let original = content.split_whitespace().count() as u32;
    let compressed_text = caveman_compress(content);
    let compressed = compressed_text.split_whitespace().count() as u32;
    let ratio = if original == 0 {
        0.0
    } else {
        ((original - compressed.min(original)) as f64 / original as f64).clamp(0.0, 1.0)
    };
    CcrStepResult {
        step_id,
        original_tokens: original,
        compressed_tokens: compressed,
        ratio,
    }
}

/// Apply three transformations in sequence, mirroring the Caveman approach:
/// 1. Strip preamble sentences (first word matches a preamble pattern).
/// 2. Strip article tokens (a, an, the).
/// 3. Strip filler words and single-word hedge phrases.
pub fn caveman_compress(text: &str) -> String {
    // Step 1: strip preamble sentences.
    let kept_sentences: Vec<&str> = text
        .split(". ")
        .filter(|s| {
            let lower = s.to_lowercase();
            let trimmed = lower.trim_start();
            !PREAMBLE_PATTERNS.iter().any(|p| trimmed.starts_with(p))
        })
        .collect();

    // Steps 2 + 3: strip articles, filler words, and single-word hedge matches.
    kept_sentences
        .iter()
        .map(|sentence| {
            sentence
                .split_whitespace()
                .filter(|w| {
                    let lw = w
                        .trim_matches(|c: char| !c.is_alphabetic())
                        .to_lowercase();
                    // Keep if it is NOT an article, filler word, or single-word hedge.
                    !matches!(lw.as_str(), "a" | "an" | "the")
                        && !FILLER_WORDS.contains(&lw.as_str())
                        && !HEDGE_PHRASES
                            .iter()
                            .any(|&p| !p.contains(' ') && lw.as_str() == p)
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|s| !s.trim().is_empty())
        .collect::<Vec<_>>()
        .join(". ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, Trace, TraceStep};
    use std::collections::HashMap;

    fn make_trace(steps: Vec<(&str, u32)>) -> Trace {
        Trace {
            trace_id: "ccr-test".into(),
            agent_name: "agent".into(),
            framework: "raw".into(),
            steps: steps
                .into_iter()
                .enumerate()
                .map(|(i, (content, tokens))| TraceStep {
                    id: (i + 1) as u32,
                    step_type: StepType::Reasoning,
                    content: content.into(),
                    tokens,
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
    fn test_ccr_clean_technical_step() {
        // Dense technical text — very little to remove.
        let trace = make_trace(vec![(
            "Order ORD-9182 fetched. Customer eligible for refund. \
             Transaction 7821 processed successfully.",
            400,
        )]);
        let result = compute(&trace);
        assert!(
            result.score < 0.30,
            "clean technical step should have CCR < 0.30, got {:.2}",
            result.score
        );
        assert!(result.pass);
    }

    #[test]
    fn test_ccr_sycophantic_step() {
        // Preamble-heavy step — should have high CCR.
        let trace = make_trace(vec![(
            "Let me think through this carefully. \
             I'd be happy to help you with a the very essentially quite long answer. \
             Certainly, I will now process the refund.",
            350,
        )]);
        let result = compute(&trace);
        assert!(
            result.score > 0.25,
            "sycophantic step should have CCR > 0.25, got {:.2}",
            result.score
        );
    }

    #[test]
    fn test_ccr_empty_content() {
        let trace = make_trace(vec![("", 0)]);
        let result = compute(&trace);
        assert_eq!(result.score, 0.0);
        assert!(result.pass);
    }

    #[test]
    fn test_caveman_compress_strips_articles() {
        let compressed = caveman_compress("The order is ready for a refund.");
        assert!(
            !compressed.contains(" the ") && !compressed.to_lowercase().starts_with("the "),
            "should strip 'the'"
        );
        assert!(!compressed.contains(" a "), "should strip 'a'");
    }

    #[test]
    fn test_caveman_compress_strips_preamble_sentence() {
        let compressed = caveman_compress("Let me help you. Order ORD-9182 is eligible.");
        // The preamble sentence should be stripped.
        assert!(
            !compressed.to_lowercase().contains("let me"),
            "preamble sentence should be stripped"
        );
        // The content sentence should remain.
        assert!(compressed.contains("ORD-9182"), "content should be kept");
    }
}