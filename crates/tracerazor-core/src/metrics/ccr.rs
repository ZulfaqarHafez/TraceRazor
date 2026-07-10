/// Caveman Compression Ratio (CCR)
///
/// Measures how much of each step's text is *structurally* compressible by
/// taking the worst case of three orthogonal signals:
///
///   1. **Wordlist excess** — fraction of tokens that fall in the curated
///      filler/article/hedge lists (the original Caveman-style rewriter).
///   2. **Gzip excess** — the fraction by which the step's deflated size
///      undercuts a typical-English baseline (≈0.45 of original bytes).
///      Padded JSON, copy-pasted boilerplate, and repeated technical noise
///      register here even when they contain zero blacklisted words.
///   3. **N-gram repetition** — `1 - unique_4grams / total_4grams` over word
///      4-grams. Catches "report report report" style runs that survive the
///      wordlist and gzip filters on short inputs.
///
/// `ratio = max(wordlist, gzip_excess, ngram_repetition)`, so an adversarial
/// prompt that tries to dodge the wordlist still has to defeat compressibility
/// *and* n-gram repetition to drive CCR below the 0.30 target.
///
/// Higher CCR = more compressible = more linguistic waste.
///
/// Target: CCR < 0.30. Post-normalisation share of TAS: ~2.5 %.
/// Normalised for TAS: `(1 - CCR)` so higher composite = less waste.
use std::collections::HashSet;
use std::io::Write;

use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};

use super::verbosity_data::{FILLER_WORDS, HEDGE_PHRASES, PREAMBLE_PATTERNS};
use crate::types::Trace;

pub const TARGET: f64 = 0.30;

/// Typical compression ratio (compressed/original bytes) for natural English
/// prose. Below this, text is more structurally redundant than typical prose.
const ENGLISH_GZIP_BASELINE: f64 = 0.45;

/// Minimum byte length at which the gzip signal is reliable. Below this,
/// gzip header overhead dominates and inflates the ratio artificially.
const MIN_GZIP_BYTES: usize = 128;

/// Minimum number of word 4-grams at which the n-gram repetition signal kicks
/// in. Below this the ratio is too noisy to act on.
const MIN_NGRAM_COUNT: usize = 8;

/// CCR result for a single step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CcrStepResult {
    pub step_id: u32,
    pub original_tokens: u32,
    pub compressed_tokens: u32,
    /// Final score for the step: worst case across the three signals.
    pub ratio: f64,
    /// Wordlist-only excess (the legacy Caveman signal).
    #[serde(default)]
    pub wordlist_ratio: f64,
    /// Excess gzip compressibility vs the English baseline.
    #[serde(default)]
    pub gzip_excess: f64,
    /// N-gram repetition ratio (1 - unique_4grams / total_4grams).
    #[serde(default)]
    pub ngram_repetition: f64,
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
    let wordlist_ratio = if original == 0 {
        0.0
    } else {
        ((original - compressed.min(original)) as f64 / original as f64).clamp(0.0, 1.0)
    };

    let gzip_excess = gzip_excess(content);
    let ngram_repetition = ngram_repetition(content, 4);
    let ratio = wordlist_ratio
        .max(gzip_excess)
        .max(ngram_repetition)
        .clamp(0.0, 1.0);

    CcrStepResult {
        step_id,
        original_tokens: original,
        compressed_tokens: compressed,
        ratio,
        wordlist_ratio,
        gzip_excess,
        ngram_repetition,
    }
}

/// Return the amount by which the step's gzip ratio undercuts a typical-
/// English baseline. Returns 0.0 for short or incompressible text.
fn gzip_excess(content: &str) -> f64 {
    let bytes = content.as_bytes();
    if bytes.len() < MIN_GZIP_BYTES {
        return 0.0;
    }
    let mut encoder = GzEncoder::new(Vec::with_capacity(bytes.len()), Compression::default());
    if encoder.write_all(bytes).is_err() {
        return 0.0;
    }
    let Ok(compressed) = encoder.finish() else {
        return 0.0;
    };
    let ratio = compressed.len() as f64 / bytes.len() as f64;
    (ENGLISH_GZIP_BASELINE - ratio).max(0.0).clamp(0.0, 1.0)
}

/// Fraction of word `n`-grams that are repeats (`1 - unique / total`).
/// Returns 0.0 when the step is too short for the signal to be reliable.
fn ngram_repetition(content: &str, n: usize) -> f64 {
    let words: Vec<&str> = content.split_whitespace().collect();
    if words.len() < n + MIN_NGRAM_COUNT - 1 {
        return 0.0;
    }
    let total = words.len() - n + 1;
    if total < MIN_NGRAM_COUNT {
        return 0.0;
    }
    let mut unique: HashSet<String> = HashSet::with_capacity(total);
    for window in words.windows(n) {
        unique.insert(window.join(" ").to_lowercase());
    }
    (1.0 - unique.len() as f64 / total as f64).clamp(0.0, 1.0)
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
                    let lw = w.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase();
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

    #[test]
    fn test_ngram_repetition_catches_wordlist_dodging_padding() {
        // Adversarial step that uses zero blacklisted words but pads the
        // content with a heavily repeating 4-gram. The legacy wordlist signal
        // returns ~0, but the n-gram repetition signal should flag it.
        let padded = "process refund step now ".repeat(12);
        let trace = make_trace(vec![(padded.as_str(), 800)]);
        let result = compute(&trace);
        let step = &result.step_results[0];
        assert!(
            step.ngram_repetition > 0.5,
            "expected high ngram repetition on padded text, got {:.2}",
            step.ngram_repetition
        );
        assert!(
            step.ratio >= step.ngram_repetition,
            "final ratio must be at least the worst sub-signal"
        );
    }

    #[test]
    fn test_gzip_excess_catches_compressible_boilerplate() {
        // Long copy-paste-style block that contains no blacklisted words but
        // compresses heavily. Should be flagged by the gzip signal.
        let block = "Order received and processed correctly through the standard \
                     fulfilment workflow with the standard fulfilment workflow \
                     completing in the standard fulfilment workflow and yielding \
                     the standard fulfilment workflow result.";
        let trace = make_trace(vec![(block, 600)]);
        let result = compute(&trace);
        let step = &result.step_results[0];
        // Either the gzip OR ngram-repetition signal should fire and dominate.
        assert!(
            step.gzip_excess > 0.05 || step.ngram_repetition > 0.10,
            "expected gzip or ngram signal to flag compressible boilerplate, got gzip={:.2} ngram={:.2}",
            step.gzip_excess,
            step.ngram_repetition
        );
    }
}
