/// Verbosity Density Index (VDI)
///
/// Measures the ratio of substantive tokens to total tokens across all trace
/// steps. Non-substantive tokens include articles, preamble phrases (weighted
/// 3× because the whole phrase is waste), and single-word filler adverbs.
///
/// Target: VDI > 0.60 (at least 60% of tokens must be substantive).
/// FAIL threshold: VDI < 0.50.
/// Weight in TAS composite: 9%.
use serde::{Deserialize, Serialize};

use super::verbosity_data::{FILLER_WORDS, PREAMBLE_PATTERNS};
use crate::types::Trace;

/// VDI result for a single step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VdiStepResult {
    pub step_id: u32,
    /// Substantive / total ratio for this step (0.0–1.0, higher = better).
    pub score: f64,
    pub filler_count: usize,
    pub total_words: usize,
    /// True when step VDI falls below the hard-fail threshold (0.50).
    pub low_density: bool,
    /// True when character-level Shannon entropy < 3.8 bits/char (low information variety).
    pub entropy_flagged: bool,
}

/// Aggregate VDI result across the full trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VdiResult {
    /// Aggregate substantive-token ratio (0.0–1.0). Higher is better.
    pub score: f64,
    pub step_results: Vec<VdiStepResult>,
    /// IDs of steps whose individual VDI < 0.50 (hard-fail per step).
    pub low_density_steps: Vec<u32>,
    /// IDs of steps with character-level Shannon entropy < 3.8 bits/char.
    pub entropy_low_steps: Vec<u32>,
    pub pass: bool,
    pub target: f64,
}

impl VdiResult {
    /// Normalised score for TAS composite (0.0–1.0, higher = better).
    /// VDI is already in [0, 1] with higher = better.
    pub fn normalised(&self) -> f64 {
        self.score.clamp(0.0, 1.0)
    }
}

pub const TARGET: f64 = 0.60;
const LOW_DENSITY_THRESHOLD: f64 = 0.50;
const ENTROPY_THRESHOLD: f64 = 3.8; // bits/char

/// Compute VDI for all steps in the trace and return an aggregate result.
pub fn compute(trace: &Trace) -> VdiResult {
    let mut step_results = Vec::with_capacity(trace.steps.len());
    let mut total_words: usize = 0;
    let mut total_filler: usize = 0;

    for step in &trace.steps {
        let result = compute_step(step.id, &step.content);
        total_words += result.total_words;
        total_filler += result.filler_count;
        step_results.push(result);
    }

    let score = if total_words == 0 {
        1.0
    } else {
        let substantive = total_words.saturating_sub(total_filler);
        (substantive as f64 / total_words as f64).clamp(0.0, 1.0)
    };

    let low_density_steps: Vec<u32> = step_results
        .iter()
        .filter(|r| r.low_density)
        .map(|r| r.step_id)
        .collect();

    let entropy_low_steps: Vec<u32> = step_results
        .iter()
        .filter(|r| r.entropy_flagged)
        .map(|r| r.step_id)
        .collect();

    VdiResult {
        score,
        entropy_low_steps,
        low_density_steps,
        step_results,
        pass: score >= TARGET,
        target: TARGET,
    }
}

/// Character-level Shannon entropy (bits/char).
/// Returns 0.0 for empty strings.
fn character_entropy(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    for b in text.bytes() {
        counts[b as usize] += 1;
    }
    let len = text.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / len;
            -p * p.log2()
        })
        .sum()
}

/// Compute VDI for a single piece of text, labelled with `step_id`.
fn compute_step(step_id: u32, content: &str) -> VdiStepResult {
    let lower = content.to_lowercase();
    let words: Vec<&str> = content.split_whitespace().collect();
    let total = words.len();

    let mut filler: usize = 0;

    // Preamble phrases: weight 3× (entire phrase is waste).
    for pat in PREAMBLE_PATTERNS {
        filler += lower.matches(*pat).count() * 3;
    }

    // Per-word filler adverbs.
    for word in &words {
        let lw = word
            .trim_matches(|c: char| !c.is_alphabetic())
            .to_lowercase();
        if FILLER_WORDS.contains(&lw.as_str()) {
            filler += 1;
        }
    }

    // Articles.
    filler += words
        .iter()
        .filter(|w| {
            matches!(
                w.trim_matches(|c: char| !c.is_alphabetic())
                    .to_lowercase()
                    .as_str(),
                "a" | "an" | "the"
            )
        })
        .count();

    // Guard against saturating below zero.
    let filler = filler.min(total);
    let substantive = total - filler;
    let score = if total == 0 {
        1.0
    } else {
        (substantive as f64 / total as f64).clamp(0.0, 1.0)
    };

    let entropy = character_entropy(content);

    VdiStepResult {
        step_id,
        score,
        filler_count: filler,
        total_words: total,
        low_density: score < LOW_DENSITY_THRESHOLD,
        entropy_flagged: total > 0 && entropy < ENTROPY_THRESHOLD,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, Trace, TraceStep};
    use std::collections::HashMap;

    fn make_trace(steps: Vec<(&str, u32)>) -> Trace {
        Trace {
            trace_id: "vdi-test".into(),
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
    fn test_vdi_clean_technical_step() {
        // Pure technical content — almost no filler.
        let trace = make_trace(vec![(
            "Order ORD-9182 fetched. Customer eligible for refund. \
             Processing refund transaction 7821.",
            500,
        )]);
        let result = compute(&trace);
        assert!(
            result.score > 0.70,
            "clean technical step should score > 0.70, got {:.2}",
            result.score
        );
        assert!(result.pass, "clean step should pass target 0.60");
    }

    #[test]
    fn test_vdi_all_filler_step() {
        // Sycophantic preamble with almost no content.
        let trace = make_trace(vec![(
            "Let me just basically actually essentially think through \
             a the very really quite quite quite",
            200,
        )]);
        let result = compute(&trace);
        assert!(
            result.score < 0.50,
            "all-filler step should score < 0.50, got {:.2}",
            result.score
        );
        assert!(
            !result.pass,
            "all-filler step should fail target 0.60"
        );
        assert_eq!(result.low_density_steps, vec![1]);
    }

    #[test]
    fn test_vdi_mixed_step_mid_range() {
        // Mix of technical content and some filler.
        let trace = make_trace(vec![(
            "I should actually look up the order details. \
             The order ORD-9182 exists and is eligible.",
            300,
        )]);
        let result = compute(&trace);
        // Should be between 0.30 and 0.80.
        assert!(
            result.score > 0.30 && result.score < 0.80,
            "mixed step should be in mid-range, got {:.2}",
            result.score
        );
    }

    #[test]
    fn test_vdi_empty_trace() {
        let trace = make_trace(vec![("", 0)]);
        let result = compute(&trace);
        assert_eq!(result.score, 1.0, "empty content should default to 1.0");
    }

    #[test]
    fn test_entropy_flagged_low_variety() {
        // Highly repetitive content has low entropy.
        let trace = make_trace(vec![("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 50)]);
        let result = compute(&trace);
        assert!(
            result.step_results[0].entropy_flagged,
            "single-char repetition should be entropy-flagged"
        );
        assert_eq!(result.entropy_low_steps, vec![1]);
    }

    #[test]
    fn test_entropy_not_flagged_rich_content() {
        // Normal English prose has entropy > 3.8.
        let trace = make_trace(vec![(
            "Order ORD-9182 fetched. Customer eligible for refund. \
             Transaction 7821 processed successfully.",
            400,
        )]);
        let result = compute(&trace);
        assert!(
            !result.step_results[0].entropy_flagged,
            "rich English prose should not be entropy-flagged"
        );
    }
}