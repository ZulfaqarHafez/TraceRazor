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
/// Maximum number of prior steps each step is compared against.
/// Caps the worst-case pair scan at O(n·LOOKBACK_WINDOW) so very long
/// traces (>~500 steps) stay responsive. A duplicate appearing further
/// back than this window will not be flagged.
pub const LOOKBACK_WINDOW: usize = 256;

/// Whether the later step of a similar pair is *responsive* rather than
/// redundant. Three real-trace patterns that lexical similarity cannot see:
///
/// 1. **New external input arrived** at or between the pair (any step in
///    `(a, b]` carries a non-empty `input_context`): a step answering a new
///    user/environment turn is never redundant with a pre-turn step, however
///    similar the wording (e.g. re-searching after the user rejected the
///    first results).
/// 2. **Fail→retry**: the earlier step is a failed call to the same tool the
///    later one completes — the retry is the productive member of the pair
///    (the failure is already penalised by TCA).
/// 3. **Verification after a state change**: two successful tool calls with
///    an intervening mutating step (an edit, a write, a booking) — re-running
///    a check after changing the world is how agents verify, not waste.
fn pair_is_responsive(steps: &[TraceStep], a: usize, b: usize) -> bool {
    use crate::types::StepType;

    // 1) New external input at or between the pair.
    if steps[a + 1..=b].iter().any(|s| {
        s.input_context
            .as_deref()
            .is_some_and(|c| !c.trim().is_empty())
    }) {
        return true;
    }

    let (pa, pb) = (&steps[a], &steps[b]);

    // 2) Fail→retry of the same tool: keep the retry.
    if pa.tool_success == Some(false)
        && pb.step_type == StepType::ToolCall
        && pa.tool_name == pb.tool_name
        && pb.tool_success != Some(false)
    {
        return true;
    }

    // 3) Verification re-run after an intervening state change.
    if pa.step_type == StepType::ToolCall
        && pb.step_type == StepType::ToolCall
        && pa.tool_success != Some(false)
        && pb.tool_success != Some(false)
        && steps[a + 1..b].iter().any(|s| s.is_mutating())
    {
        return true;
    }

    false
}

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

    // Compare every step against its most recent LOOKBACK_WINDOW prior steps.
    for i in 1..steps.len() {
        let curr = &steps[i];
        let curr_text = curr.semantic_content();
        let window_start = i.saturating_sub(LOOKBACK_WINDOW);

        // Track the *most similar* qualifying prior step (the previous code
        // broke on the first/oldest prior above threshold, contradicting its
        // own "most similar" comment).
        let mut best: Option<(usize, f64)> = None;
        for (off, prev) in steps[window_start..i].iter().enumerate() {
            let j = window_start + off;
            let sim = similarity_fn(&curr_text, &prev.semantic_content());
            if sim >= threshold
                && !pair_is_responsive(steps, j, i)
                && best.is_none_or(|(_, s)| sim > s)
            {
                best = Some((j, sim));
            }
        }
        if let Some((j, sim)) = best {
            let confidence = if sim >= HIGH_CONFIDENCE {
                Confidence::High
            } else {
                Confidence::Medium
            };
            redundant_step_ids.insert(curr.id);
            pairs.push(SrrRedundantPair {
                step_a: steps[j].id,
                step_b: curr.id,
                similarity: (sim * 100.0).round() / 100.0,
                confidence,
            });
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

    #[test]
    fn test_srr_respects_lookback_window() {
        // A duplicate outside the lookback window should not be flagged;
        // a duplicate inside it should be.
        let dup = "parse the user request about order details";
        let filler = "unique filler content";
        let mut contents: Vec<String> = vec![dup.to_string()];
        for _ in 0..(LOOKBACK_WINDOW + 5) {
            contents.push(filler.to_string());
        }
        contents.push(dup.to_string());
        let refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();
        let trace = make_trace(&refs);
        let result = compute(&trace, exact_sim, None);
        // The repeating filler within the window is still detected, but the
        // first `dup` (step 1) is outside the window of the last `dup`.
        let last_id = contents.len() as u32;
        let flagged_last = result
            .redundant_steps
            .iter()
            .any(|p| p.step_b == last_id && p.step_a == 1);
        assert!(!flagged_last, "step 1 should be outside the lookback window of the last step");
    }

    // ── Phase-1 precision rules ───────────────────────────────────────────────

    #[test]
    fn most_similar_prior_is_flagged_not_first() {
        // Two priors above threshold; the pair must point at the MORE similar
        // one (index 2), not the first/oldest above threshold (index 0).
        let trace = make_trace(&["alpha beta gamma", "unrelated", "alpha beta gamma delta", "alpha beta gamma delta"]);
        let sim = |a: &str, b: &str| {
            if a == b { 1.0 }
            else if a.starts_with("alpha") && b.starts_with("alpha") { 0.7 }
            else { 0.0 }
        };
        let result = compute(&trace, sim, None);
        let pair = result
            .redundant_steps
            .iter()
            .find(|p| p.step_b == 4)
            .expect("step 4 should be flagged");
        assert_eq!(pair.step_a, 3, "must flag the most similar prior, got {pair:?}");
    }

    #[test]
    fn new_input_between_pair_is_responsive_not_redundant() {
        // Identical wording, but a new user turn arrived at step 3: the
        // re-search answers new input and must not be flagged.
        let mut trace = make_trace(&[
            "search flights from JFK to SEA",
            "presenting the direct options",
            "search flights from JFK to SEA",
        ]);
        trace.steps[2].input_context = Some("user: nothing before 11am please".into());
        let result = compute(&trace, exact_sim, None);
        assert!(
            result.redundant_steps.is_empty(),
            "responsive re-search flagged: {:?}",
            result.redundant_steps
        );

        // Control: without the new input the identical step IS redundant.
        let control = make_trace(&[
            "search flights from JFK to SEA",
            "presenting the direct options",
            "search flights from JFK to SEA",
        ]);
        let r = compute(&control, exact_sim, None);
        assert_eq!(r.redundant_count, 1, "control pair must still be flagged");
    }

    #[test]
    fn failed_then_successful_retry_keeps_the_retry() {
        let mut trace = make_trace(&[
            "Calling book_reservation with payment 255",
            "recalculating the total",
            "Calling book_reservation with payment 255",
        ]);
        for (i, ok) in [(0, false), (2, true)] {
            trace.steps[i].step_type = StepType::ToolCall;
            trace.steps[i].tool_name = Some("book_reservation".into());
            trace.steps[i].tool_success = Some(ok);
        }
        let result = compute(&trace, exact_sim, None);
        assert!(
            !result.redundant_steps.iter().any(|p| p.step_b == 3),
            "the successful retry must not be the redundant member: {:?}",
            result.redundant_steps
        );
    }

    #[test]
    fn verification_rerun_after_mutation_is_not_redundant() {
        // run tests -> edit the file -> run tests again: the re-run verifies
        // the edit. With no intervening mutation it stays redundant.
        let mut trace = make_trace(&[
            "Action: python reproduce.py",
            "Action: edit lines 40:45",
            "Action: python reproduce.py",
        ]);
        for i in [0, 1, 2] {
            trace.steps[i].step_type = StepType::ToolCall;
            trace.steps[i].tool_success = Some(true);
        }
        trace.steps[0].tool_name = Some("python".into());
        trace.steps[1].tool_name = Some("edit".into());
        trace.steps[2].tool_name = Some("python".into());
        let result = compute(&trace, exact_sim, None);
        assert!(
            !result.redundant_steps.iter().any(|p| p.step_b == 3),
            "verification re-run flagged: {:?}",
            result.redundant_steps
        );

        // Control: same pair with a read-only step between stays redundant.
        let mut control = make_trace(&[
            "Action: python reproduce.py",
            "Action: open the file to read it",
            "Action: python reproduce.py",
        ]);
        for i in [0, 1, 2] {
            control.steps[i].step_type = StepType::ToolCall;
            control.steps[i].tool_success = Some(true);
        }
        control.steps[0].tool_name = Some("python".into());
        control.steps[1].tool_name = Some("open".into());
        control.steps[2].tool_name = Some("python".into());
        let r = compute(&control, exact_sim, None);
        assert!(r.redundant_steps.iter().any(|p| p.step_b == 3));
    }

}
