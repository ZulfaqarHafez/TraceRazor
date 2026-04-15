/// Goal Advancement Ratio (GAR) — M1
///
/// Measures how consistently each reasoning step moves the agent semantically
/// closer to its final conclusion.  Where SRR and ISR ask "is this step
/// redundant?", GAR asks "is this step *on the right path*?".
///
/// ## Algorithm
///
/// 1. The **goal proxy** is the content of the last reasoning step in the trace
///    (the final conclusion the agent produced).
/// 2. For every intermediate reasoning step (all except the last), compute its
///    cosine similarity to the goal proxy using the supplied `similarity_fn`.
/// 3. GAR is the token-weighted mean of those per-step similarities, clamped
///    to [0, 1].
///
/// A high GAR (→1.0) means every intermediate step is already closely aligned
/// with the final answer — efficient, on-topic reasoning.
/// A low GAR (→0.0) signals the agent wandered through off-topic territory
/// before converging, which is invisible to every other metric.
///
/// ## Backend note
///
/// With the Phase 1 **bag-of-words** backend the metric uses exact-word overlap,
/// so typical scores on well-formed traces range 0.30–0.70.  With the Phase 2
/// **sentence-embedding** backend (via `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or
/// `TRACERAZOR_LLM_*`) similarity is semantic rather than lexical, raising the
/// effective discrimination power substantially and allowing tighter thresholds.
///
/// ## Targets (calibrated for BoW; relax slightly for embeddings)
///
/// | Band | GAR | Interpretation |
/// |------|-----|----------------|
/// | Pass | ≥ 0.40 | Reasoning consistently on-track |
/// | Warn | 0.25–0.39 | Occasional drift |
/// | Fail | < 0.25 | Significant off-topic wandering |
///
/// Weight in TAS composite: 6% (replaces half of CCR's duplicate signal).
use serde::{Deserialize, Serialize};

use crate::types::{StepType, Trace};

/// Target: weighted-mean goal similarity must be at least this value.
pub const TARGET: f64 = 0.40;
/// Steps below this similarity to the goal are flagged as "low advancement".
pub const LOW_ADVANCEMENT_THRESHOLD: f64 = 0.20;

/// Per-step GAR result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarStepResult {
    pub step_id: u32,
    /// Cosine similarity of this step's content to the goal proxy (0.0–1.0).
    pub goal_similarity: f64,
    /// True when `goal_similarity < LOW_ADVANCEMENT_THRESHOLD`.
    pub low_advancement: bool,
}

/// Aggregate GAR result across the full trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarResult {
    /// Token-weighted mean goal similarity (0.0–1.0).  Higher is better.
    pub score: f64,
    /// Per-step breakdown (reasoning steps only, excludes the goal step itself).
    pub step_results: Vec<GarStepResult>,
    /// IDs of steps whose `goal_similarity` is below the low-advancement threshold.
    pub low_advancement_steps: Vec<u32>,
    /// ID of the step used as the goal proxy (last reasoning step).
    pub goal_step_id: Option<u32>,
    pub pass: bool,
    pub target: f64,
}

impl GarResult {
    /// Normalised score for TAS composite (0.0–1.0, higher = better).
    /// GAR is already in the right direction: higher similarity = better.
    pub fn normalised(&self) -> f64 {
        self.score.clamp(0.0, 1.0)
    }
}

/// Compute the GAR metric for a trace.
///
/// `similarity_fn` accepts two text strings and returns a cosine similarity
/// in [0.0, 1.0].  The same closure used by SRR and ISR works here.
pub fn compute(trace: &Trace, similarity_fn: impl Fn(&str, &str) -> f64) -> GarResult {
    // Collect reasoning steps in order.
    let reasoning: Vec<_> = trace
        .steps
        .iter()
        .filter(|s| s.step_type == StepType::Reasoning)
        .collect();

    // Need at least 2 reasoning steps to measure advancement.
    if reasoning.len() < 2 {
        return GarResult {
            score: 1.0,
            step_results: vec![],
            low_advancement_steps: vec![],
            goal_step_id: reasoning.last().map(|s| s.id),
            pass: true,
            target: TARGET,
        };
    }

    // Goal proxy = last reasoning step.
    let goal_step = *reasoning.last().unwrap();
    let goal_text = &goal_step.content;

    // Score each intermediate step (all except the last).
    let mut step_results = Vec::with_capacity(reasoning.len() - 1);
    let mut weighted_sum = 0.0_f64;
    let mut total_weight = 0.0_f64;

    for step in &reasoning[..reasoning.len() - 1] {
        let raw_sim = similarity_fn(&step.content, goal_text);
        let sim = raw_sim.clamp(0.0, 1.0);
        let weight = (step.tokens as f64).max(1.0);

        weighted_sum += sim * weight;
        total_weight += weight;

        step_results.push(GarStepResult {
            step_id: step.id,
            goal_similarity: (sim * 1000.0).round() / 1000.0,
            low_advancement: sim < LOW_ADVANCEMENT_THRESHOLD,
        });
    }

    let score = if total_weight > 0.0 {
        (weighted_sum / total_weight).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let score = (score * 1000.0).round() / 1000.0;

    let low_advancement_steps: Vec<u32> = step_results
        .iter()
        .filter(|r| r.low_advancement)
        .map(|r| r.step_id)
        .collect();

    GarResult {
        score,
        step_results,
        low_advancement_steps,
        goal_step_id: Some(goal_step.id),
        pass: score >= TARGET,
        target: TARGET,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, Trace, TraceStep};
    use std::collections::HashMap;

    fn step(id: u32, content: &str, tokens: u32) -> TraceStep {
        TraceStep {
            id,
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
        }
    }

    fn make_trace(steps: Vec<TraceStep>) -> Trace {
        let total = steps.iter().map(|s| s.tokens).sum();
        Trace {
            trace_id: "gar-test".into(),
            agent_name: "agent".into(),
            framework: "raw".into(),
            steps,
            total_tokens: total,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    // Similarity that returns 1.0 for identical strings, 0.0 otherwise.
    fn exact_sim(a: &str, b: &str) -> f64 {
        if a == b { 1.0 } else { 0.0 }
    }

    // Similarity that always returns a fixed value.
    fn const_sim(v: f64) -> impl Fn(&str, &str) -> f64 {
        move |_, _| v
    }

    // ── single reasoning step ─────────────────────────────────────────────────

    #[test]
    fn single_reasoning_step_returns_perfect_score() {
        // With only one reasoning step there's nothing to measure; GAR = 1.0.
        let trace = make_trace(vec![step(1, "The answer is 42.", 100)]);
        let result = compute(&trace, exact_sim);
        assert_eq!(result.score, 1.0);
        assert!(result.pass);
        assert!(result.step_results.is_empty());
        assert_eq!(result.goal_step_id, Some(1));
    }

    #[test]
    fn no_reasoning_steps_returns_perfect_score() {
        // No reasoning steps at all → GAR = 1.0 (not applicable).
        let trace = make_trace(vec![]);
        let result = compute(&trace, exact_sim);
        assert_eq!(result.score, 1.0);
        assert!(result.step_results.is_empty());
        assert!(result.goal_step_id.is_none());
    }

    // ── perfect goal alignment ────────────────────────────────────────────────

    #[test]
    fn identical_content_scores_perfect() {
        // All intermediate steps have identical content to the goal → GAR = 1.0.
        let goal_text = "order ORD-1001 shipped";
        let trace = make_trace(vec![
            step(1, goal_text, 100),
            step(2, goal_text, 100),
            step(3, goal_text, 100), // goal proxy
        ]);
        let result = compute(&trace, exact_sim);
        assert_eq!(result.score, 1.0, "identical steps should score 1.0");
        assert!(result.low_advancement_steps.is_empty());
    }

    // ── zero goal alignment ───────────────────────────────────────────────────

    #[test]
    fn zero_similarity_scores_zero() {
        // All intermediate steps are completely dissimilar to the goal.
        let trace = make_trace(vec![
            step(1, "banana", 100),
            step(2, "apple", 100),
            step(3, "goal text", 100), // goal proxy
        ]);
        let result = compute(&trace, exact_sim);
        assert_eq!(result.score, 0.0);
        assert!(!result.pass);
        assert_eq!(result.low_advancement_steps, vec![1, 2]);
    }

    // ── token weighting ───────────────────────────────────────────────────────

    #[test]
    fn heavier_steps_dominate_weighted_average() {
        // Step 1 (1000 tokens, sim 1.0) should dominate step 2 (10 tokens, sim 0.0).
        // sim_fn: step 1 content == goal → 1.0; step 2 content ≠ goal → 0.0.
        let goal = "final answer";
        let trace = make_trace(vec![
            step(1, goal, 1000),  // heavy, on-topic
            step(2, "off topic", 10),  // light, off-topic
            step(3, goal, 50),    // goal proxy
        ]);
        let result = compute(&trace, exact_sim);
        // weighted = (1.0*1000 + 0.0*10) / 1010 ≈ 0.99
        assert!(result.score > 0.90, "heavy on-topic step should dominate: {}", result.score);
    }

    #[test]
    fn heavier_off_topic_step_lowers_score() {
        let goal = "final answer";
        let trace = make_trace(vec![
            step(1, goal, 10),       // light, on-topic
            step(2, "off topic", 1000), // heavy, off-topic
            step(3, goal, 50),       // goal proxy
        ]);
        let result = compute(&trace, exact_sim);
        // weighted = (1.0*10 + 0.0*1000) / 1010 ≈ 0.0099
        assert!(result.score < 0.05, "heavy off-topic step should dominate: {}", result.score);
        assert!(!result.pass);
    }

    // ── const similarity ─────────────────────────────────────────────────────

    #[test]
    fn const_sim_above_threshold_passes() {
        let trace = make_trace(vec![
            step(1, "a", 100),
            step(2, "b", 100),
            step(3, "c", 100),
        ]);
        let result = compute(&trace, const_sim(0.60));
        assert!((result.score - 0.60).abs() < 0.01);
        assert!(result.pass, "score 0.60 ≥ target 0.40 should pass");
    }

    #[test]
    fn const_sim_below_threshold_fails() {
        let trace = make_trace(vec![
            step(1, "a", 100),
            step(2, "b", 100),
            step(3, "c", 100),
        ]);
        let result = compute(&trace, const_sim(0.10));
        assert!(!result.pass, "score 0.10 < target 0.40 should fail");
        assert_eq!(result.low_advancement_steps.len(), 2);
    }

    // ── goal step excluded from results ──────────────────────────────────────

    #[test]
    fn goal_step_excluded_from_step_results() {
        let trace = make_trace(vec![
            step(1, "step one", 100),
            step(2, "step two", 100),
            step(3, "goal step", 100),
        ]);
        let result = compute(&trace, const_sim(0.5));
        // Only steps 1 and 2 should be in step_results (step 3 is the goal).
        assert_eq!(result.step_results.len(), 2);
        let ids: Vec<u32> = result.step_results.iter().map(|r| r.step_id).collect();
        assert!(!ids.contains(&3), "goal step must not appear in step_results");
        assert_eq!(result.goal_step_id, Some(3));
    }

    // ── non-reasoning steps are ignored ──────────────────────────────────────

    #[test]
    fn tool_calls_ignored_in_computation() {
        use crate::types::StepType;
        let mut tool_step = step(2, "tool call content", 500);
        tool_step.step_type = StepType::ToolCall;

        let trace = make_trace(vec![
            step(1, "reasoning one", 100),
            tool_step,                    // should be ignored
            step(3, "goal step", 100),
        ]);
        let result = compute(&trace, const_sim(0.5));
        // Only reasoning steps: 1 and 3. Step 3 is goal → only step 1 in results.
        assert_eq!(result.step_results.len(), 1);
        assert_eq!(result.step_results[0].step_id, 1);
    }

    // ── sim clamping ─────────────────────────────────────────────────────────

    #[test]
    fn similarity_values_clamped_to_unit_interval() {
        // sim_fn returns out-of-range values — GAR must clamp them.
        let trace = make_trace(vec![
            step(1, "a", 100),
            step(2, "b", 100),
            step(3, "c", 100),
        ]);
        let result = compute(&trace, |_, _| 1.5); // above 1.0
        assert!(result.score <= 1.0, "score must not exceed 1.0");

        let result2 = compute(&trace, |_, _| -0.3); // below 0.0
        assert!(result2.score >= 0.0, "score must not go below 0.0");
    }

    // ── normalised() helper ───────────────────────────────────────────────────

    #[test]
    fn normalised_matches_score_in_unit_interval() {
        let trace = make_trace(vec![step(1, "a", 100), step(2, "b", 100)]);
        let result = compute(&trace, const_sim(0.55));
        let n = result.normalised();
        assert!((n - result.score).abs() < 0.001, "normalised() should equal score for GAR");
    }
}
