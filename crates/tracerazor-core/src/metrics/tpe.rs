/// Trajectory Path Entropy (TPE) — an information-theoretic "staying on the path" signal.
///
/// Every other path-oriented metric in TraceRazor (GAR, CSD) reduces to a *mean
/// cosine similarity*. TPE is different: it is a genuine Shannon-entropy measure
/// of how *directed* the agent's trajectory is toward its goal.
///
/// ## What it measures
///
/// We score each reasoning step by its similarity to the goal, producing a
/// sequence `g_1, g_2, … g_n`. The step-to-step increments `Δ_i = g_{i+1} − g_i`
/// are classified into three symbols:
///
/// * **ADVANCE** — `Δ_i >  ε`  (the step moved closer to the goal)
/// * **STALL**   — `|Δ_i| ≤ ε` (the step neither advanced nor regressed)
/// * **REGRESS** — `Δ_i < −ε`  (the step moved *away* from the goal)
///
/// Over the move-classes that actually occur we compute the normalised Shannon
/// entropy (Pielou's evenness — the entropy is normalised by `log2(k)` where
/// `k` is the number of *distinct* classes observed, so a trajectory that only
/// ever advances-and-regresses can still reach 1.0):
///
/// ```text
/// H = − Σ p(s) · log2 p(s)          (s ∈ observed ⊆ {ADVANCE, STALL, REGRESS})
/// path_entropy = H / log2(k)        ∈ [0, 1]   (k = |observed|, 0 when k ≤ 1)
/// ```
///
/// `path_entropy → 0` means the trajectory is *predictable* — the agent makes
/// the same kind of move every step (a clean monotonic climb to the goal, or a
/// clean failure). `path_entropy → 1` means the trajectory is a *high-surprisal
/// random walk* — the agent lurches toward and away from the goal with no
/// consistent direction. This is exactly the "entropy of an agent staying on a
/// path" that mean-similarity cannot express.
///
/// Because pure entropy cannot tell "consistently advancing" from "consistently
/// regressing" (both are low-entropy), we combine it with a signed
/// **directedness** term `D = (advances − regresses) / increments ∈ [−1, 1]` to
/// produce an actionable composite:
///
/// ```text
/// focus_score = clamp( (D + 1) / 2 − 0.25 · path_entropy , 0, 1 )
/// ```
///
/// | Trajectory                       | path_entropy | D     | focus_score |
/// |----------------------------------|--------------|-------|-------------|
/// | Monotonic climb to goal          | 0.0          | +1.0  | 1.00        |
/// | Steady drift away from goal      | 0.0          | −1.0  | 0.00        |
/// | Stalls (no movement)             | 0.0          |  0.0  | 0.50        |
/// | Erratic advance/regress lurching | ~1.0         |  ~0.0 | 0.25        |
///
/// ## Honesty note
///
/// TPE is a **diagnostic** signal reported alongside TAS; it is *not* folded
/// into the weighted TAS composite (so the published per-metric shares are
/// unaffected). The goal it measures against is the real task goal when the
/// trace carries one in `metadata` (`task` / `goal` / `objective` / …);
/// otherwise it falls back to the agent's final reasoning step — `goal_origin`
/// records which was used so the number is never silently mis-anchored.
use serde::{Deserialize, Serialize};

use crate::types::{StepType, Trace, TraceStep};

/// Similarity delta below this magnitude counts as a STALL (similarity noise floor).
pub const ADVANCE_EPSILON: f64 = 0.02;
/// `focus_score` at or above this is considered "on-path".
pub const FOCUS_TARGET: f64 = 0.50;

/// How the goal text was resolved for this computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalOrigin {
    /// Goal came from the trace's task metadata (the real objective).
    TaskGoal,
    /// No task goal available — the agent's final reasoning step was used.
    FinalStep,
    /// Not enough reasoning steps to measure a trajectory.
    NotApplicable,
}

/// Result of the TPE computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpeResult {
    /// Normalised Shannon entropy of the advance/stall/regress distribution,
    /// in [0, 1]. Higher = more disordered / off-path trajectory.
    pub path_entropy: f64,
    /// Signed directedness `(advances − regresses) / increments`, in [−1, 1].
    pub directedness: f64,
    /// Actionable composite, in [0, 1]. Higher = more focused / on-path.
    pub focus_score: f64,
    /// Descriptive normalised entropy of the agent's action-type distribution
    /// (tool names + step kinds), in [0, 1]. Reported for context; **not** part
    /// of `focus_score` (a complex task legitimately uses many distinct tools).
    pub action_entropy: f64,
    pub advances: usize,
    pub stalls: usize,
    pub regresses: usize,
    /// Number of distinct action types observed.
    pub distinct_actions: usize,
    /// Which goal the trajectory was measured against.
    pub goal_origin: GoalOrigin,
    /// True when `focus_score < FOCUS_TARGET` — the agent meaningfully drifted.
    pub high_drift: bool,
    pub pass: bool,
    pub target: f64,
}

impl Default for TpeResult {
    fn default() -> Self {
        TpeResult {
            path_entropy: 0.0,
            directedness: 0.0,
            focus_score: 1.0,
            action_entropy: 0.0,
            advances: 0,
            stalls: 0,
            regresses: 0,
            distinct_actions: 0,
            goal_origin: GoalOrigin::NotApplicable,
            high_drift: false,
            pass: true,
            target: FOCUS_TARGET,
        }
    }
}

impl TpeResult {
    /// `focus_score` normalised for an optional TAS contribution (higher = better).
    pub fn normalised(&self) -> f64 {
        self.focus_score.clamp(0.0, 1.0)
    }

    /// One-word interpretation of the trajectory.
    pub fn interpretation(&self) -> &'static str {
        if self.goal_origin == GoalOrigin::NotApplicable {
            "n/a"
        } else if self.focus_score >= 0.75 {
            "focused"
        } else if self.focus_score >= FOCUS_TARGET {
            "wandering"
        } else if self.directedness < -0.1 {
            "regressing"
        } else {
            "scattered"
        }
    }
}

/// Normalised Shannon entropy (base-2) of a symbol-count distribution.
/// Returns 0.0 for an empty or single-symbol distribution.
fn normalised_entropy(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    let observed: Vec<&usize> = counts.iter().filter(|&&c| c > 0).collect();
    if total == 0 || observed.len() <= 1 {
        return 0.0;
    }
    let total_f = total as f64;
    let h: f64 = observed
        .iter()
        .map(|&&c| {
            let p = c as f64 / total_f;
            -p * p.log2()
        })
        .sum();
    // Normalise by the maximum possible entropy for the number of *observed*
    // symbols so a perfectly uniform distribution maps to 1.0.
    let max_h = (observed.len() as f64).log2();
    if max_h > 0.0 {
        (h / max_h).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Compute Trajectory Path Entropy for a trace.
///
/// `goal` is the real task objective when known; pass `None` to fall back to the
/// agent's final reasoning step. `similarity_fn` is the same BoW/embedding
/// closure used by GAR and CSD.
pub fn compute(
    trace: &Trace,
    similarity_fn: impl Fn(&str, &str) -> f64,
    goal: Option<&str>,
) -> TpeResult {
    // ── Action-type entropy (descriptive; computed over all steps) ───────────
    let mut action_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for s in &trace.steps {
        let key = match s.step_type {
            StepType::ToolCall => s.tool_name.clone().unwrap_or_else(|| "tool".to_string()),
            StepType::Reasoning => "reasoning".to_string(),
            StepType::Handoff => "handoff".to_string(),
            StepType::Unknown => "unknown".to_string(),
        };
        *action_counts.entry(key).or_insert(0) += 1;
    }
    let action_vec: Vec<usize> = action_counts.values().copied().collect();
    let action_entropy = normalised_entropy(&action_vec);
    let distinct_actions = action_counts.len();

    // ── Goal-progress trajectory (over reasoning steps) ──────────────────────
    let reasoning: Vec<_> = trace
        .steps
        .iter()
        .filter(|s| s.step_type == StepType::Reasoning)
        .collect();

    // Need at least 3 reasoning steps to form ≥2 increments and measure entropy.
    if reasoning.len() < 3 {
        return TpeResult {
            action_entropy,
            distinct_actions,
            ..TpeResult::default()
        };
    }

    let (goal_text, goal_origin) = match goal {
        Some(g) => (g.to_string(), GoalOrigin::TaskGoal),
        None => (
            reasoning.last().unwrap().content.clone(),
            GoalOrigin::FinalStep,
        ),
    };

    // Steps that form the trajectory toward the goal. When the goal IS the final
    // step (no external goal), exclude it — scoring it against itself would
    // manufacture a spurious final ADVANCE (sim(last, last) ≈ 1.0) and bias the
    // trajectory toward "focused". This mirrors GAR's None-path handling.
    let trajectory: &[&TraceStep] = match goal_origin {
        GoalOrigin::FinalStep => &reasoning[..reasoning.len() - 1],
        _ => &reasoning[..],
    };

    // Per-step similarity to the goal.
    let g: Vec<f64> = trajectory
        .iter()
        .map(|s| similarity_fn(&s.content, &goal_text).clamp(0.0, 1.0))
        .collect();

    // Classify increments.
    let (mut advances, mut stalls, mut regresses) = (0usize, 0usize, 0usize);
    for w in g.windows(2) {
        let delta = w[1] - w[0];
        if delta > ADVANCE_EPSILON {
            advances += 1;
        } else if delta < -ADVANCE_EPSILON {
            regresses += 1;
        } else {
            stalls += 1;
        }
    }

    let increments = advances + stalls + regresses;
    let path_entropy = normalised_entropy(&[advances, stalls, regresses]);
    let directedness = if increments > 0 {
        (advances as f64 - regresses as f64) / increments as f64
    } else {
        0.0
    };

    let focus_score = (((directedness + 1.0) / 2.0) - 0.25 * path_entropy).clamp(0.0, 1.0);

    let round3 = |x: f64| (x * 1000.0).round() / 1000.0;
    let focus_score = round3(focus_score);
    let high_drift = focus_score < FOCUS_TARGET;

    TpeResult {
        path_entropy: round3(path_entropy),
        directedness: round3(directedness),
        focus_score,
        action_entropy: round3(action_entropy),
        advances,
        stalls,
        regresses,
        distinct_actions,
        goal_origin,
        high_drift,
        pass: focus_score >= FOCUS_TARGET,
        target: FOCUS_TARGET,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, TraceStep};
    use std::collections::HashMap;

    fn rstep(id: u32, content: &str) -> TraceStep {
        TraceStep {
            id,
            step_type: StepType::Reasoning,
            content: content.into(),
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
            trace_id: "tpe-test".into(),
            agent_name: "agent".into(),
            framework: "raw".into(),
            steps,
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Helper: build a trace whose i-th reasoning step has goal-similarity g[i]
    /// under `goal_progress_sim`.
    fn progress_trace(progress: &[f64]) -> Trace {
        let steps: Vec<TraceStep> = progress
            .iter()
            .enumerate()
            .map(|(i, p)| rstep((i + 1) as u32, &format!("{p}")))
            .collect();
        make_trace(steps)
    }

    // Similarity = the float parsed from step `a`'s content (goal arg ignored).
    fn goal_progress_sim(a: &str, _goal: &str) -> f64 {
        a.trim().parse::<f64>().unwrap_or(0.0)
    }

    #[test]
    fn monotonic_climb_is_focused_low_entropy() {
        // Strictly increasing similarity to goal → all ADVANCE.
        let trace = progress_trace(&[0.1, 0.3, 0.5, 0.7, 0.9]);
        let r = compute(&trace, goal_progress_sim, Some("GOAL"));
        assert_eq!(r.advances, 4);
        assert_eq!(r.regresses, 0);
        assert_eq!(r.stalls, 0);
        assert_eq!(r.path_entropy, 0.0, "single symbol → zero entropy");
        assert!(
            r.focus_score > 0.99,
            "monotonic climb → focus ~1.0, got {}",
            r.focus_score
        );
        assert!(r.pass && !r.high_drift);
        assert_eq!(r.interpretation(), "focused");
    }

    #[test]
    fn steady_regress_is_low_entropy_but_low_focus() {
        // Strictly decreasing → all REGRESS: predictable (entropy 0) but off-path.
        let trace = progress_trace(&[0.9, 0.7, 0.5, 0.3, 0.1]);
        let r = compute(&trace, goal_progress_sim, Some("GOAL"));
        assert_eq!(r.regresses, 4);
        assert_eq!(r.path_entropy, 0.0);
        assert!(
            r.focus_score < 0.01,
            "steady regress → focus ~0, got {}",
            r.focus_score
        );
        assert!(r.high_drift && !r.pass);
        assert_eq!(r.interpretation(), "regressing");
    }

    #[test]
    fn erratic_walk_is_high_entropy_low_focus() {
        // Lurching up and down → mix of advance/regress → high entropy.
        let trace = progress_trace(&[0.5, 0.9, 0.2, 0.8, 0.1, 0.7]);
        let r = compute(&trace, goal_progress_sim, Some("GOAL"));
        assert!(
            r.path_entropy > 0.8,
            "erratic walk → high entropy, got {}",
            r.path_entropy
        );
        assert!(
            r.focus_score < FOCUS_TARGET,
            "erratic → drift, got {}",
            r.focus_score
        );
        assert!(r.high_drift);
    }

    #[test]
    fn stalled_trajectory_is_neutral() {
        // No movement → all STALL → entropy 0, directedness 0 → focus 0.5.
        let trace = progress_trace(&[0.5, 0.5, 0.5, 0.5]);
        let r = compute(&trace, goal_progress_sim, Some("GOAL"));
        assert_eq!(r.stalls, 3);
        assert_eq!(r.path_entropy, 0.0);
        assert!(
            (r.focus_score - 0.5).abs() < 0.001,
            "stall → 0.5, got {}",
            r.focus_score
        );
        assert_eq!(r.interpretation(), "wandering");
    }

    #[test]
    fn fewer_than_three_reasoning_steps_is_not_applicable() {
        let trace = progress_trace(&[0.2, 0.8]);
        let r = compute(&trace, goal_progress_sim, Some("GOAL"));
        assert_eq!(r.goal_origin, GoalOrigin::NotApplicable);
        assert!(r.pass && !r.high_drift);
        assert_eq!(r.focus_score, 1.0);
    }

    #[test]
    fn goal_origin_reflects_fallback() {
        let trace = progress_trace(&[0.1, 0.4, 0.9]);
        let with_goal = compute(&trace, goal_progress_sim, Some("GOAL"));
        assert_eq!(with_goal.goal_origin, GoalOrigin::TaskGoal);
        let without = compute(&trace, goal_progress_sim, None);
        assert_eq!(without.goal_origin, GoalOrigin::FinalStep);
    }

    #[test]
    fn action_entropy_zero_for_uniform_single_action() {
        // All reasoning steps → one action type → action entropy 0.
        let trace = progress_trace(&[0.1, 0.2, 0.3, 0.4]);
        let r = compute(&trace, goal_progress_sim, Some("GOAL"));
        assert_eq!(r.distinct_actions, 1);
        assert_eq!(r.action_entropy, 0.0);
    }

    #[test]
    fn action_entropy_rises_with_tool_variety() {
        // Mix reasoning with several distinct tools → non-zero action entropy.
        let mut steps = vec![rstep(1, "0.1"), rstep(2, "0.4"), rstep(3, "0.9")];
        for (i, name) in ["a", "b", "c"].iter().enumerate() {
            let mut t = rstep((10 + i) as u32, "tool");
            t.step_type = StepType::ToolCall;
            t.tool_name = Some((*name).into());
            steps.push(t);
        }
        let trace = make_trace(steps);
        let r = compute(&trace, goal_progress_sim, Some("GOAL"));
        assert!(r.distinct_actions >= 4);
        assert!(r.action_entropy > 0.0, "varied actions → entropy > 0");
    }

    #[test]
    fn normalised_entropy_uniform_two_symbols_is_one() {
        assert!((normalised_entropy(&[5, 5]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn normalised_entropy_single_symbol_is_zero() {
        assert_eq!(normalised_entropy(&[7, 0, 0]), 0.0);
        assert_eq!(normalised_entropy(&[0]), 0.0);
    }

    #[test]
    fn default_is_applicable_and_focused() {
        let d = TpeResult::default();
        assert!(d.pass);
        assert_eq!(d.focus_score, 1.0);
        assert_eq!(d.goal_origin, GoalOrigin::NotApplicable);
    }

    #[test]
    fn final_step_path_excludes_goal_step_from_trajectory() {
        // No task goal → the final step IS the goal and must be excluded from
        // the trajectory, otherwise sim(last, last) ≈ 1.0 fabricates an extra
        // final increment (the self-comparison bug). With 4 reasoning steps the
        // trajectory is the first 3, giving exactly 2 increments, not 3.
        let trace = progress_trace(&[0.2, 0.5, 0.8, 0.95]);
        let r = compute(&trace, goal_progress_sim, None);
        assert_eq!(r.goal_origin, GoalOrigin::FinalStep);
        let increments = r.advances + r.stalls + r.regresses;
        assert_eq!(
            increments, 2,
            "goal step must be excluded from the trajectory"
        );
        assert_eq!(r.advances, 2);
        assert_eq!(r.regresses, 0);
        assert_eq!(r.stalls, 0);
    }
}
