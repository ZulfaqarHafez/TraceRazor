/// Trace Replay and Simulation (E-02)
///
/// Apply non-destructive mutations to a completed trace and project the impact
/// on TAS score, token count, and per-metric results — without re-running the
/// agent. All simulation is graph arithmetic in Rust; no LLM calls.
///
/// Supported mutations:
///   - `remove`  — delete specific steps entirely
///   - `merge`   — combine two steps into one (tokens = max(a,b) * 0.7)
///
/// # Example (CLI)
///
/// ```bash
/// tracerazor simulate trace.json --remove 3,8,9 --merge 6,7
/// ```
///
/// # Example (library)
///
/// ```rust,no_run
/// use tracerazor_core::simulate::{SimulationSpec, simulate};
/// use tracerazor_core::scoring::ScoringConfig;
/// use tracerazor_core::types::Trace;
///
/// // Assuming `trace` is a Trace loaded from JSON:
/// let trace: Trace = unimplemented!();
/// let spec = SimulationSpec {
///     remove: vec![3, 8, 9],
///     merge: vec![(6, 7)],
/// };
/// let result = simulate(&trace, &spec, &ScoringConfig::default(), |a, b| 0.0_f64);
/// println!("Original: {:.1} TAS", result.original_tas);
/// println!("Projected: {:.1} TAS  ({:+.1} delta)", result.projected_tas, result.tas_delta);
/// println!("Tokens: {} → {} ({:+})", result.original_tokens, result.projected_tokens, result.token_delta);
/// ```
use serde::{Deserialize, Serialize};

use crate::report::TraceReport;
use crate::scoring::ScoringConfig;
use crate::types::{Trace, TraceStep};

/// Specification of mutations to apply to a trace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimulationSpec {
    /// Step IDs to remove entirely.
    pub remove: Vec<u32>,
    /// Pairs of step IDs to merge. Each pair produces one combined step.
    /// Token count = max(a.tokens, b.tokens) * 0.7 (merging shortens content).
    pub merge: Vec<(u32, u32)>,
}

/// Result of a simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub original_tas: f64,
    pub projected_tas: f64,
    pub tas_delta: f64,
    pub original_tokens: u32,
    pub projected_tokens: u32,
    pub token_delta: i64,
    pub original_steps: usize,
    pub projected_steps: usize,
    /// Per-metric deltas (positive = improvement).
    pub metric_deltas: MetricDeltas,
    /// Steps kept in the simulated trace.
    pub simulated_step_ids: Vec<u32>,
}

/// Per-metric score deltas (projected − original).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDeltas {
    pub srr: f64,
    pub ldi: f64,
    pub tca: f64,
    pub tur: f64,
    pub cce: f64,
    pub rda: f64,
    pub isr: f64,
    pub dbo: f64,
    pub vdi: f64,
    pub shl: f64,
    pub ccr: f64,
}

/// Apply the spec to the trace and return the projected simulation result.
///
/// `similarity_fn` is the same BoW function used by the main analysis engine.
/// Passing `|_, _| 0.0` disables semantic comparisons (useful for fast estimates).
pub fn simulate<F>(
    trace: &Trace,
    spec: &SimulationSpec,
    config: &ScoringConfig,
    similarity_fn: F,
) -> SimulationResult
where
    F: Fn(&str, &str) -> f64,
{
    // ── Original analysis ─────────────────────────────────────────────────────
    let mut original = trace.clone();
    let original_report = crate::analyse(&mut original, |a, b| similarity_fn(a, b), config)
        .unwrap_or_else(|_| {
            // Return a zero-score report if analysis fails.
            placeholder_report(trace)
        });

    // ── Build mutated trace ───────────────────────────────────────────────────
    let mut steps: Vec<TraceStep> = trace
        .steps
        .iter()
        .filter(|s| !spec.remove.contains(&s.id))
        .cloned()
        .collect();

    // Apply merges: combine token counts, keep first step's metadata.
    for &(id_a, id_b) in &spec.merge {
        // Skip if either was already removed.
        if spec.remove.contains(&id_a) || spec.remove.contains(&id_b) {
            continue;
        }
        if let (Some(pos_a), Some(pos_b)) = (
            steps.iter().position(|s| s.id == id_a),
            steps.iter().position(|s| s.id == id_b),
        ) {
            let tokens_a = steps[pos_a].tokens;
            let tokens_b = steps[pos_b].tokens;
            // Merged token estimate: max of the two * 0.7 (merging produces shorter output).
            let merged_tokens = (tokens_a.max(tokens_b) as f64 * 0.7) as u32;
            steps[pos_a].tokens = merged_tokens;
            steps[pos_a].content = format!(
                "{} [merged with step {id_b}]",
                steps[pos_a].content
            );
            // Remove step B (mark for deletion by resetting its id temporarily).
            let _removed = steps.remove(pos_b);
        }
    }

    // Re-number steps to keep IDs contiguous.
    let simulated_ids: Vec<u32> = steps.iter().map(|s| s.id).collect();
    for (i, step) in steps.iter_mut().enumerate() {
        step.id = (i + 1) as u32;
    }

    let projected_tokens: u32 = steps.iter().map(|s| s.tokens).sum();

    // ── Projected analysis ────────────────────────────────────────────────────
    let projected_report = if steps.len() >= crate::types::MIN_TRACE_STEPS {
        let mut projected_trace = Trace {
            trace_id: format!("{}-sim", trace.trace_id),
            agent_name: trace.agent_name.clone(),
            framework: trace.framework.clone(),
            steps,
            total_tokens: projected_tokens,
            task_value_score: trace.task_value_score,
            metadata: trace.metadata.clone(),
        };
        crate::analyse(&mut projected_trace, |a, b| similarity_fn(a, b), config)
            .unwrap_or_else(|_| placeholder_report(&projected_trace))
    } else {
        // Too few steps for a full analysis — build a minimal placeholder.
        let stub = Trace {
            trace_id: format!("{}-sim", trace.trace_id),
            agent_name: trace.agent_name.clone(),
            framework: trace.framework.clone(),
            steps: steps.clone(),
            total_tokens: projected_tokens,
            task_value_score: trace.task_value_score,
            metadata: trace.metadata.clone(),
        };
        placeholder_report(&stub)
    };

    let original_tas = original_report.score.score;
    let projected_tas = projected_report.score.score;

    SimulationResult {
        original_tas,
        projected_tas,
        tas_delta: (projected_tas - original_tas * 10.0).round() / 10.0,
        original_tokens: original_report.total_tokens,
        projected_tokens: projected_report.total_tokens,
        token_delta: projected_report.total_tokens as i64 - original_report.total_tokens as i64,
        original_steps: original_report.total_steps,
        projected_steps: projected_report.total_steps,
        metric_deltas: MetricDeltas {
            srr: projected_report.score.srr.normalised()
                - original_report.score.srr.normalised(),
            ldi: projected_report.score.ldi.normalised()
                - original_report.score.ldi.normalised(),
            tca: projected_report.score.tca.normalised()
                - original_report.score.tca.normalised(),
            tur: projected_report.score.tur.normalised()
                - original_report.score.tur.normalised(),
            cce: projected_report.score.cce.normalised()
                - original_report.score.cce.normalised(),
            rda: projected_report.score.rda.normalised()
                - original_report.score.rda.normalised(),
            isr: projected_report.score.isr.normalised()
                - original_report.score.isr.normalised(),
            dbo: projected_report.score.dbo.normalised()
                - original_report.score.dbo.normalised(),
            vdi: projected_report.score.vdi.normalised()
                - original_report.score.vdi.normalised(),
            shl: projected_report.score.shl.normalised()
                - original_report.score.shl.normalised(),
            ccr: projected_report.score.ccr.normalised()
                - original_report.score.ccr.normalised(),
        },
        simulated_step_ids: simulated_ids,
    }
}

fn placeholder_report(trace: &Trace) -> TraceReport {
    use crate::metrics::{
        ccr::{CcrResult, CcrStepResult},
        dbo::DboResult,
        isr::IsrResult,
        ldi::LdiResult,
        rda::{RdaResult, TaskComplexity},
        shl::ShlResult,
        srr::SrrResult,
        tca::TcaResult,
        tur::TurResult,
        cce::CceResult,
        vdi::{VdiResult, VdiStepResult},
    };
    use crate::scoring::{Grade, SavingsEstimate, TasScore};

    let zero_srr = SrrResult {
        score: 0.0,
        redundant_steps: vec![],
        redundant_count: 0,
        total_steps: trace.steps.len(),
        pass: true,
        target: 15.0,
    };
    let zero_ldi = LdiResult {
        score: 0.0,
        loops: vec![],
        max_cycle_length: 0,
        total_steps: trace.steps.len(),
        pass: true,
        warning_threshold: 0.1,
    };
    let zero_tca = TcaResult {
        score: 100.0,
        misfires: vec![],
        successful_first_attempts: 0,
        total_tool_calls: 0,
        pass: true,
        target: 85.0,
    };
    let zero_tur = TurResult {
        score: 1.0,
        useful_tokens: 0,
        total_tokens: 0,
        wasted_tokens: 0,
        pass: true,
        target: 0.35,
    };
    let zero_cce = CceResult {
        score: 1.0,
        total_input_tokens: 0,
        duplicate_tokens: 0,
        bloated_steps: vec![],
        pass: true,
        target: 0.60,
    };
    let zero_rda = RdaResult {
        score: 0.75,
        classified_complexity: TaskComplexity::Moderate,
        expected_steps: 4.0,
        actual_steps: trace.steps.len(),
        uses_historical_baseline: false,
        pass: true,
        target: 0.75,
    };
    let zero_isr = IsrResult {
        score: 80.0,
        steps_with_novel_info: trace.steps.len(),
        total_steps: trace.steps.len(),
        low_novelty_steps: vec![],
        pass: true,
        target: 80.0,
    };
    let zero_dbo = DboResult {
        score: 0.7,
        optimal_selections: 0,
        total_branch_points: 0,
        decisions: vec![],
        cold_start: true,
        pass: true,
        target: 0.70,
    };
    let zero_vdi = VdiResult {
        score: 1.0,
        step_results: trace
            .steps
            .iter()
            .map(|s| VdiStepResult {
                step_id: s.id,
                score: 1.0,
                filler_count: 0,
                total_words: 0,
                low_density: false,
                entropy_flagged: false,
            })
            .collect(),
        low_density_steps: vec![],
        entropy_low_steps: vec![],
        pass: true,
        target: 0.60,
    };
    let zero_shl = ShlResult {
        score: 0.0,
        flagged_sentences: 0,
        total_sentences: 0,
        pass: true,
        target: 0.20,
    };
    let zero_ccr = CcrResult {
        score: 0.0,
        step_results: trace
            .steps
            .iter()
            .map(|s| CcrStepResult {
                step_id: s.id,
                original_tokens: 0,
                compressed_tokens: 0,
                ratio: 0.0,
            })
            .collect(),
        total_cuttable_tokens: 0,
        pass: true,
        target: 0.30,
    };

    let step_msg = format!(
        "Trace {} has {} steps (below minimum for analysis).",
        trace.trace_id,
        trace.steps.len()
    );
    TraceReport {
        trace_id: trace.trace_id.clone(),
        agent_name: trace.agent_name.clone(),
        framework: trace.framework.clone(),
        total_steps: trace.steps.len(),
        total_tokens: trace.effective_total_tokens(),
        analysis_duration_ms: 0,
        score: TasScore {
            score: 0.0,
            grade: Grade::Poor,
            vae: 0.0,
            passes_threshold: false,
            avs: 0.0,
            srr: zero_srr,
            ldi: zero_ldi,
            tca: zero_tca,
            tur: zero_tur,
            cce: zero_cce,
            rda: zero_rda,
            isr: zero_isr,
            dbo: zero_dbo,
            vdi: zero_vdi,
            shl: zero_shl,
            ccr: zero_ccr,
        },
        diff: vec![],
        savings: SavingsEstimate {
            tokens_saved: 0,
            reduction_pct: 0.0,
            cost_saved_per_run_usd: 0.0,
            monthly_savings_usd: 0.0,
            latency_saved_seconds: 0.0,
        },
        fixes: vec![],
        summary: step_msg.clone(),
        summary_oneliner: step_msg,
        anomalies: vec![],
        per_agent: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, TraceStep};
    use std::collections::HashMap;

    fn make_trace() -> Trace {
        Trace {
            trace_id: "t1".into(),
            agent_name: "agent".into(),
            framework: "raw".into(),
            steps: (1..=7)
                .map(|id| TraceStep {
                    id,
                    step_type: if id % 2 == 0 {
                        StepType::ToolCall
                    } else {
                        StepType::Reasoning
                    },
                    content: format!("step {id} content about order processing"),
                    tokens: 400,
                    tool_name: if id % 2 == 0 {
                        Some(format!("tool_{id}"))
                    } else {
                        None
                    },
                    tool_params: None,
                    tool_success: Some(true),
                    tool_error: None,
                    agent_id: None,
                    input_context: None,
                    output: None,
                    flags: vec![],
                    flag_details: vec![],
                })
                .collect(),
            total_tokens: 2800,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_simulate_remove_steps() {
        let trace = make_trace();
        let spec = SimulationSpec {
            remove: vec![3, 5],
            merge: vec![],
        };
        let result = simulate(&trace, &spec, &ScoringConfig::default(), |_, _| 0.0);
        assert_eq!(result.original_steps, 7);
        assert!(result.projected_steps <= 5);
        assert!(result.token_delta < 0, "Removing steps should reduce tokens");
    }

    #[test]
    fn test_simulate_merge_steps() {
        let trace = make_trace();
        let spec = SimulationSpec {
            remove: vec![],
            merge: vec![(1, 2)],
        };
        let result = simulate(&trace, &spec, &ScoringConfig::default(), |_, _| 0.0);
        assert!(result.projected_steps < result.original_steps);
        assert!(result.token_delta < 0, "Merging steps should reduce tokens");
    }
}
