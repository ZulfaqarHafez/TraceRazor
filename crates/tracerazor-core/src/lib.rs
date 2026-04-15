pub mod cost;
pub mod fixes;
pub mod graph;
pub mod metrics;
pub mod report;
pub mod scoring;
pub mod simulate;
pub mod types;

use std::time::Instant;

use anyhow::Result;

use crate::fixes::generate_fixes;
use crate::metrics::{ccr, cce, dbo, isr, ldi, rda, reformulation, shl, srr, tca, tur, vdi};
use crate::report::{AgentBreakdown, TraceReport, generate_oneliner, generate_summary};
use crate::scoring::{ScoringConfig, estimate_savings};
use crate::types::{MIN_TRACE_STEPS, Trace};

/// Analyse a trace and compute all eight TAS metrics.
///
/// All metrics are local — no network calls required. RDA uses the heuristic
/// complexity classifier (falling back to historical median if provided in
/// `config.historical_median_steps`). DBO uses historical tool sequences from
/// `config.historical_sequences`; cold-starts to 0.7 when fewer than 10
/// similar sequences are available.
///
/// `similarity_fn` is injected from `tracerazor-semantic` so that `core`
/// remains independent of the embedding backend.
pub fn analyse<F>(
    trace: &mut Trace,
    similarity_fn: F,
    config: &ScoringConfig,
) -> Result<TraceReport>
where
    F: Fn(&str, &str) -> f64,
{
    // Type-erase to break the generic recursion chain in compute_per_agent_scores.
    analyse_dyn(trace, &similarity_fn as &dyn Fn(&str, &str) -> f64, config)
}

/// Internal implementation using a `dyn Fn` reference so that
/// `compute_per_agent_scores` can call back into it without creating an
/// infinitely-deep monomorphization chain.
fn analyse_dyn(
    trace: &mut Trace,
    similarity_fn: &dyn Fn(&str, &str) -> f64,
    config: &ScoringConfig,
) -> Result<TraceReport> {
    let start = Instant::now();
    let total_tokens = trace.effective_total_tokens();

    // ── Structural metrics ────────────────────────────────────────────────────
    let srr_result = srr::compute(trace, similarity_fn, None);
    let ldi_result = ldi::compute(trace);
    let tca_result = tca::compute(trace);

    srr::annotate_steps(&mut trace.steps, &srr_result);
    ldi::annotate_steps(&mut trace.steps, &ldi_result);
    tca::annotate_steps(&mut trace.steps, &tca_result);

    let tur_result = tur::compute(trace);
    let cce_result = cce::compute(trace);
    cce::annotate_steps(&mut trace.steps, &cce_result);

    // ── Information / semantic metrics (BoW, no external calls) ───────────────
    let isr_result = isr::compute_from_similarities(trace, similarity_fn);

    // ── Local-first RDA (heuristic classifier, optional historical baseline) ──
    let rda_result = rda::compute(trace, config.historical_median_steps);

    // ── Local-first DBO (historical comparison, cold-start when < 10 traces) ──
    let dbo_result = dbo::compute(trace, &config.historical_sequences);

    // ── Verbosity metrics (v2: VDI, SHL, CCR) ────────────────────────────────
    let vdi_result = vdi::compute(trace);
    let shl_result = shl::compute(trace);
    let ccr_result = ccr::compute(trace);

    // ── Reformulation detection (P2) ──────────────────────────────────────────
    let reformulation_detected = reformulation::detect(trace);
    reformulation::annotate_steps(&mut trace.steps, &reformulation_detected);

    let score = scoring::compute(
        srr_result,
        ldi_result,
        tca_result,
        tur_result,
        cce_result,
        rda_result,
        isr_result,
        dbo_result,
        vdi_result,
        shl_result,
        ccr_result,
        trace.task_value_score,
        total_tokens,
        config,
    );

    let elapsed = start.elapsed().as_millis() as u64;
    let diff = TraceReport::build_diff(trace, &score);
    let optimal_tokens = TraceReport::optimal_tokens(&diff);
    let waste_tokens = total_tokens.saturating_sub(optimal_tokens);
    let savings = estimate_savings(total_tokens, waste_tokens, config, None);

    // ── M3: Minimum Viable Trace Gap ─────────────────────────────────────────
    // Fraction of tokens above the diff-optimal path (0.0 = perfectly lean).
    // This is a structural lower bound: it only counts steps the diff already
    // classified as DELETE or TRIM, with no speculative fix estimates.
    let mvtg = if total_tokens > 0 {
        (waste_tokens as f64 / total_tokens as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // ── E-01: auto-fix generation ─────────────────────────────────────────────
    let generated_fixes = generate_fixes(trace, &score);

    // ── E-08: template-based NL summaries ────────────────────────────────────
    let summary = generate_summary(trace, &score, &savings);
    let summary_oneliner = generate_oneliner(trace, &score, &savings);

    // ── Decision 7: per-agent breakdown for multi-agent traces ────────────────
    let per_agent = compute_per_agent_scores(trace, similarity_fn, config, total_tokens);

    Ok(TraceReport {
        trace_id: trace.trace_id.clone(),
        agent_name: trace.agent_name.clone(),
        framework: trace.framework.clone(),
        total_steps: trace.steps.len(),
        total_tokens,
        analysis_duration_ms: elapsed,
        score,
        diff,
        savings,
        mvtg,
        fixes: generated_fixes,
        summary,
        summary_oneliner,
        anomalies: vec![], // populated by the store layer after analysis
        per_agent,
    })
}

/// Compute per-agent TAS scores for multi-agent traces.
///
/// Returns an empty vec if fewer than 2 distinct `agent_id` values are present.
/// Uses `analyse_dyn` (concrete function) to break the generic recursion chain.
fn compute_per_agent_scores(
    trace: &Trace,
    similarity_fn: &dyn Fn(&str, &str) -> f64,
    config: &ScoringConfig,
    total_tokens: u32,
) -> Vec<AgentBreakdown> {
    // Collect ordered distinct agent IDs (preserving first-seen order).
    let mut seen = std::collections::HashSet::new();
    let agent_ids: Vec<String> = trace
        .steps
        .iter()
        .filter_map(|s| s.agent_id.as_ref())
        .filter(|id| seen.insert(id.as_str()))
        .map(|id| id.to_string())
        .collect();

    if agent_ids.len() < 2 {
        return vec![];
    }

    let total_f = total_tokens.max(1) as f64;

    agent_ids
        .iter()
        .map(|agent_id| {
            let steps: Vec<_> = trace
                .steps
                .iter()
                .filter(|s| s.agent_id.as_deref() == Some(agent_id.as_str()))
                .cloned()
                .collect();

            let agent_tokens: u32 = steps.iter().map(|s| s.tokens).sum();
            let token_share_pct = (agent_tokens as f64 / total_f * 1000.0).round() / 10.0;
            let step_count = steps.len();

            if step_count >= MIN_TRACE_STEPS {
                let mut sub = Trace {
                    trace_id: format!("{}-{}", trace.trace_id, agent_id),
                    agent_name: agent_id.clone(),
                    framework: trace.framework.clone(),
                    steps,
                    total_tokens: agent_tokens,
                    task_value_score: trace.task_value_score,
                    metadata: Default::default(),
                };
                // analyse_dyn is a concrete (non-generic) function — no recursion risk.
                match analyse_dyn(&mut sub, similarity_fn, config) {
                    Ok(r) => AgentBreakdown {
                        agent_id: agent_id.clone(),
                        total_steps: r.total_steps,
                        total_tokens: r.total_tokens,
                        token_share_pct,
                        tas_score: Some(r.score.score),
                        grade: Some(r.score.grade.to_string()),
                    },
                    Err(_) => AgentBreakdown {
                        agent_id: agent_id.clone(),
                        total_steps: step_count,
                        total_tokens: agent_tokens,
                        token_share_pct,
                        tas_score: None,
                        grade: None,
                    },
                }
            } else {
                AgentBreakdown {
                    agent_id: agent_id.clone(),
                    total_steps: step_count,
                    total_tokens: agent_tokens,
                    token_share_pct,
                    tas_score: None,
                    grade: None,
                }
            }
        })
        .collect()
}

/// Returns true if the trace has enough steps to be analysed.
pub fn is_analysable(trace: &Trace) -> bool {
    trace.steps.len() >= MIN_TRACE_STEPS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, Trace, TraceStep};
    use std::collections::HashMap;

    fn simple_sim(a: &str, b: &str) -> f64 {
        let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();
        if words_a.is_empty() || words_b.is_empty() {
            return 0.0;
        }
        let intersect = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();
        intersect as f64 / union as f64
    }

    fn make_trace() -> Trace {
        Trace {
            trace_id: "test-run-001".into(),
            agent_name: "test-agent".into(),
            framework: "raw".into(),
            steps: vec![
                TraceStep {
                    id: 1,
                    step_type: StepType::Reasoning,
                    content: "Parse the user request about order refund".into(),
                    tokens: 820,
                    tool_name: None,
                    tool_params: None,
                    tool_success: None,
                    tool_error: None,
                    agent_id: None,
                    input_context: None,
                    output: None,
                    flags: vec![],
                    flag_details: vec![],
                },
                TraceStep {
                    id: 2,
                    step_type: StepType::ToolCall,
                    content: "Fetch order details".into(),
                    tokens: 340,
                    tool_name: Some("get_order_details".into()),
                    tool_params: Some(serde_json::json!({"order_id": "ORD-9182"})),
                    tool_success: Some(true),
                    tool_error: None,
                    agent_id: None,
                    input_context: None,
                    output: None,
                    flags: vec![],
                    flag_details: vec![],
                },
                TraceStep {
                    id: 3,
                    step_type: StepType::ToolCall,
                    content: "Check eligibility".into(),
                    tokens: 580,
                    tool_name: Some("check_refund_eligibility".into()),
                    tool_params: Some(serde_json::json!({})),
                    tool_success: Some(false),
                    tool_error: Some("missing order_id".into()),
                    agent_id: None,
                    input_context: None,
                    output: None,
                    flags: vec![],
                    flag_details: vec![],
                },
                TraceStep {
                    id: 4,
                    step_type: StepType::ToolCall,
                    content: "Check eligibility (retry)".into(),
                    tokens: 620,
                    tool_name: Some("check_refund_eligibility".into()),
                    tool_params: Some(serde_json::json!({"order_id": "ORD-9182"})),
                    tool_success: Some(true),
                    tool_error: None,
                    agent_id: None,
                    input_context: None,
                    output: None,
                    flags: vec![],
                    flag_details: vec![],
                },
                TraceStep {
                    id: 5,
                    step_type: StepType::ToolCall,
                    content: "Process refund".into(),
                    tokens: 380,
                    tool_name: Some("process_refund".into()),
                    tool_params: Some(serde_json::json!({"order_id": "ORD-9182"})),
                    tool_success: Some(true),
                    tool_error: None,
                    agent_id: None,
                    input_context: None,
                    output: None,
                    flags: vec![],
                    flag_details: vec![],
                },
            ],
            total_tokens: 2740,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_full_analysis() {
        let mut trace = make_trace();
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, simple_sim, &config).unwrap();
        assert!(!report.trace_id.is_empty());
        assert!(report.score.score >= 0.0);
        assert!(report.score.score <= 100.0);
        assert_eq!(report.total_steps, 5);
        // All 8 metrics always present.
        assert!(report.score.score >= 0.0);
        assert!(!report.summary.is_empty());
    }

    #[test]
    fn test_minimum_steps() {
        let mut trace = make_trace();
        trace.steps.truncate(3);
        assert!(!is_analysable(&trace));
    }

    #[test]
    fn test_rda_always_computed() {
        let mut trace = make_trace();
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, simple_sim, &config).unwrap();
        // RDA is always computed, not optional.
        assert!(report.score.rda.score >= 0.0);
    }

    #[test]
    fn test_dbo_cold_start_with_empty_history() {
        let mut trace = make_trace();
        let config = ScoringConfig::default(); // historical_sequences is empty
        let report = analyse(&mut trace, simple_sim, &config).unwrap();
        assert!(report.score.dbo.cold_start);
        assert!((report.score.dbo.score - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_multi_agent_breakdown() {
        // Build a trace with two agent IDs, each with enough steps.
        let mut trace = Trace {
            trace_id: "multi-agent-test".into(),
            agent_name: "test-crew".into(),
            framework: "crewai".into(),
            steps: (1u32..=12)
                .map(|id| TraceStep {
                    id,
                    step_type: if id % 2 == 0 {
                        StepType::ToolCall
                    } else {
                        StepType::Reasoning
                    },
                    content: format!("step {} content about processing tasks", id),
                    tokens: 400,
                    tool_name: if id % 2 == 0 { Some(format!("tool_{id}")) } else { None },
                    tool_params: None,
                    tool_success: Some(true),
                    tool_error: None,
                    // First 6 steps = researcher, last 6 = resolver
                    agent_id: Some(if id <= 6 { "researcher".into() } else { "resolver".into() }),
                    input_context: None,
                    output: None,
                    flags: vec![],
                    flag_details: vec![],
                })
                .collect(),
            total_tokens: 4800,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, simple_sim, &config).unwrap();

        assert_eq!(report.per_agent.len(), 2, "should have two agent breakdowns");
        let researcher = report.per_agent.iter().find(|a| a.agent_id == "researcher").unwrap();
        let resolver = report.per_agent.iter().find(|a| a.agent_id == "resolver").unwrap();
        assert_eq!(researcher.total_steps, 6);
        assert_eq!(resolver.total_steps, 6);
        assert!((researcher.token_share_pct - 50.0).abs() < 1.0);
        // Both sub-traces have ≥ MIN_TRACE_STEPS → should have TAS scores.
        assert!(researcher.tas_score.is_some());
        assert!(resolver.tas_score.is_some());
    }

    #[test]
    fn test_single_agent_no_breakdown() {
        let mut trace = make_trace(); // no agent_id set
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, simple_sim, &config).unwrap();
        assert!(report.per_agent.is_empty(), "single-agent trace should have no breakdown");
    }

    #[test]
    fn test_summary_oneliner_populated() {
        let mut trace = make_trace();
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, simple_sim, &config).unwrap();
        assert!(!report.summary_oneliner.is_empty());
        // One-liner should contain the score.
        assert!(report.summary_oneliner.contains(&format!("{:.0}", report.score.score)));
    }

    // ── M2: Task Value Integration (integration) ──────────────────────────────

    #[test]
    fn m2_perfect_task_value_score_does_not_change_tas() {
        let mut trace = make_trace(); // task_value_score = 1.0
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, simple_sim, &config).unwrap();
        // With tvs=1.0 the multiplier is 1.0 so score == raw_tas.
        assert!(
            (report.score.score - report.score.raw_tas).abs() < 0.1,
            "score={:.1} raw_tas={:.1}",
            report.score.score,
            report.score.raw_tas
        );
        assert!((report.score.task_value_score - 1.0).abs() < 0.001);
    }

    #[test]
    fn m2_low_task_value_reduces_score() {
        let mut trace_high = make_trace();
        let mut trace_low = make_trace();
        trace_low.task_value_score = 0.0;

        let config = ScoringConfig::default();
        let report_high = analyse(&mut trace_high, simple_sim, &config).unwrap();
        let report_low = analyse(&mut trace_low, simple_sim, &config).unwrap();

        assert!(
            report_high.score.score > report_low.score.score,
            "tvs=1.0 TAS ({:.1}) should be higher than tvs=0.0 TAS ({:.1})",
            report_high.score.score,
            report_low.score.score
        );
        // raw_tas should be identical (same trace structure).
        assert!(
            (report_high.score.raw_tas - report_low.score.raw_tas).abs() < 0.1,
            "raw_tas should be the same for structurally identical traces"
        );
    }

    #[test]
    fn m2_zero_task_value_caps_score_at_70pct_of_raw() {
        let mut trace = make_trace();
        trace.task_value_score = 0.0;
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, simple_sim, &config).unwrap();

        let expected_cap = report.score.raw_tas * 0.7;
        assert!(
            (report.score.score - expected_cap).abs() < 0.2,
            "score={:.1}, expected cap={:.1} (0.7 × raw_tas {:.1})",
            report.score.score,
            expected_cap,
            report.score.raw_tas
        );
    }

    #[test]
    fn m2_score_exposed_in_json_output() {
        let mut trace = make_trace();
        trace.task_value_score = 0.6;
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, simple_sim, &config).unwrap();
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"raw_tas\""), "raw_tas must appear in JSON");
        assert!(json.contains("\"task_value_score\""), "task_value_score must appear in JSON");
    }

    // ── M3: Minimum Viable Trace Gap (integration) ────────────────────────────

    #[test]
    fn m3_mvtg_in_range() {
        let mut trace = make_trace();
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, simple_sim, &config).unwrap();
        assert!(
            (0.0..=1.0).contains(&report.mvtg),
            "MVTG must be in [0, 1], got {:.3}",
            report.mvtg
        );
    }

    #[test]
    fn m3_mvtg_matches_savings_reduction_pct() {
        // MVTG and savings.reduction_pct are derived from the same waste_tokens
        // value so they must be consistent.
        let mut trace = make_trace();
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, simple_sim, &config).unwrap();
        let pct_from_mvtg = report.mvtg * 100.0;
        assert!(
            (pct_from_mvtg - report.savings.reduction_pct).abs() < 0.5,
            "MVTG%={:.1} should match savings.reduction_pct={:.1}",
            pct_from_mvtg,
            report.savings.reduction_pct
        );
    }

    #[test]
    fn m3_mvtg_zero_for_optimal_trace() {
        // A trace where all steps are KEEP in the diff should have MVTG ≈ 0.
        // Build the simplest clean trace: no loops, no misfires, no reformulations.
        let mut trace = Trace {
            trace_id: "clean".into(),
            agent_name: "clean-agent".into(),
            framework: "raw".into(),
            steps: (1u32..=5)
                .map(|id| TraceStep {
                    id,
                    step_type: if id % 2 == 0 {
                        StepType::ToolCall
                    } else {
                        StepType::Reasoning
                    },
                    content: format!("step {id}: unique actionable content for task {id}"),
                    tokens: 100,
                    tool_name: if id % 2 == 0 { Some(format!("tool_{id}")) } else { None },
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
            total_tokens: 500,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, simple_sim, &config).unwrap();
        // This trace may have some trims but should not be massively wasteful.
        assert!(
            report.mvtg < 0.6,
            "clean trace MVTG should be < 0.6, got {:.3}",
            report.mvtg
        );
    }

    #[test]
    fn m3_mvtg_exposed_in_json_output() {
        let mut trace = make_trace();
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, simple_sim, &config).unwrap();
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"mvtg\""), "mvtg must appear in JSON output");
    }

    #[test]
    fn m3_mvtg_and_m2_together() {
        // Both features at once: partial task + some structural waste.
        let mut trace = make_trace();
        trace.task_value_score = 0.5;
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, simple_sim, &config).unwrap();
        // Score is TVI-adjusted.
        assert!(report.score.score < report.score.raw_tas + 0.1);
        // MVTG is independent of task_value_score (structural gap only).
        let mut trace2 = make_trace();
        trace2.task_value_score = 1.0;
        let report2 = analyse(&mut trace2, simple_sim, &config).unwrap();
        assert!(
            (report.mvtg - report2.mvtg).abs() < 0.01,
            "MVTG must not change with task_value_score (structural metric)"
        );
    }
}
