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
use crate::metrics::{cce, dbo, isr, ldi, rda, srr, tca, tur};
use crate::report::{TraceReport, generate_summary};
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
    let start = Instant::now();
    let total_tokens = trace.effective_total_tokens();

    // ── Structural metrics ────────────────────────────────────────────────────
    let srr_result = srr::compute(trace, &similarity_fn, None);
    let ldi_result = ldi::compute(trace);
    let tca_result = tca::compute(trace);

    srr::annotate_steps(&mut trace.steps, &srr_result);
    ldi::annotate_steps(&mut trace.steps, &ldi_result);
    tca::annotate_steps(&mut trace.steps, &tca_result);

    let tur_result = tur::compute(trace);
    let cce_result = cce::compute(trace);
    cce::annotate_steps(&mut trace.steps, &cce_result);

    // ── Information / semantic metrics (BoW, no external calls) ───────────────
    let isr_result = isr::compute_from_similarities(trace, &similarity_fn);

    // ── Local-first RDA (heuristic classifier, optional historical baseline) ──
    let rda_result = rda::compute(trace, config.historical_median_steps);

    // ── Local-first DBO (historical comparison, cold-start when < 10 traces) ──
    let dbo_result = dbo::compute(trace, &config.historical_sequences);

    let score = scoring::compute(
        srr_result,
        ldi_result,
        tca_result,
        tur_result,
        cce_result,
        rda_result,
        isr_result,
        dbo_result,
        trace.task_value_score,
        total_tokens,
        config,
    );

    let elapsed = start.elapsed().as_millis() as u64;
    let diff = TraceReport::build_diff(trace, &score);
    let optimal_tokens = TraceReport::optimal_tokens(&diff);
    let waste_tokens = total_tokens.saturating_sub(optimal_tokens);
    let savings = estimate_savings(total_tokens, waste_tokens, config, None);

    // ── E-01: auto-fix generation ─────────────────────────────────────────────
    let generated_fixes = generate_fixes(trace, &score);

    // ── E-08: template-based summary ──────────────────────────────────────────
    let summary = generate_summary(trace, &score, &savings);

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
        fixes: generated_fixes,
        summary,
        anomalies: vec![], // populated by the store layer after analysis
    })
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
}
