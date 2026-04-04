pub mod graph;
pub mod metrics;
pub mod report;
pub mod scoring;
pub mod types;

use std::time::Instant;

use anyhow::Result;

use crate::metrics::{cce, dbo, isr, ldi, rda, srr, tca, tur};
use crate::report::TraceReport;
use crate::scoring::{ScoringConfig, estimate_savings};
use crate::types::{MIN_TRACE_STEPS, Trace};

/// Phase 1: Structural-only analysis (no API calls required).
///
/// `similarity_fn` is injected from `tracerazor-semantic` so that core
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

    let srr_result = srr::compute(trace, &similarity_fn, None);
    let ldi_result = ldi::compute(trace);
    let tca_result = tca::compute(trace);

    srr::annotate_steps(&mut trace.steps, &srr_result);
    ldi::annotate_steps(&mut trace.steps, &ldi_result);
    tca::annotate_steps(&mut trace.steps, &tca_result);

    let tur_result = tur::compute(trace);
    let cce_result = cce::compute(trace);
    cce::annotate_steps(&mut trace.steps, &cce_result);

    // Also run ISR in Phase 1 using the injected similarity_fn — no extra API cost.
    let isr_result = isr::compute_from_similarities(trace, &similarity_fn);

    let score = scoring::compute(
        srr_result,
        ldi_result,
        tca_result,
        tur_result,
        cce_result,
        None, // rda — Phase 2
        Some(isr_result),
        None, // dbo — Phase 2
        trace.task_value_score,
        total_tokens,
        config,
    );

    let elapsed = start.elapsed().as_millis() as u64;
    let diff = TraceReport::build_diff(trace, &score);
    let optimal_tokens = TraceReport::optimal_tokens(&diff);
    let waste_tokens = total_tokens.saturating_sub(optimal_tokens);
    let savings = estimate_savings(total_tokens, waste_tokens, config, None);

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
    })
}

/// Phase 2: Full analysis including LLM-powered metrics (RDA, DBO).
///
/// Requires OPENAI_API_KEY. Falls back gracefully to structural-only
/// results if the API calls fail.
///
/// `similarity_fn` should use OpenAI embeddings for accurate SRR/ISR.
/// `llm_fn` is the async LLM chat completion function.
pub async fn analyse_full<F, L, Fut>(
    trace: &mut Trace,
    similarity_fn: F,
    llm_fn: L,
    config: &ScoringConfig,
) -> Result<TraceReport>
where
    F: Fn(&str, &str) -> f64,
    L: Fn(String, String) -> Fut + Clone,
    Fut: std::future::Future<Output = Result<String>>,
{
    let start = Instant::now();
    let total_tokens = trace.effective_total_tokens();

    // ── Structural metrics (always fast) ──────────────────────────────────
    let srr_result = srr::compute(trace, &similarity_fn, Some(0.85)); // full threshold with embeddings
    let ldi_result = ldi::compute(trace);
    let tca_result = tca::compute(trace);

    srr::annotate_steps(&mut trace.steps, &srr_result);
    ldi::annotate_steps(&mut trace.steps, &ldi_result);
    tca::annotate_steps(&mut trace.steps, &tca_result);

    let tur_result = tur::compute(trace);
    let cce_result = cce::compute(trace);
    cce::annotate_steps(&mut trace.steps, &cce_result);

    // ISR uses the same similarity function (embedding-based in Phase 2).
    let isr_result = isr::compute_from_similarities(trace, &similarity_fn);

    // ── LLM-powered metrics (async, batched) ──────────────────────────────
    let rda_future = rda::compute(trace, llm_fn.clone());
    let dbo_future = dbo::compute(trace, llm_fn.clone());

    let (rda_res, dbo_res) = tokio::join!(rda_future, dbo_future);

    let rda_result = rda_res
        .map_err(|e| eprintln!("RDA metric failed: {}", e))
        .ok();
    let dbo_result = dbo_res
        .map_err(|e| eprintln!("DBO metric failed: {}", e))
        .ok();

    let elapsed = start.elapsed().as_millis() as u64;

    let score = scoring::compute(
        srr_result,
        ldi_result,
        tca_result,
        tur_result,
        cce_result,
        rda_result,
        Some(isr_result),
        dbo_result,
        trace.task_value_score,
        total_tokens,
        config,
    );

    let diff = TraceReport::build_diff(trace, &score);
    let optimal_tokens = TraceReport::optimal_tokens(&diff);
    let waste_tokens = total_tokens.saturating_sub(optimal_tokens);
    let savings = estimate_savings(total_tokens, waste_tokens, config, None);

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
                    tool_name: None, tool_params: None, tool_success: None,
                    tool_error: None, agent_id: None, input_context: None,
                    output: None, flags: vec![], flag_details: vec![],
                },
                TraceStep {
                    id: 2,
                    step_type: StepType::ToolCall,
                    content: "Fetch order details".into(),
                    tokens: 340,
                    tool_name: Some("get_order_details".into()),
                    tool_params: Some(serde_json::json!({"order_id": "ORD-9182"})),
                    tool_success: Some(true),
                    tool_error: None, agent_id: None, input_context: None,
                    output: None, flags: vec![], flag_details: vec![],
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
                    agent_id: None, input_context: None,
                    output: None, flags: vec![], flag_details: vec![],
                },
                TraceStep {
                    id: 4,
                    step_type: StepType::ToolCall,
                    content: "Check eligibility (retry)".into(),
                    tokens: 620,
                    tool_name: Some("check_refund_eligibility".into()),
                    tool_params: Some(serde_json::json!({"order_id": "ORD-9182"})),
                    tool_success: Some(true),
                    tool_error: None, agent_id: None, input_context: None,
                    output: None, flags: vec![], flag_details: vec![],
                },
                TraceStep {
                    id: 5,
                    step_type: StepType::ToolCall,
                    content: "Process refund".into(),
                    tokens: 380,
                    tool_name: Some("process_refund".into()),
                    tool_params: Some(serde_json::json!({"order_id": "ORD-9182"})),
                    tool_success: Some(true),
                    tool_error: None, agent_id: None, input_context: None,
                    output: None, flags: vec![], flag_details: vec![],
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
        // ISR should now be included
        assert!(report.score.isr.is_some());
    }

    #[test]
    fn test_minimum_steps() {
        let mut trace = make_trace();
        trace.steps.truncate(3);
        assert!(!is_analysable(&trace));
    }
}
