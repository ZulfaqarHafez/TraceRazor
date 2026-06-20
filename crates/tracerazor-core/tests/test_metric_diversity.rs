// Guardrail: composite metrics must have sufficient diversity.
//
// This test uses the bundled sample trace. It is a smoke test: it verifies the
// analyser runs without panicking on a real trace and that the composite TAS
// score is not pegged at either extreme (which would hint that the metrics have
// collapsed into trivial agreement). The full correlation / collinearity
// analysis lives in Python (benchmark/).

use tracerazor_core::analyse;
use tracerazor_core::scoring::ScoringConfig;
use tracerazor_core::types::Trace;

/// Bag-of-words Jaccard similarity — a deterministic stand-in for the semantic
/// backend so this test stays offline and dependency-free.
fn bow_sim(a: &str, b: &str) -> f64 {
    let wa: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let wb: std::collections::HashSet<&str> = b.split_whitespace().collect();
    if wa.is_empty() || wb.is_empty() {
        return 0.0;
    }
    let inter = wa.intersection(&wb).count() as f64;
    let union = wa.union(&wb).count() as f64;
    inter / union
}

#[test]
fn sample_trace_metrics_are_non_trivial() {
    let trace_json = include_str!("../../../traces/support-agent-run-2847.json");
    assert!(!trace_json.is_empty(), "sample trace should not be empty");

    let mut trace: Trace =
        serde_json::from_str(trace_json).expect("sample trace must deserialize into a Trace");
    assert!(
        trace.steps.len() >= 2,
        "sample trace should have multiple steps, got {}",
        trace.steps.len()
    );

    let config = ScoringConfig::default();
    let report =
        analyse(&mut trace, bow_sim, &config).expect("analyser must run on the sample trace");

    let tas = report.score.score;
    assert!(
        (1.0..=99.0).contains(&tas),
        "TAS score {tas:.1} is pegged at an extreme — metrics may have collapsed"
    );
}
