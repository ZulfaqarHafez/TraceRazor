//! Criterion benchmarks for the top-level `analyse()` entry point.
//!
//! The README and PRD market a "<5 ms per trace" target; these benches back
//! that claim with reproducible numbers across small, medium and large traces.
//! Run with `cargo bench -p tracerazor-core`.

use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use tracerazor_core::{
    analyse,
    scoring::ScoringConfig,
    types::{StepType, Trace, TraceStep},
};

/// Build a synthetic trace of `n` steps with a deterministic mix of unique
/// content, near-duplicates and tool calls — enough variation for every
/// metric to do real work.
fn synth_trace(n: usize) -> Trace {
    let topics = [
        "parse the user request about order refund",
        "fetch order details from the database",
        "validate the refund eligibility window",
        "compose a customer-facing response with apology",
        "log the resolution to the support ticketing system",
    ];
    let mut steps = Vec::with_capacity(n);
    for i in 0..n {
        let is_tool = i % 3 == 1;
        let topic = topics[i % topics.len()];
        let content = if i % 7 == 0 && i > 0 {
            topics[(i - 1) % topics.len()].to_string()
        } else {
            format!("{topic} (iteration {i})")
        };
        steps.push(TraceStep {
            id: (i + 1) as u32,
            step_type: if is_tool {
                StepType::ToolCall
            } else {
                StepType::Reasoning
            },
            content,
            tokens: 120 + (i as u32 % 50),
            tool_name: if is_tool {
                Some(format!("tool_{}", i % 4))
            } else {
                None
            },
            tool_params: if is_tool {
                Some(serde_json::json!({ "id": i, "topic": topic }))
            } else {
                None
            },
            tool_success: if is_tool { Some(true) } else { None },
            tool_error: None,
            agent_id: None,
            input_context: Some(format!("context block {i}")),
            output: Some(format!("output for step {i}")),
            flags: vec![],
            flag_details: vec![],
        });
    }
    Trace {
        trace_id: "bench".into(),
        agent_name: "bench-agent".into(),
        framework: "raw".into(),
        steps,
        total_tokens: 0,
        task_value_score: 1.0,
        metadata: HashMap::new(),
    }
}

/// A cheap order-invariant similarity used so the benchmark measures the
/// orchestration cost of `analyse`, not the semantic backend.
fn naive_sim(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let la = a.len() as f64;
    let lb = b.len() as f64;
    if la == 0.0 || lb == 0.0 {
        return 0.0;
    }
    let shared = a.chars().filter(|c| b.contains(*c)).count() as f64;
    (shared / la.max(lb)).min(1.0)
}

fn bench_analyse(c: &mut Criterion) {
    let config = ScoringConfig::default();
    let mut group = c.benchmark_group("analyse");
    for &n in &[10usize, 50, 200, 1000] {
        let trace = synth_trace(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut t = trace.clone();
                analyse(&mut t, naive_sim, &config).unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_analyse);
criterion_main!(benches);
