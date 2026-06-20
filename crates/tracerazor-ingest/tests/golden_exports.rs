//! Golden-file tests: the adapters must parse the shapes real exporters
//! produce — flat LangSmith `list_runs()` arrays, LangChain run trees,
//! spec-compliant OTLP/JSON (string ints, message events, OpenLLMetry
//! indexed attributes) — with every run kept and tokens populated.

use tracerazor_core::report::IngestQuality;

fn fixture(name: &str) -> String {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read_to_string(p).unwrap()
}

#[test]
fn langsmith_flat_export_keeps_every_run() {
    let trace =
        tracerazor_ingest::parse(&fixture("langsmith_flat_list_runs.json"),
                                 tracerazor_ingest::TraceFormat::LangSmith)
            .unwrap();
    // 3 non-chain runs (the root chain is an orchestration wrapper).
    assert_eq!(trace.steps.len(), 3, "every run must survive: {:#?}", trace.steps);
    assert_eq!(trace.trace_id, "trace-777");
    // start_time ordering: llm-1 (10:00:01) before tool-1 before llm-2.
    assert!(trace.steps[0].content.contains("Parse the user refund"));
    // Tokens from all three real locations.
    assert_eq!(trace.steps[0].tokens, 310, "run-level prompt+completion");
    assert_eq!(trace.steps[1].tokens, 120, "run-level total_tokens");
    assert_eq!(trace.steps[2].tokens, 355, "outputs.llm_output.token_usage");
    assert!(trace.total_tokens > 0);
}

#[test]
fn langsmith_run_tree_reads_llm_output_tokens() {
    let trace = tracerazor_ingest::parse(&fixture("langsmith_run_tree.json"),
                                         tracerazor_ingest::TraceFormat::LangSmith)
        .unwrap();
    assert_eq!(trace.steps.len(), 2);
    assert_eq!(trace.steps[0].tokens, 512, "llm_output.token_usage");
    assert_eq!(trace.steps[1].tokens, 64, "run-level total_tokens");
    let q = IngestQuality::assess(&trace);
    assert!(!q.degraded, "{q:?}");
}

#[test]
fn otel_protojson_string_ints_and_event_content() {
    let trace = tracerazor_ingest::parse(&fixture("otel_protojson.json"),
                                         tracerazor_ingest::TraceFormat::Otel)
        .unwrap();
    assert_eq!(trace.steps.len(), 2);
    assert_eq!(trace.steps[0].tokens, 352, "string intValue prompt+completion");
    assert_eq!(trace.steps[1].tokens, 190, "string intValue total");
    assert!(
        trace.steps[0].content.contains("failing test"),
        "content must come from message events, not the span name: {}",
        trace.steps[0].content
    );
    assert!(
        trace.steps[1].content.contains("Re-run only the failing test"),
        "OpenLLMetry indexed attrs: {}",
        trace.steps[1].content
    );
    let q = IngestQuality::assess(&trace);
    assert!(!q.degraded);
}

#[test]
fn langsmith_3runs_tree_and_tokens() {
    // Fixture: chain root → llm child + tool child (flat list_runs export).
    // The chain wrapper must be skipped; 2 leaf steps must survive with tokens.
    let trace =
        tracerazor_ingest::parse(&fixture("langsmith_3runs.json"),
                                 tracerazor_ingest::TraceFormat::LangSmith)
            .unwrap();
    assert_eq!(trace.steps.len(), 2, "chain root is skipped; 2 leaf steps: {:#?}", trace.steps);
    // run-2 (llm) has outputs.llm_output.token_usage.total_tokens = 120.
    let llm_step = trace.steps.iter().find(|s| s.tool_name.is_none()).expect("llm step");
    assert!(llm_step.tokens > 0, "llm step must have tokens > 0, got {}", llm_step.tokens);
    assert_eq!(llm_step.tokens, 120, "tokens from llm_output.token_usage");
    // run-3 (tool) should be present.
    let tool_step = trace.steps.iter().find(|s| s.tool_name.is_some()).expect("tool step");
    assert_eq!(tool_step.tool_name.as_deref(), Some("weather_api"));
}

#[test]
fn span_name_fallback_is_detected_as_degraded() {
    let trace = tracerazor_ingest::parse(&fixture("otel_spannames_only.json"),
                                         tracerazor_ingest::TraceFormat::Otel)
        .unwrap();
    let q = IngestQuality::assess(&trace);
    assert!(q.degraded, "span-name content + zero tokens must be loud: {q:?}");
    assert!(q.zero_token_pct > 0.5);
}
