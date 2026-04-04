pub mod langsmith;
pub mod raw_json;
pub mod otel;

use anyhow::Result;
use tracerazor_core::types::Trace;

/// Supported trace input formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceFormat {
    /// TraceRazor native raw JSON (schema defined in this crate).
    RawJson,
    /// LangSmith run export format.
    LangSmith,
    /// OpenTelemetry JSON spans.
    Otel,
    /// Auto-detect from content.
    Auto,
}

/// Parse a trace file from its bytes, auto-detecting the format.
pub fn parse(data: &str, format: TraceFormat) -> Result<Trace> {
    match format {
        TraceFormat::RawJson => raw_json::parse(data),
        TraceFormat::LangSmith => langsmith::parse(data),
        TraceFormat::Otel => otel::parse(data),
        TraceFormat::Auto => detect_and_parse(data),
    }
}

/// Detect the format from JSON content and parse accordingly.
fn detect_and_parse(data: &str) -> Result<Trace> {
    let v: serde_json::Value = serde_json::from_str(data)?;

    // LangSmith: has a "run_type" field or "child_runs" at the top level.
    if v.get("run_type").is_some() || v.get("child_runs").is_some() {
        return langsmith::parse(data);
    }

    // OTEL: has a "resourceSpans" or "scopeSpans" field.
    if v.get("resourceSpans").is_some() || v.get("scopeSpans").is_some() {
        return otel::parse(data);
    }

    // Default: raw JSON.
    raw_json::parse(data)
}
