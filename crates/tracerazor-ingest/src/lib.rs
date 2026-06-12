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

    // Plain chat-completions log (the artifact most developers actually
    // have): point at the converter instead of failing with a bare
    // "missing field trace_id".
    if looks_like_chat_log(&v) {
        anyhow::bail!(
            "this looks like an OpenAI/Anthropic chat log (a \"messages\" \
             array), not a TraceRazor trace. Convert it first:\n  \
             python tools/convert_openai.py <file> -o trace.json\n\
             See docs/trace-format.md for the native schema. LangSmith and \
             OTel GenAI exports parse directly (-F langsmith / -F otel)."
        );
    }

    // Default: raw JSON.
    raw_json::parse(data)
}

/// A `{"messages": [{"role": ...}]}` envelope or a bare `[{"role": ...}]`
/// array — the shape of OpenAI/Anthropic chat-completions request logs.
fn looks_like_chat_log(v: &serde_json::Value) -> bool {
    let messages = match v.get("messages") {
        Some(m) => m,
        None => v,
    };
    messages
        .as_array()
        .and_then(|a| a.first())
        .is_some_and(|m| m.get("role").is_some() && m.get("run_type").is_none())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_log_payload_points_at_the_converter() {
        let chat = r#"{"messages": [{"role": "user", "content": "find my order"}]}"#;
        let err = parse(chat, TraceFormat::Auto).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("convert_openai"), "got: {msg}");

        let bare = r#"[{"role": "assistant", "content": "checking"}]"#;
        let err = parse(bare, TraceFormat::Auto).unwrap_err();
        assert!(format!("{err:#}").contains("convert_openai"));
    }
}
