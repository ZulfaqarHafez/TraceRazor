/// Parser for OpenTelemetry JSON spans.
///
/// OTEL traces from the OpenAI Agents SDK, Semantic Kernel, and other
/// OTEL-instrumented frameworks are exported as `resourceSpans` with nested spans.
///
/// Each span is mapped to a TraceStep based on its attributes:
///   - `gen_ai.operation.name` == "chat" → Reasoning
///   - `gen_ai.operation.name` == "execute_tool" → ToolCall
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use tracerazor_core::types::{StepType, Trace, TraceStep};

#[derive(Debug, Deserialize)]
struct OtelExport {
    #[serde(alias = "resourceSpans", default)]
    resource_spans: Vec<ResourceSpan>,
}

#[derive(Debug, Deserialize)]
struct ResourceSpan {
    resource: Option<Resource>,
    #[serde(alias = "scopeSpans", default)]
    scope_spans: Vec<ScopeSpan>,
}

#[derive(Debug, Deserialize)]
struct Resource {
    attributes: Option<Vec<Attribute>>,
}

#[derive(Debug, Deserialize)]
struct ScopeSpan {
    spans: Vec<Span>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Span {
    trace_id: String,
    #[allow(dead_code)]
    span_id: String,
    #[allow(dead_code)]
    parent_span_id: Option<String>,
    name: String,
    #[serde(default)]
    attributes: Vec<Attribute>,
    #[serde(default)]
    events: Vec<SpanEvent>,
    status: Option<SpanStatus>,
    start_time_unix_nano: Option<String>,
    #[allow(dead_code)]
    end_time_unix_nano: Option<String>,
}

/// Span event (modern gen_ai semconv puts prompt/completion messages here,
/// e.g. events named `gen_ai.user.message` / `gen_ai.choice` with a
/// `content` attribute).
#[derive(Debug, Deserialize)]
struct SpanEvent {
    #[serde(default)]
    name: String,
    #[serde(default)]
    attributes: Vec<Attribute>,
}

#[derive(Debug, Deserialize)]
struct Attribute {
    key: String,
    value: AttributeValue,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AttributeValue {
    String { #[serde(alias = "stringValue")] string_value: String },
    Int { #[serde(alias = "intValue")] int_value: serde_json::Value },
    Bool { #[allow(dead_code)] #[serde(alias = "boolValue")] bool_value: bool },
    Double { #[allow(dead_code)] #[serde(alias = "doubleValue")] double_value: f64 },
}

impl AttributeValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            AttributeValue::String { string_value } => Some(string_value.as_str()),
            _ => None,
        }
    }

    fn as_i64(&self) -> Option<i64> {
        match self {
            // Spec-compliant OTLP/JSON (protojson) encodes 64-bit ints as
            // STRINGS ("450"); accept both encodings.
            AttributeValue::Int { int_value } => int_value
                .as_i64()
                .or_else(|| int_value.as_str().and_then(|s| s.parse().ok())),
            AttributeValue::String { string_value } => string_value.parse().ok(),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SpanStatus {
    code: Option<String>,
    message: Option<String>,
}

/// Returns true for span event names that carry prompt/input content.
fn is_input_event(name: &str) -> bool {
    matches!(
        name,
        "gen_ai.content.prompt"
            | "gen_ai.input.messages"
            | "gen_ai.user.message"
            | "gen_ai.system.message"
    )
}

/// Returns true for span event names that carry completion/output content.
fn is_output_event(name: &str) -> bool {
    matches!(
        name,
        "gen_ai.content.completion"
            | "gen_ai.output.messages"
            | "gen_ai.choice"
            | "gen_ai.assistant.message"
    )
}

/// Extract the prompt/input side of a span as a string, or None.
fn extract_input_content(span: &Span, attrs: &HashMap<&str, &AttributeValue>) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();

    for key in ["gen_ai.prompt", "gen_ai.input.messages"] {
        if let Some(v) = attrs.get(key).and_then(|v| v.as_str()) {
            parts.push(v.to_string());
        }
    }

    // OpenLLMetry indexed: gen_ai.prompt.<i>.content
    let mut indexed: Vec<(usize, &str)> = Vec::new();
    for (k, v) in attrs {
        if let Some(rest) = k.strip_prefix("gen_ai.prompt.") {
            if let Some(idx) = rest.strip_suffix(".content").and_then(|i| i.parse().ok()) {
                if let Some(text) = v.as_str() {
                    indexed.push((idx, text));
                }
            }
        }
    }
    indexed.sort_by_key(|(i, _)| *i);
    parts.extend(indexed.into_iter().map(|(_, t)| t.to_string()));

    // Input events.
    for ev in &span.events {
        if is_input_event(&ev.name) {
            for a in &ev.attributes {
                if a.key == "content"
                    || a.key == "gen_ai.prompt"
                    || a.key.ends_with(".content")
                    || a.key == "body"
                {
                    if let Some(text) = a.value.as_str() {
                        parts.push(text.to_string());
                    }
                }
            }
        }
    }

    let joined = parts.join(" ").trim().to_string();
    if joined.is_empty() { None } else { Some(joined) }
}

/// Extract the completion/output side of a span as a string, or None.
fn extract_output_content(span: &Span, attrs: &HashMap<&str, &AttributeValue>) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();

    for key in ["gen_ai.completion", "gen_ai.output.messages"] {
        if let Some(v) = attrs.get(key).and_then(|v| v.as_str()) {
            parts.push(v.to_string());
        }
    }

    // OpenLLMetry indexed: gen_ai.completion.<i>.content
    let mut indexed: Vec<(usize, &str)> = Vec::new();
    for (k, v) in attrs {
        if let Some(rest) = k.strip_prefix("gen_ai.completion.") {
            if let Some(idx) = rest.strip_suffix(".content").and_then(|i| i.parse().ok()) {
                if let Some(text) = v.as_str() {
                    indexed.push((idx, text));
                }
            }
        }
    }
    indexed.sort_by_key(|(i, _)| *i);
    parts.extend(indexed.into_iter().map(|(_, t)| t.to_string()));

    // Output events.
    for ev in &span.events {
        if is_output_event(&ev.name) {
            for a in &ev.attributes {
                if a.key == "content"
                    || a.key == "gen_ai.completion"
                    || a.key.ends_with(".content")
                    || a.key == "body"
                {
                    if let Some(text) = a.value.as_str() {
                        parts.push(text.to_string());
                    }
                }
            }
        }
    }

    let joined = parts.join(" ").trim().to_string();
    if joined.is_empty() { None } else { Some(joined) }
}

/// Pull span content from the shapes real exporters use, in order:
/// flat `gen_ai.prompt`/`gen_ai.completion`, structured
/// `gen_ai.input.messages`/`gen_ai.output.messages`, OpenLLMetry indexed
/// attributes (`gen_ai.prompt.0.content`, ...), then message events.
/// Returns None when nothing content-like exists — the caller falls back to
/// the span name, and the ingest-quality check makes that fallback loud.
fn extract_content(span: &Span, attrs: &HashMap<&str, &AttributeValue>) -> Option<String> {
    let input = extract_input_content(span, attrs);
    let output = extract_output_content(span, attrs);
    match (input, output) {
        (None, None) => None,
        (Some(i), None) => Some(i),
        (None, Some(o)) => Some(o),
        (Some(i), Some(o)) => Some(format!("{i} {o}")),
    }
}

/// A span coupled with the service.name of its ResourceSpan.
struct TaggedSpan {
    span: Span,
    service_name: Option<String>,
}

/// Parse an OTEL JSON export into a Trace.
pub fn parse(data: &str) -> Result<Trace> {
    let export: OtelExport =
        serde_json::from_str(data).context("Failed to parse OTEL JSON")?;

    // Collect spans tagged with their resource's service.name.
    let mut all_spans: Vec<TaggedSpan> = Vec::new();
    let mut agent_name = "unknown".to_string();

    for rs in export.resource_spans {
        // Extract service.name from resource attributes.
        let svc_name: Option<String> = rs.resource.as_ref().and_then(|r| {
            r.attributes.as_ref()?.iter().find_map(|a| {
                if a.key == "service.name" {
                    a.value.as_str().map(|s| s.to_string())
                } else {
                    None
                }
            })
        });

        // Use the first service.name encountered as the top-level agent name.
        if agent_name == "unknown" {
            if let Some(ref s) = svc_name {
                agent_name = s.clone();
            }
        }

        for ss in rs.scope_spans {
            for span in ss.spans {
                all_spans.push(TaggedSpan {
                    span,
                    service_name: svc_name.clone(),
                });
            }
        }
    }

    if all_spans.is_empty() {
        anyhow::bail!("OTEL export contains no spans");
    }

    // Sort spans by start time (lexicographic on nanosecond timestamps works).
    all_spans.sort_by(|a, b| {
        a.span
            .start_time_unix_nano
            .as_deref()
            .unwrap_or("")
            .cmp(b.span.start_time_unix_nano.as_deref().unwrap_or(""))
    });

    // Guard against mixed-traceId exports: if spans come from more than one
    // OTEL trace, using the first span's traceId would silently merge them.
    // Emit a warning in metadata and keep only the most-common traceId's spans.
    let first_trace_id = all_spans[0].span.trace_id.clone();
    let mixed_traces = all_spans
        .iter()
        .any(|ts| ts.span.trace_id != first_trace_id);

    let (trace_id, spans_to_process): (String, Vec<&TaggedSpan>) = if mixed_traces {
        // Count spans per traceId, pick the most common.
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for ts in &all_spans {
            *counts.entry(ts.span.trace_id.as_str()).or_default() += 1;
        }
        let best_tid = counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(tid, _)| tid.to_string())
            .unwrap_or_else(|| first_trace_id.clone());
        let filtered: Vec<&TaggedSpan> = all_spans
            .iter()
            .filter(|ts| ts.span.trace_id == best_tid)
            .collect();
        (best_tid, filtered)
    } else {
        (first_trace_id, all_spans.iter().collect())
    };

    let mut steps: Vec<TraceStep> = Vec::new();
    let mut counter = 1u32;

    for tagged in &spans_to_process {
        let span = &tagged.span;
        let attrs: HashMap<&str, &AttributeValue> =
            span.attributes.iter().map(|a| (a.key.as_str(), &a.value)).collect();

        // Skip root/chain spans (those with no gen_ai operation).
        let op = attrs
            .get("gen_ai.operation.name")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let step_type = match op {
            "chat" | "generate" => StepType::Reasoning,
            "execute_tool" | "tool" => StepType::ToolCall,
            _ => {
                // Include spans with gen_ai attributes, skip pure orchestration spans.
                if attrs.keys().any(|k| k.starts_with("gen_ai.")) {
                    StepType::Reasoning
                } else {
                    continue;
                }
            }
        };

        let tokens_i64 = attrs
            .get("gen_ai.usage.total_tokens")
            .and_then(|v| v.as_i64())
            .or_else(|| {
                // input/output (current semconv) or prompt/completion
                // (older semconv + OpenLLMetry).
                let i = attrs
                    .get("gen_ai.usage.input_tokens")
                    .or_else(|| attrs.get("gen_ai.usage.prompt_tokens"))?
                    .as_i64()?;
                let o = attrs
                    .get("gen_ai.usage.output_tokens")
                    .or_else(|| attrs.get("gen_ai.usage.completion_tokens"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                // saturating_add: avoid overflow on pathological inputs.
                Some(i.saturating_add(o))
            })
            // OpenLLMetry total.
            .or_else(|| attrs.get("llm.usage.total_tokens").and_then(|v| v.as_i64()))
            .unwrap_or(0);
        // Clamp negatives to 0 and saturate at u32::MAX instead of silently
        // truncating the upper bits of an attacker-supplied token count.
        let tokens = u32::try_from(tokens_i64.max(0)).unwrap_or(u32::MAX);

        let tool_name = if step_type == StepType::ToolCall {
            attrs
                .get("gen_ai.tool.name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| Some(span.name.clone()))
        } else {
            None
        };

        let tool_success = if step_type == StepType::ToolCall {
            let is_error = span
                .status
                .as_ref()
                .and_then(|s| s.code.as_deref())
                .map(|c| c == "STATUS_CODE_ERROR")
                .unwrap_or(false);
            Some(!is_error)
        } else {
            None
        };

        // Split content into input and output for CCE and downstream metrics.
        let input_context = extract_input_content(span, &attrs);
        let output = extract_output_content(span, &attrs);
        let content = extract_content(span, &attrs).unwrap_or_else(|| span.name.clone());

        // Determine agent_id: prefer gen_ai.agent.name span attribute, then
        // fall back to service.name if it differs from the top-level agent.
        let agent_id: Option<String> = attrs
            .get("gen_ai.agent.name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                tagged.service_name.as_ref().and_then(|svc| {
                    if svc != &agent_name {
                        Some(svc.clone())
                    } else {
                        None
                    }
                })
            });

        steps.push(TraceStep {
            id: counter,
            step_type,
            content,
            tokens,
            tool_name,
            tool_params: None,
            tool_success,
            tool_error: span
                .status
                .as_ref()
                .and_then(|s| s.message.clone()),
            agent_id,
            input_context,
            output,
            flags: vec![],
            flag_details: vec![],
        });

        counter += 1;
    }

    let total_tokens: u32 = steps
        .iter()
        .map(|s| s.tokens)
        .fold(0u32, u32::saturating_add);

    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
    if mixed_traces {
        metadata.insert(
            "warning".to_string(),
            serde_json::Value::String("export contained spans from multiple OTEL traces; only the most-common traceId was kept".to_string()),
        );
    }

    Ok(Trace {
        trace_id,
        agent_name,
        framework: "otel".to_string(),
        steps,
        total_tokens,
        task_value_score: 1.0,
        metadata,
    })
}
