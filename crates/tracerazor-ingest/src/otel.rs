/// Parser for OpenTelemetry JSON spans.
///
/// OTEL traces from the OpenAI Agents SDK, Semantic Kernel, and other
/// OTEL-instrumented frameworks are exported as `resourceSpans` with nested spans.
///
/// Each span is mapped to a TraceStep based on its attributes:
///   - `gen_ai.operation.name` == "chat" → Reasoning
///   - `gen_ai.operation.name` == "execute_tool" → ToolCall
use anyhow::{Context, Result};
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use prost::Message;
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
    #[serde(default)]
    spans: Vec<Span>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Span {
    trace_id: String,
    span_id: String,
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
    String {
        #[serde(alias = "stringValue")]
        string_value: String,
    },
    Int {
        #[serde(alias = "intValue")]
        int_value: serde_json::Value,
    },
    Bool {
        #[allow(dead_code)]
        #[serde(alias = "boolValue")]
        bool_value: bool,
    },
    Double {
        #[allow(dead_code)]
        #[serde(alias = "doubleValue")]
        double_value: f64,
    },
    // OTLP permits arrays, key/value lists, bytes, and future AnyValue
    // variants. Preserve the wire object so structured GenAI messages and tool
    // payloads can be normalized without weakening forward compatibility.
    Structured(serde_json::Value),
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

    fn as_json(&self) -> Option<serde_json::Value> {
        match self {
            Self::String { string_value } => Some(serde_json::Value::String(string_value.clone())),
            Self::Int { int_value } => {
                let value = int_value
                    .as_i64()
                    .or_else(|| int_value.as_str().and_then(|value| value.parse().ok()))?;
                Some(serde_json::Value::from(value))
            }
            Self::Bool { bool_value } => Some(serde_json::Value::Bool(*bool_value)),
            Self::Double { double_value } => {
                serde_json::Number::from_f64(*double_value).map(serde_json::Value::Number)
            }
            Self::Structured(value) => decode_any_value(value),
        }
    }
}

/// Convert an OTLP `AnyValue` proto-JSON wrapper into ordinary JSON. Binary
/// content is deliberately not interpreted as text; callers mark it degraded.
fn decode_any_value(value: &serde_json::Value) -> Option<serde_json::Value> {
    let Some(object) = value.as_object() else {
        return Some(value.clone());
    };
    if let Some(value) = object.get("stringValue") {
        return value.as_str().map(|value| serde_json::json!(value));
    }
    if let Some(value) = object.get("intValue") {
        let value = value
            .as_i64()
            .or_else(|| value.as_str().and_then(|value| value.parse().ok()))?;
        return Some(serde_json::Value::from(value));
    }
    if let Some(value) = object.get("boolValue") {
        return value.as_bool().map(serde_json::Value::Bool);
    }
    if let Some(value) = object.get("doubleValue") {
        return value
            .as_f64()
            .and_then(serde_json::Number::from_f64)
            .map(serde_json::Value::Number);
    }
    if object.contains_key("bytesValue") || object.contains_key("stringValueStrindex") {
        return None;
    }
    if let Some(values) = object
        .get("arrayValue")
        .and_then(|value| value.get("values"))
        .and_then(serde_json::Value::as_array)
    {
        return values
            .iter()
            .map(decode_any_value)
            .collect::<Option<Vec<_>>>()
            .map(serde_json::Value::Array);
    }
    if let Some(values) = object
        .get("kvlistValue")
        .and_then(|value| value.get("values"))
        .and_then(serde_json::Value::as_array)
    {
        let mut decoded = serde_json::Map::new();
        for pair in values {
            let pair = pair.as_object()?;
            let key = pair.get("key")?.as_str()?.to_string();
            let value = decode_any_value(pair.get("value")?)?;
            decoded.insert(key, value);
        }
        return Some(serde_json::Value::Object(decoded));
    }

    // Some exporters place an already-structured object directly in the value
    // field. It is safe to normalize in memory; local-redacted persistence
    // hashes content later.
    Some(value.clone())
}

#[derive(Debug, Deserialize)]
struct SpanStatus {
    code: Option<SpanStatusCode>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SpanStatusCode {
    Number(i32),
    String(String),
}

impl SpanStatusCode {
    fn is_error(&self) -> bool {
        match self {
            // STATUS_CODE_ERROR is numeric value 2 in the stable OTLP schema.
            Self::Number(value) => *value == 2,
            Self::String(value) => value == "STATUS_CODE_ERROR" || value == "2",
        }
    }
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

fn push_issue(issues: &mut Vec<String>, issue: impl Into<String>) {
    let issue = issue.into();
    if !issues.contains(&issue) {
        issues.push(issue);
    }
}

fn collect_message_text(value: &serde_json::Value, parts: &mut Vec<String>) {
    match value {
        serde_json::Value::String(value) => {
            let value = value.trim();
            if value.starts_with(['[', '{']) {
                if let Ok(structured) = serde_json::from_str::<serde_json::Value>(value) {
                    collect_message_text(&structured, parts);
                    return;
                }
            }
            if !value.is_empty() {
                parts.push(value.to_string());
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_message_text(value, parts);
            }
        }
        serde_json::Value::Object(object) => {
            // Follow the stable GenAI message schemas without treating role,
            // type, IDs, or names as conversational content.
            for key in ["content", "text", "result", "arguments"] {
                if let Some(value) = object.get(key) {
                    if matches!(key, "arguments" | "result") && !value.is_string() {
                        if let Ok(value) = serde_json::to_string(value) {
                            parts.push(value);
                        }
                    } else {
                        collect_message_text(value, parts);
                    }
                }
            }
            for key in ["parts", "messages", "message"] {
                if let Some(value) = object.get(key) {
                    collect_message_text(value, parts);
                }
            }
        }
        serde_json::Value::Number(value) => parts.push(value.to_string()),
        serde_json::Value::Bool(value) => parts.push(value.to_string()),
        serde_json::Value::Null => {}
    }
}

fn message_content(_key: &str, value: &AttributeValue, issues: &mut Vec<String>) -> Option<String> {
    let Some(value) = value.as_json() else {
        push_issue(issues, "unsupported_structured_content");
        return None;
    };
    let mut parts = Vec::new();
    collect_message_text(&value, &mut parts);
    let joined = parts.join(" ").trim().to_string();
    if joined.is_empty() {
        push_issue(issues, "unusable_structured_content");
        None
    } else {
        Some(joined)
    }
}

fn join_parts(parts: Vec<String>) -> Option<String> {
    let joined = parts.join(" ").trim().to_string();
    (!joined.is_empty()).then_some(joined)
}

/// Extract prompt/input content from scalar, proto-JSON structured messages,
/// OpenLLMetry indexed attributes, and GenAI message events.
fn extract_input_content(
    span: &Span,
    attrs: &HashMap<&str, &AttributeValue>,
    issues: &mut Vec<String>,
) -> Option<String> {
    let mut parts = Vec::new();
    for key in [
        "gen_ai.prompt",
        "gen_ai.input.messages",
        "gen_ai.system_instructions",
    ] {
        if let Some(value) = attrs.get(key) {
            if let Some(value) = message_content(key, value, issues) {
                parts.push(value);
            }
        }
    }

    let mut indexed = Vec::new();
    for (key, value) in attrs {
        if let Some(rest) = key.strip_prefix("gen_ai.prompt.") {
            if let Some(index) = rest
                .strip_suffix(".content")
                .and_then(|i| i.parse::<usize>().ok())
            {
                if let Some(value) = message_content(key, value, issues) {
                    indexed.push((index, value));
                }
            }
        }
    }
    indexed.sort_by_key(|(index, _)| *index);
    parts.extend(indexed.into_iter().map(|(_, value)| value));

    for event in &span.events {
        if is_input_event(&event.name) {
            for attribute in &event.attributes {
                if attribute.key == "content"
                    || attribute.key == "gen_ai.prompt"
                    || attribute.key.ends_with(".content")
                    || attribute.key == "body"
                {
                    if let Some(value) = message_content(&attribute.key, &attribute.value, issues) {
                        parts.push(value);
                    }
                }
            }
        }
    }
    join_parts(parts)
}

/// Extract completion/output content from scalar, structured, indexed, and
/// event-carried GenAI content.
fn extract_output_content(
    span: &Span,
    attrs: &HashMap<&str, &AttributeValue>,
    issues: &mut Vec<String>,
) -> Option<String> {
    let mut parts = Vec::new();
    for key in ["gen_ai.completion", "gen_ai.output.messages"] {
        if let Some(value) = attrs.get(key) {
            if let Some(value) = message_content(key, value, issues) {
                parts.push(value);
            }
        }
    }

    let mut indexed = Vec::new();
    for (key, value) in attrs {
        if let Some(rest) = key.strip_prefix("gen_ai.completion.") {
            if let Some(index) = rest
                .strip_suffix(".content")
                .and_then(|i| i.parse::<usize>().ok())
            {
                if let Some(value) = message_content(key, value, issues) {
                    indexed.push((index, value));
                }
            }
        }
    }
    indexed.sort_by_key(|(index, _)| *index);
    parts.extend(indexed.into_iter().map(|(_, value)| value));

    for event in &span.events {
        if is_output_event(&event.name) {
            for attribute in &event.attributes {
                if attribute.key == "content"
                    || attribute.key == "gen_ai.completion"
                    || attribute.key.ends_with(".content")
                    || attribute.key == "body"
                {
                    if let Some(value) = message_content(&attribute.key, &attribute.value, issues) {
                        parts.push(value);
                    }
                }
            }
        }
    }
    join_parts(parts)
}

fn tool_value(
    attrs: &HashMap<&str, &AttributeValue>,
    keys: &[&str],
    issues: &mut Vec<String>,
) -> Option<serde_json::Value> {
    let (_key, value) = keys
        .iter()
        .find_map(|key| attrs.get(key).map(|value| (*key, *value)))?;
    let Some(mut value) = value.as_json() else {
        push_issue(issues, "unsupported_structured_content");
        return None;
    };
    if let Some(serialized) = value.as_str() {
        if let Ok(parsed) = serde_json::from_str(serialized) {
            value = parsed;
        }
    }
    Some(value)
}

fn compact_content(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(value) if !value.trim().is_empty() => {
            Some(value.trim().to_string())
        }
        serde_json::Value::Null => None,
        value => serde_json::to_string(value).ok(),
    }
}

/// A span coupled with the service.name of its ResourceSpan.
struct TaggedSpan {
    span: Span,
    service_name: Option<String>,
}

fn resource_service_name(resource_span: &ResourceSpan) -> Option<String> {
    resource_span.resource.as_ref().and_then(|resource| {
        resource.attributes.as_ref()?.iter().find_map(|attribute| {
            if attribute.key == "service.name" {
                attribute.value.as_str().map(str::to_string)
            } else {
                None
            }
        })
    })
}

fn sort_spans(spans: &mut [TaggedSpan]) {
    // OTLP encodes uint64 nanosecond timestamps as decimal strings in JSON.
    // Compare their numeric value: lexical ordering would place "10" before
    // "2" and silently reverse short synthetic or truncated timestamps.
    spans.sort_by(|a, b| {
        let a = a
            .span
            .start_time_unix_nano
            .as_deref()
            .and_then(|value| value.parse::<u128>().ok());
        let b = b
            .span
            .start_time_unix_nano
            .as_deref()
            .and_then(|value| value.parse::<u128>().ok());
        match (a, b) {
            (Some(a), Some(b)) => a.cmp(&b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
    });
}

/// Parse an OTEL JSON export into a Trace.
pub fn parse(data: &str) -> Result<Trace> {
    let export: OtelExport = serde_json::from_str(data).context("Failed to parse OTEL JSON")?;

    // Collect spans tagged with their resource's service.name.
    let mut all_spans: Vec<TaggedSpan> = Vec::new();
    let mut agent_name = "unknown".to_string();

    for rs in export.resource_spans {
        // Extract service.name from resource attributes.
        let svc_name = resource_service_name(&rs);

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

    sort_spans(&mut all_spans);

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

    Ok(normalize_trace(
        trace_id,
        agent_name,
        spans_to_process,
        mixed_traces,
    ))
}

/// Parse one OTLP/HTTP JSON export into one normalized trace per `traceId`.
///
/// Unlike [`parse`], this batch-oriented API never selects one trace and drops
/// the others when an `ExportTraceServiceRequest` contains spans from multiple
/// traces. The returned traces are ordered by the first occurrence of each
/// `traceId` in the export. A structurally valid export with no spans returns an
/// empty vector, matching OTLP's empty-export success semantics.
pub fn parse_many(data: &str) -> Result<Vec<Trace>> {
    let export: OtelExport = serde_json::from_str(data).context("Failed to parse OTEL JSON")?;

    // A Vec preserves wire order while the index provides O(1) grouping.
    let mut groups: Vec<(String, Vec<TaggedSpan>)> = Vec::new();
    let mut group_indices: HashMap<String, usize> = HashMap::new();

    for resource_span in export.resource_spans {
        let service_name = resource_service_name(&resource_span);
        for scope_span in resource_span.scope_spans {
            for span in scope_span.spans {
                let trace_id = span.trace_id.clone();
                let group_index = match group_indices.get(&trace_id).copied() {
                    Some(index) => index,
                    None => {
                        let index = groups.len();
                        groups.push((trace_id.clone(), Vec::new()));
                        group_indices.insert(trace_id, index);
                        index
                    }
                };
                groups[group_index].1.push(TaggedSpan {
                    span,
                    service_name: service_name.clone(),
                });
            }
        }
    }

    let mut traces = Vec::with_capacity(groups.len());
    for (trace_id, mut spans) in groups {
        sort_spans(&mut spans);
        let agent_name = spans
            .iter()
            .find_map(|tagged| tagged.service_name.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let spans_to_process = spans.iter().collect();
        traces.push(normalize_trace(
            trace_id,
            agent_name,
            spans_to_process,
            false,
        ));
    }

    Ok(traces)
}

/// Parse one binary OTLP `ExportTraceServiceRequest` into one normalized trace
/// per trace ID.
///
/// Protobuf decoding ignores unknown fields by design, preserving OTLP's
/// forward-compatibility contract. The generated OpenTelemetry types are then
/// converted through their canonical proto-JSON representation so both OTLP
/// transports share exactly the same normalizer and multi-trace behavior.
/// Empty exports return an empty vector. A malformed protobuf message returns
/// an error; callers must not acknowledge it as accepted.
pub fn parse_many_protobuf(data: &[u8]) -> Result<Vec<Trace>> {
    let export = ExportTraceServiceRequest::decode(data)
        .context("Failed to decode OTLP protobuf ExportTraceServiceRequest")?;
    let canonical_json = serde_json::to_string(&export)
        .context("Failed to convert OTLP protobuf request to canonical JSON")?;
    parse_many(&canonical_json)
}

struct NormalizedTokenUsage {
    step_tokens: u32,
    metadata: serde_json::Value,
    issues: Vec<String>,
    provider_total_available: bool,
    enforcement_eligible: bool,
}

fn token_field(
    attrs: &HashMap<&str, &AttributeValue>,
    keys: &[&str],
    label: &str,
    issues: &mut Vec<String>,
) -> Option<i64> {
    let (key, value) = keys
        .iter()
        .find_map(|key| attrs.get(key).map(|value| (*key, *value)))?;
    match value.as_i64() {
        Some(value) if value >= 0 => Some(value),
        _ => {
            push_issue(issues, format!("invalid_{label}:{key}"));
            None
        }
    }
}

fn field_provenance(value: Option<i64>) -> &'static str {
    if value.is_some() {
        "provider_reported"
    } else {
        "missing"
    }
}

fn normalize_token_usage(attrs: &HashMap<&str, &AttributeValue>) -> NormalizedTokenUsage {
    let mut issues = Vec::new();
    let input = token_field(
        attrs,
        &[
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.prompt_tokens",
            "llm.usage.prompt_tokens",
        ],
        "input_tokens",
        &mut issues,
    );
    let output = token_field(
        attrs,
        &[
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.completion_tokens",
            "llm.usage.completion_tokens",
        ],
        "output_tokens",
        &mut issues,
    );
    let cache_read = token_field(
        attrs,
        &[
            "gen_ai.usage.cache_read.input_tokens",
            "gen_ai.usage.cached_input_tokens",
            "llm.usage.cache_read_input_tokens",
        ],
        "cache_read_tokens",
        &mut issues,
    );
    let cache_write = token_field(
        attrs,
        &[
            "gen_ai.usage.cache_creation.input_tokens",
            "gen_ai.usage.cache_write.input_tokens",
            "llm.usage.cache_creation_input_tokens",
        ],
        "cache_write_tokens",
        &mut issues,
    );
    let reasoning = token_field(
        attrs,
        &[
            "gen_ai.usage.reasoning.output_tokens",
            "gen_ai.usage.reasoning_tokens",
            "llm.usage.reasoning_tokens",
        ],
        "reasoning_tokens",
        &mut issues,
    );
    let reported_total = token_field(
        attrs,
        &["gen_ai.usage.total_tokens", "llm.usage.total_tokens"],
        "total_tokens",
        &mut issues,
    );

    let derived_total = match (input, output) {
        (Some(input), Some(output)) => match input.checked_add(output) {
            Some(total) => Some(total),
            None => {
                push_issue(&mut issues, "token_total_overflow");
                None
            }
        },
        _ => None,
    };
    if let (Some(reported), Some(derived)) = (reported_total, derived_total) {
        if reported != derived {
            push_issue(
                &mut issues,
                "reported_total_does_not_match_input_plus_output",
            );
        }
    }
    let (total, total_source) = match (reported_total, derived_total) {
        (Some(total), _) => (Some(total), "reported_total"),
        (None, Some(total)) => (Some(total), "derived_from_reported_input_output"),
        (None, None) => {
            push_issue(&mut issues, "missing_total_token_usage");
            (None, "missing")
        }
    };

    let step_tokens = match total {
        Some(total) => match u32::try_from(total) {
            Ok(total) => total,
            Err(_) => {
                push_issue(&mut issues, "total_tokens_exceed_v1_u32_range");
                u32::MAX
            }
        },
        // The v1 TraceStep field is not optional. Keep the compatibility value
        // at zero while the adjacent ledger explicitly records total=null,
        // provenance=missing, and enforcement_eligible=false.
        None => 0,
    };
    let total_provenance = field_provenance(total);
    let metadata = serde_json::json!({
        "input_tokens": input,
        "output_tokens": output,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
        "total_source": total_source,
        "estimate_status": total_provenance,
        "provenance": {
            "input_tokens": field_provenance(input),
            "output_tokens": field_provenance(output),
            "cache_read_tokens": field_provenance(cache_read),
            "cache_write_tokens": field_provenance(cache_write),
            "reasoning_tokens": field_provenance(reasoning),
            "total_tokens": total_provenance,
        }
    });
    let provider_total_available = total.is_some();
    let enforcement_eligible = provider_total_available && issues.is_empty();
    NormalizedTokenUsage {
        step_tokens,
        metadata,
        issues,
        provider_total_available,
        enforcement_eligible,
    }
}

fn normalize_trace(
    trace_id: String,
    agent_name: String,
    spans_to_process: Vec<&TaggedSpan>,
    mixed_traces: bool,
) -> Trace {
    let mut steps: Vec<TraceStep> = Vec::new();
    let mut span_ledger = Vec::with_capacity(spans_to_process.len());
    let source_span_count = spans_to_process.len();
    let mut trace_issues = Vec::new();
    let mut provider_usage_steps = 0usize;
    let mut normalized_content_steps = 0usize;
    let mut enforcement_eligible = !mixed_traces;
    let mut counter = 1u32;

    if mixed_traces {
        push_issue(
            &mut trace_issues,
            "mixed_trace_export_filtered_by_legacy_parser",
        );
    }

    for tagged in &spans_to_process {
        let span = &tagged.span;
        let attrs: HashMap<&str, &AttributeValue> = span
            .attributes
            .iter()
            .map(|a| (a.key.as_str(), &a.value))
            .collect();
        let usage = normalize_token_usage(&attrs);

        // Skip root/chain spans (those with no gen_ai operation).
        let op = attrs
            .get("gen_ai.operation.name")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let step_type = match op {
            "chat" | "generate" | "generate_content" | "text_completion" => {
                Some(StepType::Reasoning)
            }
            "execute_tool" | "tool" => Some(StepType::ToolCall),
            _ => {
                // Include spans with gen_ai attributes, skip pure orchestration spans.
                if attrs.keys().any(|k| k.starts_with("gen_ai.")) {
                    Some(StepType::Reasoning)
                } else {
                    None
                }
            }
        };
        let parent_span_id = span
            .parent_span_id
            .as_deref()
            .filter(|value| !value.is_empty());
        let Some(step_type) = step_type else {
            span_ledger.push(serde_json::json!({
                "span_id": span.span_id,
                "parent_span_id": parent_span_id,
                "step_id": null,
                "normalization_status": "ignored_non_genai_span",
                "content_status": "not_applicable",
                "token_usage": usage.metadata,
                "issues": usage.issues,
            }));
            continue;
        };

        let mut span_issues = usage.issues;
        if usage.provider_total_available {
            provider_usage_steps += 1;
        }
        if !usage.enforcement_eligible {
            enforcement_eligible = false;
        }

        let tool_name = if step_type == StepType::ToolCall {
            attrs
                .get("gen_ai.tool.name")
                .or_else(|| attrs.get("gen_ai.tool.call.name"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| Some(span.name.clone()))
        } else {
            None
        };

        let tool_params = if step_type == StepType::ToolCall {
            tool_value(
                &attrs,
                &["gen_ai.tool.call.arguments", "gen_ai.tool.arguments"],
                &mut span_issues,
            )
        } else {
            None
        };
        let tool_result = if step_type == StepType::ToolCall {
            tool_value(
                &attrs,
                &["gen_ai.tool.call.result", "gen_ai.tool.result"],
                &mut span_issues,
            )
        } else {
            None
        };

        let tool_success = if step_type == StepType::ToolCall {
            let is_error = span
                .status
                .as_ref()
                .and_then(|s| s.code.as_ref())
                .map(SpanStatusCode::is_error)
                .unwrap_or(false);
            Some(!is_error)
        } else {
            None
        };

        // Split content into input and output for CCE and downstream metrics.
        let input_context = extract_input_content(span, &attrs, &mut span_issues);
        let model_output = extract_output_content(span, &attrs, &mut span_issues);
        let tool_result_content = tool_result.as_ref().and_then(compact_content);
        let output = join_parts(
            [model_output.clone(), tool_result_content.clone()]
                .into_iter()
                .flatten()
                .collect(),
        );
        let tool_argument_content = tool_params.as_ref().and_then(compact_content);
        let normalized_content = join_parts(
            [input_context.clone(), output.clone(), tool_argument_content]
                .into_iter()
                .flatten()
                .collect(),
        );
        let (content, content_status) = match normalized_content {
            Some(content) => {
                normalized_content_steps += 1;
                (content, "normalized")
            }
            None => {
                push_issue(&mut span_issues, "content_missing_used_span_name");
                enforcement_eligible = false;
                (span.name.clone(), "span_name_fallback")
            }
        };

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

        for issue in &span_issues {
            push_issue(&mut trace_issues, issue.clone());
        }
        let step_id = counter;
        span_ledger.push(serde_json::json!({
            "span_id": span.span_id,
            "parent_span_id": parent_span_id,
            "step_id": step_id,
            "normalization_status": "normalized",
            "content_status": content_status,
            "token_usage": usage.metadata,
            "issues": span_issues,
        }));

        steps.push(TraceStep {
            id: counter,
            step_type,
            content,
            tokens: usage.step_tokens,
            tool_name,
            tool_params,
            tool_success,
            tool_error: span.status.as_ref().and_then(|s| s.message.clone()),
            agent_id,
            input_context,
            output,
            flags: vec![],
            flag_details: vec![],
        });

        counter += 1;
    }

    if steps.is_empty() && source_span_count > 0 {
        push_issue(&mut trace_issues, "no_usable_genai_steps");
        enforcement_eligible = false;
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
    let normalized_step_count = steps.len();
    let denominator = normalized_step_count.max(1) as f64;
    let provider_token_coverage = provider_usage_steps as f64 / denominator;
    let content_coverage = normalized_content_steps as f64 / denominator;
    let degraded_ingest = !enforcement_eligible;
    metadata.insert(
        "otlp".to_string(),
        serde_json::json!({
            "schema_version": "tracerazor-otlp-normalization/v1",
            "source_span_count": source_span_count,
            "normalized_step_count": normalized_step_count,
            "provider_token_coverage": provider_token_coverage,
            "content_coverage": content_coverage,
            "estimate_status": if provider_usage_steps == normalized_step_count && normalized_step_count > 0 {
                "provider_reported"
            } else {
                "missing"
            },
            "degraded": degraded_ingest,
            "degraded_ingest": degraded_ingest,
            "enforcement_eligible": enforcement_eligible,
            "issues": trace_issues,
            "spans": span_ledger,
        }),
    );

    Trace {
        trace_id,
        agent_name,
        framework: "otel".to_string(),
        steps,
        total_tokens,
        task_value_score: 1.0,
        metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
    use prost::Message;

    const MIXED_EXPORT: &str = r#"
    {
      "resourceSpans": [
        {
          "resource": {"attributes": [
            {"key": "service.name", "value": {"stringValue": "agent-a"}}
          ]},
          "scopeSpans": [{"spans": [{
            "traceId": "trace-a",
            "spanId": "a1",
            "name": "chat a",
            "startTimeUnixNano": "1",
            "attributes": [
              {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
              {"key": "gen_ai.usage.total_tokens", "value": {"intValue": "10"}}
            ]
          }]}]
        },
        {
          "resource": {"attributes": [
            {"key": "service.name", "value": {"stringValue": "agent-b"}}
          ]},
          "scopeSpans": [{"spans": [
            {
              "traceId": "trace-b",
              "spanId": "b2",
              "name": "tool b",
              "startTimeUnixNano": "3",
              "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                {"key": "gen_ai.usage.total_tokens", "value": {"intValue": "20"}}
              ]
            },
            {
              "traceId": "trace-b",
              "spanId": "b1",
              "name": "chat b",
              "startTimeUnixNano": "2",
              "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.usage.total_tokens", "value": {"intValue": "30"}}
              ]
            }
          ]}]
        }
      ]
    }
    "#;

    #[test]
    fn parse_many_preserves_every_trace_in_first_seen_order() {
        let traces = parse_many(MIXED_EXPORT).unwrap();

        assert_eq!(traces.len(), 2);
        assert_eq!(traces[0].trace_id, "trace-a");
        assert_eq!(traces[0].agent_name, "agent-a");
        assert_eq!(traces[0].total_tokens, 10);
        assert_eq!(traces[1].trace_id, "trace-b");
        assert_eq!(traces[1].agent_name, "agent-b");
        assert_eq!(traces[1].total_tokens, 50);
        assert_eq!(traces[1].steps[0].content, "chat b");
        assert_eq!(traces[1].steps[1].content, "tool b");
        assert!(!traces
            .iter()
            .any(|trace| trace.metadata.contains_key("warning")));

        // The compatibility parser keeps its documented single-trace behavior.
        let legacy = parse(MIXED_EXPORT).unwrap();
        assert_eq!(legacy.trace_id, "trace-b");
        assert!(legacy.metadata.contains_key("warning"));
    }

    #[test]
    fn nanosecond_timestamps_sort_numerically_not_lexically() {
        let payload = r#"{
          "resourceSpans": [{"scopeSpans": [{"spans": [
            {
              "traceId": "trace-order",
              "spanId": "late",
              "name": "late at ten",
              "startTimeUnixNano": "10",
              "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.usage.total_tokens", "value": {"intValue": "1"}}
              ]
            },
            {
              "traceId": "trace-order",
              "spanId": "early",
              "name": "early at two",
              "startTimeUnixNano": "2",
              "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.usage.total_tokens", "value": {"intValue": "1"}}
              ]
            }
          ]}]}
        ]}"#;

        let trace = parse_many(payload).unwrap().remove(0);

        assert_eq!(trace.steps[0].content, "early at two");
        assert_eq!(trace.steps[1].content, "late at ten");
        assert_eq!(trace.metadata["otlp"]["spans"][0]["span_id"], "early");
        assert_eq!(trace.metadata["otlp"]["spans"][1]["span_id"], "late");
    }

    #[test]
    fn parse_many_accepts_empty_otlp_exports() {
        for payload in [
            "{}",
            r#"{"resourceSpans": []}"#,
            r#"{"resourceSpans": [{"scopeSpans": [{}]}]}"#,
        ] {
            assert!(
                parse_many(payload).unwrap().is_empty(),
                "payload: {payload}"
            );
            assert!(parse(payload).is_err(), "legacy parse contract changed");
        }
    }

    #[test]
    fn structured_messages_tools_lineage_and_token_categories_are_preserved() {
        let message = |role: &str, content: &str| {
            serde_json::json!({
                "kvlistValue": {"values": [
                    {"key": "role", "value": {"stringValue": role}},
                    {"key": "parts", "value": {"arrayValue": {"values": [
                        {"kvlistValue": {"values": [
                            {"key": "type", "value": {"stringValue": "text"}},
                            {"key": "content", "value": {"stringValue": content}}
                        ]}}
                    ]}}}
                ]}
            })
        };
        let payload = serde_json::json!({
            "resourceSpans": [{"scopeSpans": [{"spans": [
                {
                    "traceId": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "spanId": "1111111111111111",
                    "parentSpanId": "3333333333333333",
                    "name": "chat",
                    "startTimeUnixNano": "1",
                    "attributes": [
                        {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                        {"key": "gen_ai.input.messages", "value": {"arrayValue": {"values": [message("user", "Weather in Paris?")]}}},
                        {"key": "gen_ai.output.messages", "value": {"arrayValue": {"values": [message("assistant", "Rainy and 57F")]}}},
                        {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "100"}},
                        {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "50"}},
                        {"key": "gen_ai.usage.cache_read.input_tokens", "value": {"intValue": "20"}},
                        {"key": "gen_ai.usage.cache_creation.input_tokens", "value": {"intValue": "10"}},
                        {"key": "gen_ai.usage.reasoning.output_tokens", "value": {"intValue": "5"}},
                        {"key": "gen_ai.usage.total_tokens", "value": {"intValue": "150"}}
                    ]
                },
                {
                    "traceId": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "spanId": "2222222222222222",
                    "parentSpanId": "1111111111111111",
                    "name": "weather",
                    "startTimeUnixNano": "2",
                    "attributes": [
                        {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                        {"key": "gen_ai.tool.name", "value": {"stringValue": "get_weather"}},
                        {"key": "gen_ai.tool.call.arguments", "value": {"kvlistValue": {"values": [
                            {"key": "location", "value": {"stringValue": "Paris"}}
                        ]}}},
                        {"key": "gen_ai.tool.call.result", "value": {"kvlistValue": {"values": [
                            {"key": "conditions", "value": {"stringValue": "rainy"}}
                        ]}}},
                        {"key": "gen_ai.usage.total_tokens", "value": {"intValue": "12"}}
                    ]
                }
            ]}]}]
        });

        let trace = parse_many(&payload.to_string()).unwrap().remove(0);
        let protobuf_request: ExportTraceServiceRequest =
            serde_json::from_value(payload.clone()).unwrap();
        let protobuf_trace = parse_many_protobuf(&protobuf_request.encode_to_vec())
            .unwrap()
            .remove(0);
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.steps[0].tokens, 150);
        assert!(trace.steps[0]
            .input_context
            .as_deref()
            .unwrap()
            .contains("Weather in Paris?"));
        assert!(trace.steps[0]
            .output
            .as_deref()
            .unwrap()
            .contains("Rainy and 57F"));
        assert_eq!(trace.steps[1].tool_name.as_deref(), Some("get_weather"));
        assert_eq!(
            trace.steps[1].tool_params,
            Some(serde_json::json!({"location": "Paris"}))
        );
        assert!(trace.steps[1].output.as_deref().unwrap().contains("rainy"));
        assert_eq!(
            protobuf_trace.steps[0].input_context,
            trace.steps[0].input_context
        );
        assert_eq!(protobuf_trace.steps[0].output, trace.steps[0].output);
        assert_eq!(
            protobuf_trace.steps[1].tool_params,
            trace.steps[1].tool_params
        );

        let otlp = &trace.metadata["otlp"];
        assert_eq!(otlp["enforcement_eligible"], true);
        assert_eq!(otlp["estimate_status"], "provider_reported");
        assert_eq!(otlp["spans"][0]["span_id"], "1111111111111111");
        assert_eq!(otlp["spans"][0]["parent_span_id"], "3333333333333333");
        assert_eq!(otlp["spans"][1]["parent_span_id"], "1111111111111111");
        let usage = &otlp["spans"][0]["token_usage"];
        assert_eq!(usage["input_tokens"], 100);
        assert_eq!(usage["output_tokens"], 50);
        assert_eq!(usage["cache_read_tokens"], 20);
        assert_eq!(usage["cache_write_tokens"], 10);
        assert_eq!(usage["reasoning_tokens"], 5);
        assert_eq!(usage["total_tokens"], 150);
        assert_eq!(usage["provenance"]["total_tokens"], "provider_reported");
    }

    #[test]
    fn missing_usage_is_null_degraded_and_never_enforcement_eligible() {
        let payload = serde_json::json!({
            "resourceSpans": [{"scopeSpans": [{"spans": [{
                "traceId": "trace-missing",
                "spanId": "span-child",
                "parentSpanId": "span-parent",
                "name": "chat",
                "attributes": [
                    {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                    {"key": "gen_ai.prompt", "value": {"stringValue": "A real prompt with content"}}
                ]
            }]}]}]
        });
        let trace = parse_many(&payload.to_string()).unwrap().remove(0);

        assert_eq!(trace.steps[0].tokens, 0);
        let otlp = &trace.metadata["otlp"];
        assert_eq!(otlp["degraded_ingest"], true);
        assert_eq!(otlp["enforcement_eligible"], false);
        assert_eq!(otlp["estimate_status"], "missing");
        assert_eq!(otlp["spans"][0]["span_id"], "span-child");
        assert_eq!(otlp["spans"][0]["parent_span_id"], "span-parent");
        assert!(otlp["spans"][0]["token_usage"]["total_tokens"].is_null());
        assert_eq!(
            otlp["spans"][0]["token_usage"]["provenance"]["total_tokens"],
            "missing"
        );
        assert!(otlp["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue == "missing_total_token_usage"));
    }

    #[test]
    fn protobuf_parser_preserves_all_trace_ids_and_json_semantics() {
        let canonical = MIXED_EXPORT
            .replace("trace-a", "11111111111111111111111111111111")
            .replace("trace-b", "22222222222222222222222222222222")
            .replace("\"a1\"", "\"1111111111111111\"")
            .replace("\"b1\"", "\"2222222222222221\"")
            .replace("\"b2\"", "\"2222222222222222\"");
        let mut canonical: serde_json::Value = serde_json::from_str(&canonical).unwrap();
        let tool_span = &mut canonical["resourceSpans"][1]["scopeSpans"][0]["spans"][0];
        tool_span["status"] = serde_json::json!({"code": 2, "message": "tool failed"});
        tool_span["attributes"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "key": "example.tags",
                "value": {"arrayValue": {"values": [{"stringValue": "safe"}]}}
            }));
        let request: ExportTraceServiceRequest = serde_json::from_value(canonical.clone()).unwrap();
        let encoded = request.encode_to_vec();

        let protobuf = parse_many_protobuf(&encoded).unwrap();
        let json = parse_many(&canonical.to_string()).unwrap();

        assert_eq!(protobuf.len(), 2);
        assert_eq!(json.len(), protobuf.len());
        assert_eq!(protobuf[0].trace_id, "11111111111111111111111111111111");
        assert_eq!(protobuf[1].trace_id, "22222222222222222222222222222222");
        assert_eq!(protobuf[0].total_tokens, 10);
        assert_eq!(protobuf[1].total_tokens, 50);
        assert_eq!(protobuf[1].steps[1].tool_success, Some(false));
        assert_eq!(
            protobuf[1].steps[1].tool_error.as_deref(),
            Some("tool failed")
        );
        for (protobuf_trace, json_trace) in protobuf.iter().zip(&json) {
            assert_eq!(protobuf_trace.trace_id, json_trace.trace_id);
            assert_eq!(protobuf_trace.agent_name, json_trace.agent_name);
            assert_eq!(protobuf_trace.total_tokens, json_trace.total_tokens);
            assert_eq!(protobuf_trace.steps.len(), json_trace.steps.len());
            for (protobuf_step, json_step) in protobuf_trace.steps.iter().zip(&json_trace.steps) {
                assert_eq!(protobuf_step.content, json_step.content);
                assert_eq!(protobuf_step.tokens, json_step.tokens);
                assert_eq!(protobuf_step.step_type, json_step.step_type);
            }
        }
    }

    #[test]
    fn protobuf_parser_accepts_empty_and_rejects_malformed_exports() {
        assert!(parse_many_protobuf(&[]).unwrap().is_empty());
        assert!(parse_many_protobuf(&[0xff]).is_err());
    }
}
