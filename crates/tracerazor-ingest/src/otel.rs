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
    span_id: String,
    parent_span_id: Option<String>,
    name: String,
    #[serde(default)]
    attributes: Vec<Attribute>,
    status: Option<SpanStatus>,
    start_time_unix_nano: Option<String>,
    end_time_unix_nano: Option<String>,
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
    Bool { #[serde(alias = "boolValue")] bool_value: bool },
    Double { #[serde(alias = "doubleValue")] double_value: f64 },
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
            AttributeValue::Int { int_value } => int_value.as_i64(),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SpanStatus {
    code: Option<String>,
    message: Option<String>,
}

/// Parse an OTEL JSON export into a Trace.
pub fn parse(data: &str) -> Result<Trace> {
    let export: OtelExport =
        serde_json::from_str(data).context("Failed to parse OTEL JSON")?;

    let mut all_spans: Vec<Span> = Vec::new();
    let mut trace_id = String::new();
    let mut agent_name = "unknown".to_string();

    for rs in export.resource_spans {
        // Extract agent name from resource attributes.
        if let Some(resource) = rs.resource {
            if let Some(attrs) = resource.attributes {
                for attr in &attrs {
                    if attr.key == "service.name" {
                        if let Some(s) = attr.value.as_str() {
                            agent_name = s.to_string();
                        }
                    }
                }
            }
        }
        for ss in rs.scope_spans {
            all_spans.extend(ss.spans);
        }
    }

    if all_spans.is_empty() {
        anyhow::bail!("OTEL export contains no spans");
    }

    // Use the first span's trace_id.
    trace_id = all_spans[0].trace_id.clone();

    // Sort spans by start time (lexicographic on nanosecond timestamps works).
    all_spans.sort_by(|a, b| {
        a.start_time_unix_nano
            .as_deref()
            .unwrap_or("")
            .cmp(b.start_time_unix_nano.as_deref().unwrap_or(""))
    });

    let mut steps: Vec<TraceStep> = Vec::new();
    let mut counter = 1u32;

    for span in &all_spans {
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

        let tokens = attrs
            .get("gen_ai.usage.total_tokens")
            .and_then(|v| v.as_i64())
            .or_else(|| {
                let i = attrs.get("gen_ai.usage.input_tokens")?.as_i64()?;
                let o = attrs
                    .get("gen_ai.usage.output_tokens")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                Some(i + o)
            })
            .unwrap_or(0) as u32;

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

        let content = attrs
            .get("gen_ai.prompt")
            .and_then(|v| v.as_str())
            .unwrap_or(&span.name)
            .to_string();

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
            agent_id: None,
            input_context: None,
            output: None,
            flags: vec![],
            flag_details: vec![],
        });

        counter += 1;
    }

    let total_tokens: u32 = steps.iter().map(|s| s.tokens).sum();

    Ok(Trace {
        trace_id,
        agent_name,
        framework: "otel".to_string(),
        steps,
        total_tokens,
        task_value_score: 1.0,
        metadata: HashMap::new(),
    })
}
