//! Flexible parser for Langfuse JSON trace/observation exports.
//!
//! Langfuse exports vary by route/version. This parser accepts common shapes:
//! `{ "observations": [...] }`, `{ "traces": [{..., "observations": [...] }] }`,
//! a single trace object with `observations`, or a bare observations array.

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use tracerazor_core::types::{StepType, Trace, TraceStep};

pub fn parse(data: &str) -> Result<Trace> {
    let v: Value = serde_json::from_str(data).context("Invalid JSON in Langfuse export")?;
    let observations = collect_observations(&v);
    if observations.is_empty() {
        anyhow::bail!("Langfuse export contains no observations");
    }

    let mut obs = observations;
    obs.sort_by(|a, b| {
        string_field(a, &["startTime", "start_time", "createdAt", "timestamp"]).cmp(&string_field(
            b,
            &["startTime", "start_time", "createdAt", "timestamp"],
        ))
    });

    let trace_id = string_field(&v, &["id", "traceId", "trace_id"])
        .or_else(|| {
            obs.iter()
                .find_map(|o| string_field(o, &["traceId", "trace_id"]))
        })
        .unwrap_or_else(|| "langfuse-trace".into());
    let agent_name = string_field(&v, &["name", "sessionId", "userId"])
        .or_else(|| obs.iter().find_map(|o| string_field(o, &["name"])))
        .unwrap_or_else(|| "langfuse-agent".into());

    let mut steps = Vec::new();
    for item in obs {
        let step_type = match string_field(&item, &["type"])
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "generation" | "llm" | "completion" | "chat" => StepType::Reasoning,
            "span" | "tool" | "retriever" => StepType::ToolCall,
            _ => {
                if item.get("model").is_some() {
                    StepType::Reasoning
                } else {
                    StepType::ToolCall
                }
            }
        };
        let content = content_from(&item);
        let output = textish(item.get("output").or_else(|| item.get("completion")));
        steps.push(TraceStep {
            id: steps.len() as u32 + 1,
            step_type: step_type.clone(),
            content,
            tokens: extract_tokens(&item),
            tool_name: (step_type == StepType::ToolCall).then(|| {
                string_field(&item, &["name"]).unwrap_or_else(|| "langfuse_observation".into())
            }),
            tool_params: item.get("input").cloned(),
            tool_success: (step_type == StepType::ToolCall).then(|| {
                item.get("level")
                    .and_then(Value::as_str)
                    .unwrap_or("DEFAULT")
                    != "ERROR"
            }),
            tool_error: item
                .get("statusMessage")
                .or_else(|| item.get("error"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            input_context: textish(item.get("input").or_else(|| item.get("prompt"))),
            output,
            ..Default::default()
        });
    }
    let total_tokens = steps.iter().map(|s| s.tokens).sum();
    let mut metadata = HashMap::new();
    metadata.insert("source".into(), Value::String("langfuse-export".into()));
    let model = v
        .get("model")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| observations_model(data));
    if let Some(model) = model {
        metadata.insert("model".into(), Value::String(model));
    }

    Ok(Trace {
        trace_id,
        agent_name,
        framework: "langfuse".into(),
        steps,
        total_tokens,
        task_value_score: 1.0,
        metadata,
    })
}

fn collect_observations(v: &Value) -> Vec<Value> {
    if let Some(obs) = v.get("observations").and_then(Value::as_array) {
        return obs.clone();
    }
    if let Some(traces) = v.get("traces").and_then(Value::as_array) {
        return traces
            .iter()
            .flat_map(|t| {
                t.get("observations")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_else(|| vec![t.clone()])
            })
            .collect();
    }
    if let Some(arr) = v.as_array() {
        if arr.iter().any(|x| x.get("observations").is_some()) {
            return arr
                .iter()
                .flat_map(|t| {
                    t.get("observations")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_else(|| vec![t.clone()])
                })
                .collect();
        }
        return arr.clone();
    }
    vec![v.clone()]
}

fn extract_tokens(v: &Value) -> u32 {
    let usage = v
        .get("usage")
        .or_else(|| v.get("usageDetails"))
        .or_else(|| v.get("usage_details"))
        .unwrap_or(v);
    let total = number(
        usage,
        &["total", "totalTokens", "total_tokens", "totalUsage"],
    )
    .or_else(|| {
        let input = number(
            usage,
            &["input", "inputTokens", "promptTokens", "prompt_tokens"],
        );
        let output = number(
            usage,
            &[
                "output",
                "outputTokens",
                "completionTokens",
                "completion_tokens",
            ],
        );
        match (input, output) {
            (None, None) => None,
            (a, b) => Some(a.unwrap_or(0) + b.unwrap_or(0)),
        }
    })
    .unwrap_or(0);
    u32::try_from(total).unwrap_or(u32::MAX)
}

fn content_from(v: &Value) -> String {
    let mut parts = Vec::new();
    for key in ["input", "prompt", "output", "completion"] {
        if let Some(text) = textish(v.get(key)) {
            if !text.is_empty() {
                parts.push(text);
            }
        }
    }
    if parts.is_empty() {
        string_field(v, &["name"]).unwrap_or_else(|| "langfuse observation".into())
    } else {
        parts.join("\n")
    }
}

fn number(v: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| {
        v.get(*key).and_then(|n| {
            n.as_u64()
                .or_else(|| n.as_str().and_then(|s| s.parse().ok()))
        })
    })
}

fn string_field(v: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| v.get(*key).and_then(Value::as_str))
        .map(ToOwned::to_owned)
}

fn textish(v: Option<&Value>) -> Option<String> {
    let v = v?;
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    if v.is_null() {
        return None;
    }
    serde_json::to_string(v).ok()
}

fn observations_model(data: &str) -> Option<String> {
    let v: Value = serde_json::from_str(data).ok()?;
    collect_observations(&v).iter().find_map(|o| {
        o.get("model")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_langfuse_shape() {
        let json = r#"{
          "id":"trace-1",
          "name":"support",
          "observations":[
            {"id":"obs-1","type":"GENERATION","name":"llm","input":"hello","output":"hi","usageDetails":{"input":10,"output":20}},
            {"id":"obs-2","type":"SPAN","name":"lookup","input":{"id":"1"},"output":"found","usage":{"total":5}}
          ]
        }"#;
        let trace = parse(json).unwrap();
        assert_eq!(trace.trace_id, "trace-1");
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.total_tokens, 35);
        assert_eq!(trace.steps[1].tool_name.as_deref(), Some("lookup"));
    }
}
