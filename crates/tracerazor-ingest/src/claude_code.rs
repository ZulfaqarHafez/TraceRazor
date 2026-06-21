//! Parser for Claude Code local transcript JSONL.
//!
//! Claude Code stores one JSONL transcript per session. Assistant API messages
//! can be split across several transcript lines that repeat the same `usage`
//! object, so this parser groups lines by message id before counting tokens.

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use tracerazor_core::types::{StepType, Trace, TraceStep};

const MAX_CONTENT_CHARS: usize = 2_000;
const MAX_OUTPUT_CHARS: usize = 2_000;
const MAX_CONTEXT_CHARS: usize = 4_000;

#[derive(Debug, Clone)]
struct AssistantMessage {
    model: String,
    blocks: Vec<Value>,
    usage: Value,
    sidechain: bool,
    context: String,
}

pub fn parse(data: &str) -> Result<Trace> {
    let entries = load_entries(data)?;
    if entries.is_empty() {
        anyhow::bail!("Claude Code transcript contains no assistant/user entries");
    }

    let mut tool_results: HashMap<String, Value> = HashMap::new();
    for entry in &entries {
        if entry.get("type").and_then(Value::as_str) != Some("user") {
            continue;
        }
        if let Some(blocks) = entry
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(Value::as_array)
        {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                    if let Some(id) = block.get("tool_use_id").and_then(Value::as_str) {
                        tool_results.insert(id.to_string(), block.clone());
                    }
                }
            }
        }
    }

    let mut messages: Vec<AssistantMessage> = Vec::new();
    let mut by_id: HashMap<String, usize> = HashMap::new();
    let mut pending_context: Vec<String> = Vec::new();
    let mut first_user_prompt: Option<String> = None;
    let mut session_id: Option<String> = None;

    for entry in &entries {
        if session_id.is_none() {
            session_id = entry
                .get("session_id")
                .or_else(|| entry.get("sessionId"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
        }
        let null_message = Value::Null;
        let msg = entry.get("message").unwrap_or(&null_message);
        match entry.get("type").and_then(Value::as_str) {
            Some("user") => {
                collect_user_context(
                    msg.get("content").unwrap_or(&Value::Null),
                    &mut pending_context,
                    &mut first_user_prompt,
                    entry.get("isSidechain").and_then(Value::as_bool).unwrap_or(false),
                );
            }
            Some("assistant") => {
                let Some(mid) = msg.get("id").and_then(Value::as_str) else {
                    continue;
                };
                let idx = if let Some(idx) = by_id.get(mid).copied() {
                    idx
                } else {
                    let rec = AssistantMessage {
                        model: msg
                            .get("model")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        blocks: Vec::new(),
                        usage: msg.get("usage").cloned().unwrap_or(Value::Null),
                        sidechain: entry
                            .get("isSidechain")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                        context: clip(&pending_context.join("\n"), MAX_CONTEXT_CHARS),
                    };
                    pending_context.clear();
                    messages.push(rec);
                    let idx = messages.len() - 1;
                    by_id.insert(mid.to_string(), idx);
                    idx
                };
                if let Some(blocks) = msg.get("content").and_then(Value::as_array) {
                    messages[idx].blocks.extend(blocks.iter().cloned());
                }
            }
            _ => {}
        }
    }

    let mut steps: Vec<TraceStep> = Vec::new();
    let mut total_tokens = 0u32;
    let mut first_main_seen = false;
    let mut model = String::new();
    for rec in messages {
        if model.is_empty() && !rec.model.is_empty() {
            model = rec.model.clone();
        }
        let first_turn = !rec.sidechain && !first_main_seen;
        if first_turn {
            first_main_seen = true;
        }
        let msg_tokens = usage_tokens(&rec.usage, false, first_turn);
        total_tokens = total_tokens.saturating_add(msg_tokens);

        let mut reasoning_parts: Vec<String> = Vec::new();
        let mut tool_blocks: Vec<Value> = Vec::new();
        for block in &rec.blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("thinking") => {
                    if let Some(text) = block.get("thinking").and_then(Value::as_str) {
                        reasoning_parts.push(text.to_string());
                    }
                }
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        reasoning_parts.push(text.to_string());
                    }
                }
                Some("tool_use") => tool_blocks.push(block.clone()),
                _ => {}
            }
        }

        let reasoning_text = reasoning_parts
            .into_iter()
            .filter(|s| !s.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        let n_steps = usize::from(!reasoning_text.trim().is_empty()) + tool_blocks.len();
        if n_steps == 0 {
            continue;
        }
        let share = msg_tokens / n_steps as u32;
        let mut rem = msg_tokens % n_steps as u32;
        let mut first_in_message = true;
        let mut push_step = |mut step: TraceStep| {
            step.id = steps.len() as u32 + 1;
            step.tokens = share + if rem > 0 { 1 } else { 0 };
            rem = rem.saturating_sub(1);
            if rec.sidechain {
                step.agent_id = Some("subagent".into());
            }
            if first_in_message && !rec.context.is_empty() {
                step.input_context = Some(rec.context.clone());
            }
            first_in_message = false;
            steps.push(step);
        };

        if !reasoning_text.trim().is_empty() {
            push_step(TraceStep {
                step_type: StepType::Reasoning,
                content: clip(&reasoning_text, MAX_CONTENT_CHARS),
                ..Default::default()
            });
        }

        for tool in tool_blocks {
            let tool_id = tool.get("id").and_then(Value::as_str).unwrap_or_default();
            let result = tool_results.get(tool_id);
            let is_error = result
                .and_then(|r| r.get("is_error").or_else(|| r.get("isError")))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let output = result
                .map(|r| block_text(r.get("content").unwrap_or(&Value::Null)))
                .unwrap_or_default();
            push_step(TraceStep {
                step_type: StepType::ToolCall,
                content: format!(
                    "Call {} with {}",
                    tool.get("name").and_then(Value::as_str).unwrap_or("tool"),
                    tool.get("input").unwrap_or(&Value::Null)
                ),
                tool_name: tool.get("name").and_then(Value::as_str).map(ToOwned::to_owned),
                tool_params: tool.get("input").cloned(),
                tool_success: Some(!is_error),
                tool_error: is_error.then(|| clip(&output, MAX_OUTPUT_CHARS)),
                output: (!is_error && !output.is_empty()).then(|| clip(&output, MAX_OUTPUT_CHARS)),
                ..Default::default()
            });
        }
    }

    if steps.is_empty() {
        anyhow::bail!("Claude Code transcript produced no auditable steps");
    }

    let mut metadata = HashMap::new();
    metadata.insert("source".into(), Value::String("claude-code-transcript".into()));
    metadata.insert(
        "token_accounting".into(),
        Value::String(
            "marginal: input + cache_creation + output; cache reads excluded; first-turn cache_creation excluded"
                .into(),
        ),
    );
    if !model.is_empty() {
        metadata.insert("model".into(), Value::String(model.clone()));
    }
    if let Some(task) = first_user_prompt.clone().filter(|s| !s.trim().is_empty()) {
        metadata.insert("task".into(), Value::String(task));
    }

    Ok(Trace {
        trace_id: session_id.unwrap_or_else(|| "claude-code-transcript".into()),
        agent_name: if model.is_empty() {
            "claude-code".into()
        } else {
            format!("claude-code ({model})")
        },
        framework: "claude-code".into(),
        steps,
        total_tokens,
        task_value_score: 1.0,
        metadata,
    })
}

fn load_entries(data: &str) -> Result<Vec<Value>> {
    let trimmed = data.trim();
    if trimmed.starts_with('[') {
        let entries: Vec<Value> = serde_json::from_str(trimmed)
            .context("Invalid Claude Code transcript JSON array")?;
        return Ok(entries
            .into_iter()
            .filter(|v| matches!(v.get("type").and_then(Value::as_str), Some("assistant" | "user")))
            .collect());
    }
    let mut entries = Vec::new();
    for line in data.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            if matches!(v.get("type").and_then(Value::as_str), Some("assistant" | "user")) {
                entries.push(v);
            }
        }
    }
    Ok(entries)
}

fn collect_user_context(
    content: &Value,
    pending_context: &mut Vec<String>,
    first_user_prompt: &mut Option<String>,
    sidechain: bool,
) {
    match content {
        Value::String(s) => {
            if first_user_prompt.is_none() && !sidechain {
                *first_user_prompt = Some(s.clone());
            }
            pending_context.push(s.clone());
        }
        Value::Array(blocks) => {
            for block in blocks {
                if !block.is_object() {
                    continue;
                }
                match block.get("type").and_then(Value::as_str) {
                    Some("tool_result") => {
                        pending_context.push(block_text(block.get("content").unwrap_or(&Value::Null)));
                    }
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(Value::as_str) {
                            if first_user_prompt.is_none() && !sidechain {
                                *first_user_prompt = Some(text.to_string());
                            }
                            pending_context.push(text.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

fn block_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                item.as_str().map(ToOwned::to_owned).or_else(|| {
                    item.get("text")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned)
                })
            })
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(_) => content
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

fn usage_tokens(usage: &Value, include_cache_read: bool, first_turn: bool) -> u32 {
    let mut total = usage_u32(usage, "input_tokens")
        .saturating_add(usage_u32(usage, "output_tokens"));
    if !first_turn || include_cache_read {
        total = total.saturating_add(usage_u32(usage, "cache_creation_input_tokens"));
    }
    if include_cache_read {
        total = total.saturating_add(usage_u32(usage, "cache_read_input_tokens"));
    }
    total
}

fn usage_u32(usage: &Value, key: &str) -> u32 {
    usage
        .get(key)
        .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
        .map(|v| u32::try_from(v).unwrap_or(u32::MAX))
        .unwrap_or(0)
}

fn clip(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let clipped: String = text.chars().take(limit).collect();
    format!("{clipped}…[+{} chars]", text.chars().count() - limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant(mid: &str, block: Value) -> Value {
        serde_json::json!({
            "type": "assistant",
            "message": {
                "id": mid,
                "model": "claude-haiku-4-5-20251001",
                "content": [block],
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 90,
                    "cache_read_input_tokens": 500,
                    "output_tokens": 100
                }
            }
        })
    }

    #[test]
    fn counts_usage_once_per_split_message() {
        let entries = vec![
            serde_json::json!({"type":"user","message":{"content":"Fix the test"}}),
            assistant("m1", serde_json::json!({"type":"text","text":"Let me inspect."})),
            assistant("m1", serde_json::json!({"type":"tool_use","id":"tu1","name":"Read","input":{"file_path":"a.py"}})),
            serde_json::json!({"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tu1","content":[{"type":"text","text":"file contents"}]}]}}),
            assistant("m2", serde_json::json!({"type":"text","text":"Found it."})),
        ];
        let data = entries.into_iter().map(|v| v.to_string()).collect::<Vec<_>>().join("\n");
        let trace = parse(&data).unwrap();
        assert_eq!(trace.total_tokens, 310);
        assert_eq!(trace.steps.len(), 3);
        assert_eq!(trace.steps[1].tool_name.as_deref(), Some("Read"));
        assert_eq!(trace.steps[2].input_context.as_deref(), Some("file contents"));
    }

    #[test]
    fn sidechain_steps_get_agent_id() {
        let data = serde_json::json!([
            {"type":"user","message":{"content":"go"}},
            {"type":"assistant","isSidechain":true,"message":{"id":"s1","model":"m","content":[{"type":"text","text":"subagent"}],"usage":{"input_tokens":1,"output_tokens":2}}}
        ])
        .to_string();
        let trace = parse(&data).unwrap();
        assert_eq!(trace.steps[0].agent_id.as_deref(), Some("subagent"));
    }
}
