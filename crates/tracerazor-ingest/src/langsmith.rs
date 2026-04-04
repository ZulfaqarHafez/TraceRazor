/// Parser for LangSmith run export format.
///
/// LangSmith exports traces as a tree of "runs" where each run has:
///   - run_type: "chain" | "llm" | "tool" | "retriever"
///   - inputs / outputs: the data flowing through
///   - child_runs: nested sub-runs
///
/// We flatten the tree into a sequential list of TraceSteps.
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracerazor_core::types::{StepType, Trace, TraceStep};

/// LangSmith run object (simplified).
#[derive(Debug, Deserialize)]
struct LangSmithRun {
    #[serde(default)]
    id: String,
    name: String,
    run_type: String,
    #[serde(default)]
    inputs: serde_json::Value,
    #[serde(default)]
    outputs: serde_json::Value,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    extra: Option<serde_json::Value>,
    #[serde(default)]
    child_runs: Vec<LangSmithRun>,
    #[serde(default)]
    parent_run_id: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

/// Parse a LangSmith run export (single root run or array of runs).
pub fn parse(data: &str) -> Result<Trace> {
    // LangSmith can export as a single run object or an array of runs.
    let v: serde_json::Value =
        serde_json::from_str(data).context("Invalid JSON in LangSmith trace")?;

    let root: LangSmithRun = if v.is_array() {
        // Array of runs — wrap in a synthetic root.
        let runs: Vec<LangSmithRun> = serde_json::from_value(v)
            .context("Failed to parse LangSmith run array")?;
        if runs.is_empty() {
            anyhow::bail!("LangSmith trace contains no runs");
        }
        // Use the first run as root; remaining become siblings (treat as children).
        let mut root = runs.into_iter().next().unwrap();
        root
    } else {
        serde_json::from_value(v).context("Failed to parse LangSmith run")?
    };

    let mut steps: Vec<TraceStep> = Vec::new();
    let mut counter = 1u32;
    flatten_run(&root, &mut steps, &mut counter, None);

    // Derive framework from tags or extra.
    let framework = root
        .extra
        .as_ref()
        .and_then(|e| e.get("metadata"))
        .and_then(|m| m.get("framework"))
        .and_then(|f| f.as_str())
        .unwrap_or("langgraph")
        .to_string();

    let total_tokens: u32 = steps.iter().map(|s| s.tokens).sum();

    Ok(Trace {
        trace_id: root.id.clone(),
        agent_name: root.name.clone(),
        framework,
        steps,
        total_tokens,
        task_value_score: 1.0,
        metadata: HashMap::new(),
    })
}

/// Recursively flatten a LangSmith run tree into sequential TraceSteps.
fn flatten_run(
    run: &LangSmithRun,
    steps: &mut Vec<TraceStep>,
    counter: &mut u32,
    agent_id: Option<&str>,
) {
    let step_type = match run.run_type.as_str() {
        "llm" => StepType::Reasoning,
        "tool" | "retriever" => StepType::ToolCall,
        "chain" => {
            // Chain runs are orchestration wrappers — skip the wrapper itself
            // and only include children.
            for child in &run.child_runs {
                flatten_run(child, steps, counter, Some(&run.name));
            }
            return;
        }
        _ => StepType::Reasoning,
    };

    // Extract token count from usage metadata if available.
    let tokens = extract_tokens(&run.extra);

    // Build a content string from inputs/outputs.
    let content = build_content(&run.inputs, &run.outputs, &run.run_type);

    // Build tool params from inputs.
    let tool_params = if step_type == StepType::ToolCall {
        Some(run.inputs.clone())
    } else {
        None
    };

    // Determine success: no error = success.
    let tool_success = if step_type == StepType::ToolCall {
        Some(run.error.is_none())
    } else {
        None
    };

    steps.push(TraceStep {
        id: *counter,
        step_type,
        content,
        tokens,
        tool_name: if run.run_type == "tool" || run.run_type == "retriever" {
            Some(run.name.clone())
        } else {
            None
        },
        tool_params,
        tool_success,
        tool_error: run.error.clone(),
        agent_id: agent_id.map(|s| s.to_string()),
        input_context: run
            .inputs
            .get("messages")
            .and_then(|m| serde_json::to_string(m).ok()),
        output: run
            .outputs
            .get("output")
            .and_then(|o| o.as_str())
            .map(|s| s.to_string()),
        flags: vec![],
        flag_details: vec![],
    });

    *counter += 1;

    for child in &run.child_runs {
        flatten_run(child, steps, counter, agent_id);
    }
}

fn extract_tokens(extra: &Option<serde_json::Value>) -> u32 {
    extra
        .as_ref()
        .and_then(|e| e.get("usage_metadata").or_else(|| e.get("token_usage")))
        .and_then(|u| {
            u.get("total_tokens")
                .or_else(|| u.get("totalTokens"))
                .and_then(|t| t.as_u64())
        })
        .map(|t| t as u32)
        .unwrap_or(0)
}

fn build_content(
    inputs: &serde_json::Value,
    outputs: &serde_json::Value,
    run_type: &str,
) -> String {
    match run_type {
        "llm" => {
            // Extract the last message content from inputs.
            let input_text = inputs
                .get("messages")
                .and_then(|m| m.as_array())
                .and_then(|arr| arr.last())
                .and_then(|msg| msg.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or_default();

            let output_text = outputs
                .get("generations")
                .and_then(|g| g.as_array())
                .and_then(|arr| arr.first())
                .and_then(|gen| gen.as_array())
                .and_then(|arr| arr.first())
                .and_then(|g| g.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or_default();

            format!("{} {}", input_text, output_text)
                .trim()
                .to_string()
        }
        "tool" | "retriever" => {
            let output_text = outputs.get("output").and_then(|o| o.as_str()).unwrap_or_default();
            output_text.to_string()
        }
        _ => serde_json::to_string(inputs).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_langsmith() {
        let json = r#"
        {
          "id": "run-abc123",
          "name": "RunnableSequence",
          "run_type": "chain",
          "inputs": {},
          "outputs": {},
          "child_runs": [
            {
              "id": "run-llm1",
              "name": "ChatAnthropic",
              "run_type": "llm",
              "inputs": {"messages": [{"content": "Parse the user request"}]},
              "outputs": {"generations": [[{"text": "The user wants a refund"}]]},
              "extra": {"usage_metadata": {"total_tokens": 450}},
              "child_runs": []
            },
            {
              "id": "run-tool1",
              "name": "get_order_details",
              "run_type": "tool",
              "inputs": {"order_id": "ORD-9182"},
              "outputs": {"output": "Order found: blue jacket"},
              "extra": {"usage_metadata": {"total_tokens": 120}},
              "child_runs": []
            }
          ]
        }
        "#;
        let trace = parse(json).unwrap();
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.steps[0].step_type, StepType::Reasoning);
        assert_eq!(trace.steps[1].step_type, StepType::ToolCall);
        assert_eq!(trace.steps[1].tool_name.as_deref(), Some("get_order_details"));
    }
}
