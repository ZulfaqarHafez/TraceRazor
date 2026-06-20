/// Parser for LangSmith run export format.
///
/// LangSmith exports traces as a tree of "runs" where each run has:
///   - run_type: "chain" | "llm" | "tool" | "retriever"
///   - inputs / outputs: the data flowing through
///   - child_runs: nested sub-runs
///
/// We flatten the tree into a sequential list of TraceSteps.
use anyhow::{Context, Result};
use serde::Deserialize;
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
    trace_id: Option<String>,
    #[serde(default)]
    start_time: Option<String>,
    /// Run-level token counts (present on `client.list_runs()` exports).
    #[serde(default)]
    total_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    tags: Vec<String>,
}

/// Parse a LangSmith run export (single root run or array of runs).
pub fn parse(data: &str) -> Result<Trace> {
    // LangSmith can export as a single run object or an array of runs.
    let v: serde_json::Value =
        serde_json::from_str(data).context("Invalid JSON in LangSmith trace")?;

    let root: LangSmithRun = if v.is_array() {
        // Flat `client.list_runs()` export: EVERY run must survive. Rebuild
        // the tree from `parent_run_id` (the old code silently kept only the
        // first run and discarded the rest).
        let runs: Vec<LangSmithRun> = serde_json::from_value(v)
            .context("Failed to parse LangSmith run array")?;
        if runs.is_empty() {
            anyhow::bail!("LangSmith trace contains no runs");
        }
        rebuild_tree(runs)?
    } else {
        serde_json::from_value(v).context("Failed to parse LangSmith run")?
    };

    let mut steps: Vec<TraceStep> = Vec::new();
    let mut counter = 1u32;
    flatten_run(&root, &mut steps, &mut counter, None, 0)?;

    // Derive framework from tags or extra.
    let framework = root
        .extra
        .as_ref()
        .and_then(|e| e.get("metadata"))
        .and_then(|m| m.get("framework"))
        .and_then(|f| f.as_str())
        .unwrap_or("langgraph")
        .to_string();

    let total_tokens: u32 = steps
        .iter()
        .map(|s| s.tokens)
        .fold(0u32, u32::saturating_add);

    Ok(Trace {
        trace_id: root.trace_id.clone().unwrap_or_else(|| root.id.clone()),
        agent_name: root.name.clone(),
        framework,
        steps,
        total_tokens,
        task_value_score: 1.0,
        metadata: HashMap::new(),
    })
}

/// Maximum run-tree depth accepted from an export. Real agent traces nest at
/// most a few dozen levels; an unbounded `parent_run_id` chain in a malformed
/// or adversarial export would otherwise overflow the stack in
/// `attach`/`flatten_run`.
const MAX_RUN_DEPTH: usize = 128;

/// Rebuild a run tree from a flat `list_runs()` array via `parent_run_id`.
/// Runs are ordered by `start_time` (exports are often reverse-chronological);
/// roots are runs whose parent is absent from the export. Multiple roots are
/// wrapped in a synthetic chain so all of them are flattened.
fn rebuild_tree(mut runs: Vec<LangSmithRun>) -> Result<LangSmithRun> {
    runs.sort_by(|a, b| a.start_time.cmp(&b.start_time));
    let ids: std::collections::HashSet<String> = runs.iter().map(|r| r.id.clone()).collect();

    // Pop children into parent buckets, keep roots.
    let mut children: HashMap<String, Vec<LangSmithRun>> = HashMap::new();
    let mut roots: Vec<LangSmithRun> = Vec::new();
    for run in runs {
        match run.parent_run_id.clone().filter(|p| ids.contains(p)) {
            Some(parent) => children.entry(parent).or_default().push(run),
            None => roots.push(run),
        }
    }
    fn attach(
        run: &mut LangSmithRun,
        children: &mut HashMap<String, Vec<LangSmithRun>>,
        depth: usize,
    ) -> Result<()> {
        if depth >= MAX_RUN_DEPTH {
            anyhow::bail!(
                "LangSmith run tree exceeds the maximum depth of {MAX_RUN_DEPTH} \
                 (malformed parent_run_id chain?)"
            );
        }
        if let Some(mut kids) = children.remove(&run.id) {
            for k in &mut kids {
                attach(k, children, depth + 1)?;
            }
            run.child_runs.append(&mut kids);
        }
        Ok(())
    }
    for r in &mut roots {
        attach(r, &mut children, 0)?;
    }
    if roots.is_empty() {
        anyhow::bail!("LangSmith export has no root runs (cyclic parent_run_id?)");
    }
    if roots.len() == 1 {
        return Ok(roots.into_iter().next().expect("len checked"));
    }
    // Multiple traces in one export: wrap them under a synthetic chain.
    //
    // If all roots share the same trace_id, use that shared id (the common
    // case when a list_runs export captures parallel sub-agents of one run).
    // If the roots belong to *different* traces, using the first root's
    // trace_id would falsely attribute the whole export to that one trace;
    // instead generate a synthetic id so the audit report is not misleading.
    let first_tid = roots[0].trace_id.clone().unwrap_or_else(|| roots[0].id.clone());
    let all_same_trace = roots
        .iter()
        .all(|r| r.trace_id.as_deref().unwrap_or(&r.id) == first_tid);
    let (trace_id, name) = if all_same_trace {
        (first_tid, roots[0].name.clone())
    } else {
        (format!("multi-{}-runs", roots.len()), "multi-trace-export".to_string())
    };
    Ok(LangSmithRun {
        id: trace_id.clone(),
        name,
        run_type: "chain".into(),
        inputs: serde_json::Value::Null,
        outputs: serde_json::Value::Null,
        error: None,
        extra: None,
        child_runs: roots,
        parent_run_id: None,
        trace_id: Some(trace_id),
        start_time: None,
        total_tokens: None,
        prompt_tokens: None,
        completion_tokens: None,
        tags: vec![],
    })
}

/// Recursively flatten a LangSmith run tree into sequential TraceSteps.
fn flatten_run(
    run: &LangSmithRun,
    steps: &mut Vec<TraceStep>,
    counter: &mut u32,
    agent_id: Option<&str>,
    depth: usize,
) -> Result<()> {
    if depth >= MAX_RUN_DEPTH {
        anyhow::bail!(
            "LangSmith run tree exceeds the maximum depth of {MAX_RUN_DEPTH} \
             (malformed parent_run_id chain?)"
        );
    }
    let step_type = match run.run_type.as_str() {
        "llm" => StepType::Reasoning,
        "tool" | "retriever" => StepType::ToolCall,
        "chain" => {
            // Chain runs are orchestration wrappers — skip the wrapper itself
            // and only include children.
            for child in &run.child_runs {
                flatten_run(child, steps, counter, Some(&run.name), depth + 1)?;
            }
            return Ok(());
        }
        _ => StepType::Reasoning,
    };

    // Extract token count from all the places real exports put it.
    let tokens = extract_tokens(run);

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
        flatten_run(child, steps, counter, agent_id, depth + 1)?;
    }
    Ok(())
}

/// Token count, checked in the order real exports actually use:
/// run-level `total_tokens` (list_runs), `prompt+completion` pair,
/// `outputs.llm_output.token_usage` (LangChain run trees),
/// `outputs.usage_metadata`, then `extra.usage_metadata`/`extra.token_usage`.
fn extract_tokens(run: &LangSmithRun) -> u32 {
    fn total_of(u: &serde_json::Value) -> Option<u64> {
        u.get("total_tokens")
            .or_else(|| u.get("totalTokens"))
            .and_then(|t| t.as_u64())
            .or_else(|| {
                let p = u.get("prompt_tokens").and_then(|t| t.as_u64());
                let c = u.get("completion_tokens").and_then(|t| t.as_u64());
                match (p, c) {
                    (None, None) => None,
                    (p, c) => Some(p.unwrap_or(0) + c.unwrap_or(0)),
                }
            })
    }

    let from_run = run.total_tokens.or_else(|| {
        match (run.prompt_tokens, run.completion_tokens) {
            (None, None) => None,
            (p, c) => Some(p.unwrap_or(0) + c.unwrap_or(0)),
        }
    });

    let found = from_run
        .or_else(|| {
            run.outputs
                .get("llm_output")
                .and_then(|l| l.get("token_usage"))
                .and_then(total_of)
        })
        .or_else(|| run.outputs.get("usage_metadata").and_then(total_of))
        .or_else(|| {
            run.extra
                .as_ref()
                .and_then(|e| e.get("usage_metadata").or_else(|| e.get("token_usage")))
                .and_then(total_of)
        });

    // Saturate rather than silently truncating the upper 32 bits.
    found.map(|t| u32::try_from(t).unwrap_or(u32::MAX)).unwrap_or(0)
}

fn build_content(
    inputs: &serde_json::Value,
    outputs: &serde_json::Value,
    run_type: &str,
) -> String {
    match run_type {
        "llm" => {
            // Concatenate ALL messages so multi-turn histories are not
            // silently truncated to a single exchange (bug: arr.last() only).
            // LangChain llm runs include the full conversation in messages[].
            let input_text: String = inputs
                .get("messages")
                .and_then(|m| m.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|msg| {
                            // messages can be plain objects {"content":…}
                            // or wrapped arrays [[{"content":…}]]
                            msg.get("content")
                                .and_then(|c| c.as_str())
                                .map(|s| s.to_string())
                                .or_else(|| {
                                    msg.as_array().and_then(|inner| {
                                        inner.first()
                                            .and_then(|m| m.get("content"))
                                            .and_then(|c| c.as_str())
                                            .map(|s| s.to_string())
                                    })
                                })
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();
            let input_text = input_text.as_str();

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

    #[test]
    fn deep_parent_chain_errors_instead_of_overflowing() {
        // 600 runs chained via parent_run_id — far past MAX_RUN_DEPTH. The
        // tree rebuild must fail with a clean parse error, not blow the stack.
        let mut runs = String::from("[");
        for i in 0..600 {
            if i > 0 {
                runs.push(',');
            }
            if i == 0 {
                runs.push_str(&format!(
                    r#"{{"id":"run-{i}","name":"step","run_type":"llm","inputs":{{}},"outputs":{{}}}}"#
                ));
            } else {
                runs.push_str(&format!(
                    r#"{{"id":"run-{i}","name":"step","run_type":"llm","inputs":{{}},"outputs":{{}},"parent_run_id":"run-{}"}}"#,
                    i - 1
                ));
            }
        }
        runs.push(']');

        let err = parse(&runs).unwrap_err();
        assert!(
            format!("{err:#}").contains("maximum depth"),
            "expected a depth-limit error, got: {err:#}"
        );
    }
}
