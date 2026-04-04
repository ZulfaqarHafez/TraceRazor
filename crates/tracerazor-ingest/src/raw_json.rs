/// Parser for TraceRazor's native raw JSON format.
///
/// Schema:
/// ```json
/// {
///   "trace_id": "string",
///   "agent_name": "string",
///   "framework": "string",
///   "steps": [
///     {
///       "id": number,
///       "type": "reasoning" | "tool_call" | "handoff",
///       "content": "string",
///       "tokens": number,
///       "tool_name": "string?",
///       "tool_params": object?,
///       "tool_success": boolean?,
///       "tool_error": "string?",
///       "agent_id": "string?",
///       "input_context": "string?",
///       "output": "string?"
///     }
///   ],
///   "task_value_score": number?,
///   "metadata": object?
/// }
/// ```
use anyhow::{Context, Result};
use tracerazor_core::types::Trace;

/// Parse a raw JSON string into a Trace.
pub fn parse(data: &str) -> Result<Trace> {
    let trace: Trace =
        serde_json::from_str(data).context("Failed to parse raw JSON trace")?;
    validate(&trace)?;
    Ok(trace)
}

fn validate(trace: &Trace) -> Result<()> {
    use anyhow::bail;
    if trace.trace_id.is_empty() {
        bail!("trace_id is required");
    }
    if trace.steps.is_empty() {
        bail!("trace must contain at least one step");
    }
    for step in &trace.steps {
        if step.id == 0 {
            bail!("step id must be >= 1");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal() {
        let json = r#"
        {
          "trace_id": "test-001",
          "agent_name": "test-agent",
          "framework": "raw",
          "steps": [
            {"id": 1, "type": "reasoning", "content": "first step", "tokens": 100},
            {"id": 2, "type": "tool_call", "content": "call tool", "tokens": 200,
             "tool_name": "my_tool", "tool_success": true}
          ]
        }
        "#;
        let trace = parse(json).unwrap();
        assert_eq!(trace.trace_id, "test-001");
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.task_value_score, 1.0);
    }

    #[test]
    fn test_parse_with_metadata() {
        let json = r#"
        {
          "trace_id": "test-002",
          "agent_name": "my-agent",
          "framework": "langgraph",
          "task_value_score": 0.9,
          "steps": [
            {
              "id": 1, "type": "reasoning", "content": "parse request",
              "tokens": 500, "agent_id": "agent-1",
              "input_context": "user wants a refund"
            }
          ],
          "metadata": {"model": "claude-3-5-sonnet"}
        }
        "#;
        let trace = parse(json).unwrap();
        assert_eq!(trace.task_value_score, 0.9);
        assert!(trace.steps[0].input_context.is_some());
    }
}
