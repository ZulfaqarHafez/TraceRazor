use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The type of a reasoning step in an agent trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepType {
    Reasoning,
    ToolCall,
    Handoff,
    Unknown,
}

impl Default for StepType {
    fn default() -> Self {
        StepType::Unknown
    }
}

impl std::fmt::Display for StepType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepType::Reasoning => write!(f, "reasoning"),
            StepType::ToolCall => write!(f, "tool_call"),
            StepType::Handoff => write!(f, "handoff"),
            StepType::Unknown => write!(f, "unknown"),
        }
    }
}

/// Flags that can be applied to a step during analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StepFlag {
    /// Step is semantically redundant with a prior step.
    Redundant,
    /// Step is part of a detected loop.
    Loop,
    /// Step is the start of a detected loop.
    LoopStart,
    /// Tool call with wrong parameters, required retry.
    Misfire,
    /// Retry of a misfired tool call.
    Retry,
    /// Reasoning depth is excessive for the task complexity.
    OverDepth,
    /// Input context contains significant duplicated content.
    ContextBloat,
}

impl std::fmt::Display for StepFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepFlag::Redundant => write!(f, "REDUNDANT"),
            StepFlag::Loop => write!(f, "LOOP"),
            StepFlag::LoopStart => write!(f, "LOOP-START"),
            StepFlag::Misfire => write!(f, "MISFIRE"),
            StepFlag::Retry => write!(f, "RETRY"),
            StepFlag::OverDepth => write!(f, "OVER-DEPTH"),
            StepFlag::ContextBloat => write!(f, "CONTEXT-BLOAT"),
        }
    }
}

/// Confidence level for redundancy detection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    /// Cosine similarity >= 0.95
    High,
    /// Cosine similarity 0.85-0.94
    Medium,
    /// Cosine similarity 0.75-0.84 (shown in verbose mode)
    Low,
}

impl std::fmt::Display for Confidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Confidence::High => write!(f, "high"),
            Confidence::Medium => write!(f, "medium"),
            Confidence::Low => write!(f, "low"),
        }
    }
}

/// A single step in an agent trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    /// 1-based step index.
    pub id: u32,
    #[serde(rename = "type", default)]
    pub step_type: StepType,
    /// Primary text content of the step (reasoning text, tool description, etc.)
    pub content: String,
    /// Total tokens consumed by this step (input + output).
    pub tokens: u32,
    /// Tool name if this is a tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Parameters passed to the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_params: Option<serde_json::Value>,
    /// Whether the tool call succeeded on this attempt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_success: Option<bool>,
    /// Error message if the tool call failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_error: Option<String>,
    /// Agent identifier for multi-agent traces.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// The full input context fed into this step (for CCE computation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_context: Option<String>,
    /// The output produced by this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Flags applied by the analysis engine.
    #[serde(default)]
    pub flags: Vec<StepFlag>,
    /// Additional flag detail (e.g., "97% similar to step 1", "cycle: 8→9→10→8").
    #[serde(default)]
    pub flag_details: Vec<String>,
}

impl TraceStep {
    /// Returns the most informative text content for semantic analysis.
    pub fn semantic_content(&self) -> String {
        let mut parts = vec![self.content.clone()];
        if let Some(out) = &self.output {
            parts.push(out.clone());
        }
        parts.join(" ")
    }

    /// Generates a state hash for loop detection.
    /// Hash is based on tool name + serialised key params + step type.
    pub fn state_hash(&self) -> String {
        let tool = self.tool_name.as_deref().unwrap_or("none");
        let params_str = self
            .tool_params
            .as_ref()
            .map(|p| p.to_string())
            .unwrap_or_default();
        format!("{}:{}:{}", self.step_type, tool, params_str)
    }
}

/// A complete agent execution trace, parsed into the internal representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub trace_id: String,
    pub agent_name: String,
    pub framework: String,
    pub steps: Vec<TraceStep>,
    /// Total tokens across all steps. If not provided, summed from steps.
    #[serde(default)]
    pub total_tokens: u32,
    /// Optional task value score (0.0–1.0). Defaults to 1.0 if absent.
    #[serde(default = "default_task_value")]
    pub task_value_score: f64,
    /// Additional metadata (timestamps, model, etc.)
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

fn default_task_value() -> f64 {
    1.0
}

impl Trace {
    /// Compute total tokens from steps if the field is zero.
    pub fn effective_total_tokens(&self) -> u32 {
        if self.total_tokens > 0 {
            self.total_tokens
        } else {
            self.steps.iter().map(|s| s.tokens).sum()
        }
    }

    /// All unique agent IDs in this trace (empty for single-agent).
    pub fn agent_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .steps
            .iter()
            .filter_map(|s| s.agent_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        ids.sort();
        ids
    }
}

/// Minimum steps required for a trace to be analysed (per PRD Decision 8).
pub const MIN_TRACE_STEPS: usize = 5;
