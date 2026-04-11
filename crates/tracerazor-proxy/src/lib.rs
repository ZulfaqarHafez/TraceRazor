/// TraceRazor LLM Proxy — three-layer guardrail system.
///
/// The proxy intercepts LLM API calls and applies:
///
///   Layer 1 — Semantic Preservation Guard
///     Checks that the outgoing prompt is semantically aligned with the
///     original task (cosine similarity ≥ 0.55). Blocks requests that have
///     drifted too far from the original goal.
///
///   Layer 2 — Scope Whitelist
///     Validates tool calls against a statically-configured whitelist.
///     Blocks disallowed tool names before they reach the LLM.
///
///   Layer 3 — Token Budget Injection
///     Injects a `<budget>` header into system prompts if the trace is
///     projected to exceed the configured token budget. Nudges the model
///     to be more concise without hard-blocking.
///
///   Layer 4 — Verbosity Directive Injection
///     When the caller reports a rolling average CCR of the last three steps
///     ≥ 0.35, injects a conciseness directive into the system prompt.
///     Standard directive at CCR ≥ 0.35; ultra-concise directive at CCR > 0.50.
///
/// Usage:
/// ```ignore
/// let proxy = ProxyConfig::default();
/// let result = proxy.intercept(request).await?;
/// ```
pub mod budget;
pub mod guardrail;
pub mod scope;
pub mod verbosity;

pub use guardrail::{GuardrailDecision, GuardrailResult, ProxyConfig};

/// A simplified LLM request that the proxy inspects.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProxyRequest {
    /// The original high-level task description (set at trace start).
    pub task_description: String,
    /// The system prompt for this LLM call.
    pub system_prompt: String,
    /// The user message for this LLM call.
    pub user_message: String,
    /// Tool names being requested in this call (may be empty).
    pub requested_tools: Vec<String>,
    /// Cumulative token count so far in this trace.
    pub tokens_used: u32,
    /// Rolling average CCR (Caveman Compression Ratio) of the last three steps.
    /// `None` when fewer than three steps have been completed. Used by Layer 4.
    #[serde(default)]
    pub rolling_ccr: Option<f64>,
}

/// A proxy response — either approved (possibly modified) or blocked.
#[derive(Debug, Clone)]
pub enum ProxyResponse {
    /// Request approved. The system_prompt may have been modified
    /// (budget injection). The request can proceed.
    Approved {
        system_prompt: String,
        user_message: String,
    },
    /// Request blocked by one of the guardrail layers.
    Blocked {
        reason: String,
        layer: u8,
    },
}
