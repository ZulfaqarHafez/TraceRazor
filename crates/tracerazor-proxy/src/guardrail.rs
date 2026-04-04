/// Central guardrail orchestrator — applies all three layers in order.
use crate::{ProxyRequest, ProxyResponse};
use crate::budget::BudgetConfig;
use crate::scope::ScopeConfig;
use tracerazor_semantic::default_similarity_fn;

/// Configuration for the proxy guardrail system.
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Layer 1: minimum cosine similarity to original task.
    pub semantic_threshold: f64,
    /// Layer 2: allowed tool names (empty = allow all).
    pub scope: ScopeConfig,
    /// Layer 3: token budget settings.
    pub budget: BudgetConfig,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        ProxyConfig {
            semantic_threshold: 0.55,
            scope: ScopeConfig::default(),
            budget: BudgetConfig::default(),
        }
    }
}

/// The outcome of a single guardrail check.
#[derive(Debug, Clone)]
pub enum GuardrailDecision {
    Pass,
    Block { reason: String, layer: u8 },
    Warn { reason: String },
}

/// Aggregated result from all guardrail layers.
#[derive(Debug, Clone)]
pub struct GuardrailResult {
    pub decision: GuardrailDecision,
    /// Similarity score from Layer 1.
    pub semantic_similarity: f64,
    /// Whether budget injection was applied.
    pub budget_injected: bool,
}

impl ProxyConfig {
    /// Run all three guardrail layers and return the final ProxyResponse.
    pub fn intercept(&self, req: &ProxyRequest) -> ProxyResponse {
        let result = self.evaluate(req);

        match result.decision {
            GuardrailDecision::Block { reason, layer } => {
                ProxyResponse::Blocked { reason, layer }
            }
            _ => {
                // Apply budget injection if flagged.
                let system_prompt = if result.budget_injected {
                    crate::budget::inject_budget_header(
                        &req.system_prompt,
                        self.budget.max_tokens,
                        req.tokens_used,
                    )
                } else {
                    req.system_prompt.clone()
                };

                ProxyResponse::Approved {
                    system_prompt,
                    user_message: req.user_message.clone(),
                }
            }
        }
    }

    /// Evaluate all layers and return a GuardrailResult (without mutating the request).
    pub fn evaluate(&self, req: &ProxyRequest) -> GuardrailResult {
        // ── Layer 1: Semantic Preservation ────────────────────────────────
        let sim_fn = default_similarity_fn();
        let combined_prompt = format!("{} {}", req.system_prompt, req.user_message);
        let similarity = sim_fn(&req.task_description, &combined_prompt);

        if similarity < self.semantic_threshold {
            return GuardrailResult {
                decision: GuardrailDecision::Block {
                    reason: format!(
                        "Semantic drift detected: prompt similarity {:.2} < threshold {:.2}",
                        similarity, self.semantic_threshold
                    ),
                    layer: 1,
                },
                semantic_similarity: similarity,
                budget_injected: false,
            };
        }

        // ── Layer 2: Scope Whitelist ──────────────────────────────────────
        if let Some(blocked_tool) = self.scope.check_tools(&req.requested_tools) {
            return GuardrailResult {
                decision: GuardrailDecision::Block {
                    reason: format!("Tool '{}' is not in the scope whitelist", blocked_tool),
                    layer: 2,
                },
                semantic_similarity: similarity,
                budget_injected: false,
            };
        }

        // ── Layer 3: Token Budget ─────────────────────────────────────────
        let budget_injected = self.budget.should_inject(req.tokens_used);

        GuardrailResult {
            decision: GuardrailDecision::Pass,
            semantic_similarity: similarity,
            budget_injected,
        }
    }
}
