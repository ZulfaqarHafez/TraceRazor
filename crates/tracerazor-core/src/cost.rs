/// Cost Projection Engine (E-05)
///
/// Projects monthly and annual costs at a given run volume, compares current
/// efficiency against the optimised projection, and identifies the worst-
/// offending agent by waste percentage.
///
/// Pure arithmetic — no model calls, no store dependency.
///
/// # Example
///
/// ```rust
/// use tracerazor_core::cost::{CostConfig, project_cost, ProviderPreset};
///
/// let config = CostConfig::from_preset(ProviderPreset::AnthropicClaude35Sonnet);
/// let result = project_cost(
///     &[(2800, 620)],   // (total_tokens, tokens_saved) per trace
///     50_000,           // runs per month
///     &config,
/// );
/// println!("Current:   ${:.2}/month", result.current_monthly_usd);
/// println!("Optimised: ${:.2}/month", result.optimised_monthly_usd);
/// println!("Savings:   ${:.2}/month", result.savings_monthly_usd);
/// ```
use serde::{Deserialize, Serialize};

/// Provider pricing presets (input-token cost; output is 3× by default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderPreset {
    OpenAiGpt4o,
    OpenAiGpt4oMini,
    AnthropicClaude35Sonnet,
    AnthropicClaude3Haiku,
    GoogleGemini15Flash,
    Custom,
}

impl ProviderPreset {
    /// Returns (cost_per_1k_input_usd, cost_per_1k_output_usd).
    pub fn pricing(self) -> (f64, f64) {
        match self {
            ProviderPreset::OpenAiGpt4o => (0.005, 0.015),
            ProviderPreset::OpenAiGpt4oMini => (0.00015, 0.0006),
            ProviderPreset::AnthropicClaude35Sonnet => (0.003, 0.015),
            ProviderPreset::AnthropicClaude3Haiku => (0.00025, 0.00125),
            ProviderPreset::GoogleGemini15Flash => (0.000075, 0.0003),
            ProviderPreset::Custom => (0.003, 0.015),
        }
    }
}

impl std::fmt::Display for ProviderPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderPreset::OpenAiGpt4o => write!(f, "openai-gpt4o"),
            ProviderPreset::OpenAiGpt4oMini => write!(f, "openai-gpt4o-mini"),
            ProviderPreset::AnthropicClaude35Sonnet => write!(f, "anthropic-claude-3-5-sonnet"),
            ProviderPreset::AnthropicClaude3Haiku => write!(f, "anthropic-claude-3-haiku"),
            ProviderPreset::GoogleGemini15Flash => write!(f, "google-gemini-1-5-flash"),
            ProviderPreset::Custom => write!(f, "custom"),
        }
    }
}

/// Pricing configuration for cost projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostConfig {
    /// Cost per 1,000 input tokens in USD.
    pub cost_per_1k_input_usd: f64,
    /// Cost per 1,000 output tokens in USD.
    pub cost_per_1k_output_usd: f64,
    /// Fraction of tokens that are output tokens (default: 0.25 — 1:3 ratio).
    pub output_fraction: f64,
}

impl Default for CostConfig {
    fn default() -> Self {
        let (input, output) = ProviderPreset::AnthropicClaude35Sonnet.pricing();
        CostConfig {
            cost_per_1k_input_usd: input,
            cost_per_1k_output_usd: output,
            output_fraction: 0.25,
        }
    }
}

impl CostConfig {
    /// Build a config from a provider preset.
    pub fn from_preset(preset: ProviderPreset) -> Self {
        let (input, output) = preset.pricing();
        CostConfig {
            cost_per_1k_input_usd: input,
            cost_per_1k_output_usd: output,
            output_fraction: 0.25,
        }
    }

    /// Build a config from explicit per-1K rates.
    pub fn custom(cost_per_1k_input_usd: f64, cost_per_1k_output_usd: f64) -> Self {
        CostConfig {
            cost_per_1k_input_usd,
            cost_per_1k_output_usd,
            output_fraction: 0.25,
        }
    }

    /// Cost of one token in USD (blended input/output rate).
    pub fn cost_per_token(&self) -> f64 {
        let input_rate = self.cost_per_1k_input_usd / 1000.0;
        let output_rate = self.cost_per_1k_output_usd / 1000.0;
        input_rate * (1.0 - self.output_fraction) + output_rate * self.output_fraction
    }
}

/// Per-agent cost breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCostBreakdown {
    pub agent_index: usize,
    pub total_tokens: u32,
    pub tokens_saved: u32,
    pub waste_pct: f64,
    pub current_cost_per_run_usd: f64,
    pub optimised_cost_per_run_usd: f64,
}

/// Full cost projection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostProjection {
    pub runs_per_month: u32,
    /// Total cost per month at current efficiency.
    pub current_monthly_usd: f64,
    /// Projected cost per month if all savings recommendations are applied.
    pub optimised_monthly_usd: f64,
    /// Monthly savings (current − optimised).
    pub savings_monthly_usd: f64,
    /// Annual savings extrapolated from monthly.
    pub savings_annual_usd: f64,
    /// Overall waste percentage across all input traces.
    pub overall_waste_pct: f64,
    /// Index of the worst-offending trace by waste percentage.
    pub worst_offender_index: Option<usize>,
    pub worst_offender_waste_pct: f64,
    /// Per-trace breakdown.
    pub per_agent: Vec<AgentCostBreakdown>,
}

/// Project monthly and annual costs for a fleet of agent traces.
///
/// `traces` is a slice of `(total_tokens, tokens_saved)` tuples — one per
/// trace or agent. Values come from `TraceReport.total_tokens` and
/// `TraceReport.savings.tokens_saved`.
///
/// All traces are assumed to run at `runs_per_month` volume each.
pub fn project_cost(
    traces: &[(u32, u32)],
    runs_per_month: u32,
    config: &CostConfig,
) -> CostProjection {
    let rate = config.cost_per_token();

    let per_agent: Vec<AgentCostBreakdown> = traces
        .iter()
        .enumerate()
        .map(|(i, &(total, saved))| {
            let optimised = total.saturating_sub(saved);
            let waste_pct = if total == 0 {
                0.0
            } else {
                saved as f64 / total as f64 * 100.0
            };
            AgentCostBreakdown {
                agent_index: i,
                total_tokens: total,
                tokens_saved: saved,
                waste_pct: (waste_pct * 10.0).round() / 10.0,
                current_cost_per_run_usd: total as f64 * rate,
                optimised_cost_per_run_usd: optimised as f64 * rate,
            }
        })
        .collect();

    let total_current: f64 = per_agent.iter().map(|a| a.current_cost_per_run_usd).sum();
    let total_optimised: f64 = per_agent.iter().map(|a| a.optimised_cost_per_run_usd).sum();

    let current_monthly = total_current * runs_per_month as f64;
    let optimised_monthly = total_optimised * runs_per_month as f64;
    let savings_monthly = current_monthly - optimised_monthly;
    let savings_annual = savings_monthly * 12.0;

    let total_tokens: u32 = traces.iter().map(|(t, _)| t).sum();
    let total_saved: u32 = traces.iter().map(|(_, s)| s).sum();
    let overall_waste_pct = if total_tokens == 0 {
        0.0
    } else {
        total_saved as f64 / total_tokens as f64 * 100.0
    };

    let worst = per_agent
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.waste_pct.partial_cmp(&b.waste_pct).unwrap());
    let worst_offender_index = worst.map(|(i, _)| i);
    let worst_offender_waste_pct = worst.map(|(_, a)| a.waste_pct).unwrap_or(0.0);

    CostProjection {
        runs_per_month,
        current_monthly_usd: (current_monthly * 100.0).round() / 100.0,
        optimised_monthly_usd: (optimised_monthly * 100.0).round() / 100.0,
        savings_monthly_usd: (savings_monthly * 100.0).round() / 100.0,
        savings_annual_usd: (savings_annual * 100.0).round() / 100.0,
        overall_waste_pct: (overall_waste_pct * 10.0).round() / 10.0,
        worst_offender_index,
        worst_offender_waste_pct,
        per_agent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_projection_basic() {
        let config = CostConfig::from_preset(ProviderPreset::AnthropicClaude35Sonnet);
        // One trace: 2800 tokens total, 620 tokens saved.
        let result = project_cost(&[(2800, 620)], 50_000, &config);
        assert!(result.savings_monthly_usd > 0.0);
        assert!(result.savings_annual_usd > result.savings_monthly_usd);
        assert_eq!(result.worst_offender_index, Some(0));
    }

    #[test]
    fn test_zero_waste() {
        let config = CostConfig::default();
        let result = project_cost(&[(1000, 0)], 10_000, &config);
        assert_eq!(result.savings_monthly_usd, 0.0);
        assert_eq!(result.overall_waste_pct, 0.0);
    }

    #[test]
    fn test_multi_agent_worst_offender() {
        let config = CostConfig::default();
        // Agent B has 60% waste, Agent A has 20% waste.
        let result = project_cost(&[(1000, 200), (1000, 600)], 1_000, &config);
        assert_eq!(result.worst_offender_index, Some(1));
        assert!((result.worst_offender_waste_pct - 60.0).abs() < 0.5);
    }

    #[test]
    fn test_preset_pricing() {
        let (inp, out) = ProviderPreset::OpenAiGpt4oMini.pricing();
        assert!(inp < ProviderPreset::OpenAiGpt4o.pricing().0);
        assert!(out < ProviderPreset::OpenAiGpt4o.pricing().1);
    }
}
