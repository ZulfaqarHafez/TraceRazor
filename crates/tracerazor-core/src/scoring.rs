/// Scoring engine: composites all metrics into the TraceRazor Score (TAS)
/// and computes the Value-Adjusted Efficiency (VAE) multiplier.
use serde::{Deserialize, Serialize};

use crate::metrics::{CceResult, DboResult, IsrResult, LdiResult, RdaResult, SrrResult, TcaResult, TurResult};

/// Grade bands for the composite TAS score (mirrors Google Lighthouse).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Grade {
    /// 90–100: Agent is highly optimised.
    Excellent,
    /// 70–89: Minor inefficiencies present.
    Good,
    /// 50–69: Actionable waste present.
    Fair,
    /// 0–49: Significant restructuring recommended.
    Poor,
}

impl Grade {
    pub fn from_score(score: f64) -> Self {
        if score >= 90.0 {
            Grade::Excellent
        } else if score >= 70.0 {
            Grade::Good
        } else if score >= 50.0 {
            Grade::Fair
        } else {
            Grade::Poor
        }
    }
}

impl std::fmt::Display for Grade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Grade::Excellent => write!(f, "EXCELLENT"),
            Grade::Good => write!(f, "GOOD"),
            Grade::Fair => write!(f, "FAIR"),
            Grade::Poor => write!(f, "POOR"),
        }
    }
}

/// Weight configuration for the composite score.
/// Weights must sum to 1.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weights {
    pub srr: f64,
    pub ldi: f64,
    pub tca: f64,
    pub rda: f64,
    pub isr: f64,
    pub tur: f64,
    pub cce: f64,
    pub dbo: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Weights {
            srr: 0.20,
            ldi: 0.15,
            tca: 0.15,
            rda: 0.10,
            isr: 0.10,
            tur: 0.10,
            cce: 0.10,
            dbo: 0.10,
        }
    }
}

/// Phase 1 computes 5 structural metrics. The remaining 3 (rda, isr, dbo)
/// are added in Phase 2. In Phase 1 we re-normalise over the available weights.
impl Weights {
    pub fn phase1_normalised(&self) -> f64 {
        self.srr + self.ldi + self.tca + self.tur + self.cce
    }
}

/// The composite TraceRazor Score and all component results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TasScore {
    /// Composite score (0–100). Higher is better.
    pub score: f64,
    pub grade: Grade,
    /// Value-Adjusted Efficiency score.
    pub vae: f64,
    /// Whether TAS meets the configured threshold.
    pub passes_threshold: bool,

    // Component metrics (Phase 1)
    pub srr: SrrResult,
    pub ldi: LdiResult,
    pub tca: TcaResult,
    pub tur: TurResult,
    pub cce: CceResult,
    // Phase 2 metrics (None when run without API key)
    pub rda: Option<RdaResult>,
    pub isr: Option<IsrResult>,
    pub dbo: Option<DboResult>,
}

/// Configuration for the scoring engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    pub weights: Weights,
    /// Minimum TAS to "pass" (used for CI/CD gating).
    pub threshold: f64,
    /// Per-token cost in USD (for savings estimates).
    pub cost_per_million_tokens: f64,
    /// Historical median token count for this task type (for VAE normalisation).
    pub baseline_tokens: Option<u32>,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        ScoringConfig {
            weights: Weights::default(),
            threshold: 70.0,
            cost_per_million_tokens: 3.0,
            baseline_tokens: None,
        }
    }
}

/// Compute the composite TAS score from structural (Phase 1) metrics
/// plus optional semantic (Phase 2) metrics.
pub fn compute(
    srr: SrrResult,
    ldi: LdiResult,
    tca: TcaResult,
    tur: TurResult,
    cce: CceResult,
    rda: Option<RdaResult>,
    isr: Option<IsrResult>,
    dbo: Option<DboResult>,
    task_value_score: f64,
    total_tokens: u32,
    config: &ScoringConfig,
) -> TasScore {
    let w = &config.weights;

    // Normalise component scores to 0.0–1.0 (higher = better).
    let srr_n = srr.normalised();
    let ldi_n = ldi.normalised();
    let tca_n = tca.normalised();
    let tur_n = tur.normalised();
    let cce_n = cce.normalised();

    // Accumulate weighted scores over available metrics.
    let mut weighted_sum = srr_n * w.srr
        + ldi_n * w.ldi
        + tca_n * w.tca
        + tur_n * w.tur
        + cce_n * w.cce;
    let mut weight_total = w.srr + w.ldi + w.tca + w.tur + w.cce;

    // Add Phase 2 metrics when available.
    if let Some(ref r) = rda {
        weighted_sum += r.normalised() * w.rda;
        weight_total += w.rda;
    }
    if let Some(ref i) = isr {
        weighted_sum += i.normalised() * w.isr;
        weight_total += w.isr;
    }
    if let Some(ref d) = dbo {
        weighted_sum += d.normalised() * w.dbo;
        weight_total += w.dbo;
    }

    let raw_efficiency = weighted_sum / weight_total;

    // TAS in 0–100
    let tas = (raw_efficiency * 100.0 * 10.0).round() / 10.0;

    // VAE = (task_value_score * raw_efficiency) / normalised_token_cost
    let baseline = config.baseline_tokens.unwrap_or(total_tokens).max(1) as f64;
    let normalised_cost = (total_tokens as f64 / baseline).max(0.001);
    let vae = ((task_value_score * raw_efficiency) / normalised_cost * 100.0).round() / 100.0;
    let vae = vae.min(1.0);

    let grade = Grade::from_score(tas);
    let passes = tas >= config.threshold;

    TasScore {
        score: tas,
        grade,
        vae,
        passes_threshold: passes,
        srr,
        ldi,
        tca,
        tur,
        cce,
        rda,
        isr,
        dbo,
    }
}

/// Estimate token and cost savings if all recommendations are applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavingsEstimate {
    pub tokens_saved: u32,
    pub reduction_pct: f64,
    pub cost_saved_per_run_usd: f64,
    pub monthly_savings_usd: f64,
    pub latency_saved_seconds: f64,
}

pub fn estimate_savings(
    original_tokens: u32,
    waste_tokens: u32,
    config: &ScoringConfig,
    monthly_runs: Option<u32>,
) -> SavingsEstimate {
    let savings = waste_tokens.min(original_tokens);
    let reduction_pct = if original_tokens == 0 {
        0.0
    } else {
        (savings as f64 / original_tokens as f64) * 100.0
    };
    let cost_per_token = config.cost_per_million_tokens / 1_000_000.0;
    let cost_saved = savings as f64 * cost_per_token;
    let monthly = monthly_runs.unwrap_or(50_000);
    let monthly_savings = cost_saved * monthly as f64;
    // Rough latency estimate: ~0.67ms per token throughput.
    let latency = savings as f64 * 0.00067;

    SavingsEstimate {
        tokens_saved: savings,
        reduction_pct: (reduction_pct * 10.0).round() / 10.0,
        cost_saved_per_run_usd: (cost_saved * 10000.0).round() / 10000.0,
        monthly_savings_usd: (monthly_savings * 100.0).round() / 100.0,
        latency_saved_seconds: (latency * 10.0).round() / 10.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grade_bands() {
        assert_eq!(Grade::from_score(95.0), Grade::Excellent);
        assert_eq!(Grade::from_score(75.0), Grade::Good);
        assert_eq!(Grade::from_score(55.0), Grade::Fair);
        assert_eq!(Grade::from_score(40.0), Grade::Poor);
    }
}
