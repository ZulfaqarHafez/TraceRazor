/// Scoring engine: composites all eight metrics into the TraceRazor Score (TAS)
/// and computes the Value-Adjusted Efficiency (VAE) multiplier.
///
/// All eight metrics are now always computed (no API key required). RDA and DBO
/// use local heuristics / historical data; ISR uses the BoW similarity backend.
use serde::{Deserialize, Serialize};

use crate::metrics::{
    dbo::{DboResult, HistoricalSequence},
    CceResult, CcrResult, IsrResult, LdiResult, RdaResult, ShlResult, SrrResult, TcaResult,
    TurResult, VdiResult,
};

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

/// Weight configuration for the eleven-metric composite score.
///
/// Weights are normalised by their sum in `compute()`, so they do not need to
/// equal exactly 1.0. The relative proportions determine each metric's
/// contribution to TAS. Defaults match the v2 distribution from the product
/// spec (structural metrics reduced to make room for three verbosity metrics).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weights {
    // Structural metrics (reduced from v1).
    pub srr: f64, // 17%
    pub ldi: f64, // 13%
    pub tca: f64, // 13%
    // Context / semantic metrics (unchanged from v1).
    pub rda: f64, // 10%
    pub isr: f64, // 10%
    pub tur: f64, // 10%
    pub cce: f64, // 10%
    pub dbo: f64, //  9%
    // Verbosity metrics (new in v2).
    pub vdi: f64, //  9%
    pub shl: f64, //  5%
    pub ccr: f64, //  4%
}

impl Default for Weights {
    fn default() -> Self {
        Weights {
            srr: 0.17,
            ldi: 0.13,
            tca: 0.13,
            rda: 0.10,
            isr: 0.10,
            tur: 0.10,
            cce: 0.10,
            dbo: 0.09,
            vdi: 0.09,
            shl: 0.05,
            ccr: 0.04,
        }
    }
}

/// The composite TraceRazor Score and all component results.
/// All eleven metrics are always present — no Option wrappers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TasScore {
    /// Composite score (0–100). Higher is better.
    pub score: f64,
    pub grade: Grade,
    /// Value-Adjusted Efficiency score.
    pub vae: f64,
    /// Whether TAS meets the configured threshold.
    pub passes_threshold: bool,
    /// Aggregate Verbosity Score (0.0–1.0). Higher = more verbose waste.
    /// AVS > 0.40 triggers a VERBOSITY ALERT in the report.
    pub avs: f64,

    // Structural metrics.
    pub srr: SrrResult,
    pub ldi: LdiResult,
    pub tca: TcaResult,
    pub tur: TurResult,
    pub cce: CceResult,
    pub rda: RdaResult,
    pub isr: IsrResult,
    pub dbo: DboResult,

    // Verbosity metrics (v2).
    pub vdi: VdiResult,
    pub shl: ShlResult,
    pub ccr: CcrResult,
}

/// Configuration for the scoring engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    pub weights: Weights,
    /// Minimum TAS to "pass" (used for CI/CD gating).
    pub threshold: f64,
    /// Per-token cost in USD, expressed per million tokens.
    pub cost_per_million_tokens: f64,
    /// Historical median token count for this task type (for VAE normalisation).
    pub baseline_tokens: Option<u32>,
    /// Historical tool sequences for DBO computation.
    /// Empty slice = cold start (neutral 0.7). Populated from the store.
    #[serde(default)]
    pub historical_sequences: Vec<HistoricalSequence>,
    /// Historical median step count for RDA accuracy improvement.
    /// None = use heuristic classifier. Populated from the store.
    pub historical_median_steps: Option<f64>,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        ScoringConfig {
            weights: Weights::default(),
            threshold: 70.0,
            cost_per_million_tokens: 3.0,
            baseline_tokens: None,
            historical_sequences: vec![],
            historical_median_steps: None,
        }
    }
}

/// Compute the composite TAS score from all eleven metrics.
#[allow(clippy::too_many_arguments)]
pub fn compute(
    srr: SrrResult,
    ldi: LdiResult,
    tca: TcaResult,
    tur: TurResult,
    cce: CceResult,
    rda: RdaResult,
    isr: IsrResult,
    dbo: DboResult,
    vdi: VdiResult,
    shl: ShlResult,
    ccr: CcrResult,
    task_value_score: f64,
    total_tokens: u32,
    config: &ScoringConfig,
) -> TasScore {
    let w = &config.weights;

    // Normalise all component scores to 0.0–1.0 (higher = better).
    let srr_n = srr.normalised();
    let ldi_n = ldi.normalised();
    let tca_n = tca.normalised();
    let tur_n = tur.normalised();
    let cce_n = cce.normalised();
    let rda_n = rda.normalised();
    let isr_n = isr.normalised();
    let dbo_n = dbo.normalised();
    let vdi_n = vdi.normalised();
    let shl_n = shl.normalised();
    let ccr_n = ccr.normalised();

    // Sum weights so the composite remains in [0, 1] even if weights don't
    // add up to exactly 1.0 (the spec sums to 1.10 due to rounding).
    let weight_total = w.srr
        + w.ldi
        + w.tca
        + w.tur
        + w.cce
        + w.rda
        + w.isr
        + w.dbo
        + w.vdi
        + w.shl
        + w.ccr;

    let weighted_sum = srr_n * w.srr
        + ldi_n * w.ldi
        + tca_n * w.tca
        + tur_n * w.tur
        + cce_n * w.cce
        + rda_n * w.rda
        + isr_n * w.isr
        + dbo_n * w.dbo
        + vdi_n * w.vdi
        + shl_n * w.shl
        + ccr_n * w.ccr;

    let raw_efficiency = weighted_sum / weight_total;

    // TAS in 0–100.
    let tas = (raw_efficiency * 100.0 * 10.0).round() / 10.0;

    // VAE = (task_value_score * raw_efficiency) / normalised_token_cost.
    let baseline = config.baseline_tokens.unwrap_or(total_tokens).max(1) as f64;
    let normalised_cost = (total_tokens as f64 / baseline).max(0.001);
    let vae = ((task_value_score * raw_efficiency) / normalised_cost * 100.0).round() / 100.0;
    let vae = vae.min(1.0);

    // Aggregate Verbosity Score: weighted combination of three verbosity waste signals.
    // vdi_waste = 1 - vdi density (higher = more filler)
    // shl.score = fraction of sycophantic/hedged sentences (higher = more hedging)
    // ccr.score = compression ratio waste (higher = more compressible)
    let avs = ((1.0 - vdi_n) * 0.45 + shl.score * 0.30 + ccr.score * 0.25)
        .clamp(0.0, 1.0);
    let avs = (avs * 1000.0).round() / 1000.0;

    let grade = Grade::from_score(tas);
    let passes = tas >= config.threshold;

    TasScore {
        score: tas,
        grade,
        vae,
        passes_threshold: passes,
        avs,
        srr,
        ldi,
        tca,
        tur,
        cce,
        rda,
        isr,
        dbo,
        vdi,
        shl,
        ccr,
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