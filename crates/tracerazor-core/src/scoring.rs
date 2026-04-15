/// Scoring engine: composites all eight metrics into the TraceRazor Score (TAS)
/// and computes the Value-Adjusted Efficiency (VAE) multiplier.
///
/// All eight metrics are now always computed (no API key required). RDA and DBO
/// use local heuristics / historical data; ISR uses the BoW similarity backend.
use serde::{Deserialize, Serialize};

use crate::metrics::{
    dbo::{DboResult, HistoricalSequence},
    CceResult, CcrResult, GarResult, IsrResult, LdiResult, RdaResult, ShlResult, SrrResult,
    TcaResult, TurResult, VdiResult,
};

/// Serde default helper returning 1.0 (used for backward-compat deserialization
/// of reports that pre-date the task_value_score field).
fn default_one() -> f64 {
    1.0
}

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
    pub vdi: f64, //  8%
    pub shl: f64, //  5%
    pub ccr: f64, //  3%
    // Goal advancement (M1).
    pub gar: f64, //  6%
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
            vdi: 0.08,
            shl: 0.05,
            ccr: 0.03,
            gar: 0.07,
        }
    }
}

/// The composite TraceRazor Score and all component results.
/// All eleven metrics are always present — no Option wrappers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TasScore {
    /// Composite score (0–100), **after** the Task Value Integration multiplier.
    /// `score = raw_tas × (0.7 + 0.3 × task_value_score)`.
    /// Higher is better.
    pub score: f64,
    /// Raw structural-efficiency score before the TVI adjustment (0–100).
    /// Equal to `score` when `task_value_score == 1.0`.
    #[serde(default)]
    pub raw_tas: f64,
    /// The task-completion quality supplied by the caller (0.0–1.0).
    /// A trace that is structurally clean but failed the task is penalised:
    /// at 0.0 the multiplier is 0.7; at 1.0 it is 1.0 (no change).
    #[serde(default = "default_one")]
    pub task_value_score: f64,
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
    // Goal advancement metric (M1).
    pub gar: GarResult,
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
    gar: GarResult,
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
    let gar_n = gar.normalised();

    // Sum weights so the composite remains in [0, 1] even if weights don't
    // add up to exactly 1.0.  GAR uses 7% of the composite; CCR reduced from
    // 4% to 3% to partially offset (both overlap with verbosity waste signal).
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
        + w.ccr
        + w.gar;

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
        + ccr_n * w.ccr
        + gar_n * w.gar;

    let raw_efficiency = weighted_sum / weight_total;

    // Raw TAS (0–100) — structural efficiency only, before task-value adjustment.
    let raw_tas = (raw_efficiency * 100.0 * 10.0).round() / 10.0;

    // ── M2: Task Value Integration (TVI) ─────────────────────────────────────
    // TAS_final = TAS_raw × (0.7 + 0.3 × task_value_score)
    //   task_value_score = 1.0 → multiplier = 1.0  (no change for complete tasks)
    //   task_value_score = 0.5 → multiplier = 0.85 (15% penalty for partial failure)
    //   task_value_score = 0.0 → multiplier = 0.7  (30% penalty for total failure)
    // This ensures a structurally clean but failed trace cannot exceed ~70 TAS,
    // and task quality directly gates the ceiling of the composite score.
    let tvs = task_value_score.clamp(0.0, 1.0);
    let tvi_multiplier = 0.7 + 0.3 * tvs;
    let tas = (raw_tas * tvi_multiplier * 10.0).round() / 10.0;

    // VAE = (task_value_score * raw_efficiency) / normalised_token_cost.
    let baseline = config.baseline_tokens.unwrap_or(total_tokens).max(1) as f64;
    let normalised_cost = (total_tokens as f64 / baseline).max(0.001);
    let vae = ((tvs * raw_efficiency) / normalised_cost * 100.0).round() / 100.0;
    let vae = vae.min(1.0);

    // Aggregate Verbosity Score: weighted combination of three verbosity waste signals.
    // vdi_waste = 1 - vdi density (higher = more filler)
    // shl.score = fraction of sycophantic/hedged sentences (higher = more hedging)
    // ccr.score = compression ratio waste (higher = more compressible)
    let avs = ((1.0 - vdi_n) * 0.45 + shl.score * 0.30 + ccr.score * 0.25)
        .clamp(0.0, 1.0);
    let avs = (avs * 1000.0).round() / 1000.0;

    // Grade is based on the TVI-adjusted score so quality gating is reflected.
    let grade = Grade::from_score(tas);
    let passes = tas >= config.threshold;

    TasScore {
        score: tas,
        raw_tas,
        task_value_score: tvs,
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
        gar,
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

    // ── M2: Task Value Integration ────────────────────────────────────────────

    #[test]
    fn m2_perfect_task_value_leaves_score_unchanged() {
        // task_value_score = 1.0 → multiplier = 1.0; TAS must not change.
        let raw = 80.0_f64;
        let tvs = 1.0_f64;
        let tvi = 0.7 + 0.3 * tvs;
        let adjusted = (raw * tvi * 10.0).round() / 10.0;
        assert_eq!(adjusted, raw, "perfect task should not change TAS");
    }

    #[test]
    fn m2_zero_task_value_applies_30pct_ceiling() {
        // task_value_score = 0.0 → multiplier = 0.7; max reachable TAS ≈ 70.
        let raw = 100.0_f64;
        let tvs = 0.0_f64;
        let tvi = 0.7 + 0.3 * tvs;
        let adjusted = (raw * tvi * 10.0).round() / 10.0;
        assert!(
            adjusted <= 70.0,
            "failed task should cap TAS at ~70, got {adjusted}"
        );
    }

    #[test]
    fn m2_half_task_value_applies_15pct_penalty() {
        let raw = 80.0_f64;
        let tvs = 0.5_f64;
        let tvi = 0.7 + 0.3 * tvs;
        let adjusted = (raw * tvi * 10.0).round() / 10.0;
        // 80 × 0.85 = 68.0
        let expected = (80.0_f64 * 0.85 * 10.0).round() / 10.0;
        assert!(
            (adjusted - expected).abs() < 0.2,
            "half-value task: expected ≈ {expected}, got {adjusted}"
        );
    }

    #[test]
    fn m2_task_value_score_clamped_above_one() {
        // Values > 1.0 are clamped; should behave same as 1.0.
        let tvs_clamped = 1.5_f64.clamp(0.0, 1.0);
        assert_eq!(tvs_clamped, 1.0);
        let tvi = 0.7 + 0.3 * tvs_clamped;
        assert_eq!(tvi, 1.0);
    }

    #[test]
    fn m2_raw_tas_always_geq_adjusted_tas() {
        // For any valid task_value_score ≤ 1.0, raw_tas ≥ score.
        for tvs_tenth in 0..=10 {
            let tvs = tvs_tenth as f64 / 10.0;
            let tvi = 0.7 + 0.3 * tvs;
            let raw = 75.0_f64;
            let adjusted = (raw * tvi * 10.0).round() / 10.0;
            assert!(
                raw >= adjusted - 0.1,
                "raw_tas {raw} should be >= adjusted {adjusted} at tvs={tvs}"
            );
        }
    }

    #[test]
    fn m2_grade_reflects_adjusted_score() {
        // A structurally excellent (raw 92) but partially-failed (tvs 0.5) trace
        // should grade as Fair/Good after TVI, not Excellent.
        let raw = 92.0_f64;
        let tvs = 0.5_f64;
        let tvi = 0.7 + 0.3 * tvs;
        let adjusted = (raw * tvi * 10.0).round() / 10.0;
        // 92 × 0.85 = 78.2 → Good (not Excellent)
        let grade = Grade::from_score(adjusted);
        assert_ne!(
            grade,
            Grade::Excellent,
            "partial-failure trace (tvs=0.5) should not grade as Excellent"
        );
    }
}