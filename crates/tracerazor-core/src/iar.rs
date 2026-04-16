/// Instruction Adherence Rate (M5)
///
/// Measures whether applying fixes from `tracerazor optimize` actually improved the
/// targeted waste patterns on re-audit. Compares a "before" audit (with fixes generated)
/// against an "after" audit (re-run with optimized prompt).
///
/// IAR = improved_count / addressed_count
///
/// where:
/// - `addressed_count` = unique FixType values in `before.fixes`
/// - `improved_count` = those whose target metric's normalised score increased in `after`
///
/// If the LLM-rewritten prompt didn't reduce SHL (for HedgeReduction) but optimize
/// claimed it would, IAR surfaces that by marking HedgeReduction as not improved.
///
/// # Example (library usage)
///
/// ```rust,no_run
/// use tracerazor_core::iar;
/// use tracerazor_core::report::TraceReport;
///
/// // Assuming `before_report` and `after_report` are loaded from JSON:
/// let before_report: TraceReport = unimplemented!();
/// let after_report: TraceReport = unimplemented!();
///
/// let result = iar::compute(&before_report, &after_report);
/// println!("IAR: {:.1}%  ({}/{})", result.score * 100.0, result.improved_count, result.addressed_count);
/// for adherence in &result.fix_adherence {
///     let status = if adherence.improved { "✓" } else { "✗" };
///     println!("  {status} {:?}: {:.3} → {:.3}", adherence.fix_type, adherence.before_score, adherence.after_score);
/// }
/// ```
use serde::{Deserialize, Serialize};

use crate::fixes::FixType;
use crate::report::TraceReport;

pub const TARGET: f64 = 0.75;  // 3 out of 4 fix types must improve to pass
const MIN_DELTA: f64 = 0.01;   // ignore rounding noise

/// Per-fix-type adherence details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixAdherence {
    /// Which fix type this tracks.
    pub fix_type: FixType,
    /// True if the targeted metric improved (delta > MIN_DELTA).
    pub improved: bool,
    /// Normalised score (0–1, higher=better) from before report.
    pub before_score: f64,
    /// Normalised score (0–1, higher=better) from after report.
    pub after_score: f64,
    /// after_score - before_score (positive = improvement).
    pub delta: f64,
}

/// Result of instruction adherence check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IarResult {
    /// Fraction of addressed fix types that improved, 0.0–1.0.
    pub score: f64,
    /// Number of unique fix types addressed in `before.fixes`.
    pub addressed_count: usize,
    /// Number of those that showed improvement in `after`.
    pub improved_count: usize,
    /// Per-fix-type adherence details.
    pub fix_adherence: Vec<FixAdherence>,
    /// True if score >= TARGET (0.75).
    pub pass: bool,
    /// Target threshold.
    pub target: f64,
}

/// Compute IAR by comparing before and after reports.
///
/// Returns a perfect score (1.0) if there were no fixes to address.
pub fn compute(before: &TraceReport, after: &TraceReport) -> IarResult {
    // Collect unique FixTypes from before.fixes (preserving order of first occurrence).
    let mut unique_fixes: Vec<&FixType> = Vec::new();
    for fix in &before.fixes {
        if !unique_fixes.iter().any(|f| f == &&fix.fix_type) {
            unique_fixes.push(&fix.fix_type);
        }
    }

    let addressed_count = unique_fixes.len();

    // If no fixes were addressed, return perfect score (trivially excellent adherence).
    if addressed_count == 0 {
        return IarResult {
            score: 1.0,
            addressed_count: 0,
            improved_count: 0,
            fix_adherence: vec![],
            pass: true,
            target: TARGET,
        };
    }

    // Compute adherence for each unique fix type.
    let mut fix_adherence = Vec::new();
    let mut improved_count = 0;

    for fix_type in unique_fixes {
        let before_score = metric_score(fix_type, before);
        let after_score = metric_score(fix_type, after);
        let delta = after_score - before_score;
        let improved = delta > MIN_DELTA;

        if improved {
            improved_count += 1;
        }

        fix_adherence.push(FixAdherence {
            fix_type: fix_type.clone(),
            improved,
            before_score: (before_score * 1000.0).round() / 1000.0,
            after_score: (after_score * 1000.0).round() / 1000.0,
            delta: (delta * 1000.0).round() / 1000.0,
        });
    }

    let score = if addressed_count > 0 {
        let raw = improved_count as f64 / addressed_count as f64;
        (raw * 1000.0).round() / 1000.0
    } else {
        1.0
    };

    IarResult {
        score,
        addressed_count,
        improved_count,
        fix_adherence,
        pass: score >= TARGET,
        target: TARGET,
    }
}

/// Look up the normalised metric score for a given fix type from a report.
///
/// All scores are normalised to 0–1, with higher = better.
fn metric_score(fix_type: &FixType, report: &TraceReport) -> f64 {
    match fix_type {
        FixType::ToolSchema => report.score.tca.normalised(),
        FixType::PromptInsert => report.score.rda.normalised(),
        FixType::TerminationGuard => report.score.ldi.normalised(),
        FixType::ContextCompression => report.score.cce.normalised(),
        FixType::VerbosityReduction => report.score.vdi.normalised(),
        FixType::HedgeReduction => report.score.shl.normalised(),
        FixType::CavemanPromptInsert => report.score.ccr.normalised(),
        FixType::ReformulationGuard => report.score.isr.normalised(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixes::Fix;

    /// Create a minimal TraceReport with the given fixes, for testing.
    fn minimal_report(fixes: Vec<Fix>) -> TraceReport {
        use crate::metrics::{
            ccr::CcrResult,
            cce::CceResult,
            csd::CsdResult,
            dbo::DboResult,
            gar::GarResult,
            isr::IsrResult,
            ldi::LdiResult,
            rda::{RdaResult, TaskComplexity},
            shl::ShlResult,
            srr::SrrResult,
            tca::TcaResult,
            tur::TurResult,
            vdi::VdiResult,
        };
        use crate::scoring::{Grade, SavingsEstimate, TasScore};

        let score = TasScore {
            score: 60.0,
            raw_tas: 60.0,
            task_value_score: 1.0,
            grade: Grade::Fair,
            vae: 0.60,
            passes_threshold: false,
            avs: 0.3,
            srr: SrrResult {
                score: 10.0,
                redundant_steps: vec![],
                redundant_count: 0,
                total_steps: 5,
                pass: true,
                target: 15.0,
            },
            ldi: LdiResult {
                score: 0.1,
                loops: vec![],
                max_cycle_length: 0,
                total_steps: 5,
                pass: false,
                warning_threshold: 0.1,
            },
            tca: TcaResult {
                score: 80.0,
                misfires: vec![],
                successful_first_attempts: 0,
                total_tool_calls: 0,
                pass: false,
                target: 85.0,
            },
            tur: TurResult {
                score: 0.5,
                useful_tokens: 0,
                total_tokens: 0,
                wasted_tokens: 0,
                pass: true,
                target: 0.35,
            },
            cce: CceResult {
                score: 0.65,
                total_input_tokens: 0,
                duplicate_tokens: 0,
                bloated_steps: vec![],
                pass: true,
                target: 0.60,
            },
            rda: RdaResult {
                score: 0.7,
                classified_complexity: TaskComplexity::Moderate,
                expected_steps: 4.0,
                actual_steps: 6,
                uses_historical_baseline: false,
                pass: false,
                target: 0.75,
            },
            isr: IsrResult {
                score: 75.0,
                steps_with_novel_info: 4,
                total_steps: 5,
                low_novelty_steps: vec![],
                pass: false,
                target: 80.0,
            },
            dbo: DboResult {
                score: 0.65,
                optimal_selections: 0,
                total_branch_points: 0,
                decisions: vec![],
                cold_start: true,
                pass: false,
                target: 0.70,
            },
            vdi: VdiResult {
                score: 0.65,
                step_results: vec![],
                low_density_steps: vec![],
                entropy_low_steps: vec![],
                pass: false,
                target: 0.60,
            },
            shl: ShlResult {
                score: 0.25,
                flagged_sentences: 1,
                total_sentences: 5,
                pass: false,
                target: 0.20,
            },
            ccr: CcrResult {
                score: 0.35,
                step_results: vec![],
                total_cuttable_tokens: 0,
                pass: false,
                target: 0.30,
            },
            gar: GarResult {
                score: 0.5,
                step_results: vec![],
                low_advancement_steps: vec![],
                goal_step_id: None,
                pass: false,
                target: 0.40,
            },
            csd: CsdResult {
                score: 0.55,
                step_results: vec![],
                high_drift_pairs: vec![],
                pass: false,
                target: 0.60,
            },
        };

        TraceReport {
            trace_id: "test".into(),
            agent_name: "test".into(),
            framework: "test".into(),
            total_steps: 5,
            total_tokens: 1000,
            analysis_duration_ms: 10,
            score,
            diff: vec![],
            savings: SavingsEstimate {
                tokens_saved: 0,
                reduction_pct: 0.0,
                cost_saved_per_run_usd: 0.0,
                monthly_savings_usd: 0.0,
                latency_saved_seconds: 0.0,
            },
            mvtg: 0.0,
            fixes,
            summary: String::new(),
            summary_oneliner: String::new(),
            anomalies: vec![],
            per_agent: vec![],
            iar: None,
        }
    }

    #[test]
    fn empty_fixes_returns_perfect_score() {
        let before = minimal_report(vec![]);
        let after = minimal_report(vec![]);
        let result = compute(&before, &after);
        assert_eq!(result.score, 1.0);
        assert!(result.pass);
        assert_eq!(result.addressed_count, 0);
        assert_eq!(result.improved_count, 0);
    }

    #[test]
    fn all_fixes_improved_returns_one() {
        let before = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        let mut after = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        // Improve TCA score in after report
        after.score.tca.score = 100.0;
        let result = compute(&before, &after);
        assert_eq!(result.score, 1.0);
        assert!(result.pass);
        assert_eq!(result.addressed_count, 1);
        assert_eq!(result.improved_count, 1);
    }

    #[test]
    fn no_fixes_improved_returns_zero() {
        let before = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        let after = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        // Scores stay the same → no improvement
        let result = compute(&before, &after);
        assert_eq!(result.score, 0.0);
        assert!(!result.pass);
        assert_eq!(result.improved_count, 0);
    }

    #[test]
    fn partial_improvement_scores_correctly() {
        let before = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::TerminationGuard,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::ContextCompression,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        let mut after = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::TerminationGuard,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::ContextCompression,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        // Improve TCA and CCE, but not LDI
        after.score.tca.score = 100.0;
        after.score.cce.score = 0.9;
        // LDI stays at 0.05
        let result = compute(&before, &after);
        assert_eq!(result.addressed_count, 3);
        assert_eq!(result.improved_count, 2);
        let expected_score = 2.0 / 3.0;
        assert!((result.score - expected_score).abs() < 0.01);
        assert!(!result.pass); // 0.667 < 0.75
    }

    #[test]
    fn deduplicates_same_fix_type() {
        let before = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "tool_a".into(),
                patch: "fix a".into(),
                estimated_token_savings: 50,
            },
            Fix {
                fix_type: FixType::ToolSchema,
                target: "tool_b".into(),
                patch: "fix b".into(),
                estimated_token_savings: 50,
            },
        ]);
        let mut after = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "tool_a".into(),
                patch: "fix a".into(),
                estimated_token_savings: 50,
            },
            Fix {
                fix_type: FixType::ToolSchema,
                target: "tool_b".into(),
                patch: "fix b".into(),
                estimated_token_savings: 50,
            },
        ]);
        after.score.tca.score = 100.0;
        let result = compute(&before, &after);
        // Two ToolSchema fixes → counted as 1 unique type
        assert_eq!(result.addressed_count, 1);
        assert_eq!(result.improved_count, 1);
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn exact_threshold_passes() {
        let before = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::TerminationGuard,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::ContextCompression,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::VerbosityReduction,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        let mut after = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::TerminationGuard,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::ContextCompression,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::VerbosityReduction,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        // 3 out of 4 improve
        after.score.tca.score = 100.0;
        after.score.ldi.score = 0.02;
        after.score.cce.score = 0.9;
        // vdi stays at 0.8
        let result = compute(&before, &after);
        assert_eq!(result.addressed_count, 4);
        assert_eq!(result.improved_count, 3);
        assert_eq!(result.score, 0.75);
        assert!(result.pass); // >= TARGET
    }

    #[test]
    fn below_threshold_fails() {
        let before = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::TerminationGuard,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::ContextCompression,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::VerbosityReduction,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        let mut after = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::TerminationGuard,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::ContextCompression,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::VerbosityReduction,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        // Only 2 out of 4 improve
        after.score.tca.score = 100.0;
        after.score.cce.score = 0.9;
        // ldi and vdi stay the same
        let result = compute(&before, &after);
        assert_eq!(result.addressed_count, 4);
        assert_eq!(result.improved_count, 2);
        assert_eq!(result.score, 0.5);
        assert!(!result.pass); // < TARGET
    }

    #[test]
    fn min_delta_filters_noise() {
        let before = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        let mut after = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        // Increase TCA by 0.005 (less than MIN_DELTA of 0.01)
        after.score.tca.score = before.score.tca.score + 0.5; // ~0.005 normalised
        let result = compute(&before, &after);
        // The improvement should be filtered out
        assert_eq!(result.improved_count, 0);
    }

    #[test]
    fn hedge_reduction_uses_inverted_shl() {
        let before = minimal_report(vec![
            Fix {
                fix_type: FixType::HedgeReduction,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        let mut after = minimal_report(vec![
            Fix {
                fix_type: FixType::HedgeReduction,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        // SHL is "lower is better", so normalised() inverts it.
        // Decreasing raw SHL score → increasing normalised SHL → improvement.
        after.score.shl.score = 0.05; // was 0.1, so raw decreased
        let result = compute(&before, &after);
        // Normalised SHL went from (1-0.1)=0.9 to (1-0.05)=0.95, so delta=+0.05 > MIN_DELTA
        assert_eq!(result.improved_count, 1);
        assert!(result.fix_adherence[0].improved);
    }

    #[test]
    fn reformulation_guard_uses_isr() {
        let before = minimal_report(vec![
            Fix {
                fix_type: FixType::ReformulationGuard,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        let mut after = minimal_report(vec![
            Fix {
                fix_type: FixType::ReformulationGuard,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        // ReformulationGuard targets ISR.
        after.score.isr.score = 90.0; // was 85.0
        let result = compute(&before, &after);
        assert_eq!(result.improved_count, 1);
        assert!(result.fix_adherence[0].improved);
    }

    #[test]
    fn correct_fix_types_in_adherence() {
        let before = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::ContextCompression,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        let mut after = minimal_report(vec![
            Fix {
                fix_type: FixType::ToolSchema,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
            Fix {
                fix_type: FixType::ContextCompression,
                target: "test".into(),
                patch: "test".into(),
                estimated_token_savings: 100,
            },
        ]);
        after.score.tca.score = 100.0;
        after.score.cce.score = 0.9;
        let result = compute(&before, &after);
        assert_eq!(result.fix_adherence.len(), 2);
        assert!(matches!(result.fix_adherence[0].fix_type, FixType::ToolSchema));
        assert!(matches!(result.fix_adherence[1].fix_type, FixType::ContextCompression));
    }
}
