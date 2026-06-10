/// Auto-Fix Generation (E-01)
///
/// For each flagged issue in the trace, generates an actionable fix that
/// the user can apply directly to their agent configuration.
///
/// Fix types:
///   - `ToolSchema`         — correct a tool description's required parameters
///   - `PromptInsert`       — insert an instruction into the system prompt
///   - `TerminationGuard`   — add a loop-breaking condition to the system prompt
///   - `ContextCompression` — add a context summarisation instruction
///   - `VerbosityReduction` — reduce filler words and low-density content
///   - `HedgeReduction`     — strip sycophantic preambles and hedging phrases
///   - `CavemanPromptInsert`— add a directive to keep output maximally concise
///   - `ReformulationGuard` — prevent the agent from re-stating its input context
///   - `GoalAnchor`         — re-anchor a drifting agent on its task objective
use serde::{Deserialize, Serialize};

use crate::metrics::TpeResult;
use crate::scoring::TasScore;
use crate::types::{StepFlag, Trace};

const AVS_FIX_THRESHOLD: f64 = 0.40;
/// Fraction of a drifting agent's off-path token spend assumed recoverable once
/// it is re-anchored on its goal. Deliberately conservative; see `GoalAnchor`.
const GOAL_ANCHOR_RECOVERY_FRACTION: f64 = 0.25;

/// The kind of fix generated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FixType {
    /// Corrected tool schema with required parameters explicitly marked.
    ToolSchema,
    /// Instruction to insert into the agent's system prompt.
    PromptInsert,
    /// Termination guard condition to break a detected loop.
    TerminationGuard,
    /// Context summarisation instruction to reduce CCE bloat.
    ContextCompression,
    /// Remove filler words and low-density content from reasoning steps.
    VerbosityReduction,
    /// Eliminate sycophantic openers and excessive hedging phrases.
    HedgeReduction,
    /// Inject a Caveman-style conciseness directive into the system prompt.
    CavemanPromptInsert,
    /// Prevent the agent from re-stating its input context verbatim.
    ReformulationGuard,
    /// Re-anchor a drifting agent on its stated task objective.
    GoalAnchor,
}

impl std::fmt::Display for FixType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FixType::ToolSchema => write!(f, "tool_schema"),
            FixType::PromptInsert => write!(f, "prompt_insert"),
            FixType::TerminationGuard => write!(f, "termination_guard"),
            FixType::ContextCompression => write!(f, "context_compression"),
            FixType::VerbosityReduction => write!(f, "verbosity_reduction"),
            FixType::HedgeReduction => write!(f, "hedge_reduction"),
            FixType::CavemanPromptInsert => write!(f, "caveman_prompt_insert"),
            FixType::ReformulationGuard => write!(f, "reformulation_guard"),
            FixType::GoalAnchor => write!(f, "goal_anchor"),
        }
    }
}

/// How much human review a fix needs before it can be applied.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FixRisk {
    /// Pure prompt-text hygiene (verbosity, hedging, compression directives).
    Safe,
    /// Changes behaviour-shaping config (schemas, loop guards): review first.
    #[default]
    NeedsReview,
    /// Could suppress legitimate behaviour (e.g. retries/verification runs):
    /// `apply` refuses these without an explicit override.
    Dangerous,
}

impl FixRisk {
    fn for_type(t: &FixType) -> FixRisk {
        match t {
            FixType::ContextCompression
            | FixType::VerbosityReduction
            | FixType::HedgeReduction
            | FixType::CavemanPromptInsert
            | FixType::ReformulationGuard
            | FixType::PromptInsert
            | FixType::GoalAnchor => FixRisk::Safe,
            FixType::ToolSchema => FixRisk::NeedsReview,
            // A termination guard can suppress exactly the re-run an agent
            // uses to verify a fix; never auto-apply.
            FixType::TerminationGuard => FixRisk::Dangerous,
        }
    }
}

/// A generated fix with estimated token impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fix {
    pub fix_type: FixType,
    /// The config, schema, or prompt section to modify.
    pub target: String,
    /// The suggested correction or instruction text.
    pub patch: String,
    /// Estimated tokens saved per run if this fix is applied.
    pub estimated_token_savings: u32,
    /// Review class: safe / needs_review / dangerous (see [`FixRisk`]).
    #[serde(default)]
    pub risk: FixRisk,
}

impl Fix {
    /// The text that should actually land in a prompt file when this fix is
    /// applied. Report patches are written for human readers ("Task
    /// complexity classified as ... Add to system prompt: \"...\" (37% of
    /// sentences ...)"); appending that meta-prose verbatim into a live
    /// system prompt injects analysis noise into the agent. This extracts
    /// just the quoted directive; patches without the marker are returned
    /// unchanged.
    pub fn prompt_directive(&self) -> String {
        const MARKER: &str = "Add to system prompt: \"";
        if let Some(pos) = self.patch.find(MARKER) {
            let rest = &self.patch[pos + MARKER.len()..];
            if let Some(end) = rest.rfind('\"') {
                let directive = rest[..end].trim();
                if !directive.is_empty() {
                    return directive.to_string();
                }
            }
        }
        self.patch.clone()
    }
}

/// Generate fixes for all flagged issues in the trace.
///
/// Returns an empty vec if no actionable issues were found. `tpe` is the
/// Trajectory Path Entropy diagnostic; together with GAR it drives the
/// `GoalAnchor` remediation for off-path trajectories.
pub fn generate_fixes(trace: &Trace, score: &TasScore, tpe: &TpeResult) -> Vec<Fix> {
    let mut fixes = Vec::new();

    // ── TCA: tool misfires → tool schema fixes ──────────────────────────────
    for misfire in &score.tca.misfires {
        let tool = &misfire.tool_name;
        let savings = estimate_misfire_savings(trace, misfire.failed_step);
        // Diagnose from the actual error text instead of assuming a schema
        // problem: most real failures are value errors (wrong amount, bad id),
        // which marking parameters required would not prevent.
        let error = misfire.error.as_deref().unwrap_or("");
        let elow = error.to_lowercase();
        let patch = if elow.contains("required") || elow.contains("missing") {
            format!(
                "Tool \"{tool}\" failed at step {} with: \"{error}\". Mark the \
                 missing parameter(s) as required in the tool schema so the model \
                 cannot omit them.",
                misfire.failed_step,
            )
        } else if !error.is_empty() {
            format!(
                "Tool \"{tool}\" failed at step {} with: \"{error}\". Add a \
                 pre-call check for this condition to the system prompt (e.g.\
                 recompute derived values such as totals/fees and validate \
                 identifiers against prior observations before calling \
                 \"{tool}\").",
                misfire.failed_step,
            )
        } else {
            format!(
                "Tool \"{tool}\" failed at step {} with no recorded error. Log \
                 tool errors into the trace (tool_error) so failures are \
                 diagnosable, and add a retry-once-with-validation policy for \
                 \"{tool}\".",
                misfire.failed_step,
            )
        };
        fixes.push(Fix {
            fix_type: FixType::ToolSchema,
            target: tool.clone(),
            patch,
            estimated_token_savings: savings,
            risk: FixRisk::for_type(&FixType::ToolSchema),
        });
    }

    // ── CCE: context bloat → compression instruction ─────────────────────────
    if !score.cce.bloated_steps.is_empty() {
        let total_bloat_tokens: u32 = score
            .cce
            .bloated_steps
            .iter()
            .filter_map(|b| {
                trace.steps.iter().find(|s| s.id == b.step_id).map(|s| {
                    (s.tokens as f64 * b.duplicate_pct / 100.0) as u32
                })
            })
            .sum();
        fixes.push(Fix {
            fix_type: FixType::ContextCompression,
            target: "system_prompt".into(),
            patch: "Before each tool call, summarise the conversation to the last three \
                    relevant facts. Do not re-include information that has already been \
                    established earlier in this session."
                .into(),
            estimated_token_savings: total_bloat_tokens,
            risk: FixRisk::for_type(&FixType::ContextCompression),
        });
    }

    // ── LDI: detected loops → termination guards ─────────────────────────────
    for detected_loop in &score.ldi.loops {
        if detected_loop.step_ids.is_empty() {
            continue;
        }
        let ids: Vec<String> = detected_loop.step_ids.iter().map(|id| id.to_string()).collect();
        let loop_desc = ids.join(", ");

        // Estimate savings: token cost of all but the first iteration.
        let loop_tokens: u32 = trace
            .steps
            .iter()
            .filter(|s| detected_loop.step_ids.contains(&s.id))
            .map(|s| s.tokens)
            .sum();
        let iters = detected_loop.step_ids.len().max(2);
        let save_tokens = loop_tokens.saturating_sub(loop_tokens / iters as u32);

        fixes.push(Fix {
            fix_type: FixType::TerminationGuard,
            target: "system_prompt".into(),
            patch: format!(
                "Add termination condition for steps [{loop_desc}]: once the action \
                 at these steps succeeds, do not repeat it. Proceed directly to the \
                 next distinct task step."
            ),
            estimated_token_savings: save_tokens,
            risk: FixRisk::for_type(&FixType::TerminationGuard),
        });
    }

    // ── RDA: over-depth → step-count instruction ─────────────────────────────
    if !score.rda.pass && score.rda.actual_steps > score.rda.expected_steps as usize {
        let overdepth_tokens: u32 = trace
            .steps
            .iter()
            .filter(|s| s.flags.contains(&StepFlag::OverDepth))
            .map(|s| s.tokens * 3 / 4) // removing 75% of flagged step tokens
            .sum();

        // Even if no OverDepth flags exist, estimate from excess steps.
        let excess = score.rda.actual_steps.saturating_sub(score.rda.expected_steps as usize);
        let avg_tokens = if trace.steps.is_empty() {
            0
        } else {
            trace.steps.iter().map(|s| s.tokens).sum::<u32>() / trace.steps.len() as u32
        };
        let estimated = if overdepth_tokens > 0 {
            overdepth_tokens
        } else {
            excess as u32 * avg_tokens
        };

        if estimated > 0 {
            fixes.push(Fix {
                fix_type: FixType::PromptInsert,
                target: "system_prompt".into(),
                patch: format!(
                    "Task complexity classified as {} (expected ~{:.0} steps, used {} steps). \
                     Add to system prompt: \"Complete this task in {:.0} steps or fewer. \
                     Do not re-verify results that have already been confirmed.\"",
                    score.rda.classified_complexity,
                    score.rda.expected_steps,
                    score.rda.actual_steps,
                    score.rda.expected_steps + 1.0,
                ),
                estimated_token_savings: estimated,
                risk: FixRisk::for_type(&FixType::PromptInsert),
            });
        }
    }

    // ── Verbosity metrics (VDI, SHL, CCR) → verbosity fixes ─────────────────
    if score.avs > AVS_FIX_THRESHOLD {
        // VDI: low density → VerbosityReduction
        if !score.vdi.pass {
            let low_steps = &score.vdi.low_density_steps;
            let wasted: u32 = trace
                .steps
                .iter()
                .filter(|s| low_steps.contains(&s.id))
                .map(|s| (s.tokens as f64 * (1.0 - score.vdi.score)).round() as u32)
                .sum();
            if wasted > 0 {
                // Quote the top three filler phrases actually observed in this
                // trace, with their counts, so the patch is trace-specific and
                // the agent's prompt names real offenders rather than a static
                // dictionary of "basically / actually / essentially".
                let observed = score
                    .vdi
                    .top_offenders
                    .iter()
                    .take(3)
                    .map(|(phrase, n)| format!("\"{phrase}\" ({n}x)"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let offenders_clause = if observed.is_empty() {
                    "filler adverbs and vague qualifiers".to_string()
                } else {
                    format!("the highest-frequency filler patterns observed in this trace: {observed}")
                };
                fixes.push(Fix {
                    fix_type: FixType::VerbosityReduction,
                    target: "system_prompt".into(),
                    patch: format!(
                        "Add to system prompt: \"Every reasoning step must consist of \
                         substantive content only. In particular, eliminate {offenders_clause}. \
                         Steps [{low_steps_str}] had VDI below target.\"",
                        low_steps_str = low_steps
                            .iter()
                            .map(|id| id.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    estimated_token_savings: wasted,
                    risk: FixRisk::for_type(&FixType::VerbosityReduction),
                });
            }
        }

        // SHL: high sycophancy/hedging → HedgeReduction
        if !score.shl.pass {
            let shl_waste = (score.shl.score * trace.steps.iter().map(|s| s.tokens).sum::<u32>() as f64)
                .round() as u32;
            fixes.push(Fix {
                fix_type: FixType::HedgeReduction,
                target: "system_prompt".into(),
                patch: format!(
                    "Add to system prompt: \"Do not begin responses with preamble phrases \
                     (let me, I'd be happy to, certainly, absolutely, of course). \
                     Do not use more than one hedging phrase per sentence (might, could, \
                     perhaps, possibly). State conclusions directly.\" \
                     ({:.0}% of sentences were flagged as sycophantic or over-hedged.)",
                    score.shl.score * 100.0
                ),
                estimated_token_savings: shl_waste / 5, // ~20% of flagged content is preamble
                risk: FixRisk::for_type(&FixType::HedgeReduction),
            });
        }

        // CCR: high compression ratio → CavemanPromptInsert
        if !score.ccr.pass {
            let ccr_waste = score.ccr.total_cuttable_tokens;
            if ccr_waste > 0 {
                fixes.push(Fix {
                    fix_type: FixType::CavemanPromptInsert,
                    target: "system_prompt".into(),
                    patch: format!(
                        "Add to system prompt: \"Be maximally concise. Output only the \
                         information necessary for the next step. Skip re-stating context, \
                         avoid throat-clearing sentences, and omit preamble sentences entirely. \
                         ~{ccr_waste} tokens of your current output are estimated to be \
                         compressible without information loss.\""
                    ),
                    estimated_token_savings: ccr_waste,
                    risk: FixRisk::for_type(&FixType::CavemanPromptInsert),
                });
            }
        }
    }

    // ── Reformulation detection → ReformulationGuard ─────────────────────────
    let reformulation_steps: Vec<u32> = trace
        .steps
        .iter()
        .filter(|s| s.flags.contains(&StepFlag::Reformulation))
        .map(|s| s.id)
        .collect();

    if !reformulation_steps.is_empty() {
        let wasted: u32 = trace
            .steps
            .iter()
            .filter(|s| reformulation_steps.contains(&s.id))
            .map(|s| s.tokens / 3) // ~33% of reformulation step is the redundant restate
            .sum();
        let ids_str = reformulation_steps
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        fixes.push(Fix {
            fix_type: FixType::ReformulationGuard,
            target: "system_prompt".into(),
            patch: format!(
                "Add to system prompt: \"Do not re-state, paraphrase, or summarise the \
                 user's request at the start of your reasoning. Proceed directly to your \
                 analysis or first action.\" \
                 (Steps [{ids_str}] were detected as reformulating their input context.)"
            ),
            estimated_token_savings: wasted,
            risk: FixRisk::for_type(&FixType::ReformulationGuard),
        });
    }

    // ── GAR / TPE: off-path drift → goal-anchoring directive ─────────────────
    // Previously GAR and the trajectory metrics were detection-only: they could
    // tell you an agent wandered but emitted no remediation. A GoalAnchor fix is
    // produced when the trajectory drifts (high path entropy / low focus) or the
    // agent fails to advance toward its goal. Savings are estimated
    // conservatively from the tokens spent on concretely low-advancement steps —
    // this is primarily a *coherence* fix, not a verbosity trim.
    let drifting = tpe.high_drift || !score.gar.pass;
    if drifting {
        // Tokens in steps GAR flagged as not advancing toward the goal.
        let off_path_tokens: u32 = trace
            .steps
            .iter()
            .filter(|s| score.gar.low_advancement_steps.contains(&s.id))
            .map(|s| s.tokens)
            .sum();
        // When drift is detected purely by TPE (GAR passing, so no
        // low-advancement steps), fall back to the trajectory's regressing-step
        // count × the mean reasoning-step size as the off-path proxy, so the
        // estimate is non-degenerate rather than a misleading 0.
        let off_path_tokens = if off_path_tokens == 0 && tpe.regresses > 0 {
            let reasoning_tokens: u32 = trace
                .steps
                .iter()
                .filter(|s| s.step_type == crate::types::StepType::Reasoning)
                .map(|s| s.tokens)
                .sum();
            let reasoning_steps = trace
                .steps
                .iter()
                .filter(|s| s.step_type == crate::types::StepType::Reasoning)
                .count()
                .max(1);
            let avg = reasoning_tokens / reasoning_steps as u32;
            avg.saturating_mul(tpe.regresses as u32)
        } else {
            off_path_tokens
        };
        // Conservative: assume a quarter of off-path step tokens are recoverable
        // exploration once the agent is re-anchored. Never inflate beyond that.
        let estimated = (off_path_tokens as f64 * GOAL_ANCHOR_RECOVERY_FRACTION).round() as u32;

        let goal_clause = match trace.task_goal() {
            Some(g) => {
                let snippet: String = g.chars().take(120).collect();
                format!("the stated task objective (\"{}\")", snippet.trim())
            }
            None => "the task objective established at the start of the trace".to_string(),
        };
        let drift_note = if tpe.goal_origin == crate::metrics::GoalOrigin::NotApplicable {
            format!("goal advancement is below target ({:.2})", score.gar.score)
        } else {
            format!(
                "trajectory path entropy {:.2} / focus {:.2} ({})",
                tpe.path_entropy,
                tpe.focus_score,
                tpe.interpretation()
            )
        };

        // Measured live (docs/case_study.md): an earlier wording that asked the
        // agent to *restate the objective before each reasoning step* added a
        // per-turn output cost that exceeded the recovered drift on on-track
        // runs (mean -5.6% tokens). The anchor must never add a standing
        // ritual: anchor silently, skip non-advancing steps, forbid restating.
        fixes.push(Fix {
            fix_type: FixType::GoalAnchor,
            target: "system_prompt".into(),
            patch: format!(
                "Add to system prompt: \"Keep {goal_clause} as your working objective. \
                 Before acting, check that the action moves measurably closer to it; if it \
                 does not, skip it and take the next concrete action instead. Do not restate \
                 the objective or summarise progress unless explicitly asked.\" \
                 (Detected drift: {drift_note}.)"
            ),
            estimated_token_savings: estimated,
            risk: FixRisk::for_type(&FixType::GoalAnchor),
        });
    }

    fixes
}

fn estimate_misfire_savings(trace: &Trace, failed_step_id: u32) -> u32 {
    if let Some(failed) = trace.steps.iter().find(|s| s.id == failed_step_id) {
        // The misfire itself is the wasted cost — the retry was necessary.
        return failed.tokens;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::ScoringConfig;
    use crate::types::{StepType, Trace, TraceStep};
    use std::collections::HashMap;

    fn make_trace(steps: Vec<TraceStep>) -> Trace {
        let tokens: u32 = steps.iter().map(|s| s.tokens).sum();
        Trace {
            trace_id: "t1".into(),
            agent_name: "agent".into(),
            framework: "raw".into(),
            steps,
            total_tokens: tokens,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_no_fixes_for_clean_trace() {
        let trace = make_trace(vec![
            TraceStep {
                id: 1,
                step_type: StepType::Reasoning,
                content: "parse request".into(),
                tokens: 300,
                tool_name: None,
                tool_params: None,
                tool_success: None,
                tool_error: None,
                agent_id: None,
                input_context: None,
                output: None,
                flags: vec![],
                flag_details: vec![],
            },
            TraceStep {
                id: 2,
                step_type: StepType::ToolCall,
                content: "call tool".into(),
                tokens: 200,
                tool_name: Some("get_order".into()),
                tool_params: None,
                tool_success: Some(true),
                tool_error: None,
                agent_id: None,
                input_context: None,
                output: None,
                flags: vec![],
                flag_details: vec![],
            },
        ]);
        let mut t = trace.clone();
        let config = ScoringConfig::default();
        let sim = |_: &str, _: &str| 0.0_f64;
        let report = crate::analyse(&mut t, sim, &config).unwrap();
        let fixes = generate_fixes(&trace, &report.score, &report.path_entropy);
        // Clean trace with no misfire, no bloat, no loops → likely empty or only RDA.
        assert!(fixes.iter().all(|f| !matches!(f.fix_type, FixType::ToolSchema)));
    }

    #[test]
    fn test_verbosity_fix_quotes_observed_offenders() {
        // Build a verbose trace stuffed with specific filler patterns so the
        // VerbosityReduction patch should quote them by name.
        let verbose_step = |id: u32| TraceStep {
            id,
            step_type: StepType::Reasoning,
            content:
                "Let me think through this carefully. I'd be happy to help. \
                 Basically the order is, actually, essentially in the system. \
                 Basically I will now proceed. Actually let me also confirm. \
                 Essentially we should basically actually proceed actually now."
                    .into(),
            tokens: 600,
            tool_name: None,
            tool_params: None,
            tool_success: None,
            tool_error: None,
            agent_id: None,
            input_context: None,
            output: None,
            flags: vec![],
            flag_details: vec![],
        };
        let trace = make_trace(vec![
            verbose_step(1),
            verbose_step(2),
            verbose_step(3),
            verbose_step(4),
            verbose_step(5),
        ]);
        let mut t = trace.clone();
        let config = ScoringConfig::default();
        let sim = |_: &str, _: &str| 0.0_f64;
        let report = crate::analyse(&mut t, sim, &config).unwrap();
        let fixes = generate_fixes(&trace, &report.score, &report.path_entropy);
        let verbosity_fix = fixes
            .iter()
            .find(|f| matches!(f.fix_type, FixType::VerbosityReduction))
            .expect("verbosity fix should be emitted");
        // The patch must quote at least one of the observed offenders with a count.
        assert!(
            verbosity_fix.patch.contains("\"basically\"")
                || verbosity_fix.patch.contains("\"actually\"")
                || verbosity_fix.patch.contains("\"essentially\""),
            "patch should quote observed filler with count, got: {}",
            verbosity_fix.patch
        );
        assert!(
            verbosity_fix.patch.contains("x)"),
            "patch should include the (Nx) count format"
        );
    }
}
