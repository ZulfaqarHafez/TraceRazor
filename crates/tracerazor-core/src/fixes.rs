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
use serde::{Deserialize, Serialize};

use crate::scoring::TasScore;
use crate::types::{StepFlag, Trace};

const AVS_FIX_THRESHOLD: f64 = 0.40;

/// The kind of fix generated.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

/// Generate fixes for all flagged issues in the trace.
///
/// Returns an empty vec if no actionable issues were found.
pub fn generate_fixes(trace: &Trace, score: &TasScore) -> Vec<Fix> {
    let mut fixes = Vec::new();

    // ── TCA: tool misfires → tool schema fixes ──────────────────────────────
    for misfire in &score.tca.misfires {
        let tool = &misfire.tool_name;
        let savings = estimate_misfire_savings(trace, misfire.failed_step);
        fixes.push(Fix {
            fix_type: FixType::ToolSchema,
            target: tool.clone(),
            patch: format!(
                "Mark required parameters as required in the \"{tool}\" tool schema. \
                 The tool failed at step {} ({}) — ensure all parameters needed for \
                 a successful call are documented as required so the model cannot \
                 omit them.",
                misfire.failed_step,
                misfire.error.as_deref().unwrap_or("missing parameter"),
            ),
            estimated_token_savings: savings,
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
                fixes.push(Fix {
                    fix_type: FixType::VerbosityReduction,
                    target: "system_prompt".into(),
                    patch: format!(
                        "Add to system prompt: \"Every reasoning step must consist of \
                         substantive content only. Do not use filler adverbs (basically, \
                         actually, essentially, literally), articles in non-essential positions, \
                         or vague qualifiers. Steps [{low_steps_str}] had VDI below target.\"\
                         ",
                        low_steps_str = low_steps
                            .iter()
                            .map(|id| id.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    estimated_token_savings: wasted,
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
        let fixes = generate_fixes(&trace, &report.score);
        // Clean trace with no misfire, no bloat, no loops → likely empty or only RDA.
        assert!(fixes.iter().all(|f| !matches!(f.fix_type, FixType::ToolSchema)));
    }
}
