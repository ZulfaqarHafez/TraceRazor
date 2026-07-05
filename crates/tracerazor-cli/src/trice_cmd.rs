use super::OutputFormat;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use tracerazor_core::{
    provenance::sha256_hex,
    types::{StepFlag, Trace, TraceStep},
};
use tracerazor_ingest::{parse as ingest_parse, TraceFormat};
const TRICE_BUCKET_SIZE: u32 = 128;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TriceSegment {
    segment_id: String,
    step_id: u32,
    kind: String,
    state: String,
    tokens: u32,
    locked: bool,
    receipt: String,
    identifiers: Vec<String>,
    rehydrate_pointer: Option<String>,
    rationale: String,
}

#[derive(Debug, Clone)]
struct TriceCandidate {
    action: String,
    tokens: u32,
    value: f64,
    rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TriceDecision {
    segment_id: String,
    step_id: u32,
    state: String,
    action: String,
    original_tokens: u32,
    policy_tokens: u32,
    locked: bool,
    receipt: String,
    rehydrate_pointer: Option<String>,
    value: f64,
    rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TricePolicy {
    algorithm: String,
    budget_ratio: f64,
    bucket_size: u32,
    baseline_input_tokens: u32,
    budget_tokens: u32,
    policy_tokens: u32,
    projected_input_savings_pct: f64,
    budget_exceeded: bool,
    constraints: serde_json::Value,
    decisions: Vec<TriceDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TriceReplayMetrics {
    evidence_recall: f64,
    action_divergence: f64,
    expired_info_retention: f64,
    rehydration_success: f64,
    compression_overhead: f64,
    pass_noninferior: bool,
}

pub(super) async fn cmd_trice_optimize(
    trace_path: PathBuf,
    budget_ratio: f64,
    output_path: Option<PathBuf>,
    format: OutputFormat,
) -> Result<()> {
    if !(0.05..=1.0).contains(&budget_ratio) {
        anyhow::bail!("--budget-ratio must be between 0.05 and 1.0");
    }
    let data = std::fs::read_to_string(&trace_path)
        .with_context(|| format!("Cannot read trace: {}", trace_path.display()))?;
    let trace = ingest_parse(&data, TraceFormat::Auto)
        .with_context(|| format!("Failed to parse trace: {}", trace_path.display()))?;
    let segments = trice_segments_from_trace(&trace);
    let policy = trice_solve_policy(&segments, budget_ratio);

    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&policy)?),
        OutputFormat::Markdown => println!("{}", render_trice_policy_markdown(&trace, &policy)),
    }
    if let Some(path) = output_path {
        std::fs::write(&path, serde_json::to_string_pretty(&policy)? + "\n")
            .with_context(|| format!("Cannot write {}", path.display()))?;
        eprintln!("Wrote TRICE context policy -> {}", path.display());
    }
    Ok(())
}

pub(super) async fn cmd_trice_replay(
    trace_path: PathBuf,
    policy_path: PathBuf,
    format: OutputFormat,
) -> Result<()> {
    let trace_data = std::fs::read_to_string(&trace_path)
        .with_context(|| format!("Cannot read trace: {}", trace_path.display()))?;
    let trace = ingest_parse(&trace_data, TraceFormat::Auto)
        .with_context(|| format!("Failed to parse trace: {}", trace_path.display()))?;
    let policy_data = std::fs::read_to_string(&policy_path)
        .with_context(|| format!("Cannot read policy: {}", policy_path.display()))?;
    let policy: TricePolicy = serde_json::from_str(&policy_data)
        .with_context(|| format!("Failed to parse policy JSON: {}", policy_path.display()))?;
    let segments = trice_segments_from_trace(&trace);
    let metrics = trice_evaluate_policy(&segments, &policy);

    match format {
        OutputFormat::Json => {
            let out = json!({
                "trace_id": trace.trace_id,
                "policy": {
                    "algorithm": policy.algorithm,
                    "baseline_input_tokens": policy.baseline_input_tokens,
                    "policy_tokens": policy.policy_tokens,
                    "projected_input_savings_pct": policy.projected_input_savings_pct,
                },
                "replay": metrics,
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        OutputFormat::Markdown => println!(
            "{}",
            render_trice_replay_markdown(&trace, &policy, &metrics)
        ),
    }
    Ok(())
}

fn trice_segments_from_trace(trace: &Trace) -> Vec<TriceSegment> {
    trace
        .steps
        .iter()
        .enumerate()
        .map(|(idx, step)| {
            let text = trice_step_text(step);
            let state = trice_classify_step(trace, step, idx);
            let locked = state == "essential";
            TriceSegment {
                segment_id: format!("s{}", step.id),
                step_id: step.id,
                kind: step.step_type.to_string(),
                state: state.to_string(),
                tokens: step.tokens.max(1),
                locked,
                receipt: sha256_hex(text.as_bytes()),
                identifiers: trice_identifiers(&text),
                rehydrate_pointer: Some(format!("trace:{}:step:{}", trace.trace_id, step.id)),
                rationale: trice_state_rationale(state, step),
            }
        })
        .collect()
}

fn trice_step_text(step: &TraceStep) -> String {
    let mut parts = Vec::new();
    if !step.content.trim().is_empty() {
        parts.push(step.content.trim().to_string());
    }
    if let Some(output) = &step.output {
        if !output.trim().is_empty() {
            parts.push(output.trim().to_string());
        }
    }
    if let Some(ctx) = &step.input_context {
        if !ctx.trim().is_empty() {
            parts.push(ctx.trim().to_string());
        }
    }
    if let Some(tool) = &step.tool_name {
        parts.push(format!("tool:{tool}"));
    }
    if let Some(err) = &step.tool_error {
        parts.push(format!("error:{err}"));
    }
    if let Some(params) = &step.tool_params {
        parts.push(params.to_string());
    }
    parts.join("\n")
}

fn trice_classify_step(trace: &Trace, step: &TraceStep, idx: usize) -> &'static str {
    let n = trace.steps.len();
    if trice_failed_then_retried(trace, idx, step) {
        return "expired";
    }
    if step.tool_success == Some(false) {
        return "essential";
    }
    if idx == 0 || idx + 1 == n {
        return "essential";
    }
    if step.is_mutating() && step.tool_success == Some(true) {
        return "essential";
    }
    if step.flags.iter().any(|f| {
        matches!(
            f,
            StepFlag::Redundant | StepFlag::Loop | StepFlag::Reformulation
        )
    }) || trice_looks_redundant(step)
    {
        return "redundant";
    }
    if step.tool_name.is_some() && step.tool_success == Some(true) {
        return "rehydratable";
    }
    if trice_looks_distracting(step) {
        return "distractor";
    }
    "unknown"
}

fn trice_failed_then_retried(trace: &Trace, idx: usize, step: &TraceStep) -> bool {
    if step.tool_success != Some(false) {
        return false;
    }
    let Some(tool) = &step.tool_name else {
        return false;
    };
    trace.steps.iter().skip(idx + 1).any(|later| {
        later.tool_name.as_deref() == Some(tool.as_str()) && later.tool_success == Some(true)
    })
}

fn trice_state_rationale(state: &str, step: &TraceStep) -> String {
    match state {
        "essential" if step.tool_success == Some(false) => {
            "unresolved failure/error evidence".into()
        }
        "essential" if step.is_mutating() => "successful mutating tool call".into(),
        "essential" => "task, final-state, or quality anchor".into(),
        "rehydratable" => "successful read/tool observation can be re-fetched".into(),
        "expired" => "failed tool call followed by a successful retry".into(),
        "redundant" => "redundant, looping, or reformulation-like step".into(),
        "distractor" => "filler-heavy low-signal step".into(),
        _ => "no conservative state rule matched".into(),
    }
}

fn trice_looks_redundant(step: &TraceStep) -> bool {
    let c = step.content.to_lowercase();
    c.contains("parse the user request again")
        || c.contains("re-evaluating whether")
        || c.contains("re-evaluate whether")
        || (c.contains("again") && c.matches("refund").count() > 3)
}

fn trice_looks_distracting(step: &TraceStep) -> bool {
    let c = step.content.to_lowercase();
    [
        "let me",
        "basically",
        "essentially",
        "to be honest",
        "actually",
        "think deeply",
        "double check",
    ]
    .iter()
    .filter(|term| c.contains(**term))
    .count()
        >= 2
}

fn trice_identifiers(text: &str) -> Vec<String> {
    let mut ids = Vec::new();
    for raw in text
        .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':')))
    {
        let token = raw.trim_matches(|c: char| matches!(c, '.' | ',' | ':' | ';' | ')' | '('));
        if token.len() < 3 {
            continue;
        }
        let looks_like_id = token.chars().any(|c| c.is_ascii_digit())
            || token.contains('.')
            || token.contains('/')
            || token.contains("::")
            || token
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || matches!(c, '_' | '-'));
        if looks_like_id && !ids.iter().any(|seen| seen == token) {
            ids.push(token.to_string());
        }
        if ids.len() >= 12 {
            break;
        }
    }
    ids
}

fn trice_solve_policy(segments: &[TriceSegment], budget_ratio: f64) -> TricePolicy {
    let baseline_input_tokens: u32 = segments.iter().map(|s| s.tokens).sum();
    let budget_tokens = ((baseline_input_tokens as f64 * budget_ratio).round() as u32).max(1);
    let candidate_sets: Vec<Vec<TriceCandidate>> =
        segments.iter().map(trice_action_candidates).collect();
    let min_required: u32 = candidate_sets
        .iter()
        .map(|cs| cs.iter().map(|c| c.tokens).min().unwrap_or(1))
        .sum();
    let effective_budget = budget_tokens.max(min_required);
    let budget_buckets = trice_buckets(effective_budget) as usize;

    let mut dp: Vec<Option<(f64, Vec<usize>)>> = vec![None; budget_buckets + 1];
    dp[0] = Some((0.0, Vec::new()));
    for candidates in &candidate_sets {
        let mut next: Vec<Option<(f64, Vec<usize>)>> = vec![None; budget_buckets + 1];
        for (used, state) in dp.iter().enumerate() {
            let Some((score, picks)) = state else {
                continue;
            };
            for (idx, candidate) in candidates.iter().enumerate() {
                let nb = used + trice_buckets(candidate.tokens) as usize;
                if nb > budget_buckets {
                    continue;
                }
                let new_score = *score + candidate.value;
                let replace = next[nb]
                    .as_ref()
                    .map(|(best, _)| new_score > *best)
                    .unwrap_or(true);
                if replace {
                    let mut new_picks = picks.clone();
                    new_picks.push(idx);
                    next[nb] = Some((new_score, new_picks));
                }
            }
        }
        dp = next;
    }

    let picks = dp
        .into_iter()
        .flatten()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, picks)| picks)
        .unwrap_or_else(|| {
            candidate_sets
                .iter()
                .map(|cs| {
                    cs.iter()
                        .enumerate()
                        .min_by_key(|(_, c)| c.tokens)
                        .map(|(idx, _)| idx)
                        .unwrap_or(0)
                })
                .collect()
        });

    let decisions: Vec<TriceDecision> = segments
        .iter()
        .zip(candidate_sets.iter())
        .zip(picks.iter())
        .map(|((segment, candidates), pick)| {
            let candidate = &candidates[*pick];
            TriceDecision {
                segment_id: segment.segment_id.clone(),
                step_id: segment.step_id,
                state: segment.state.clone(),
                action: candidate.action.clone(),
                original_tokens: segment.tokens,
                policy_tokens: candidate.tokens,
                locked: segment.locked,
                receipt: segment.receipt.clone(),
                rehydrate_pointer: segment.rehydrate_pointer.clone(),
                value: (candidate.value * 1_000_000.0).round() / 1_000_000.0,
                rationale: candidate.rationale.clone(),
            }
        })
        .collect();
    let policy_tokens: u32 = decisions.iter().map(|d| d.policy_tokens).sum();
    let projected_input_savings_pct = if baseline_input_tokens == 0 {
        0.0
    } else {
        ((baseline_input_tokens as f64 - policy_tokens as f64) / baseline_input_tokens as f64
            * 10_000.0)
            .round()
            / 100.0
    };
    TricePolicy {
        algorithm: "trice-v0.1-multi-choice-knapsack".into(),
        budget_ratio,
        bucket_size: TRICE_BUCKET_SIZE,
        baseline_input_tokens,
        budget_tokens,
        policy_tokens,
        projected_input_savings_pct,
        budget_exceeded: policy_tokens > budget_tokens,
        constraints: json!({
            "evidence_recall_min": 0.95,
            "pass_rate_noninferiority_pp": -2,
            "locked_anchors_unchanged": true,
        }),
        decisions,
    }
}

fn trice_action_candidates(segment: &TriceSegment) -> Vec<TriceCandidate> {
    if segment.locked {
        let action = if segment.step_id <= 2 {
            "anchor_prefix"
        } else {
            "keep"
        };
        return vec![TriceCandidate {
            action: action.into(),
            tokens: segment.tokens,
            value: 1.35,
            rationale: "locked anchor kept byte-for-byte".into(),
        }];
    }

    let (utility, risk, cache, hallucination) = match segment.state.as_str() {
        "rehydratable" => (0.72, 0.30, 0.18, 0.25),
        "expired" => (0.24, 0.16, 0.05, 0.18),
        "redundant" => (0.18, 0.12, 0.06, 0.22),
        "distractor" => (0.12, 0.10, 0.04, 0.22),
        _ => (0.58, 0.55, 0.10, 0.55),
    };

    let mut out = Vec::new();
    let mut push = |action: &str, tokens: u32, u: f64, r: f64, k: f64, h: f64, rationale: &str| {
        let cost = tokens as f64 / segment.tokens.max(1) as f64;
        let value = u - 1.4 * r - 0.8 * cost + 0.5 * k - 1.1 * h;
        out.push(TriceCandidate {
            action: action.into(),
            tokens: tokens.max(1).min(segment.tokens),
            value,
            rationale: rationale.into(),
        });
    };
    push(
        "keep",
        segment.tokens,
        utility,
        risk,
        cache,
        hallucination,
        "full segment retained",
    );
    push(
        "extract",
        trice_ratio_tokens(segment.tokens, 0.45, 32),
        utility * 0.88,
        risk * 0.55,
        cache * 0.55,
        hallucination * 0.42,
        "extractive compression",
    );
    push(
        "summarize",
        trice_ratio_tokens(segment.tokens, 0.25, 24),
        utility * 0.70,
        risk * 0.42,
        cache * 0.40,
        hallucination * 0.70,
        "short natural-language state summary",
    );
    push(
        "lazy_recall",
        trice_ratio_tokens(segment.tokens, 0.12, 20),
        utility * 0.55,
        risk * 0.35,
        cache * 0.85,
        hallucination * 0.45,
        "receipt plus rehydration pointer",
    );
    push(
        "mask_with_receipt",
        trice_ratio_tokens(segment.tokens, 0.07, 12),
        utility * 0.28,
        risk * 0.18,
        cache * 0.55,
        hallucination * 0.38,
        "drop text, retain cryptographic receipt",
    );
    out
}

fn trice_ratio_tokens(tokens: u32, ratio: f64, floor: u32) -> u32 {
    if tokens <= floor {
        tokens.max(1)
    } else {
        ((tokens as f64 * ratio).round() as u32).max(floor)
    }
}

fn trice_buckets(tokens: u32) -> u32 {
    tokens.max(1).div_ceil(TRICE_BUCKET_SIZE).max(1)
}

fn trice_evaluate_policy(segments: &[TriceSegment], policy: &TricePolicy) -> TriceReplayMetrics {
    let mut required = Vec::<String>::new();
    let mut available = Vec::<String>::new();
    let mut locked_count = 0usize;
    let mut destructive_changes = 0usize;
    let mut expired_original = 0u32;
    let mut expired_kept = 0u32;
    let mut lazy_total = 0usize;
    let mut lazy_valid = 0usize;

    for decision in &policy.decisions {
        let Some(segment) = segments
            .iter()
            .find(|s| s.segment_id == decision.segment_id)
        else {
            continue;
        };
        if decision.locked {
            locked_count += 1;
            for id in &segment.identifiers {
                if !required.contains(id) {
                    required.push(id.clone());
                }
            }
            if !matches!(decision.action.as_str(), "keep" | "anchor_prefix") {
                destructive_changes += 1;
            }
        }
        if matches!(
            decision.action.as_str(),
            "keep" | "anchor_prefix" | "extract" | "summarize"
        ) {
            for id in &segment.identifiers {
                if !available.contains(id) {
                    available.push(id.clone());
                }
            }
        }
        if decision.action == "lazy_recall" {
            lazy_total += 1;
            if decision.receipt == segment.receipt && decision.rehydrate_pointer.is_some() {
                lazy_valid += 1;
            }
        }
        if decision.state == "expired" {
            expired_original += decision.original_tokens;
            if matches!(
                decision.action.as_str(),
                "keep" | "anchor_prefix" | "extract" | "summarize"
            ) {
                expired_kept += decision.policy_tokens;
            }
        }
    }

    let recalled = required.iter().filter(|id| available.contains(*id)).count();
    let evidence_recall = if required.is_empty() {
        1.0
    } else {
        recalled as f64 / required.len() as f64
    };
    let action_divergence = destructive_changes as f64 / locked_count.max(1) as f64;
    let expired_info_retention = if expired_original == 0 {
        0.0
    } else {
        expired_kept as f64 / expired_original as f64
    };
    let rehydration_success = if lazy_total == 0 {
        1.0
    } else {
        lazy_valid as f64 / lazy_total as f64
    };
    let compression_overhead =
        policy.policy_tokens as f64 / policy.baseline_input_tokens.max(1) as f64;
    TriceReplayMetrics {
        evidence_recall: trice_round4(evidence_recall),
        action_divergence: trice_round4(action_divergence),
        expired_info_retention: trice_round4(expired_info_retention),
        rehydration_success: trice_round4(rehydration_success),
        compression_overhead: trice_round4(compression_overhead),
        pass_noninferior: evidence_recall >= 0.95 && action_divergence == 0.0,
    }
}

fn trice_round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

fn render_trice_policy_markdown(trace: &Trace, policy: &TricePolicy) -> String {
    let mut lines = Vec::new();
    lines.push(format!("# TRICE optimize - {}", trace.trace_id));
    lines.push(String::new());
    lines.push(format!(
        "Baseline input tokens: {} | Policy tokens: {} | Projected savings: {:.2}%",
        policy.baseline_input_tokens, policy.policy_tokens, policy.projected_input_savings_pct
    ));
    if policy.budget_exceeded {
        lines.push(format!(
            "Budget note: locked anchors force {} tokens, above requested budget {}.",
            policy.policy_tokens, policy.budget_tokens
        ));
    }
    lines.push(String::new());
    lines.push("| Step | State | Action | Tokens | Rationale |".into());
    lines.push("|---:|---|---|---:|---|".into());
    for d in &policy.decisions {
        lines.push(format!(
            "| {} | {} | {} | {} -> {} | {} |",
            d.step_id, d.state, d.action, d.original_tokens, d.policy_tokens, d.rationale
        ));
    }
    lines.join("\n") + "\n"
}

fn render_trice_replay_markdown(
    trace: &Trace,
    policy: &TricePolicy,
    metrics: &TriceReplayMetrics,
) -> String {
    format!(
        "# TRICE replay - {}\n\n\
         Policy: {} | projected savings {:.2}%\n\n\
         | Metric | Value |\n\
         |---|---:|\n\
         | evidence_recall | {:.3} |\n\
         | action_divergence | {:.3} |\n\
         | expired_info_retention | {:.3} |\n\
         | rehydration_success | {:.3} |\n\
         | compression_overhead | {:.3} |\n\
         | pass_noninferior | {} |\n",
        trace.trace_id,
        policy.algorithm,
        policy.projected_input_savings_pct,
        metrics.evidence_recall,
        metrics.action_divergence,
        metrics.expired_info_retention,
        metrics.rehydration_success,
        metrics.compression_overhead,
        metrics.pass_noninferior
    )
}
