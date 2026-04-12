/// Report generation: produces JSON and Markdown output from a TasScore.
use serde::{Deserialize, Serialize};

use crate::fixes::Fix;
use crate::scoring::{SavingsEstimate, TasScore};
use crate::types::{StepFlag, Trace};

/// An entry in the optimal path diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffAction {
    Keep,
    Delete,
    Trim,
}

impl std::fmt::Display for DiffAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiffAction::Keep => write!(f, "KEEP"),
            DiffAction::Delete => write!(f, "DEL "),
            DiffAction::Trim => write!(f, "TRIM"),
        }
    }
}

/// One line in the optimal path diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffLine {
    pub action: DiffAction,
    pub step_id: u32,
    pub step_type: String,
    pub description: String,
    pub justification: Option<String>,
    pub tokens_actual: u32,
    pub tokens_suggested: Option<u32>,
}

/// A detected anomaly when this trace deviates from the agent's historical baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Metric that triggered the anomaly.
    pub metric: String,
    /// Observed value.
    pub value: f64,
    /// z-score (signed: negative = regression, positive = improvement).
    pub z_score: f64,
    /// Rolling mean of this metric for the agent.
    pub baseline_mean: f64,
    /// Rolling standard deviation.
    pub baseline_std: f64,
}

/// Per-agent efficiency breakdown for multi-agent traces (Decision 7).
///
/// Populated only when the trace contains steps with at least two distinct
/// `agent_id` values. Each entry represents one agent thread within the
/// overall trace DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBreakdown {
    /// The agent identifier (matches `TraceStep::agent_id`).
    pub agent_id: String,
    pub total_steps: usize,
    pub total_tokens: u32,
    /// This agent's share of total trace tokens (0–100%).
    pub token_share_pct: f64,
    /// Individual TAS score for this agent's sub-trace (0–100).
    /// `None` if the sub-trace had fewer than the minimum required steps.
    pub tas_score: Option<f64>,
    pub grade: Option<String>,
}

/// A complete TraceRazor report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceReport {
    pub trace_id: String,
    pub agent_name: String,
    pub framework: String,
    pub total_steps: usize,
    pub total_tokens: u32,
    pub analysis_duration_ms: u64,
    pub score: TasScore,
    pub diff: Vec<DiffLine>,
    pub savings: SavingsEstimate,
    /// Auto-generated fix patches (E-01).
    #[serde(default)]
    pub fixes: Vec<Fix>,
    /// Plain-English one-paragraph summary of the report.
    pub summary: String,
    /// Executive one-liner for stakeholder communication (E-08).
    pub summary_oneliner: String,
    /// Anomalies detected against the agent's historical baseline (E-04).
    #[serde(default)]
    pub anomalies: Vec<Anomaly>,
    /// Per-agent thread breakdown (populated for multi-agent traces only).
    #[serde(default)]
    pub per_agent: Vec<AgentBreakdown>,
}

impl TraceReport {
    /// Build the optimal-path diff from annotated trace steps.
    pub fn build_diff(trace: &Trace, _score: &TasScore) -> Vec<DiffLine> {
        let mut diff = Vec::new();

        for step in &trace.steps {
            let has_flag = |f: &StepFlag| step.flags.contains(f);

            let (action, justification, tokens_suggested) = if has_flag(&StepFlag::Redundant) {
                let detail = step.flag_details.first().cloned().unwrap_or_default();
                (DiffAction::Delete, Some(format!("Redundant: {}", detail)), Some(0))
            } else if has_flag(&StepFlag::Loop) {
                let detail = step.flag_details.first().cloned().unwrap_or("loop".into());
                (DiffAction::Delete, Some(format!("Loop: {}", detail)), Some(0))
            } else if has_flag(&StepFlag::LoopStart) {
                let detail = step
                    .flag_details
                    .first()
                    .cloned()
                    .unwrap_or("loop start".into());
                (
                    DiffAction::Keep,
                    Some(format!("Loop start (keep first): {}", detail)),
                    None,
                )
            } else if has_flag(&StepFlag::Misfire) {
                let detail = step.flag_details.first().cloned().unwrap_or_default();
                (DiffAction::Delete, Some(format!("Misfired: {}", detail)), Some(0))
            } else if has_flag(&StepFlag::OverDepth) {
                let trimmed = (step.tokens / 4).max(100);
                (
                    DiffAction::Trim,
                    Some("Reduce reasoning depth (simple task)".into()),
                    Some(trimmed),
                )
            } else if has_flag(&StepFlag::ContextBloat) {
                let detail = step.flag_details.first().cloned().unwrap_or_default();
                let kept = (step.tokens as f64 * 0.44) as u32;
                (
                    DiffAction::Trim,
                    Some(format!("Compress context: {}", detail)),
                    Some(kept),
                )
            } else if has_flag(&StepFlag::Reformulation) {
                let detail = step.flag_details.first().cloned().unwrap_or_default();
                let trimmed = (step.tokens * 2 / 3).max(50);
                (
                    DiffAction::Trim,
                    Some(format!("Reformulation: {}", detail)),
                    Some(trimmed),
                )
            } else {
                (DiffAction::Keep, None, None)
            };

            diff.push(DiffLine {
                action,
                step_id: step.id,
                step_type: step.step_type.to_string(),
                description: step
                    .tool_name
                    .as_deref()
                    .map(|n| format!("Call {n}"))
                    .unwrap_or_else(|| step.content.chars().take(50).collect()),
                justification,
                tokens_actual: step.tokens,
                tokens_suggested,
            });
        }

        diff
    }

    /// Tokens in the suggested optimal path.
    pub fn optimal_tokens(diff: &[DiffLine]) -> u32 {
        diff.iter()
            .map(|d| match d.action {
                DiffAction::Keep => d.tokens_actual,
                DiffAction::Delete => 0,
                DiffAction::Trim => d.tokens_suggested.unwrap_or(d.tokens_actual),
            })
            .sum()
    }

    /// Render the report as a Markdown string.
    pub fn to_markdown(&self) -> String {
        let s = &self.score;
        let sep = "-".repeat(54);

        // Header
        let mut out = format!(
            "TRACERAZOR REPORT\n{sep}\n\
             Trace:     {}\n\
             Agent:     {}\n\
             Framework: {}\n\
             Steps:     {}   Tokens: {}\n\
             Analysed:  {}ms\n\
             {sep}\n",
            self.trace_id,
            self.agent_name,
            self.framework,
            self.total_steps,
            self.total_tokens,
            self.analysis_duration_ms,
        );

        // Score
        out += &format!(
            "TRACERAZOR SCORE:  {:.0} / 100  [{}]\n\
             VAE SCORE:         {:.2}\n\
             {sep}\n",
            s.score, s.grade, s.vae
        );

        // VERBOSITY ALERT (when AVS > 0.40)
        if s.avs > 0.40 {
            // Identify the primary verbosity driver.
            let vdi_waste = 1.0 - s.vdi.normalised();
            let drivers = [
                ("VDI (verbosity density)", vdi_waste * 0.45),
                ("SHL (sycophancy/hedging)", s.shl.score * 0.30),
                ("CCR (compression ratio)", s.ccr.score * 0.25),
            ];
            let primary = drivers
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(name, _)| *name)
                .unwrap_or("VDI");
            let estimated_verbose_tokens = ((s.avs
                * self.total_tokens as f64)
                .round() as u32)
                .min(self.total_tokens);
            out += &format!(
                "!! VERBOSITY ALERT  AVS: {:.3}  Primary driver: {}  \
                 Est. verbose tokens: {}\n\
                 {sep}\n",
                s.avs, primary, estimated_verbose_tokens
            );
        }

        // Metric breakdown table
        out += "METRIC BREAKDOWN\n";
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "Code", "Metric", "Score", "Target", "Status"
        );

        fn pass_str(pass: bool) -> &'static str {
            if pass { "PASS" } else { "FAIL" }
        }

        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "SRR",
            "Step Redundancy Rate",
            format!("{:.1}%", s.srr.score),
            "<15%",
            pass_str(s.srr.pass)
        );
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "LDI",
            "Loop Detection Index",
            format!("{:.3}", s.ldi.score),
            "<0.10",
            pass_str(s.ldi.pass)
        );
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "TCA",
            "Tool Call Accuracy",
            format!("{:.1}%", s.tca.score),
            ">85%",
            pass_str(s.tca.pass)
        );
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}{}\n",
            "RDA",
            "Reasoning Depth Approp.",
            format!("{:.3}", s.rda.score),
            ">0.75",
            pass_str(s.rda.pass),
            if s.rda.uses_historical_baseline { " [hist]" } else { "" }
        );
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "ISR",
            "Info Sufficiency Rate",
            format!("{:.1}%", s.isr.score),
            ">80%",
            pass_str(s.isr.pass)
        );
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "TUR",
            "Token Utilisation Ratio",
            format!("{:.3}", s.tur.score),
            ">0.35",
            pass_str(s.tur.pass)
        );
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "CCE",
            "Context Carry-over Eff.",
            format!("{:.3}", s.cce.score),
            ">0.60",
            pass_str(s.cce.pass)
        );
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}{}\n",
            "DBO",
            "Decision Branch Optimality",
            format!("{:.3}", s.dbo.score),
            ">0.70",
            pass_str(s.dbo.pass),
            if s.dbo.cold_start { " [cold]" } else { "" }
        );

        // Verbosity metrics separator + three new rows.
        out += &format!("-- Verbosity Metrics {}\n", "-".repeat(34));
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "VDI",
            "Verbosity Density Index",
            format!("{:.3}", s.vdi.score),
            ">0.60",
            pass_str(s.vdi.pass)
        );
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "SHL",
            "Sycophancy/Hedging Level",
            format!("{:.3}", s.shl.score),
            "<0.20",
            pass_str(s.shl.pass)
        );
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "CCR",
            "Caveman Compression Ratio",
            format!("{:.3}", s.ccr.score),
            "<0.30",
            pass_str(s.ccr.pass)
        );

        out += &format!("{sep}\n");

        // Summary (plain-English)
        if !self.summary.is_empty() {
            out += "SUMMARY\n";
            out += &self.summary;
            out += "\n";
            out += &format!("{sep}\n");
        }

        // Per-step annotations
        out += "PER-STEP ANNOTATIONS\n";
        out += &format!("{:>3}  {:<12} {:<8}  {}\n", "#", "Type", "Tokens", "Flags");

        for line in &self.diff {
            let flags_str = line
                .justification
                .as_deref()
                .unwrap_or("-")
                .to_string();
            out += &format!(
                "{:>3}  {:<12} {:>8}  {}\n",
                line.step_id, line.step_type, line.tokens_actual, flags_str
            );
        }

        out += &format!("{sep}\n");

        // Optimal path
        let optimal_tokens = Self::optimal_tokens(&self.diff);
        let kept = self
            .diff
            .iter()
            .filter(|d| matches!(d.action, DiffAction::Keep | DiffAction::Trim))
            .count();
        out += &format!(
            "OPTIMAL PATH RECOMMENDATION\n\
             Suggested: {} steps (vs {} actual)  |  Est. tokens: {} (vs {})\n\n",
            kept, self.total_steps, optimal_tokens, self.total_tokens
        );

        for line in &self.diff {
            let marker = match line.action {
                DiffAction::Keep => "  KEEP",
                DiffAction::Delete => "- DEL ",
                DiffAction::Trim => "~ TRIM",
            };
            let just = line.justification.as_deref().unwrap_or("");
            out += &format!(
                "{}  Step {:>2}  {:<12}  {}",
                marker, line.step_id, line.step_type, line.description
            );
            if !just.is_empty() {
                out += &format!("  [{}]", just);
            }
            out += "\n";
        }

        out += &format!("{sep}\n");

        // Fixes (E-01)
        if !self.fixes.is_empty() {
            out += "AUTO-GENERATED FIXES\n";
            for (i, fix) in self.fixes.iter().enumerate() {
                out += &format!(
                    "  Fix {}: [{}] → {}\n  Patch: {}\n  Est. savings: {} tokens/run\n\n",
                    i + 1,
                    fix.fix_type,
                    fix.target,
                    fix.patch,
                    fix.estimated_token_savings,
                );
            }
            out += &format!("{sep}\n");
        }

        // Anomalies (E-04)
        if !self.anomalies.is_empty() {
            out += "ANOMALY ALERTS\n";
            for a in &self.anomalies {
                let direction = if a.z_score < 0.0 { "REGRESSION" } else { "IMPROVEMENT" };
                out += &format!(
                    "  [{}] {}: {:.1} (baseline {:.1} ± {:.1}, z={:.1})\n",
                    direction, a.metric, a.value, a.baseline_mean, a.baseline_std, a.z_score
                );
            }
            out += &format!("{sep}\n");
        }

        // Savings
        let sv = &self.savings;
        out += &format!(
            "SAVINGS ESTIMATE\n\
             Tokens saved:      {}  ({:.1}% reduction)\n\
             Cost saved:        ${:.4} per run\n\
             At 50K runs/month: ${:.2}/month saved\n\
             Latency saved:     ~{:.1}s per run\n\
             {sep}\n",
            sv.tokens_saved,
            sv.reduction_pct,
            sv.cost_saved_per_run_usd,
            sv.monthly_savings_usd,
            sv.latency_saved_seconds
        );

        // Multi-agent breakdown
        if !self.per_agent.is_empty() {
            out += "MULTI-AGENT BREAKDOWN\n";
            out += &format!(
                "{:<24} {:>6} {:>8} {:>7} {:>7}  {}\n",
                "Agent", "Steps", "Tokens", "Share", "TAS", "Grade"
            );
            out += &"-".repeat(64);
            out += "\n";
            for ab in &self.per_agent {
                let tas_str = ab
                    .tas_score
                    .map(|t| format!("{:.1}", t))
                    .unwrap_or_else(|| "N/A".into());
                let grade_str = ab.grade.as_deref().unwrap_or("N/A");
                out += &format!(
                    "{:<24} {:>6} {:>8} {:>6.1}% {:>7}  {}\n",
                    ab.agent_id,
                    ab.total_steps,
                    ab.total_tokens,
                    ab.token_share_pct,
                    tas_str,
                    grade_str,
                );
            }
            out += &format!("{sep}\n");
        }

        // Executive one-liner (E-08)
        if !self.summary_oneliner.is_empty() {
            out += &format!("EXECUTIVE SUMMARY\n{}\n{sep}\n", self.summary_oneliner);
        }

        out
    }
}

/// Identify the single worst-performing metric by its normalised score.
fn worst_metric(score: &TasScore) -> (&'static str, f64) {
    let metrics = [
        ("SRR (step redundancy)", score.srr.normalised()),
        ("LDI (loop detection)", score.ldi.normalised()),
        ("TCA (tool accuracy)", score.tca.normalised()),
        ("RDA (reasoning depth)", score.rda.normalised()),
        ("ISR (info sufficiency)", score.isr.normalised()),
        ("TUR (token utilisation)", score.tur.normalised()),
        ("CCE (context carry-over)", score.cce.normalised()),
        ("DBO (branch optimality)", score.dbo.normalised()),
        ("VDI (verbosity density)", score.vdi.normalised()),
        ("SHL (sycophancy/hedging)", score.shl.normalised()),
        ("CCR (compression ratio)", score.ccr.normalised()),
    ];
    metrics
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or(("TAS", 0.0))
}

/// Generate a plain-English one-paragraph summary (E-08 template engine).
///
/// Leads with the single biggest problem, includes specific token numbers,
/// and ends with the cost impact. Suitable for a PR comment or Slack alert.
pub fn generate_summary(trace: &Trace, score: &TasScore, savings: &SavingsEstimate) -> String {
    let grade_desc = match score.grade {
        crate::scoring::Grade::Excellent => "highly optimised",
        crate::scoring::Grade::Good => "reasonably efficient with minor inefficiencies",
        crate::scoring::Grade::Fair => "wasting a significant portion of its token budget",
        crate::scoring::Grade::Poor => "consuming far more tokens than necessary",
    };

    // Lead with the worst metric.
    let (worst_name, worst_val) = worst_metric(score);
    let worst_sentence = format!(
        "The biggest efficiency gap is {} (score {:.2}/1.0).",
        worst_name,
        worst_val
    );

    // Build specific issue sentences.
    let mut issues = Vec::new();

    if !score.ldi.pass && !score.ldi.loops.is_empty() {
        let loop_tokens: u32 = score
            .ldi
            .loops
            .iter()
            .flat_map(|l| l.step_ids.iter())
            .filter_map(|&id| trace.steps.iter().find(|s| s.id == id))
            .map(|s| s.tokens)
            .sum();
        issues.push(format!(
            "{} reasoning loop(s) detected consuming ~{} tokens unnecessarily",
            score.ldi.loops.len(),
            loop_tokens
        ));
    }
    if !score.srr.pass {
        let redundant_tokens: u32 = score
            .srr
            .redundant_steps
            .iter()
            .filter_map(|p| trace.steps.iter().find(|s| s.id == p.step_b))
            .map(|s| s.tokens)
            .sum();
        issues.push(format!(
            "{:.0}% of steps are redundant ({} tokens wasted)",
            score.srr.score,
            redundant_tokens
        ));
    }
    if !score.tca.pass {
        let misfire_tokens: u32 = score
            .tca
            .misfires
            .iter()
            .filter_map(|m| trace.steps.iter().find(|s| s.id == m.failed_step))
            .map(|s| s.tokens)
            .sum();
        issues.push(format!(
            "{} tool misfire(s) wasted ~{} tokens on failed calls",
            score.tca.misfires.len(),
            misfire_tokens
        ));
    }
    if !score.cce.pass {
        let bloat_tokens: u32 = score
            .cce
            .bloated_steps
            .iter()
            .filter_map(|b| {
                trace.steps.iter().find(|s| s.id == b.step_id).map(|s| {
                    (s.tokens as f64 * b.duplicate_pct / 100.0) as u32
                })
            })
            .sum();
        issues.push(format!(
            "context bloat duplicated ~{} tokens across LLM calls",
            bloat_tokens
        ));
    }
    if !score.rda.pass {
        let direction = if score.rda.actual_steps > score.rda.expected_steps as usize {
            "over-reasoned"
        } else {
            "under-reasoned"
        };
        issues.push(format!(
            "{} ({} steps used vs ~{:.0} expected for a {} task)",
            direction,
            score.rda.actual_steps,
            score.rda.expected_steps,
            score.rda.classified_complexity
        ));
    }

    let issues_text = if issues.is_empty() {
        "No major issues detected.".into()
    } else {
        format!("Issues found: {}.", issues.join("; "))
    };

    let savings_text = if savings.tokens_saved > 0 {
        format!(
            " Applying the recommended fixes would save {} tokens per run \
             (${:.4}/run, ${:.2}/month at 50K runs).",
            savings.tokens_saved,
            savings.cost_saved_per_run_usd,
            savings.monthly_savings_usd
        )
    } else {
        String::new()
    };

    format!(
        "The {} agent ({}) scored {:.0}/100 [{}] — it is {}. {} {}{}",
        trace.agent_name,
        trace.framework,
        score.score,
        score.grade,
        grade_desc,
        worst_sentence,
        issues_text,
        savings_text,
    )
    .trim()
    .to_string()
}

/// Generate an executive one-liner for stakeholder communication (E-08).
///
/// Format: "<Agent> scores <N>/100 [<Grade>]. Biggest issue: <worst metric>.
/// Fix saves $<Z>/month."
pub fn generate_oneliner(trace: &Trace, score: &TasScore, savings: &SavingsEstimate) -> String {
    let (worst_name, _) = worst_metric(score);

    if savings.monthly_savings_usd > 0.0 {
        format!(
            "{} scores {:.0}/100 [{}]. Biggest issue: {}. \
             Apply fixes to save ${:.0}/month at 50K runs.",
            trace.agent_name,
            score.score,
            score.grade,
            worst_name,
            savings.monthly_savings_usd,
        )
    } else {
        format!(
            "{} scores {:.0}/100 [{}]. {}",
            trace.agent_name,
            score.score,
            score.grade,
            if score.score >= 90.0 {
                "No significant waste detected.".into()
            } else {
                format!("Primary concern: {}.", worst_name)
            }
        )
    }
}
