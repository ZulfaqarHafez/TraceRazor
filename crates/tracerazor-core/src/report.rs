/// Report generation: produces JSON and Markdown output from a TasScore.
use serde::{Deserialize, Serialize};

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
                // Loop *repeat* steps are deleted.
                let detail = step.flag_details.first().cloned().unwrap_or("loop".into());
                (DiffAction::Delete, Some(format!("Loop: {}", detail)), Some(0))
            } else if has_flag(&StepFlag::LoopStart) {
                // Loop *start* step is kept (it's the first, valid occurrence).
                // The loop is documented in a note.
                let detail = step.flag_details.first().cloned().unwrap_or("loop start".into());
                (DiffAction::Keep, Some(format!("Loop start (keep first): {}", detail)), None)
            } else if has_flag(&StepFlag::Misfire) {
                let detail = step.flag_details.first().cloned().unwrap_or_default();
                (DiffAction::Delete, Some(format!("Misfired: {}", detail)), Some(0))
            } else if has_flag(&StepFlag::OverDepth) {
                let trimmed = (step.tokens / 4).max(100); // suggest 25% of original
                (
                    DiffAction::Trim,
                    Some("Reduce reasoning depth (simple task)".into()),
                    Some(trimmed),
                )
            } else if has_flag(&StepFlag::ContextBloat) {
                let detail = step.flag_details.first().cloned().unwrap_or_default();
                let kept = (step.tokens as f64 * 0.44) as u32; // remove ~56% bloat
                (
                    DiffAction::Trim,
                    Some(format!("Compress context: {}", detail)),
                    Some(kept),
                )
            } else if has_flag(&StepFlag::Retry) {
                // The retry itself is OK (needed) — keep it.
                (DiffAction::Keep, None, None)
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
             Steps:     {}   Tokens: {}   \n\
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
        // RDA — show actual value if Phase 2 computed it, else PENDING
        let (rda_score_str, rda_status) = match &s.rda {
            Some(r) => (format!("{:.3}", r.score), pass_str(r.pass).to_string()),
            None => ("N/A".into(), "PENDING".into()),
        };
        out += &format!("{:<6} {:<30} {:<8} {:<8} {}\n", "RDA", "Reasoning Depth Approp.", rda_score_str, ">0.75", rda_status);

        // ISR — available in both phases (Phase 1 uses BoW, Phase 2 uses embeddings)
        let (isr_score_str, isr_status) = match &s.isr {
            Some(r) => (format!("{:.1}%", r.score), pass_str(r.pass).to_string()),
            None => ("N/A".into(), "PENDING".into()),
        };
        out += &format!("{:<6} {:<30} {:<8} {:<8} {}\n", "ISR", "Info Sufficiency Rate", isr_score_str, ">80%", isr_status);

        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "TUR", "Token Utilisation Ratio",
            format!("{:.3}", s.tur.score), ">0.35", pass_str(s.tur.pass)
        );
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "CCE", "Context Carry-over Eff.",
            format!("{:.3}", s.cce.score), ">0.60", pass_str(s.cce.pass)
        );

        // DBO — show actual value if Phase 2 computed it
        let (dbo_score_str, dbo_status) = match &s.dbo {
            Some(r) => (format!("{:.3}", r.score), pass_str(r.pass).to_string()),
            None => ("N/A".into(), "PENDING".into()),
        };
        out += &format!("{:<6} {:<30} {:<8} {:<8} {}\n", "DBO", "Decision Branch Optimality", dbo_score_str, ">0.70", dbo_status);

        out += &format!("{sep}\n");

        // Per-step annotations
        out += "PER-STEP ANNOTATIONS\n";
        out += &format!("{:>3}  {:<12} {:<8}  {}\n", "#", "Type", "Tokens", "Flags");

        // We need the trace steps — re-derive from score's waste data.
        // The report stores all info needed.
        for line in &self.diff {
            let flags_str = if line.justification.is_some() {
                line.justification.as_deref().unwrap_or("").to_string()
            } else {
                "-".into()
            };
            out += &format!(
                "{:>3}  {:<12} {:>8}  {}\n",
                line.step_id,
                line.step_type,
                line.tokens_actual,
                flags_str
            );
        }

        out += &format!("{sep}\n");

        // Optimal path
        let optimal_tokens = Self::optimal_tokens(&self.diff);
        let kept = self.diff.iter().filter(|d| matches!(d.action, DiffAction::Keep | DiffAction::Trim)).count();
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

        // Savings
        let sv = &self.savings;
        out += &format!(
            "SAVINGS ESTIMATE\n\
             Tokens saved:      {}  ({:.1}% reduction)\n\
             Cost saved:        ${:.4} per run  (at ${:.2}/1M tokens)\n\
             At 50K runs/month: ${:.2}/month saved\n\
             Latency saved:     ~{:.1}s per run\n\
             {sep}\n",
            sv.tokens_saved,
            sv.reduction_pct,
            sv.cost_saved_per_run_usd,
            3.0_f64, // default display
            sv.monthly_savings_usd,
            sv.latency_saved_seconds
        );

        out
    }
}
