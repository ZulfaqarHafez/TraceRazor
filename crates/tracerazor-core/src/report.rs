/// Report generation: produces JSON and Markdown output from a TasScore.
use serde::{Deserialize, Serialize};

use crate::fixes::Fix;
use crate::iar::IarResult;
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
    /// Minimum Viable Trace Gap (M3).
    ///
    /// Fraction of tokens above the optimal path suggested by the step diff:
    ///   `mvtg = (actual_tokens − optimal_tokens) / actual_tokens`
    ///
    /// 0.0 = the trace was already at minimum viable token count.
    /// 1.0 = all tokens were in deletable/trimmable steps (fully wasteful).
    ///
    /// Unlike `savings.reduction_pct` (which projects fix-based estimates),
    /// MVTG is derived directly from the per-step diff classifications
    /// (KEEP / DELETE / TRIM), making it a structural lower bound on waste.
    #[serde(default)]
    pub mvtg: f64,
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
    /// Trajectory Path Entropy — an information-theoretic "staying on the path"
    /// diagnostic. Reported alongside TAS but **not** folded into the composite
    /// score (see `metrics::tpe`).
    #[serde(default)]
    pub path_entropy: crate::metrics::TpeResult,
    /// Instruction Adherence Rate (M5) — populated only when comparing before/after reports.
    #[serde(default)]
    pub iar: Option<IarResult>,
    /// Experimental context-accumulation features (see `crate::features`).
    /// Diagnostic only — emitted next to the score for calibration research, and
    /// **not** part of the TAS composite. Keys are stable snake_case strings.
    #[serde(default)]
    pub features: std::collections::BTreeMap<String, f64>,
    /// Action/Claim Grounding Fidelity (AGF) — deterministic provenance
    /// diagnostic: the share of tool-call argument literals and final-answer
    /// claims that are traceable to prior context/observations. Reported
    /// alongside TAS but **not** folded into the composite pending weight
    /// calibration (see `metrics::agf`).
    #[serde(default)]
    pub agf: Option<crate::metrics::AgfResult>,
    /// Run manifest binding this report to its inputs: trace content hash,
    /// tool version, weights, similarity backend, and the store-derived
    /// baselines that influenced scoring. `None` on reports produced by
    /// embedding callers that do not supply one.
    #[serde(default)]
    pub manifest: Option<RunManifest>,
}

/// Provenance manifest: everything needed to attribute — and, for hermetic
/// bag-of-words runs, byte-for-byte re-verify — a TraceRazor score.
///
/// Auditable-run rationale: a score is only evidence if a third party can
/// check *what* was scored (`trace_sha256`), *by what* (`tool_version`,
/// `similarity_backend`), *under which configuration* (`weights` + hash,
/// `threshold`, `min_steps`), and *with which hidden inputs* (the store-derived
/// baselines). `tracerazor verify` consumes this block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    /// SHA-256 of the raw input trace file bytes (before parsing).
    pub trace_sha256: String,
    /// `CARGO_PKG_VERSION` of the binary that produced the report.
    pub tool_version: String,
    /// RFC 3339 UTC timestamp of the audit.
    pub created_at: String,
    /// Similarity backend actually used: `"bow"` or `"embeddings:<model>"`.
    /// Silent embedding→BoW fallbacks are recorded here as fact.
    pub similarity_backend: String,
    /// The exact composite weights used (inline — small and self-contained).
    pub weights: crate::scoring::Weights,
    /// SHA-256 of the canonical JSON serialisation of `weights`.
    pub weights_sha256: String,
    /// TAS pass threshold in force.
    pub threshold: f64,
    /// Cost basis used for savings estimates (USD per million tokens).
    pub cost_per_million_tokens: f64,
    /// Step floor the audit ran with.
    pub min_steps: usize,
    /// True when the run read nothing from and wrote nothing to the local
    /// store — scoring was a pure function of (trace, config, version).
    pub hermetic: bool,
    /// Store-derived baseline token count that influenced VAE, if any.
    pub baseline_tokens: Option<u32>,
    /// Store-derived historical median steps that influenced RDA, if any.
    pub historical_median_steps: Option<f64>,
    /// Number of store-derived historical tool sequences fed to DBO.
    pub n_historical_sequences: usize,
    /// Parse-quality assessment of the input (see [`IngestQuality`]).
    #[serde(default)]
    pub ingest_quality: Option<IngestQuality>,
    /// Ed25519 verifying key (base64url, no padding) — present when the
    /// audit was run with `TRACERAZOR_SIGNING_KEY` set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signing_public_key: Option<String>,
    /// Ed25519 signature over the canonical report bytes (base64url, no
    /// padding). Canonical bytes: full report serialised to compact JSON
    /// with `analysis_duration_ms = 0` and both signature fields set to
    /// `null` before serialising.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report_signature: Option<String>,
}

impl RunManifest {
    /// A non-hermetic run with any store-derived input is not exactly
    /// reproducible from the manifest alone; verification is then limited to
    /// hash/version checks.
    pub fn store_influenced(&self) -> bool {
        self.baseline_tokens.is_some()
            || self.historical_median_steps.is_some()
            || self.n_historical_sequences > 0
    }
}

/// Compute the canonical byte representation of a report that is used as the
/// Ed25519 signing input.
///
/// Canonical form: full report serialised to compact JSON with two fields
/// normalised to remove sources of non-determinism:
///   - `analysis_duration_ms` → 0  (wall-clock timing is not reproducible)
///   - `manifest.signing_public_key` → None  (signing a field that includes
///     the public key would create a circularity; it is recovered at verify time)
///   - `manifest.report_signature` → None  (signature cannot sign itself)
///
/// The report is first serialised to JSON then immediately re-parsed to ensure
/// that all floating-point values are in their "round-tripped" form. This
/// eliminates a 1-ULP serde_json parsing discrepancy where
/// `to_string` → `from_str` may produce a slightly different f64 bit pattern
/// than the original in-memory value, causing sign/verify to diverge.
/// By going through the same JSON round-trip on both paths the canonical bytes
/// are always derived from the same stable representation.
pub fn canonical_report_bytes(report: &TraceReport) -> anyhow::Result<Vec<u8>> {
    // Normalise mutable fields before serialising.
    let mut r = report.clone();
    r.analysis_duration_ms = 0;
    if let Some(ref mut m) = r.manifest {
        m.report_signature = None;
        m.signing_public_key = None;
    }
    // Round-trip through JSON so every f64 is in its serde_json-stable form.
    // Without this, a value like 0.9566563467492261 may re-parse to a 1-ULP-
    // different float, making the canonical bytes differ between sign and verify.
    let json_str = serde_json::to_string(&r)?;
    let r2: TraceReport = serde_json::from_str(&json_str)?;
    Ok(serde_json::to_vec(&r2)?)
}

/// How much of the parsed trace carries real data. A TAS computed over steps
/// with zero token counts or placeholder content (e.g. an OTel parse that
/// fell back to span names) must never look authoritative — the audit
/// surfaces this loudly and records it next to the score.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestQuality {
    /// Share of steps with a zero token count (0.0–1.0).
    pub zero_token_pct: f64,
    /// Share of steps whose content is a placeholder: empty, a bare tool /
    /// span name, or fewer than three words (0.0–1.0).
    pub placeholder_content_pct: f64,
    /// True when either share exceeds 50% — token- and content-derived
    /// metrics are then unreliable for this trace.
    pub degraded: bool,
}

impl IngestQuality {
    pub fn assess(trace: &crate::types::Trace) -> IngestQuality {
        let n = trace.steps.len().max(1) as f64;
        let zero_tokens = trace.steps.iter().filter(|s| s.tokens == 0).count() as f64;
        let placeholder = trace
            .steps
            .iter()
            .filter(|s| {
                let c = s.content.trim();
                c.is_empty()
                    || s.tool_name.as_deref() == Some(c)
                    || c.split_whitespace().count() < 3
            })
            .count() as f64;
        let zero_token_pct = zero_tokens / n;
        let placeholder_content_pct = placeholder / n;
        IngestQuality {
            zero_token_pct: (zero_token_pct * 1000.0).round() / 1000.0,
            placeholder_content_pct: (placeholder_content_pct * 1000.0).round() / 1000.0,
            degraded: zero_token_pct > 0.5 || placeholder_content_pct > 0.5,
        }
    }
}

impl TraceReport {
    /// Canonical serialisation bytes for Ed25519 signing/verification.
    ///
    /// `analysis_duration_ms` is zeroed (non-deterministic wall-clock field);
    /// `manifest.signature` and `manifest.signing_key_pub` are excluded (the
    /// signature cannot sign itself). Every other field — including
    /// `manifest.similarity_backend`, `agf`, `savings`, `fixes`, `summary` —
    /// is included, so any edit to any field invalidates the signature.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        let mut r = self.clone();
        r.analysis_duration_ms = 0;
        if let Some(ref mut m) = r.manifest {
            m.report_signature = None;
            m.signing_public_key = None;
        }
        serde_json::to_vec(&r)
    }

    /// Build the optimal-path diff from annotated trace steps.
    pub fn build_diff(trace: &Trace, _score: &TasScore) -> Vec<DiffLine> {
        let mut diff = Vec::new();

        for step in &trace.steps {
            let has_flag = |f: &StepFlag| step.flags.contains(f);

            // A successful state-changing call (booking, edit, write) is never
            // a delete candidate: removing it breaks the task the trace
            // completed, whatever its lexical similarity to earlier steps.
            let protected = step.is_mutating() && step.tool_success != Some(false);

            let (action, justification, tokens_suggested) = if protected
                && (has_flag(&StepFlag::Redundant) || has_flag(&StepFlag::Loop))
            {
                (
                    DiffAction::Keep,
                    Some("Successful state-changing call (kept; not deletable)".into()),
                    None,
                )
            } else if has_flag(&StepFlag::Redundant) {
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

        // Score — show TVI-adjusted TAS; also show raw_tas when task was imperfect.
        let tvi_note = if (s.task_value_score - 1.0).abs() > 0.001 {
            format!(
                "  (raw structural: {:.0}, task value: {:.2})",
                s.raw_tas, s.task_value_score
            )
        } else {
            String::new()
        };
        out += &format!(
            "TRACERAZOR SCORE:  {:.0} / 100  [{}]{}\n\
             VAE SCORE:         {:.2}\n\
             MVTG:              {:.1}%  (trace is {:.1}% above minimum viable token count)\n\
             Note: TAS is an *ordinal* heuristic score — compare runs within one\n\
             project over time, not as an absolute efficiency percentage.\n\
             {sep}\n",
            s.score, s.grade, tvi_note,
            s.vae,
            self.mvtg * 100.0,
            self.mvtg * 100.0,
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
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "CCR",
            "Caveman Compression Ratio",
            format!("{:.3}", s.ccr.score),
            "<0.30",
            pass_str(s.ccr.pass)
        );
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}\n",
            "OBS", "Observation Token Share",
            format!("{:.3}", s.obs.score),
            "≥0.30",
            pass_str(s.obs.pass),
        );

        // Detection-only metrics: their detectors, annotations and fixes all
        // run, but they carry no composite weight by default — the metric
        // self-evaluation over real traces found them non-discriminative or
        // range-broken (see docs/metric_effectiveness.md).
        out += &format!(
            "-- Diagnostics (no default composite weight) {}\n",
            "-".repeat(10)
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

        let gar_note = if s.gar.low_advancement_steps.is_empty() {
            String::new()
        } else {
            format!(
                "  [steps off-track: {}]",
                s.gar
                    .low_advancement_steps
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}{}  (goal proxy: step {})\n",
            "GAR",
            "Goal Advancement Ratio",
            format!("{:.3}", s.gar.score),
            "≥0.40",
            pass_str(s.gar.pass),
            gar_note,
            s.gar.goal_step_id.map(|id| id.to_string()).unwrap_or_else(|| "—".into()),
        );

        let csd_drift_note = if s.csd.high_drift_pairs.is_empty() {
            String::new()
        } else {
            format!(
                "  [drifting pairs: {}]",
                s.csd.high_drift_pairs.iter()
                    .map(|(a, b)| format!("{a}→{b}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        out += &format!(
            "{:<6} {:<30} {:<8} {:<8} {}{}\n",
            "CSD", "Cross-Step Semantic Drift",
            format!("{:.3}", s.csd.score),
            "≥0.60",
            pass_str(s.csd.pass),
            csd_drift_note,
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

        // Path Entropy (information-theoretic on-path diagnostic)
        let tpe = &self.path_entropy;
        if tpe.goal_origin != crate::metrics::GoalOrigin::NotApplicable {
            let goal_src = match tpe.goal_origin {
                crate::metrics::GoalOrigin::TaskGoal => "task goal",
                crate::metrics::GoalOrigin::FinalStep => "final step (no task goal in trace)",
                crate::metrics::GoalOrigin::NotApplicable => "n/a",
            };
            out += &format!(
                "PATH ENTROPY  (staying-on-path diagnostic, not part of TAS)\n\
                 Path entropy:      {:.3}   (0 = directed, 1 = random walk)\n\
                 Focus score:       {:.3}   [{}]   target ≥ {:.2}\n\
                 Trajectory:        {} advance / {} stall / {} regress   (vs {})\n",
                tpe.path_entropy,
                tpe.focus_score,
                tpe.interpretation(),
                tpe.target,
                tpe.advances,
                tpe.stalls,
                tpe.regresses,
                goal_src,
            );
            // On the default lexical (BoW) backend, step-to-goal similarity is
            // noisy, so a genuinely on-track agent can still read as "scattered".
            // Point drifting traces at the embedding backend before trusting it.
            if tpe.high_drift {
                out += "   note:           lexical backend — re-run with --enhanced for embedding-based drift\n";
            }
            out += &format!("{sep}\n");
        }

        // Savings (heuristic projection — see note below)
        let sv = &self.savings;
        out += &format!(
            "SAVINGS ESTIMATE  (heuristic projection from flagged waste, not a measured re-run)\n\
             Tokens saved:      {}  ({:.1}% reduction)\n\
             Cost saved:        ${:.4} per run\n\
             Projected/month:   ${:.2}  (at the configured run count & token price)\n\
             Latency saved:     ~{:.1}s per run\n\
             {sep}\n",
            sv.tokens_saved,
            sv.reduction_pct,
            sv.cost_saved_per_run_usd,
            sv.monthly_savings_usd,
            sv.latency_saved_seconds
        );

        // Instruction Adherence Rate (M5)
        if let Some(ref iar) = self.iar {
            out += "-- Instruction Adherence (M5) ----\n";
            out += &format!(
                "IAR    Instruction Adherence Rate    {:.3}    ≥0.75    {}\n",
                iar.score,
                pass_str(iar.pass),
            );
            if !iar.fix_adherence.is_empty() {
                out += &format!("       {}/{} addressed fix types improved:\n", iar.improved_count, iar.addressed_count);
                for adherence in &iar.fix_adherence {
                    let status = if adherence.improved { "✓" } else { "✗" };
                    out += &format!(
                        "         {status} {:?}  ({:+.3})\n",
                        adherence.fix_type, adherence.delta
                    );
                }
            }
            out += &format!("{sep}\n");
        }

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
        ("GAR (goal advancement)", score.gar.normalised()),
        ("CSD (semantic drift)", score.csd.normalised()),
        ("OBS (observation share)", score.obs.normalised()),
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
            " Applying the recommended fixes is estimated to save ~{} tokens per run \
             (${:.4}/run; ~${:.0}/month projected at 50K runs — a heuristic estimate, \
             not a measured re-run).",
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
             Est. ~${:.0}/month at 50K runs (heuristic projection).",
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
