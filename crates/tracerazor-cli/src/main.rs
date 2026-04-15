use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracerazor_core::{
    cost::{CostConfig, ProviderPreset, project_cost},
    fixes::{Fix, FixType},
    is_analysable,
    scoring::ScoringConfig,
    simulate::{SimulationSpec, simulate},
    types::MIN_TRACE_STEPS,
};
use tracerazor_ingest::{TraceFormat, parse as ingest_parse};
use tracerazor_semantic::{LlmConfig, default_similarity_fn};
use tracerazor_store::TraceStore;

/// Open the persistent file-backed store at `~/.tracerazor/store`.
///
/// Falls back to in-memory if the home directory cannot be determined or if
/// the file store fails to open (e.g. permissions error). This ensures the
/// CLI always works even in CI environments without a writable home directory.
async fn open_store() -> TraceStore {
    let path = (|| -> Option<PathBuf> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()?;
        let dir = PathBuf::from(home).join(".tracerazor");
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir.join("store"))
    })();

    if let Some(p) = path {
        match TraceStore::connect_file(p.to_string_lossy().as_ref()).await {
            Ok(store) => return store,
            Err(e) => eprintln!("Warning: could not open persistent store ({e}), using in-memory."),
        }
    }

    TraceStore::connect_mem().await.expect("in-memory store failed")
}

/// TraceRazor — Token Efficiency Auditor for AI Agents
#[derive(Parser)]
#[command(
    name = "tracerazor",
    version,
    author = "Zulfaqar Hafez",
    about = "Lighthouse score for AI agents. Audit reasoning traces and eliminate token waste.",
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Audit a trace file and produce an efficiency report.
    Audit {
        /// Path to the trace file (JSON).
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Output format.
        #[arg(short, long, default_value = "markdown")]
        format: OutputFormat,

        /// Minimum TAS score to pass (CI/CD gating). Returns exit code 1 if below.
        #[arg(short = 't', long)]
        threshold: Option<f64>,

        /// Trace format. Auto-detects if not specified.
        #[arg(short = 'F', long, default_value = "auto")]
        trace_format: InputFormat,

        /// Cost per million tokens in USD (for savings estimates).
        #[arg(long, default_value = "3.0")]
        cost_per_million: f64,

        /// Save trace and report to the store for historical benchmarking.
        #[arg(long, default_value = "true")]
        store: bool,

        /// Enable enhanced semantic analysis using configured LLM embeddings.
        /// Significantly improves SRR and ISR accuracy by replacing bag-of-words
        /// with dense sentence embeddings. Supports OpenAI, Anthropic (chat-only,
        /// falls back to BoW for embeddings), and OpenAI-compatible endpoints via
        /// TRACERAZOR_LLM_* / OPENAI_API_KEY / ANTHROPIC_API_KEY env vars.
        #[arg(long, default_value = "false")]
        enhanced: bool,
    },

    /// List all stored traces in the current session.
    List {
        /// Filter by agent name.
        #[arg(short, long)]
        agent: Option<String>,
    },

    /// Compare two trace files: TAS delta, per-metric breakdown, regression detection.
    ///
    /// Returns exit code 1 if any metric regresses by more than the configured threshold.
    Compare {
        /// Baseline trace file.
        #[arg(value_name = "BASELINE")]
        baseline: PathBuf,
        /// Target (newer) trace file.
        #[arg(value_name = "TARGET")]
        target: PathBuf,
        /// Output format.
        #[arg(short, long, default_value = "markdown")]
        format: OutputFormat,
        /// Percentage regression threshold that triggers a non-zero exit code.
        /// Default: 10% — a 10-point TAS drop exits with code 1.
        #[arg(long, default_value = "10.0")]
        regression_threshold: f64,
    },

    /// Project monthly and annual costs at a given run volume (E-05).
    ///
    /// Provide one or more trace files. Each file contributes one data point.
    Cost {
        /// Trace file(s) to project costs for.
        #[arg(value_name = "FILE", required = true, num_args = 1..)]
        files: Vec<PathBuf>,
        /// Monthly run volume per agent.
        #[arg(long, default_value = "50000")]
        runs: u32,
        /// Cost per 1K input tokens in USD. Overrides --provider.
        #[arg(long)]
        input_cost: Option<f64>,
        /// Cost per 1K output tokens in USD. Overrides --provider.
        #[arg(long)]
        output_cost: Option<f64>,
        /// Provider preset (sets input/output costs automatically).
        #[arg(long, default_value = "anthropic-claude-3-5-sonnet")]
        provider: ProviderArg,
        /// Output format.
        #[arg(short, long, default_value = "markdown")]
        format: OutputFormat,
    },

    /// Simulate removing or merging steps and project the TAS/token impact (E-02).
    Simulate {
        /// Trace file to simulate.
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Comma-separated step IDs to remove (e.g. 3,8,9).
        #[arg(long, value_delimiter = ',')]
        remove: Vec<u32>,
        /// Comma-separated pair of step IDs to merge (e.g. 6,7).
        #[arg(long, value_delimiter = ',')]
        merge: Vec<u32>,
        /// Output format.
        #[arg(short, long, default_value = "markdown")]
        format: OutputFormat,
    },

    /// Apply safe fix patches from an audit JSON onto a target prompt file.
    ///
    /// By default only "safe" patches (system_prompt-only, non-functional)
    /// are applied: hedge_reduction, verbosity_reduction, caveman_prompt_insert,
    /// reformulation_guard. Pass `--all` to apply every fix in the file.
    ///
    /// The input JSON may be either a raw `[Fix, ...]` array or a full audit
    /// report (output of `tracerazor audit --format json`).
    Apply {
        /// Path to fixes JSON (audit report or raw fix array).
        #[arg(value_name = "FIXES")]
        fixes: PathBuf,
        /// Target file to append patches to (e.g. system_prompt.txt).
        #[arg(long, value_name = "FILE")]
        to: PathBuf,
        /// Apply every fix type, not just the safe subset.
        #[arg(long, default_value = "false")]
        all: bool,
        /// Preview the patches without writing to disk.
        #[arg(long, default_value = "false")]
        dry_run: bool,
    },

    /// Benchmark actual savings between a before and after trace.
    ///
    /// Reports measured token and TAS deltas and — if the fixes JSON from the
    /// baseline audit is supplied — compares those measured savings against the
    /// fixes' `estimated_token_savings` so you can validate the recommendation.
    Bench {
        /// Baseline trace captured before fixes were applied.
        #[arg(long, value_name = "FILE")]
        before: PathBuf,
        /// Target trace captured after fixes were applied.
        #[arg(long, value_name = "FILE")]
        after: PathBuf,
        /// Optional fixes JSON from the baseline audit (for estimated-vs-actual).
        #[arg(long, value_name = "FIXES")]
        fixes: Option<PathBuf>,
        /// Output format.
        #[arg(short, long, default_value = "markdown")]
        format: OutputFormat,
    },

    /// Rewrite the agent's system prompt using an LLM to eliminate detected waste.
    ///
    /// Audits the trace, identifies the top waste patterns, then iteratively
    /// asks the configured LLM to produce a tighter system prompt.  After each
    /// iteration the simulator projects the TAS improvement; the loop stops
    /// early when the target is met or the iteration cap is reached.
    ///
    /// Requires LLM credentials — see `tracerazor-semantic` docs for env vars:
    ///   OPENAI_API_KEY  /  ANTHROPIC_API_KEY  /  TRACERAZOR_LLM_*
    Optimize {
        /// Trace file to optimise.
        #[arg(value_name = "TRACE")]
        file: PathBuf,
        /// Existing system-prompt file to rewrite. If omitted a prompt is
        /// generated from scratch based on the trace's detected issues.
        #[arg(long, value_name = "FILE")]
        system_prompt: Option<PathBuf>,
        /// Write the optimised prompt to this file (stdout if omitted).
        #[arg(long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// Maximum optimisation iterations (each calls the LLM once).
        #[arg(long, default_value = "3")]
        iterations: u8,
        /// Stop early once the projected TAS reaches this score.
        #[arg(long, default_value = "85.0")]
        target_tas: f64,
        /// Output format.
        #[arg(short, long, default_value = "markdown")]
        format: OutputFormat,
    },

    /// Export a report to an observability platform or webhook (E-07).
    Export {
        /// Trace file to audit and export.
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// OTEL collector endpoint (e.g. http://localhost:4317).
        #[arg(long)]
        otel: Option<String>,
        /// Webhook URL (receives a JSON POST with the full report).
        #[arg(long)]
        webhook: Option<String>,
        /// Also print the report locally.
        #[arg(long, default_value = "false")]
        print: bool,
        /// Output format for local print.
        #[arg(short, long, default_value = "markdown")]
        format: OutputFormat,
    },
}

#[derive(ValueEnum, Clone, Debug)]
enum OutputFormat {
    Markdown,
    Json,
}

#[derive(ValueEnum, Clone, Debug)]
enum InputFormat {
    Auto,
    Raw,
    Langsmith,
    Otel,
}

#[derive(ValueEnum, Clone, Debug)]
enum ProviderArg {
    #[value(name = "openai-gpt4o")]
    OpenAiGpt4o,
    #[value(name = "openai-gpt4o-mini")]
    OpenAiGpt4oMini,
    #[value(name = "anthropic-claude-3-5-sonnet")]
    AnthropicClaude35Sonnet,
    #[value(name = "anthropic-claude-3-haiku")]
    AnthropicClaude3Haiku,
    #[value(name = "google-gemini-1-5-flash")]
    GoogleGemini15Flash,
}

impl From<ProviderArg> for ProviderPreset {
    fn from(p: ProviderArg) -> Self {
        match p {
            ProviderArg::OpenAiGpt4o => ProviderPreset::OpenAiGpt4o,
            ProviderArg::OpenAiGpt4oMini => ProviderPreset::OpenAiGpt4oMini,
            ProviderArg::AnthropicClaude35Sonnet => ProviderPreset::AnthropicClaude35Sonnet,
            ProviderArg::AnthropicClaude3Haiku => ProviderPreset::AnthropicClaude3Haiku,
            ProviderArg::GoogleGemini15Flash => ProviderPreset::GoogleGemini15Flash,
        }
    }
}

impl From<InputFormat> for TraceFormat {
    fn from(f: InputFormat) -> Self {
        match f {
            InputFormat::Auto => TraceFormat::Auto,
            InputFormat::Raw => TraceFormat::RawJson,
            InputFormat::Langsmith => TraceFormat::LangSmith,
            InputFormat::Otel => TraceFormat::Otel,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let cli = Cli::parse();

    match cli.command {
        Commands::Audit { file, format, threshold, trace_format, cost_per_million, store, enhanced } => {
            cmd_audit(file, format, threshold, trace_format, cost_per_million, store, enhanced).await?;
        }
        Commands::List { agent } => {
            cmd_list(agent).await?;
        }
        Commands::Compare { baseline, target, format, regression_threshold } => {
            cmd_compare(baseline, target, format, regression_threshold).await?;
        }
        Commands::Cost { files, runs, input_cost, output_cost, provider, format } => {
            cmd_cost(files, runs, input_cost, output_cost, provider, format).await?;
        }
        Commands::Simulate { file, remove, merge, format } => {
            cmd_simulate(file, remove, merge, format).await?;
        }
        Commands::Apply { fixes, to, all, dry_run } => {
            cmd_apply(fixes, to, all, dry_run).await?;
        }
        Commands::Bench { before, after, fixes, format } => {
            cmd_bench(before, after, fixes, format).await?;
        }
        Commands::Optimize { file, system_prompt, output, iterations, target_tas, format } => {
            cmd_optimize(file, system_prompt, output, iterations, target_tas, format).await?;
        }
        Commands::Export { file, otel, webhook, print, format } => {
            cmd_export(file, otel, webhook, print, format).await?;
        }
    }

    Ok(())
}

// ── audit ─────────────────────────────────────────────────────────────────────

async fn cmd_audit(
    file: PathBuf,
    format: OutputFormat,
    threshold: Option<f64>,
    trace_format: InputFormat,
    cost_per_million: f64,
    do_store: bool,
    enhanced: bool,
) -> Result<()> {
    let data = std::fs::read_to_string(&file)
        .with_context(|| format!("Cannot read file: {}", file.display()))?;

    let mut trace = ingest_parse(&data, trace_format.into())
        .with_context(|| format!("Failed to parse trace: {}", file.display()))?;

    if !is_analysable(&trace) {
        eprintln!(
            "Notice: Trace '{}' has {} steps (minimum {} required).",
            trace.trace_id,
            trace.steps.len(),
            MIN_TRACE_STEPS
        );
        return Ok(());
    }

    let store = open_store().await;

    let mut config = ScoringConfig {
        cost_per_million_tokens: cost_per_million,
        ..Default::default()
    };
    if let Some(t) = threshold {
        config.threshold = t;
    }
    if let Ok(Some(baseline)) = store.baseline_tokens(&trace.agent_name).await {
        config.baseline_tokens = Some(baseline);
    }
    if let Ok(Some(median)) = store.historical_median_steps(&trace.agent_name).await {
        config.historical_median_steps = Some(median);
    }
    if let Ok(sequences) = store.historical_sequences(&trace.agent_name).await {
        config.historical_sequences = sequences;
    }

    let mut report = if enhanced {
        // Build embedding cache from all step texts in one batched call.
        let texts: Vec<String> = trace.steps.iter().map(|s| s.content.clone()).collect();
        if tracerazor_semantic::LlmConfig::from_env().is_none() {
            eprintln!(
                "Warning: --enhanced found no LLM credentials \
                 (OPENAI_API_KEY / ANTHROPIC_API_KEY / TRACERAZOR_LLM_*). \
                 Falling back to BoW similarity."
            );
        }
        let sim_fn = tracerazor_semantic::embedding_similarity_fn(texts).await;
        tracerazor_core::analyse(&mut trace, sim_fn, &config)?
    } else {
        let sim_fn = default_similarity_fn();
        tracerazor_core::analyse(&mut trace, sim_fn, &config)?
    };

    // Detect anomalies against historical baseline (E-04) — all 8 metrics + TAS.
    if let Ok(anomalies) = store.detect_all_anomalies(&trace.agent_name, &report).await {
        report.anomalies = anomalies;
    }

    match format {
        OutputFormat::Markdown => println!("{}", report.to_markdown()),
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
    }

    if do_store {
        store.save_trace(&trace, Some(&report)).await?;
    }

    if !report.score.passes_threshold {
        eprintln!(
            "FAIL: TAS {:.1} is below threshold {:.1}",
            report.score.score, config.threshold
        );
        std::process::exit(1);
    }

    Ok(())
}

// ── list ──────────────────────────────────────────────────────────────────────

async fn cmd_list(agent_filter: Option<String>) -> Result<()> {
    let store = open_store().await;
    let summaries = store.list_traces().await?;

    if summaries.is_empty() {
        println!("No traces stored in this session.");
        println!("Run `tracerazor audit <file>` to analyse and store a trace.");
        return Ok(());
    }

    let summaries: Vec<_> = summaries
        .into_iter()
        .filter(|s| {
            agent_filter
                .as_ref()
                .map(|a| s.agent_name.contains(a.as_str()))
                .unwrap_or(true)
        })
        .collect();

    println!(
        "{:<36} {:<22} {:<10} {:<8} TAS",
        "TRACE ID", "AGENT", "FRAMEWORK", "STEPS"
    );
    println!("{}", "-".repeat(90));
    for s in &summaries {
        println!(
            "{:<36} {:<22} {:<10} {:<8} {}",
            s.trace_id,
            s.agent_name,
            s.framework,
            s.total_steps,
            s.tas_score
                .map(|t| format!("{:.1} ({})", t, s.grade.as_deref().unwrap_or("?")))
                .unwrap_or("N/A".into())
        );
    }

    Ok(())
}

// ── compare ───────────────────────────────────────────────────────────────────

async fn cmd_compare(
    baseline: PathBuf,
    target: PathBuf,
    format: OutputFormat,
    regression_threshold: f64,
) -> Result<()> {
    let config = ScoringConfig::default();
    let sim_fn = default_similarity_fn();

    let mut baseline_trace = ingest_parse(
        &std::fs::read_to_string(&baseline)
            .with_context(|| format!("Cannot read {}", baseline.display()))?,
        TraceFormat::Auto,
    )?;
    let mut target_trace = ingest_parse(
        &std::fs::read_to_string(&target)
            .with_context(|| format!("Cannot read {}", target.display()))?,
        TraceFormat::Auto,
    )?;

    let baseline_report =
        tracerazor_core::analyse(&mut baseline_trace, default_similarity_fn(), &config)?;
    let target_report = tracerazor_core::analyse(&mut target_trace, sim_fn, &config)?;

    let tas_delta = target_report.score.score - baseline_report.score.score;
    let token_delta =
        target_report.total_tokens as i64 - baseline_report.total_tokens as i64;

    // Per-metric deltas.
    let srr_d = target_report.score.srr.normalised() - baseline_report.score.srr.normalised();
    let ldi_d = target_report.score.ldi.normalised() - baseline_report.score.ldi.normalised();
    let tca_d = target_report.score.tca.normalised() - baseline_report.score.tca.normalised();
    let tur_d = target_report.score.tur.normalised() - baseline_report.score.tur.normalised();
    let cce_d = target_report.score.cce.normalised() - baseline_report.score.cce.normalised();
    let rda_d = target_report.score.rda.normalised() - baseline_report.score.rda.normalised();
    let isr_d = target_report.score.isr.normalised() - baseline_report.score.isr.normalised();
    let dbo_d = target_report.score.dbo.normalised() - baseline_report.score.dbo.normalised();

    // Regression detection: any metric drop > threshold.
    let regressions: Vec<(&str, f64)> = [
        ("SRR", srr_d),
        ("LDI", ldi_d),
        ("TCA", tca_d),
        ("TUR", tur_d),
        ("CCE", cce_d),
        ("RDA", rda_d),
        ("ISR", isr_d),
        ("DBO", dbo_d),
    ]
    .into_iter()
    .filter(|(_, d)| *d * 100.0 < -regression_threshold)
    .collect();

    match format {
        OutputFormat::Markdown => {
            let sep = "-".repeat(60);
            println!("TRACERAZOR COMPARISON REPORT");
            println!("{sep}");
            println!(
                "Baseline: {} | TAS {:.1} [{}]",
                baseline_report.trace_id,
                baseline_report.score.score,
                baseline_report.score.grade
            );
            println!(
                "Target:   {} | TAS {:.1} [{}]",
                target_report.trace_id,
                target_report.score.score,
                target_report.score.grade
            );
            println!("{sep}");

            let tas_arrow = if tas_delta >= 0.0 { "▲" } else { "▼" };
            println!("TAS delta:    {tas_arrow} {:.1}", tas_delta.abs());
            let tok_arrow = if token_delta <= 0 { "▼" } else { "▲" };
            println!("Token delta:  {tok_arrow} {}", token_delta.abs());
            println!("{sep}");

            println!("METRIC BREAKDOWN (target − baseline)");
            println!("{:<6}  {:>10}  {:>10}  {:>10}", "Metric", "Baseline", "Target", "Delta");
            println!("{}", "-".repeat(44));
            print_metric_row("SRR", baseline_report.score.srr.normalised(), target_report.score.srr.normalised(), srr_d);
            print_metric_row("LDI", baseline_report.score.ldi.normalised(), target_report.score.ldi.normalised(), ldi_d);
            print_metric_row("TCA", baseline_report.score.tca.normalised(), target_report.score.tca.normalised(), tca_d);
            print_metric_row("TUR", baseline_report.score.tur.normalised(), target_report.score.tur.normalised(), tur_d);
            print_metric_row("CCE", baseline_report.score.cce.normalised(), target_report.score.cce.normalised(), cce_d);
            print_metric_row("RDA", baseline_report.score.rda.normalised(), target_report.score.rda.normalised(), rda_d);
            print_metric_row("ISR", baseline_report.score.isr.normalised(), target_report.score.isr.normalised(), isr_d);
            print_metric_row("DBO", baseline_report.score.dbo.normalised(), target_report.score.dbo.normalised(), dbo_d);
            println!("{sep}");

            if tas_delta > 0.0 {
                println!("RESULT: Target is MORE efficient (+{:.1} TAS)", tas_delta);
            } else if tas_delta < 0.0 {
                println!("RESULT: Target is LESS efficient ({:.1} TAS)", tas_delta);
            } else {
                println!("RESULT: No change in efficiency");
            }

            if !regressions.is_empty() {
                println!();
                println!("REGRESSIONS (> {:.0}% drop):", regression_threshold);
                for (metric, delta) in &regressions {
                    println!(
                        "  [REGRESSION] {metric}: {:.1}% drop — investigate this metric",
                        delta.abs() * 100.0
                    );
                }
            }
        }
        OutputFormat::Json => {
            let output = serde_json::json!({
                "baseline": {
                    "trace_id": baseline_report.trace_id,
                    "tas": baseline_report.score.score,
                    "grade": baseline_report.score.grade.to_string(),
                    "tokens": baseline_report.total_tokens
                },
                "target": {
                    "trace_id": target_report.trace_id,
                    "tas": target_report.score.score,
                    "grade": target_report.score.grade.to_string(),
                    "tokens": target_report.total_tokens
                },
                "delta": {
                    "tas": tas_delta,
                    "tokens": token_delta,
                    "srr": srr_d, "ldi": ldi_d, "tca": tca_d, "tur": tur_d,
                    "cce": cce_d, "rda": rda_d, "isr": isr_d, "dbo": dbo_d
                },
                "regressions": regressions.iter().map(|(m, d)| serde_json::json!({"metric": m, "delta": d})).collect::<Vec<_>>(),
                "regression_detected": !regressions.is_empty()
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
    }

    if !regressions.is_empty() {
        eprintln!(
            "FAIL: {} metric(s) regressed by more than {:.0}%",
            regressions.len(),
            regression_threshold
        );
        std::process::exit(1);
    }

    Ok(())
}

fn print_metric_row(name: &str, baseline: f64, target: f64, delta: f64) {
    let arrow = if delta > 0.01 {
        "▲"
    } else if delta < -0.01 {
        "▼"
    } else {
        "="
    };
    println!(
        "{:<6}  {:>10.3}  {:>10.3}  {}{:>+8.3}",
        name, baseline, target, arrow, delta
    );
}

// ── cost ──────────────────────────────────────────────────────────────────────

async fn cmd_cost(
    files: Vec<PathBuf>,
    runs: u32,
    input_cost: Option<f64>,
    output_cost: Option<f64>,
    provider: ProviderArg,
    format: OutputFormat,
) -> Result<()> {
    let cost_config = match (input_cost, output_cost) {
        (Some(inp), Some(out)) => CostConfig::custom(inp, out),
        _ => CostConfig::from_preset(provider.into()),
    };

    let config = ScoringConfig::default();
    let mut traces_data: Vec<(u32, u32, String)> = Vec::new();

    for file in &files {
        let data = std::fs::read_to_string(file)
            .with_context(|| format!("Cannot read {}", file.display()))?;
        let mut trace = ingest_parse(&data, TraceFormat::Auto)
            .with_context(|| format!("Failed to parse {}", file.display()))?;

        if !is_analysable(&trace) {
            eprintln!(
                "Notice: {} has too few steps for analysis.",
                file.display()
            );
            continue;
        }

        let sim_fn = default_similarity_fn();
        let report = tracerazor_core::analyse(&mut trace, sim_fn, &config)?;
        traces_data.push((
            report.total_tokens,
            report.savings.tokens_saved,
            trace.agent_name.clone(),
        ));
    }

    if traces_data.is_empty() {
        eprintln!("No analysable traces found.");
        return Ok(());
    }

    let pairs: Vec<(u32, u32)> = traces_data.iter().map(|(t, s, _)| (*t, *s)).collect();
    let projection = project_cost(&pairs, runs, &cost_config);

    match format {
        OutputFormat::Markdown => {
            let sep = "-".repeat(54);
            println!("TRACERAZOR COST PROJECTION");
            println!("{sep}");
            println!("Provider:  ${:.4}/1K in  ${:.4}/1K out",
                cost_config.cost_per_1k_input_usd, cost_config.cost_per_1k_output_usd);
            println!("Volume:    {:>10} runs/month", runs);
            println!("{sep}");
            for (i, (total, saved, agent)) in traces_data.iter().enumerate() {
                println!(
                    "  [{}] {} — {} tokens, {} saved ({:.0}% waste)",
                    i + 1,
                    agent,
                    total,
                    saved,
                    projection.per_agent[i].waste_pct
                );
            }
            println!("{sep}");
            println!("Current monthly:   ${:.2}", projection.current_monthly_usd);
            println!("Optimised monthly: ${:.2}", projection.optimised_monthly_usd);
            println!("Monthly savings:   ${:.2}", projection.savings_monthly_usd);
            println!("Annual savings:    ${:.2}", projection.savings_annual_usd);
            println!("Overall waste:     {:.1}%", projection.overall_waste_pct);
            if let Some(idx) = projection.worst_offender_index {
                println!(
                    "Worst offender:    {} ({:.0}% waste)",
                    traces_data[idx].2, projection.worst_offender_waste_pct
                );
            }
            println!("{sep}");
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&projection)?);
        }
    }

    Ok(())
}

// ── simulate ──────────────────────────────────────────────────────────────────

async fn cmd_simulate(
    file: PathBuf,
    remove: Vec<u32>,
    merge_flat: Vec<u32>,
    format: OutputFormat,
) -> Result<()> {
    let data = std::fs::read_to_string(&file)
        .with_context(|| format!("Cannot read {}", file.display()))?;
    let trace = ingest_parse(&data, TraceFormat::Auto)
        .with_context(|| format!("Failed to parse {}", file.display()))?;

    // Convert flat merge list [a, b, c, d] to pairs [(a,b), (c,d)].
    let merge: Vec<(u32, u32)> = merge_flat
        .chunks(2)
        .filter_map(|c| if c.len() == 2 { Some((c[0], c[1])) } else { None })
        .collect();

    if remove.is_empty() && merge.is_empty() {
        eprintln!("No mutations specified. Use --remove or --merge.");
        eprintln!("Example: tracerazor simulate trace.json --remove 3,8,9 --merge 6,7");
        return Ok(());
    }

    let spec = SimulationSpec { remove: remove.clone(), merge: merge.clone() };
    let config = ScoringConfig::default();
    let sim_fn = default_similarity_fn();
    let result = simulate(&trace, &spec, &config, sim_fn);

    match format {
        OutputFormat::Markdown => {
            let sep = "-".repeat(54);
            println!("TRACERAZOR SIMULATION");
            println!("{sep}");
            if !remove.is_empty() {
                println!(
                    "Remove steps:  {}",
                    remove.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
                );
            }
            if !merge.is_empty() {
                let pairs: Vec<String> =
                    merge.iter().map(|(a, b)| format!("{a}+{b}")).collect();
                println!("Merge pairs:   {}", pairs.join(", "));
            }
            println!("{sep}");
            println!(
                "TAS:           {:.1} → {:.1}  ({:+.1})",
                result.original_tas, result.projected_tas, result.tas_delta
            );
            println!(
                "Steps:         {} → {}",
                result.original_steps, result.projected_steps
            );
            println!(
                "Tokens:        {} → {}  ({:+})",
                result.original_tokens, result.projected_tokens, result.token_delta
            );
            println!("{sep}");
            println!("METRIC DELTAS (projected − original)");
            let d = &result.metric_deltas;
            for (name, val) in [
                ("SRR", d.srr),
                ("LDI", d.ldi),
                ("TCA", d.tca),
                ("TUR", d.tur),
                ("CCE", d.cce),
                ("RDA", d.rda),
                ("ISR", d.isr),
                ("DBO", d.dbo),
            ] {
                let arrow = if val > 0.005 { "▲" } else if val < -0.005 { "▼" } else { "=" };
                println!("  {:<6} {}{:+.3}", name, arrow, val);
            }
            println!("{sep}");
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }

    Ok(())
}

// ── apply ─────────────────────────────────────────────────────────────────────

/// Fix types that only patch the system prompt with non-functional changes.
///
/// These can be auto-applied without risk of breaking tool wiring or agent
/// control flow. Fixes like `tool_schema` and `termination_guard` are *not*
/// included because they alter behaviour and require human review.
fn is_safe_fix(fix: &Fix) -> bool {
    matches!(
        fix.fix_type,
        FixType::HedgeReduction
            | FixType::VerbosityReduction
            | FixType::CavemanPromptInsert
            | FixType::ReformulationGuard
    ) && fix.target == "system_prompt"
}

/// Load `[Fix, ...]` from either a raw fix array JSON file or a full audit
/// report JSON file (which has a top-level `fixes` field).
fn load_fixes(path: &PathBuf) -> Result<Vec<Fix>> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read fixes file: {}", path.display()))?;
    if let Ok(fixes) = serde_json::from_str::<Vec<Fix>>(&data) {
        return Ok(fixes);
    }
    let value: serde_json::Value = serde_json::from_str(&data)
        .with_context(|| format!("Invalid JSON in {}", path.display()))?;
    if let Some(arr) = value.get("fixes") {
        let fixes: Vec<Fix> = serde_json::from_value(arr.clone())
            .context("`fixes` field is not a Fix array")?;
        return Ok(fixes);
    }
    anyhow::bail!(
        "{} is neither a Fix array nor an audit report with a `fixes` field",
        path.display()
    )
}

async fn cmd_apply(
    fixes_path: PathBuf,
    target: PathBuf,
    all: bool,
    dry_run: bool,
) -> Result<()> {
    let fixes = load_fixes(&fixes_path)?;
    if fixes.is_empty() {
        println!("No fixes found in {}. Nothing to apply.", fixes_path.display());
        return Ok(());
    }

    let selected: Vec<&Fix> = if all {
        fixes.iter().collect()
    } else {
        fixes.iter().filter(|f| is_safe_fix(f)).collect()
    };

    if selected.is_empty() {
        println!(
            "No {} fixes found in {}. Use --all to apply non-safe fixes.",
            if all { "any" } else { "safe" },
            fixes_path.display()
        );
        return Ok(());
    }

    let total_savings: u32 = selected.iter().map(|f| f.estimated_token_savings).sum();

    let sep = "-".repeat(60);
    println!("TRACERAZOR APPLY");
    println!("{sep}");
    println!("Target:       {}", target.display());
    println!("Fixes file:   {}", fixes_path.display());
    println!(
        "Mode:         {}{}",
        if all { "all" } else { "safe-only" },
        if dry_run { " (dry-run)" } else { "" }
    );
    println!("Patches:      {} of {} in file", selected.len(), fixes.len());
    println!("Est. savings: {} tokens/run", total_savings);
    println!("{sep}");

    let mut appended = String::new();
    appended.push_str("\n\n# ── TraceRazor auto-applied patches ──\n");
    for (i, fix) in selected.iter().enumerate() {
        println!(
            "  [{}/{}] {} (~{} tokens)",
            i + 1,
            selected.len(),
            fix.fix_type,
            fix.estimated_token_savings
        );
        appended.push_str(&format!(
            "# {} (est. {} tokens/run)\n{}\n\n",
            fix.fix_type, fix.estimated_token_savings, fix.patch
        ));
    }

    if dry_run {
        println!("{sep}");
        println!("DRY RUN — patches below would be appended to {}:", target.display());
        println!("{sep}");
        println!("{appended}");
        return Ok(());
    }

    let existing = std::fs::read_to_string(&target).unwrap_or_default();
    let new_contents = format!("{existing}{appended}");
    std::fs::write(&target, new_contents)
        .with_context(|| format!("Cannot write to {}", target.display()))?;

    println!("{sep}");
    println!("Applied {} patch(es) to {}", selected.len(), target.display());
    println!(
        "Next step: re-run your agent, capture a new trace, then validate with:"
    );
    println!(
        "  tracerazor bench --before <old>.json --after <new>.json --fixes {}",
        fixes_path.display()
    );

    Ok(())
}

// ── bench ─────────────────────────────────────────────────────────────────────

async fn cmd_bench(
    before: PathBuf,
    after: PathBuf,
    fixes_path: Option<PathBuf>,
    format: OutputFormat,
) -> Result<()> {
    let config = ScoringConfig::default();

    let mut before_trace = ingest_parse(
        &std::fs::read_to_string(&before)
            .with_context(|| format!("Cannot read {}", before.display()))?,
        TraceFormat::Auto,
    )?;
    let mut after_trace = ingest_parse(
        &std::fs::read_to_string(&after)
            .with_context(|| format!("Cannot read {}", after.display()))?,
        TraceFormat::Auto,
    )?;

    let before_report =
        tracerazor_core::analyse(&mut before_trace, default_similarity_fn(), &config)?;
    let after_report =
        tracerazor_core::analyse(&mut after_trace, default_similarity_fn(), &config)?;

    let tokens_before = before_report.total_tokens as i64;
    let tokens_after = after_report.total_tokens as i64;
    let actual_tokens_saved = tokens_before - tokens_after;
    let pct_saved = if tokens_before > 0 {
        (actual_tokens_saved as f64 / tokens_before as f64) * 100.0
    } else {
        0.0
    };
    let tas_delta = after_report.score.score - before_report.score.score;

    let estimated: Option<u32> = match &fixes_path {
        Some(p) => Some(load_fixes(p)?.iter().map(|f| f.estimated_token_savings).sum()),
        None => None,
    };
    let accuracy_pct = estimated.and_then(|est| {
        if est == 0 {
            None
        } else {
            Some((actual_tokens_saved as f64 / est as f64) * 100.0)
        }
    });

    match format {
        OutputFormat::Markdown => {
            let sep = "-".repeat(60);
            println!("TRACERAZOR BENCHMARK");
            println!("{sep}");
            println!(
                "Before: {} | TAS {:.1} | {} tokens",
                before_report.trace_id, before_report.score.score, tokens_before
            );
            println!(
                "After:  {} | TAS {:.1} | {} tokens",
                after_report.trace_id, after_report.score.score, tokens_after
            );
            println!("{sep}");
            let tok_arrow = if actual_tokens_saved >= 0 { "▼" } else { "▲" };
            println!(
                "Tokens saved:  {} {} ({:+.1}%)",
                tok_arrow,
                actual_tokens_saved.abs(),
                -pct_saved
            );
            let tas_arrow = if tas_delta >= 0.0 { "▲" } else { "▼" };
            println!("TAS delta:     {} {:.1}", tas_arrow, tas_delta.abs());
            if let Some(est) = estimated {
                println!("{sep}");
                println!("Estimated savings: {est} tokens");
                println!("Measured savings:  {} tokens", actual_tokens_saved);
                if let Some(acc) = accuracy_pct {
                    let verdict = if (80.0..=120.0).contains(&acc) {
                        "MATCH"
                    } else if acc > 120.0 {
                        "UNDER-ESTIMATED"
                    } else if acc >= 0.0 {
                        "OVER-ESTIMATED"
                    } else {
                        "REGRESSION"
                    };
                    println!("Accuracy:          {:.0}% [{}]", acc, verdict);
                }
            }
            println!("{sep}");
            if actual_tokens_saved > 0 && tas_delta >= 0.0 {
                println!("RESULT: Fixes are working. Keep them.");
            } else if actual_tokens_saved > 0 && tas_delta < 0.0 {
                println!("RESULT: Tokens down, but TAS regressed. Review which metric dropped.");
            } else if actual_tokens_saved < 0 {
                println!("RESULT: After-trace uses MORE tokens. Revert the patches.");
            } else {
                println!("RESULT: No measurable change.");
            }
        }
        OutputFormat::Json => {
            let output = serde_json::json!({
                "before": {
                    "trace_id": before_report.trace_id,
                    "tas": before_report.score.score,
                    "tokens": tokens_before,
                },
                "after": {
                    "trace_id": after_report.trace_id,
                    "tas": after_report.score.score,
                    "tokens": tokens_after,
                },
                "actual_tokens_saved": actual_tokens_saved,
                "pct_tokens_saved": pct_saved,
                "tas_delta": tas_delta,
                "estimated_tokens_saved": estimated,
                "estimate_accuracy_pct": accuracy_pct,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
    }

    Ok(())
}

// ── optimize ──────────────────────────────────────────────────────────────────

async fn cmd_optimize(
    file: PathBuf,
    system_prompt_path: Option<PathBuf>,
    output_path: Option<PathBuf>,
    iterations: u8,
    target_tas: f64,
    format: OutputFormat,
) -> Result<()> {
    // ── 1. Audit the trace ───────────────────────────────────────────────────
    let data = std::fs::read_to_string(&file)
        .with_context(|| format!("Cannot read trace: {}", file.display()))?;
    let mut trace = ingest_parse(&data, tracerazor_ingest::TraceFormat::Auto)
        .with_context(|| format!("Failed to parse trace: {}", file.display()))?;

    if !is_analysable(&trace) {
        anyhow::bail!(
            "Trace '{}' has {} steps (minimum {} required for analysis).",
            trace.trace_id, trace.steps.len(), MIN_TRACE_STEPS
        );
    }

    let sim_fn = default_similarity_fn();
    let config = ScoringConfig::default();
    let report = tracerazor_core::analyse(&mut trace, sim_fn, &config)?;

    let original_tas = report.score.score;
    let original_tokens = report.total_tokens;

    // ── 2. Check if already optimal ─────────────────────────────────────────
    if original_tas >= target_tas {
        eprintln!(
            "TAS {:.1} already meets target {:.1}. Nothing to do.",
            original_tas, target_tas
        );
        return Ok(());
    }

    // ── 3. Require LLM credentials ──────────────────────────────────────────
    let llm = LlmConfig::from_env().ok_or_else(|| {
        anyhow::anyhow!(
            "No LLM credentials found.\n\
             Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or TRACERAZOR_LLM_* env vars.\n\
             Example: OPENAI_API_KEY=sk-... tracerazor optimize trace.json"
        )
    })?;

    // ── 4. Load existing system prompt (if any) ──────────────────────────────
    let current_prompt = match &system_prompt_path {
        Some(p) => std::fs::read_to_string(p)
            .with_context(|| format!("Cannot read system prompt: {}", p.display()))?,
        None => String::new(),
    };

    // ── 5. Build waste summary for the LLM ──────────────────────────────────
    let mut fixes_by_savings = report.fixes.clone();
    fixes_by_savings.sort_by(|a, b| b.estimated_token_savings.cmp(&a.estimated_token_savings));
    let waste_summary = build_waste_summary(&report, &fixes_by_savings);

    // ── 6. Derive simulation spec from the diff ──────────────────────────────
    // Steps marked Delete in the diff are candidates the optimizer can eliminate.
    let delete_ids: Vec<u32> = report
        .diff
        .iter()
        .filter(|d| matches!(d.action, tracerazor_core::report::DiffAction::Delete))
        .map(|d| d.step_id)
        .collect();

    // ── 7. Optimization loop ──────────────────────────────────────────────────
    let mut best_prompt = current_prompt.clone();
    let mut best_projected_tas = original_tas;
    let mut best_projected_tokens = original_tokens;
    let mut iteration_log: Vec<IterationRow> = Vec::new();

    eprintln!(
        "Optimizing '{}' (TAS {:.1} → target {:.1}) using {}…",
        trace.agent_name, original_tas, target_tas, llm.model
    );

    for i in 1..=iterations {
        eprint!("  Iteration {i}/{iterations} — calling LLM… ");

        let new_prompt = match ask_llm_to_optimize(
            &llm, &best_prompt, &waste_summary, &trace.agent_name,
            original_tas, report.total_tokens,
        ).await {
            Ok(p) => p,
            Err(e) => {
                eprintln!("FAILED ({e})");
                break;
            }
        };

        // Project improvement: simulate removing the wasteful steps.
        let spec = SimulationSpec { remove: delete_ids.clone(), merge: vec![] };
        let sim = simulate(&trace, &spec, &config, default_similarity_fn());
        let projected_tas = sim.projected_tas;
        let projected_tokens = sim.projected_tokens;
        let token_delta = sim.token_delta;

        eprintln!(
            "projected TAS {:.1} ({:+.1}), tokens {:+}",
            projected_tas, projected_tas - original_tas, token_delta
        );

        iteration_log.push(IterationRow {
            iteration: i,
            projected_tas,
            projected_tokens,
            token_delta,
        });

        // Keep the best prompt seen so far.
        if projected_tas > best_projected_tas {
            best_projected_tas = projected_tas;
            best_projected_tokens = projected_tokens;
        }
        best_prompt = new_prompt;

        if projected_tas >= target_tas {
            eprintln!("  Target reached — stopping early.");
            break;
        }
    }

    // ── 8. Write the optimized prompt ────────────────────────────────────────
    match &output_path {
        Some(p) => {
            std::fs::write(p, &best_prompt)
                .with_context(|| format!("Cannot write output: {}", p.display()))?;
            eprintln!("Wrote optimised prompt → {}", p.display());
        }
        None => {
            // Print raw prompt to stdout so it can be piped/redirected.
            println!("{best_prompt}");
        }
    }

    // ── 9. Print the summary report ──────────────────────────────────────────
    match format {
        OutputFormat::Markdown => {
            eprintln!("{}", render_optimize_markdown(
                &trace.agent_name, original_tas, original_tokens,
                best_projected_tas, best_projected_tokens, &iteration_log,
                &report.fixes,
            ));
        }
        OutputFormat::Json => {
            let out = serde_json::json!({
                "agent_name": trace.agent_name,
                "original_tas": original_tas,
                "original_tokens": original_tokens,
                "projected_tas": best_projected_tas,
                "projected_tokens": best_projected_tokens,
                "tas_delta": best_projected_tas - original_tas,
                "token_delta": best_projected_tokens as i64 - original_tokens as i64,
                "iterations": iteration_log.len(),
                "fixes_addressed": report.fixes.len(),
                "model": llm.model,
            });
            eprintln!("{}", serde_json::to_string_pretty(&out)?);
        }
    }

    Ok(())
}

struct IterationRow {
    iteration: u8,
    projected_tas: f64,
    projected_tokens: u32,
    token_delta: i64,
}

/// Build a structured waste summary the LLM can act on.
fn build_waste_summary(
    report: &tracerazor_core::report::TraceReport,
    fixes: &[Fix],
) -> String {
    use std::fmt::Write as FmtWrite;
    let mut s = String::new();

    let _ = writeln!(s, "Current TAS: {:.1}/100 ({})", report.score.score, report.score.grade);
    let _ = writeln!(s, "Total tokens: {}", report.total_tokens);
    let _ = writeln!(
        s, "Estimated waste: {} tokens ({:.0}%)",
        report.savings.tokens_saved,
        if report.total_tokens > 0 {
            report.savings.tokens_saved as f64 / report.total_tokens as f64 * 100.0
        } else { 0.0 }
    );
    let _ = writeln!(s, "\nTop waste patterns detected:");
    for fix in fixes.iter().take(5) {
        let _ = writeln!(
            s, "  - [{}] {} (est. {} tokens/run)",
            fix.fix_type, fix.patch, fix.estimated_token_savings
        );
    }
    s
}

/// Prompt the LLM to generate an optimised system prompt.
async fn ask_llm_to_optimize(
    llm: &LlmConfig,
    current_prompt: &str,
    waste_summary: &str,
    agent_name: &str,
    original_tas: f64,
    total_tokens: u32,
) -> Result<String> {
    let system = "\
You are an expert AI agent system-prompt optimizer. \
Your sole job is to rewrite a system prompt so that the agent \
produces shorter, more direct reasoning traces with less token waste — \
without removing any existing capabilities or business logic.\n\
Rules:\n\
- Keep all tool descriptions and business constraints verbatim.\n\
- Eliminate hedge phrases, preambles, and unnecessary meta-commentary.\n\
- Add an EFFICIENCY RULES section with 3-5 concise bullet directives.\n\
- Return ONLY the rewritten system prompt text — no explanation, no markdown fences.";

    let user = format!(
        "## Agent: {agent_name}\n\
         ## Efficiency audit\n\
         {waste_summary}\n\
         ## Current system prompt\n\
         {current}\n\
         ## Task\n\
         Rewrite the system prompt above to eliminate the detected waste patterns. \
         The current TAS is {original_tas:.1}/100 with {total_tokens} tokens. \
         Target: reduce token waste by at least 30% while keeping all capabilities.",
        current = if current_prompt.is_empty() {
            "(no system prompt — generate one from scratch based on the waste patterns)"
        } else {
            current_prompt
        },
    );

    llm.complete(system, &user).await
}

fn render_optimize_markdown(
    agent_name: &str,
    original_tas: f64,
    original_tokens: u32,
    projected_tas: f64,
    projected_tokens: u32,
    iterations: &[IterationRow],
    fixes: &[Fix],
) -> String {
    use std::fmt::Write as FmtWrite;
    let mut s = String::new();
    let _ = writeln!(s, "# ⚡ TraceRazor Optimize — {agent_name}");
    let _ = writeln!(s);
    let _ = writeln!(s, "| | Before | After (projected) | Delta |");
    let _ = writeln!(s, "|---|---:|---:|---:|");
    let _ = writeln!(
        s, "| TAS | {:.1} | {:.1} | {:+.1} |",
        original_tas, projected_tas, projected_tas - original_tas
    );
    let _ = writeln!(
        s, "| Tokens | {} | {} | {:+} |",
        original_tokens, projected_tokens,
        projected_tokens as i64 - original_tokens as i64
    );
    let waste_pct = if original_tokens > 0 {
        (original_tokens - projected_tokens) as f64 / original_tokens as f64 * 100.0
    } else { 0.0 };
    let _ = writeln!(s, "| Est. waste removed | — | — | {:.0}% |", waste_pct);
    let _ = writeln!(s);
    let _ = writeln!(s, "## Iteration log");
    let _ = writeln!(s);
    let _ = writeln!(s, "| Iter | Projected TAS | Projected tokens | Token delta |");
    let _ = writeln!(s, "|---:|---:|---:|---:|");
    for row in iterations {
        let _ = writeln!(
            s, "| {} | {:.1} | {} | {:+} |",
            row.iteration, row.projected_tas, row.projected_tokens, row.token_delta
        );
    }
    let _ = writeln!(s);
    let _ = writeln!(s, "## Waste patterns addressed ({})", fixes.len());
    let _ = writeln!(s);
    for fix in fixes {
        let _ = writeln!(
            s, "- **{}**: {} *(est. {} tokens/run)*",
            fix.fix_type, fix.patch, fix.estimated_token_savings
        );
    }
    s
}

// ── export ────────────────────────────────────────────────────────────────────

async fn cmd_export(
    file: PathBuf,
    otel_endpoint: Option<String>,
    webhook_url: Option<String>,
    print_report: bool,
    format: OutputFormat,
) -> Result<()> {
    if otel_endpoint.is_none() && webhook_url.is_none() {
        eprintln!("Specify at least one export target: --otel <url> or --webhook <url>");
        eprintln!("Example: tracerazor export trace.json --otel http://localhost:4317");
        return Ok(());
    }

    let data = std::fs::read_to_string(&file)
        .with_context(|| format!("Cannot read {}", file.display()))?;
    let mut trace = ingest_parse(&data, TraceFormat::Auto)?;

    let config = ScoringConfig::default();
    let sim_fn = default_similarity_fn();
    let report = tracerazor_core::analyse(&mut trace, sim_fn, &config)?;

    if print_report {
        match format {
            OutputFormat::Markdown => println!("{}", report.to_markdown()),
            OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
        }
    }

    // ── OTEL export ───────────────────────────────────────────────────────────
    if let Some(ref endpoint) = otel_endpoint {
        export_otel(&report, &trace, endpoint).await?;
        eprintln!("Exported OTEL spans to {endpoint}");
    }

    // ── Webhook export ────────────────────────────────────────────────────────
    if let Some(ref url) = webhook_url {
        export_webhook(&report, url).await?;
        eprintln!("Posted report to {url}");
    }

    Ok(())
}

/// POST a JSON report payload to a webhook URL.
async fn export_webhook(
    report: &tracerazor_core::report::TraceReport,
    url: &str,
) -> Result<()> {
    let payload = serde_json::json!({
        "source": "tracerazor",
        "trace_id": report.trace_id,
        "agent_name": report.agent_name,
        "tas_score": report.score.score,
        "grade": report.score.grade.to_string(),
        "tokens_saved": report.savings.tokens_saved,
        "summary": report.summary,
        "anomalies": report.anomalies,
    });

    let client = reqwest::Client::new();
    client
        .post(url)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .with_context(|| format!("Webhook POST to {url} failed"))?;

    Ok(())
}

/// Emit TraceRazor metrics as OTEL span attributes.
///
/// Posts to the OTEL HTTP/JSON endpoint (`/v1/traces`).
/// Each TAS metric is emitted as a span attribute:
///   tracerazor.tas_score, tracerazor.srr, tracerazor.ldi, etc.
async fn export_otel(
    report: &tracerazor_core::report::TraceReport,
    trace: &tracerazor_core::types::Trace,
    endpoint: &str,
) -> Result<()> {
    let span_id = format!("{:016x}", report.score.score as u64 * 100);
    let trace_id_hex = report
        .trace_id
        .chars()
        .filter(|c| c.is_ascii_hexdigit())
        .take(32)
        .collect::<String>();
    let trace_id_padded = format!("{:0>32}", trace_id_hex);

    let attributes = serde_json::json!([
        {"key": "tracerazor.tas_score",   "value": {"doubleValue": report.score.score}},
        {"key": "tracerazor.grade",        "value": {"stringValue": report.score.grade.to_string()}},
        {"key": "tracerazor.srr",          "value": {"doubleValue": report.score.srr.score}},
        {"key": "tracerazor.ldi",          "value": {"doubleValue": report.score.ldi.score}},
        {"key": "tracerazor.tca",          "value": {"doubleValue": report.score.tca.score}},
        {"key": "tracerazor.tur",          "value": {"doubleValue": report.score.tur.score}},
        {"key": "tracerazor.cce",          "value": {"doubleValue": report.score.cce.score}},
        {"key": "tracerazor.rda",          "value": {"doubleValue": report.score.rda.score}},
        {"key": "tracerazor.isr",          "value": {"doubleValue": report.score.isr.score}},
        {"key": "tracerazor.dbo",          "value": {"doubleValue": report.score.dbo.score}},
        {"key": "tracerazor.tokens_saved", "value": {"intValue": report.savings.tokens_saved}},
        {"key": "tracerazor.agent_name",   "value": {"stringValue": trace.agent_name.clone()}},
        {"key": "tracerazor.anomaly",      "value": {"boolValue": !report.anomalies.is_empty()}},
    ]);

    let payload = serde_json::json!({
        "resourceSpans": [{
            "resource": {
                "attributes": [{"key": "service.name", "value": {"stringValue": "tracerazor"}}]
            },
            "scopeSpans": [{
                "spans": [{
                    "traceId": trace_id_padded,
                    "spanId": span_id,
                    "name": format!("tracerazor.audit.{}", trace.agent_name),
                    "kind": 1,
                    "attributes": attributes
                }]
            }]
        }]
    });

    let otel_url = format!(
        "{}/v1/traces",
        endpoint.trim_end_matches('/')
    );

    let client = reqwest::Client::new();
    client
        .post(&otel_url)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .with_context(|| format!("OTEL export to {otel_url} failed"))?;

    Ok(())
}
