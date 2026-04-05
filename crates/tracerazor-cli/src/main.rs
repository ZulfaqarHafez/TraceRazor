use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracerazor_core::{
    cost::{CostConfig, ProviderPreset, project_cost},
    is_analysable,
    scoring::ScoringConfig,
    simulate::{SimulationSpec, simulate},
    types::MIN_TRACE_STEPS,
};
use tracerazor_ingest::{TraceFormat, parse as ingest_parse};
use tracerazor_semantic::default_similarity_fn;
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

        /// Enable enhanced semantic analysis using OpenAI embeddings.
        /// Significantly improves SRR and ISR accuracy by replacing bag-of-words
        /// with dense sentence embeddings. Requires OPENAI_API_KEY env var.
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
        if std::env::var("OPENAI_API_KEY").is_err() {
            eprintln!("Warning: --enhanced requires OPENAI_API_KEY. Falling back to BoW.");
        }
        let sim_fn = tracerazor_semantic::openai_similarity_fn(texts).await;
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
