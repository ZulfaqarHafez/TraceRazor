use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracerazor_core::{is_analysable, scoring::ScoringConfig, types::MIN_TRACE_STEPS};
use tracerazor_ingest::{TraceFormat, parse as ingest_parse};
use tracerazor_semantic::default_similarity_fn;
#[allow(unused_imports)]
use tracerazor_core::report::TraceReport;
use tracerazor_store::TraceStore;

/// TraceRazor — Agentic Reasoning Path Efficiency Auditor
///
/// Analyses completed AI agent reasoning traces and produces actionable efficiency
/// scores, redundancy reports, and optimal-path recommendations.
#[derive(Parser)]
#[command(
    name = "tracerazor",
    version,
    author = "Zulfaqar Hafez",
    about = "Lighthouse score for AI agents. Audit reasoning traces and eliminate token waste.",
    long_about = None
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

        /// Minimum TAS score to pass. Returns non-zero exit code if below threshold.
        /// Useful for CI/CD gating.
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

        /// Show verbose output including low-confidence redundancy flags (0.75–0.84).
        #[arg(short, long)]
        verbose: bool,
    },
    /// List all stored traces.
    List {
        /// Filter by agent name.
        #[arg(short, long)]
        agent: Option<String>,
    },
    /// Compare two traces: show TAS delta and diff.
    Compare {
        /// Baseline trace file.
        #[arg(value_name = "BASELINE")]
        baseline: PathBuf,
        /// Target trace file (typically the new version).
        #[arg(value_name = "TARGET")]
        target: PathBuf,
        /// Output format.
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
    let cli = Cli::parse();

    match cli.command {
        Commands::Audit {
            file,
            format,
            threshold,
            trace_format,
            cost_per_million,
            store,
            verbose: _,
        } => {
            cmd_audit(file, format, threshold, trace_format, cost_per_million, store).await?;
        }
        Commands::List { agent } => {
            cmd_list(agent).await?;
        }
        Commands::Compare {
            baseline,
            target,
            format,
        } => {
            cmd_compare(baseline, target, format).await?;
        }
    }

    Ok(())
}

async fn cmd_audit(
    file: PathBuf,
    format: OutputFormat,
    threshold: Option<f64>,
    trace_format: InputFormat,
    cost_per_million: f64,
    do_store: bool,
) -> Result<()> {
    // Read and parse the trace file.
    let data = std::fs::read_to_string(&file)
        .with_context(|| format!("Cannot read file: {}", file.display()))?;

    let mut trace = ingest_parse(&data, trace_format.into())
        .with_context(|| format!("Failed to parse trace: {}", file.display()))?;

    // Check minimum step requirement.
    if !is_analysable(&trace) {
        eprintln!(
            "Notice: Trace '{}' has {} steps (minimum {} required for analysis).",
            trace.trace_id,
            trace.steps.len(),
            MIN_TRACE_STEPS
        );
        eprintln!(
            "Token usage: {} tokens. Trace stored for historical reference.",
            trace.effective_total_tokens()
        );
        return Ok(());
    }

    // Build scoring config.
    let mut config = ScoringConfig::default();
    config.cost_per_million_tokens = cost_per_million;
    if let Some(t) = threshold {
        config.threshold = t;
    }

    // Get baseline from store if available.
    let store = TraceStore::connect_mem().await?;
    if let Ok(Some(baseline)) = store.baseline_tokens(&trace.agent_name).await {
        config.baseline_tokens = Some(baseline);
    }

    // Run analysis.
    let sim_fn = default_similarity_fn();
    let report = tracerazor_core::analyse(&mut trace, sim_fn, &config)
        .context("Analysis failed")?;

    // Output the report.
    match format {
        OutputFormat::Markdown => {
            println!("{}", report.to_markdown());
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&report)?;
            println!("{}", json);
        }
    }

    // Persist to store.
    if do_store {
        store.save_trace(&trace, Some(&report)).await?;
    }

    // CI/CD gating: exit non-zero if below threshold.
    if !report.score.passes_threshold {
        eprintln!(
            "FAIL: TAS {:.1} is below threshold {:.1}",
            report.score.score, config.threshold
        );
        std::process::exit(1);
    }

    Ok(())
}

async fn cmd_list(agent_filter: Option<String>) -> Result<()> {
    let store = TraceStore::connect_mem().await?;
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

    println!("{:<36} {:<20} {:<10} {:<8} {:<8}", "TRACE ID", "AGENT", "FRAMEWORK", "STEPS", "TAS");
    println!("{}", "-".repeat(90));
    for s in &summaries {
        println!(
            "{:<36} {:<20} {:<10} {:<8} {}",
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

async fn cmd_compare(baseline: PathBuf, target: PathBuf, format: OutputFormat) -> Result<()> {
    let config = ScoringConfig::default();

    let parse_and_analyse = |path: &PathBuf| -> Result<tracerazor_core::report::TraceReport> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("Cannot read file: {}", path.display()))?;
        let mut trace = ingest_parse(&data, TraceFormat::Auto)
            .with_context(|| format!("Parse error: {}", path.display()))?;
        tracerazor_core::analyse(&mut trace, default_similarity_fn(), &config)
    };

    let baseline_report = parse_and_analyse(&baseline)?;
    let target_report = parse_and_analyse(&target)?;

    let delta = target_report.score.score - baseline_report.score.score;
    let token_delta = target_report.total_tokens as i64 - baseline_report.total_tokens as i64;

    match format {
        OutputFormat::Markdown => {
            println!("TRACERAZOR COMPARISON REPORT");
            println!("{}", "-".repeat(54));
            println!(
                "Baseline: {} | TAS: {:.1} [{}]",
                baseline_report.trace_id,
                baseline_report.score.score,
                baseline_report.score.grade
            );
            println!(
                "Target:   {} | TAS: {:.1} [{}]",
                target_report.trace_id,
                target_report.score.score,
                target_report.score.grade
            );
            println!("{}", "-".repeat(54));
            let arrow = if delta >= 0.0 { "▲" } else { "▼" };
            println!("TAS delta:    {} {:.1}", arrow, delta.abs());
            let token_arrow = if token_delta <= 0 { "▼" } else { "▲" };
            println!("Token delta:  {} {}", token_arrow, token_delta.abs());
            println!("{}", "-".repeat(54));
            if delta > 0.0 {
                println!("RESULT: Target is MORE efficient than baseline (+{:.1} TAS)", delta);
            } else if delta < 0.0 {
                println!("RESULT: Target is LESS efficient than baseline ({:.1} TAS)", delta);
            } else {
                println!("RESULT: No change in efficiency");
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
                    "tas": delta,
                    "tokens": token_delta
                }
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
    }

    Ok(())
}
