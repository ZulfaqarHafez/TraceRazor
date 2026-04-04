use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracerazor_core::{is_analysable, scoring::ScoringConfig, types::MIN_TRACE_STEPS};
use tracerazor_ingest::{TraceFormat, parse as ingest_parse};
use tracerazor_semantic::{default_similarity_fn, llm};
use tracerazor_store::TraceStore;

/// TraceRazor — Agentic Reasoning Path Efficiency Auditor
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

        /// Enable Phase 2 semantic analysis (requires OPENAI_API_KEY).
        /// Activates RDA, DBO metrics and OpenAI embeddings for accurate SRR/ISR.
        #[arg(long)]
        semantic: bool,

        /// Save trace and report to the store for historical benchmarking.
        #[arg(long, default_value = "true")]
        store: bool,
    },
    /// List all stored traces in the current session.
    List {
        /// Filter by agent name.
        #[arg(short, long)]
        agent: Option<String>,
    },
    /// Compare two traces: TAS delta and per-metric diff.
    Compare {
        /// Baseline trace file.
        #[arg(value_name = "BASELINE")]
        baseline: PathBuf,
        /// Target trace file.
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
    // Load .env file if present (silently ignore if not found).
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();

    match cli.command {
        Commands::Audit { file, format, threshold, trace_format, cost_per_million, semantic, store } => {
            cmd_audit(file, format, threshold, trace_format, cost_per_million, semantic, store).await?;
        }
        Commands::List { agent } => {
            cmd_list(agent).await?;
        }
        Commands::Compare { baseline, target, format } => {
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
    semantic: bool,
    do_store: bool,
) -> Result<()> {
    let data = std::fs::read_to_string(&file)
        .with_context(|| format!("Cannot read file: {}", file.display()))?;

    let mut trace = ingest_parse(&data, trace_format.into())
        .with_context(|| format!("Failed to parse trace: {}", file.display()))?;

    if !is_analysable(&trace) {
        eprintln!(
            "Notice: Trace '{}' has {} steps (minimum {} required for analysis).",
            trace.trace_id, trace.steps.len(), MIN_TRACE_STEPS
        );
        eprintln!("Token usage: {} tokens. Trace stored for historical reference.", trace.effective_total_tokens());
        return Ok(());
    }

    let mut config = ScoringConfig { cost_per_million_tokens: cost_per_million, ..Default::default() };
    if let Some(t) = threshold {
        config.threshold = t;
    }

    let store = TraceStore::connect_mem().await?;
    if let Ok(Some(baseline)) = store.baseline_tokens(&trace.agent_name).await {
        config.baseline_tokens = Some(baseline);
    }

    let report = if semantic && std::env::var("OPENAI_API_KEY").is_ok() {
        // Phase 2: OpenAI embeddings + LLM metrics.
        let texts: Vec<String> = trace.steps.iter().map(|s| s.semantic_content()).collect();
        let sim_fn = tracerazor_semantic::openai_similarity_fn(texts).await;

        let llm_fn = |system: String, user: String| async move {
            llm::complete(&system, &user).await
        };

        tracerazor_core::analyse_full(&mut trace, |a, b| sim_fn(a, b), llm_fn, &config).await?
    } else {
        if semantic {
            eprintln!("Warning: --semantic requires OPENAI_API_KEY. Running structural analysis only.");
        }
        // Phase 1: structural analysis only.
        let sim_fn = default_similarity_fn();
        tracerazor_core::analyse(&mut trace, sim_fn, &config)?
    };

    match format {
        OutputFormat::Markdown => println!("{}", report.to_markdown()),
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
    }

    if do_store {
        store.save_trace(&trace, Some(&report)).await?;
    }

    if !report.score.passes_threshold {
        eprintln!("FAIL: TAS {:.1} is below threshold {:.1}", report.score.score, config.threshold);
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
        .filter(|s| agent_filter.as_ref().map(|a| s.agent_name.contains(a.as_str())).unwrap_or(true))
        .collect();

    println!("{:<36} {:<22} {:<10} {:<8} TAS", "TRACE ID", "AGENT", "FRAMEWORK", "STEPS");
    println!("{}", "-".repeat(90));
    for s in &summaries {
        println!(
            "{:<36} {:<22} {:<10} {:<8} {}",
            s.trace_id, s.agent_name, s.framework, s.total_steps,
            s.tas_score.map(|t| format!("{:.1} ({})", t, s.grade.as_deref().unwrap_or("?"))).unwrap_or("N/A".into())
        );
    }

    Ok(())
}

async fn cmd_compare(baseline: PathBuf, target: PathBuf, format: OutputFormat) -> Result<()> {
    let config = ScoringConfig::default();
    let sim_fn = default_similarity_fn();

    let mut baseline_trace = ingest_parse(
        &std::fs::read_to_string(&baseline)?,
        TraceFormat::Auto,
    )?;
    let mut target_trace = ingest_parse(
        &std::fs::read_to_string(&target)?,
        TraceFormat::Auto,
    )?;

    let baseline_report = tracerazor_core::analyse(&mut baseline_trace, default_similarity_fn(), &config)?;
    let target_report = tracerazor_core::analyse(&mut target_trace, sim_fn, &config)?;

    let delta = target_report.score.score - baseline_report.score.score;
    let token_delta = target_report.total_tokens as i64 - baseline_report.total_tokens as i64;

    match format {
        OutputFormat::Markdown => {
            println!("TRACERAZOR COMPARISON REPORT");
            println!("{}", "-".repeat(54));
            println!("Baseline: {} | TAS: {:.1} [{}]", baseline_report.trace_id, baseline_report.score.score, baseline_report.score.grade);
            println!("Target:   {} | TAS: {:.1} [{}]", target_report.trace_id, target_report.score.score, target_report.score.grade);
            println!("{}", "-".repeat(54));
            let arrow = if delta >= 0.0 { "▲" } else { "▼" };
            println!("TAS delta:    {} {:.1}", arrow, delta.abs());
            let token_arrow = if token_delta <= 0 { "▼" } else { "▲" };
            println!("Token delta:  {} {}", token_arrow, token_delta.abs());
            println!("{}", "-".repeat(54));
            if delta > 0.0 {
                println!("RESULT: Target is MORE efficient (+{:.1} TAS)", delta);
            } else if delta < 0.0 {
                println!("RESULT: Target is LESS efficient ({:.1} TAS)", delta);
            } else {
                println!("RESULT: No change in efficiency");
            }
        }
        OutputFormat::Json => {
            let output = serde_json::json!({
                "baseline": {"trace_id": baseline_report.trace_id, "tas": baseline_report.score.score, "tokens": baseline_report.total_tokens},
                "target": {"trace_id": target_report.trace_id, "tas": target_report.score.score, "tokens": target_report.total_tokens},
                "delta": {"tas": delta, "tokens": token_delta}
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
    }

    Ok(())
}
