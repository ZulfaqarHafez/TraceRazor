use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::Read;
use std::path::{Path, PathBuf};
use tracerazor_core::{
    cost::{project_cost, CostConfig, ProviderPreset},
    fixes::{Fix, FixType},
    is_analysable,
    provenance::{hex_encode, sha256_hex},
    scoring::ScoringConfig,
    simulate::{simulate, SimulationSpec},
    types::MIN_TRACE_STEPS,
};
use tracerazor_ingest::{parse as ingest_parse, TraceFormat};
use tracerazor_semantic::{default_similarity_fn, LlmConfig};
use tracerazor_store::TraceStore;

mod agent;
mod commands;
mod trice_cmd;

use agent::{cmd_agent, AgentCommand};
use commands::{
    cmd_apply, cmd_audit, cmd_audit_batch, cmd_bench, cmd_claude, cmd_compare, cmd_cost,
    cmd_export, cmd_import, cmd_keygen, cmd_list, cmd_optimize, cmd_simulate, cmd_verify,
    expand_trace_paths,
};

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
        Some(dir.join("store.db"))
    })();

    if let Some(p) = path {
        match TraceStore::connect_file(p.to_string_lossy().as_ref()).await {
            Ok(store) => return store,
            Err(e) => eprintln!("Warning: could not open persistent store ({e}), using in-memory."),
        }
    }

    TraceStore::connect_mem()
        .await
        .expect("in-memory store failed")
}

/// TraceRazor - Token Efficiency Auditor for AI Agents
#[derive(Parser)]
#[command(
    name = "tracerazor",
    version,
    author = "Zulfaqar Hafez",
    about = "Lighthouse score for AI agents. Audit reasoning traces and eliminate token waste.",
    long_about = "Lighthouse score for AI agents. Audit reasoning traces and eliminate token waste.\n\n\
        Exit codes:\n  \
        0  success - the command ran (an audit completed; any gate that was set passed)\n  \
        1  gate failure - an explicit gate failed: audit/compare --threshold, compare regression, or verify tamper/mismatch\n  \
        2  error - bad input, IO, or parse failure (distinct from a failed gate)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Audit a trace file and produce an efficiency report.
    ///
    /// Gate semantics: without --threshold a low score is reported but still
    /// exits 0; --threshold N exits 1 when TAS < N (batch mode gates the mean
    /// TAS). A bad/unparseable trace exits 2, keeping "inefficient" distinct
    /// from "broken input".
    Audit {
        /// Trace file(s) or directories (JSON). A directory or multiple
        /// files switches to batch mode: every trace is audited hermetically
        /// and one aggregate fleet report is produced.
        #[arg(value_name = "FILE", num_args = 1..)]
        files: Vec<PathBuf>,

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
        /// Pass `--store false` to disable.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set,
              num_args = 1, value_name = "BOOL")]
        store: bool,

        /// Hermetic mode: read nothing from and write nothing to the local
        /// store, making the score a pure function of (trace, config,
        /// version). Recorded in the report's run manifest - required for
        /// exact third-party re-verification with `tracerazor verify`.
        #[arg(long, default_value_t = false)]
        hermetic: bool,

        /// Enable enhanced semantic analysis using configured LLM embeddings.
        /// Significantly improves SRR and ISR accuracy by replacing bag-of-words
        /// with dense sentence embeddings. Supports OpenAI, Anthropic (chat-only,
        /// falls back to BoW for embeddings), and OpenAI-compatible endpoints via
        /// TRACERAZOR_LLM_* / OPENAI_API_KEY / ANTHROPIC_API_KEY env vars.
        #[arg(long, default_value = "false")]
        enhanced: bool,

        /// Path to a calibrated weights JSON file (as produced by the
        /// calibration tool, `calibration/calibrate.py`). Falls back to the
        /// TRACERAZOR_WEIGHTS env var, then the built-in default weights.
        #[arg(long, value_name = "FILE")]
        weights: Option<PathBuf>,

        /// Minimum trace steps required to audit (clamped to >= 2). The
        /// default keeps the statistically conservative floor; lower it to
        /// audit short real-world trajectories (most ReAct task runs finish
        /// in 3-4 steps). Pair-based metrics carry less evidence on very
        /// short traces - interpret scores accordingly.
        #[arg(long, value_name = "N", default_value_t = MIN_TRACE_STEPS)]
        min_steps: usize,
    },

    /// Configure TraceRazor for agent hosts and run child agents with trace context.
    Agent {
        #[command(subcommand)]
        command: AgentCommand,
    },

    /// Install and run the Claude Code TraceRazor coach hooks.
    Claude {
        #[command(subcommand)]
        command: ClaudeCommand,
    },

    /// Normalize external trace exports into TraceRazor traces, optionally auditing them.
    #[command(name = "import")]
    ImportTrace {
        /// Trace export file(s) or directories.
        #[arg(value_name = "INPUT", num_args = 1..)]
        inputs: Vec<PathBuf>,
        /// Source format. Auto-detects if not specified.
        #[arg(long = "from", default_value = "auto")]
        source_format: InputFormat,
        /// Output file for a single input, or output directory for multiple inputs.
        #[arg(long, value_name = "PATH")]
        out: Option<PathBuf>,
        /// Run a hermetic audit and emit report/fixes/coach artifacts next to the trace.
        #[arg(long, default_value = "false")]
        audit: bool,
    },

    /// Verify a historical report against its trace and run manifest.
    ///
    /// Checks the trace file hash against the manifest, and - for hermetic
    /// bag-of-words runs - re-scores the trace with the manifest's exact
    /// configuration and compares every metric. Exit 0 = verified.
    ///
    /// For evidence bundles (zip files produced by `export --bundle`), the
    /// trace argument is optional - the bundle contains the trace internally.
    ///
    /// Gate semantics: exit 0 = verified; exit 1 = tamper/mismatch (signature
    /// invalid, trace-hash mismatch, or re-score divergence); exit 2 = error.
    Verify {
        /// The report JSON produced by `audit --format json`, or an evidence
        /// bundle zip produced by `export --bundle`.
        #[arg(value_name = "REPORT")]
        report: PathBuf,
        /// The original trace file the report claims to describe.
        /// Not required when REPORT is an evidence bundle (.zip).
        #[arg(value_name = "TRACE")]
        trace: Option<PathBuf>,
        /// Output format. `text` (default) prints the human-readable audit
        /// trail byte-for-byte; `json` prints a single machine-readable verdict
        /// object. Exit codes are identical in both modes.
        #[arg(short, long, default_value = "text")]
        format: VerifyFormat,
    },

    /// List all stored traces in the current session.
    List {
        /// Filter by agent name.
        #[arg(short, long)]
        agent: Option<String>,
        /// Output format.
        #[arg(short, long, default_value = "markdown")]
        format: OutputFormat,
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
        /// Default: 10% - a 10-point TAS drop exits with code 1.
        #[arg(long, default_value = "10.0")]
        regression_threshold: f64,
    },

    /// Project monthly and annual costs at a given run volume .
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

    /// Simulate removing or merging steps and project the TAS/token impact.
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
    /// reformulation_guard, goal_anchor. Pass `--all` to apply every fix in the file.
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
        /// Also apply fixes classified `dangerous` (e.g. termination guards
        /// that can suppress legitimate verification re-runs). Off by default
        /// even with --all.
        #[arg(long, default_value = "false")]
        force: bool,
        /// Preview the patches without writing to disk.
        #[arg(long, default_value = "false")]
        dry_run: bool,
    },

    /// Benchmark actual savings between a before and after trace.
    ///
    /// Reports measured token and TAS deltas and - if the fixes JSON from the
    /// baseline audit is supplied - compares those measured savings against the
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

    /// Optimize a trace with TRICE, or rewrite a system prompt with the legacy LLM optimizer.
    ///
    /// Audits the trace, identifies the top waste patterns, then iteratively
    /// asks the configured LLM to produce a tighter system prompt.  After each
    /// iteration the simulator projects the TAS improvement; the loop stops
    /// early when the target is met or the iteration cap is reached.
    ///
    /// Requires LLM credentials - see `tracerazor-semantic` docs for env vars:
    ///   OPENAI_API_KEY  /  ANTHROPIC_API_KEY  /  TRACERAZOR_LLM_*
    Optimize {
        /// Trace file to optimise (legacy positional form).
        #[arg(value_name = "TRACE")]
        file: Option<PathBuf>,
        /// Trace file to optimize with TRICE's runtime compressor.
        #[arg(long, value_name = "TRACE")]
        trace: Option<PathBuf>,
        /// Target input-token budget as a fraction of the original trace.
        #[arg(long, default_value = "0.40")]
        budget_ratio: f64,
        /// Write the TRICE context policy JSON here.
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,
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

    /// Replay a TRICE context policy against a recorded trace.
    Replay {
        /// Trace file used to build or evaluate the policy.
        #[arg(long, value_name = "TRACE")]
        trace: PathBuf,
        /// TRICE context policy JSON produced by `tracerazor optimize`.
        #[arg(long, value_name = "POLICY")]
        policy: PathBuf,
        /// Output format.
        #[arg(short, long, default_value = "markdown")]
        format: OutputFormat,
    },

    /// Export a report to an observability platform or webhook.
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
        /// Create a verifiable evidence bundle (zip: trace, signed report,
        /// weights, SHA256SUMS). Verify with: tracerazor verify bundle.zip
        #[arg(long, value_name = "FILE")]
        bundle: Option<PathBuf>,
    },

    /// Generate an Ed25519 keypair for cryptographic report signing.
    ///
    /// Prints TRACERAZOR_SIGNING_KEY (private, keep secret) and
    /// TRACERAZOR_VERIFY_KEY (public, safe to distribute) to stdout.
    ///
    /// To sign every audit: export TRACERAZOR_SIGNING_KEY=<key>
    /// To verify a signed report: tracerazor verify report.json trace.json
    Keygen,

    /// Start the TraceRazor HTTP server (REST API + dashboard).
    ///
    /// Alias for the `tracerazor-server` binary. POST a native trace to
    /// /api/audit or OTLP/HTTP JSON to /v1/traces. Set TRACERAZOR_API_TOKEN to
    /// require `Authorization: Bearer <token>` on protected routes - mandatory
    /// before exposing a non-loopback bind address. TRACERAZOR_OTLP_SPOOL_DIR
    /// selects the durable local-redacted OTLP receipt directory.
    Serve {
        /// Port to listen on.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Bind address. Loopback by default; pass 0.0.0.0 to expose
        /// externally (set TRACERAZOR_API_TOKEN first).
        #[arg(long, default_value = "127.0.0.1")]
        bind: String,
        /// SQLite database path.
        #[arg(long, value_name = "FILE")]
        db: Option<String>,
    },
}

#[derive(Subcommand)]
enum ClaudeCommand {
    /// Install the Claude Code SessionEnd + SessionStart TraceRazor hooks.
    Install {
        /// Settings scope to modify. Defaults to per-project local settings.
        #[arg(long, default_value = "local")]
        scope: ClaudeScope,
        /// Hook behavior. Coach mode still never auto-edits prompts/settings.
        #[arg(long, default_value = "coach")]
        mode: ClaudeMode,
        /// Also install the packaged `tracerazor` skill into the same scope.
        #[arg(long)]
        with_skill: bool,
    },
    /// Remove the TraceRazor Claude Code hooks from settings.
    Uninstall {
        /// Settings scope to modify. Defaults to per-project local settings.
        #[arg(long, default_value = "local")]
        scope: ClaudeScope,
        /// Also remove the packaged `tracerazor` skill from the same scope.
        #[arg(long)]
        with_skill: bool,
    },
    /// Convert a Claude Code transcript JSONL into a TraceRazor trace.
    Convert {
        /// Claude Code transcript JSONL path.
        #[arg(value_name = "TRANSCRIPT")]
        transcript: PathBuf,
        /// Output trace JSON path. Prints to stdout if omitted.
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,
    },
    /// Hook entrypoints called by Claude Code.
    Hook {
        #[command(subcommand)]
        command: ClaudeHookCommand,
    },
}

#[derive(Subcommand)]
enum ClaudeHookCommand {
    /// Handle a Claude Code SessionEnd hook event from stdin.
    #[command(name = "session-end")]
    SessionEnd {
        /// Hook behavior. Both modes write audit artifacts and apply nothing.
        #[arg(long, default_value = "coach")]
        mode: ClaudeMode,
    },
    /// Handle a Claude Code SessionStart hook event from stdin.
    ///
    /// Reads the SessionStart payload on stdin and, when the previous audited
    /// session is fresh and actionable, prints a compact coach advisory to
    /// STDOUT (Claude Code injects plain stdout into the new session's context).
    #[command(name = "session-start")]
    SessionStart,
}

#[derive(ValueEnum, Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ClaudeScope {
    Local,
    Project,
    User,
}

#[derive(ValueEnum, Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ClaudeMode {
    Passive,
    Coach,
}

#[derive(ValueEnum, Clone, Debug)]
enum OutputFormat {
    Markdown,
    Json,
}

/// Output format for `verify`: prose audit trail vs machine-readable verdict.
#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum VerifyFormat {
    Text,
    Json,
}

#[derive(ValueEnum, Clone, Debug)]
enum InputFormat {
    Auto,
    Raw,
    Langsmith,
    Otel,
    #[value(name = "claude-code")]
    ClaudeCode,
    Langfuse,
    Phoenix,
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
            InputFormat::ClaudeCode => TraceFormat::ClaudeCode,
            InputFormat::Langfuse => TraceFormat::Langfuse,
            InputFormat::Phoenix => TraceFormat::Phoenix,
        }
    }
}

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();
    // Exit-code contract: 0 = success / gate passed, 1 = an explicit gate
    // failed (threshold, regression, tamper), 2 = error (bad input, IO,
    // parse). Batch jobs can rely on the distinction.
    if let Err(e) = run().await {
        eprintln!("Error: {e:#}");
        std::process::exit(2);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Audit {
            files,
            format,
            threshold,
            trace_format,
            cost_per_million,
            store,
            hermetic,
            enhanced,
            weights,
            min_steps,
        } => {
            let expanded = expand_trace_paths(&files)?;
            if expanded.len() == 1 {
                cmd_audit(
                    expanded.into_iter().next().expect("len checked"),
                    format,
                    threshold,
                    trace_format,
                    cost_per_million,
                    store,
                    hermetic,
                    enhanced,
                    weights,
                    min_steps,
                )
                .await?;
            } else {
                cmd_audit_batch(
                    expanded,
                    format,
                    threshold,
                    trace_format,
                    cost_per_million,
                    weights,
                    min_steps,
                )?;
            }
        }
        Commands::Claude { command } => {
            cmd_claude(command).await?;
        }
        Commands::Agent { command } => {
            if let Some(code) = cmd_agent(command)? {
                std::process::exit(code);
            }
        }
        Commands::ImportTrace {
            inputs,
            source_format,
            out,
            audit,
        } => {
            cmd_import(inputs, source_format, out, audit).await?;
        }
        Commands::Verify {
            report,
            trace,
            format,
        } => {
            cmd_verify(report, trace, format)?;
        }
        Commands::List { agent, format } => {
            cmd_list(agent, format).await?;
        }
        Commands::Compare {
            baseline,
            target,
            format,
            regression_threshold,
        } => {
            cmd_compare(baseline, target, format, regression_threshold).await?;
        }
        Commands::Cost {
            files,
            runs,
            input_cost,
            output_cost,
            provider,
            format,
        } => {
            cmd_cost(files, runs, input_cost, output_cost, provider, format).await?;
        }
        Commands::Simulate {
            file,
            remove,
            merge,
            format,
        } => {
            cmd_simulate(file, remove, merge, format).await?;
        }
        Commands::Apply {
            fixes,
            to,
            all,
            force,
            dry_run,
        } => {
            cmd_apply(fixes, to, all, force, dry_run).await?;
        }
        Commands::Bench {
            before,
            after,
            fixes,
            format,
        } => {
            cmd_bench(before, after, fixes, format).await?;
        }
        Commands::Optimize {
            file,
            trace,
            budget_ratio,
            out,
            system_prompt,
            output,
            iterations,
            target_tas,
            format,
        } => {
            if trace.is_some() || out.is_some() {
                let trace_path = trace
                    .or(file)
                    .context("optimize needs --trace <TRACE> (or legacy positional TRACE)")?;
                trice_cmd::cmd_trice_optimize(trace_path, budget_ratio, out.or(output), format)
                    .await?;
            } else {
                let trace_path = file.context("legacy optimize needs a TRACE argument")?;
                cmd_optimize(
                    trace_path,
                    system_prompt,
                    output,
                    iterations,
                    target_tas,
                    format,
                )
                .await?;
            }
        }
        Commands::Replay {
            trace,
            policy,
            format,
        } => {
            trice_cmd::cmd_trice_replay(trace, policy, format).await?;
        }
        Commands::Export {
            file,
            otel,
            webhook,
            print,
            format,
            bundle,
        } => {
            cmd_export(file, otel, webhook, print, format, bundle).await?;
        }
        Commands::Keygen => {
            cmd_keygen();
        }
        Commands::Serve { port, bind, db } => {
            let db_path = db
                .or_else(|| std::env::var("TRACERAZOR_DB_PATH").ok())
                .unwrap_or_else(|| "./tracerazor.db".to_string());
            tracerazor_server::run_server(tracerazor_server::ServeOptions {
                port,
                bind,
                db_path,
            })
            .await?;
        }
    }

    Ok(())
}
