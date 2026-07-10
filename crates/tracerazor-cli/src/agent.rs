//! Agent-host bootstrap, lifecycle capture, and child-process propagation.
//!
//! This module deliberately performs no work at package-install time. Every
//! filesystem mutation is behind an explicit `tracerazor agent install` or
//! `uninstall` command and is tracked in a per-scope ownership ledger.

use anyhow::{bail, Context, Result};
use chrono::Utc;
use clap::{Subcommand, ValueEnum};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tracerazor_core::{
    provenance::{
        hex_decode_32, sha256_hex, sign_run_receipt, verify_run_receipt_json,
        RunReceiptSignatureStatus, RunReceiptV1,
    },
    report::{IngestQuality, RunManifest},
    scoring::ScoringConfig,
    types::{StepType, Trace, TraceStep},
};
use tracerazor_ingest::{parse as ingest_parse, TraceFormat};
use tracerazor_semantic::{default_similarity_fn, BOW_BACKEND_ID};

const STATE_SCHEMA_VERSION: u32 = 1;
const EVENT_SCHEMA_VERSION: &str = "tracerazor-event/v1";
const STATE_RELATIVE_PATH: &str = ".tracerazor/agent-install.json";
const MAX_HOST_TRANSCRIPT_BYTES: u64 = 64 * 1024 * 1024;
const CODEX_MCP_BLOCK: &str = "# >>> tracerazor managed mcp >>>\n[mcp_servers.tracerazor]\ncommand = \"tracerazor-mcp\"\n# <<< tracerazor managed mcp <<<\n";
const TRACERAZOR_SKILL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/tracerazor-skill/SKILL.md"
));

#[derive(Subcommand, Debug)]
pub(crate) enum AgentCommand {
    /// Inspect host availability, policy, and the current installation.
    Doctor {
        /// Output format.
        #[arg(long, default_value = "text")]
        format: AgentOutputFormat,
    },
    /// Explicitly install TraceRazor integration for an agent host.
    Install {
        /// Host to configure. `auto` selects the first detected supported host.
        #[arg(long, default_value = "auto")]
        host: AgentHost,
        /// Configuration scope.
        #[arg(long, default_value = "project")]
        scope: AgentScope,
        /// Capture/advice policy. No mode automatically edits project files.
        #[arg(long, default_value = "coach")]
        mode: AgentMode,
        /// Preview every change without writing files.
        #[arg(long)]
        dry_run: bool,
        /// Output format.
        #[arg(long, default_value = "text")]
        format: AgentOutputFormat,
    },
    /// Report whether the recorded integration is still healthy.
    Status {
        /// Filter by host. `auto` reports every installation in the scope.
        #[arg(long, default_value = "auto")]
        host: AgentHost,
        /// Configuration scope.
        #[arg(long, default_value = "project")]
        scope: AgentScope,
        /// Output format.
        #[arg(long, default_value = "text")]
        format: AgentOutputFormat,
    },
    /// Remove only TraceRazor-owned content for a host.
    Uninstall {
        /// Host to remove. `auto` removes all recorded hosts in the scope.
        #[arg(long, default_value = "auto")]
        host: AgentHost,
        /// Configuration scope.
        #[arg(long, default_value = "project")]
        scope: AgentScope,
        /// Preview every change without writing files.
        #[arg(long)]
        dry_run: bool,
        /// Output format.
        #[arg(long, default_value = "text")]
        format: AgentOutputFormat,
    },
    /// Run a child agent while propagating run, parent, policy, and W3C context.
    Run {
        /// Command and arguments. Put `--` before commands with flags.
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        command: Vec<OsString>,
    },
    /// Record a host lifecycle event. Intended for trusted host hook configs.
    Hook {
        /// Host whose hook protocol invoked the command.
        #[arg(long, default_value = "generic")]
        host: AgentHost,
        /// Host lifecycle event being recorded.
        #[arg(long)]
        event: AgentHookEvent,
    },
    /// Verify an offline run receipt and any sibling artifacts that are present.
    VerifyReceipt {
        /// Path to `run-receipt.json`.
        receipt: PathBuf,
        /// Expected Ed25519 public key. Defaults to TRACERAZOR_VERIFY_KEY.
        #[arg(long)]
        verify_key: Option<String>,
        /// Output format.
        #[arg(long, default_value = "text")]
        format: AgentOutputFormat,
    },
}

#[derive(ValueEnum, Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AgentOutputFormat {
    Text,
    Json,
}

#[derive(ValueEnum, Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AgentHost {
    Auto,
    Codex,
    Claude,
    Gemini,
    Generic,
}

#[derive(ValueEnum, Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AgentScope {
    Project,
    User,
    Image,
}

#[derive(ValueEnum, Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AgentMode {
    Off,
    Passive,
    Coach,
    Enforce,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CaptureMode {
    Auto,
    Manual,
    Off,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum PrivacyMode {
    LocalRedacted,
    Raw,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct AgentPolicy {
    schema_version: u32,
    mode: AgentMode,
    capture: CaptureMode,
    hermetic: bool,
    privacy: PrivacyMode,
    persist_raw_content: bool,
    artifact_dir: String,
    min_steps: usize,
    quality: QualityPolicy,
    enforcement: EnforcementPolicy,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct QualityPolicy {
    verifier: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct EnforcementPolicy {
    enabled: bool,
}

impl Default for AgentPolicy {
    fn default() -> Self {
        Self::with_mode(AgentMode::Coach)
    }
}

impl AgentPolicy {
    fn with_mode(mode: AgentMode) -> Self {
        Self {
            schema_version: 1,
            mode,
            capture: CaptureMode::Auto,
            hermetic: true,
            privacy: PrivacyMode::LocalRedacted,
            persist_raw_content: false,
            artifact_dir: ".tracerazor/runs".to_string(),
            min_steps: 5,
            quality: QualityPolicy::default(),
            enforcement: EnforcementPolicy {
                enabled: mode == AgentMode::Enforce,
            },
        }
    }

    fn automatic_capture(&self) -> bool {
        self.mode != AgentMode::Off && self.capture == CaptureMode::Auto
    }

    fn privacy_str(&self) -> &'static str {
        match self.privacy {
            PrivacyMode::LocalRedacted => "local-redacted",
            PrivacyMode::Raw => "raw",
        }
    }

    fn capture_str(&self) -> &'static str {
        match self.capture {
            CaptureMode::Auto => "auto",
            CaptureMode::Manual => "manual",
            CaptureMode::Off => "off",
        }
    }

    fn validate(&self, source: &Path) -> Result<()> {
        if self.schema_version != 1 {
            bail!("{} must use schema_version = 1", source.display());
        }
        if self.min_steps < 2 {
            bail!("{} min_steps must be at least 2", source.display());
        }
        if self.artifact_dir.trim().is_empty() {
            bail!("{} artifact_dir must not be empty", source.display());
        }
        let artifact = Path::new(&self.artifact_dir);
        if artifact.is_absolute()
            || artifact
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            bail!(
                "{} artifact_dir must be a confined relative path",
                source.display()
            );
        }
        if self.persist_raw_content && self.privacy != PrivacyMode::Raw {
            bail!(
                "{} persist_raw_content requires privacy = \"raw\"",
                source.display()
            );
        }
        Ok(())
    }
}

struct LoadedPolicy {
    policy: AgentPolicy,
    path: Option<PathBuf>,
    root: PathBuf,
}

#[derive(ValueEnum, Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum AgentHookEvent {
    SessionStart,
    SessionEnd,
    AfterAgent,
    SubagentStart,
    SubagentStop,
    PreCompress,
    Stop,
}

impl AgentHost {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Codex => "codex",
            Self::Claude => "claude",
            Self::Gemini => "gemini",
            Self::Generic => "generic",
        }
    }
}

impl AgentScope {
    fn as_str(self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::User => "user",
            Self::Image => "image",
        }
    }
}

impl AgentMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Passive => "passive",
            Self::Coach => "coach",
            Self::Enforce => "enforce",
        }
    }
}

impl AgentHookEvent {
    fn as_str(self) -> &'static str {
        match self {
            Self::SessionStart => "session_start",
            Self::SessionEnd => "session_end",
            Self::AfterAgent => "after_agent",
            Self::SubagentStart => "subagent_start",
            Self::SubagentStop => "subagent_stop",
            Self::PreCompress => "pre_compress",
            Self::Stop => "stop",
        }
    }

    fn as_cli_str(self) -> &'static str {
        match self {
            Self::SessionStart => "session-start",
            Self::SessionEnd => "session-end",
            Self::AfterAgent => "after-agent",
            Self::SubagentStart => "subagent-start",
            Self::SubagentStop => "subagent-stop",
            Self::PreCompress => "pre-compress",
            Self::Stop => "stop",
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
struct InstallState {
    #[serde(default = "state_schema_version")]
    schema_version: u32,
    #[serde(default)]
    installations: Vec<InstallRecord>,
}

fn state_schema_version() -> u32 {
    STATE_SCHEMA_VERSION
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct InstallRecord {
    host: AgentHost,
    scope: AgentScope,
    mode: AgentMode,
    root: String,
    installed_at: String,
    #[serde(default)]
    owned_paths: Vec<OwnedPath>,
    #[serde(default)]
    modified_paths: Vec<String>,
    #[serde(default)]
    managed_hooks: Vec<ManagedHook>,
    #[serde(default)]
    managed_configs: Vec<ManagedConfig>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct OwnedPath {
    path: String,
    sha256: String,
    kind: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct ManagedHook {
    path: String,
    event: String,
    fingerprint: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct ManagedConfig {
    path: String,
    kind: String,
    fingerprint: String,
}

#[derive(Debug)]
struct StateRead {
    state: InstallState,
    invalid: bool,
}

#[derive(Serialize)]
struct HostDetection {
    host: &'static str,
    detected: bool,
    executable: Option<String>,
    project_config: bool,
}

struct CaptureOutcome {
    status: &'static str,
    reason: Option<String>,
    step_count: usize,
    total_tokens: u32,
    ingest_status: &'static str,
    provider_token_coverage: f64,
    issues: Vec<String>,
    audit_trace_sha256: Option<String>,
    persisted_trace_sha256: Option<String>,
    replayable: bool,
}

struct AuditSummary {
    step_count: usize,
    total_tokens: u32,
    ingest_status: &'static str,
    provider_token_coverage: f64,
    issues: Vec<String>,
    audit_trace_sha256: String,
    persisted_trace_sha256: String,
    replayable: bool,
}

struct ParsedHostTranscript {
    trace: Trace,
    format: &'static str,
    issues: Vec<String>,
}

#[derive(Clone, Copy)]
struct RunIdentity<'a> {
    run_id: &'a str,
    trace_id: &'a str,
    session_id: &'a str,
    agent_id: &'a str,
    parent_agent_id: Option<&'a str>,
}

pub(crate) fn cmd_agent(command: AgentCommand) -> Result<Option<i32>> {
    match command {
        AgentCommand::Doctor { format } => {
            cmd_doctor(format)?;
            Ok(None)
        }
        AgentCommand::Install {
            host,
            scope,
            mode,
            dry_run,
            format,
        } => {
            cmd_install(host, scope, mode, dry_run, format)?;
            Ok(None)
        }
        AgentCommand::Status {
            host,
            scope,
            format,
        } => {
            cmd_status(host, scope, format)?;
            Ok(None)
        }
        AgentCommand::Uninstall {
            host,
            scope,
            dry_run,
            format,
        } => {
            cmd_uninstall(host, scope, dry_run, format)?;
            Ok(None)
        }
        AgentCommand::Run { command } => Ok(Some(cmd_run(command)?)),
        AgentCommand::Hook { host, event } => {
            // Hooks must never fail the host. Diagnostics go to stderr and the
            // host always receives a successful exit status.
            let loaded_policy = load_effective_policy();
            let recorded = match &loaded_policy {
                Ok(loaded) if loaded.policy.automatic_capture() => {
                    record_hook_event(host, event, loaded)
                }
                Ok(_) => Ok(()),
                Err(error) => Err(anyhow::anyhow!("invalid TraceRazor policy: {error:#}")),
            };
            if let Err(error) = &recorded {
                eprintln!("TraceRazor agent hook warning: {error:#}");
            }
            let coach = if recorded.is_ok()
                && event == AgentHookEvent::SessionStart
                && loaded_policy.as_ref().is_ok_and(|loaded| {
                    loaded.policy.automatic_capture()
                        && matches!(loaded.policy.mode, AgentMode::Coach | AgentMode::Enforce)
                }) {
                loaded_policy
                    .as_ref()
                    .ok()
                    .and_then(|loaded| latest_coach_context(loaded).ok().flatten())
                    .unwrap_or_else(|| {
                        "TraceRazor coach is active; advice is local and advisory.".to_string()
                    })
            } else {
                String::new()
            };
            if host == AgentHost::Gemini {
                let output = if event == AgentHookEvent::SessionStart {
                    json!({
                        "hookSpecificOutput": {
                            "hookEventName": "SessionStart",
                            "additionalContext": coach
                        },
                        "suppressOutput": true
                    })
                } else {
                    json!({"suppressOutput": true})
                };
                println!("{}", serde_json::to_string(&output)?);
            } else if host == AgentHost::Codex
                && matches!(event, AgentHookEvent::Stop | AgentHookEvent::SubagentStop)
            {
                // Codex requires JSON stdout for Stop/SubagentStop on exit 0.
                // An empty object is an explicit non-blocking decision.
                println!("{{}}");
            } else if !coach.is_empty() {
                println!("{coach}");
            }
            Ok(None)
        }
        AgentCommand::VerifyReceipt {
            receipt,
            verify_key,
            format,
        } => Ok(Some(cmd_verify_receipt(
            &receipt,
            verify_key.as_deref(),
            format,
        ))),
    }
}

fn cmd_verify_receipt(path: &Path, verify_key: Option<&str>, format: AgentOutputFormat) -> i32 {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) => {
            emit_receipt_verification(
                format,
                &json!({
                    "schema_version": "tracerazor-receipt-verification/v1",
                    "status": "malformed",
                    "receipt": path.to_string_lossy(),
                    "authenticated": false,
                    "error": format!("cannot read receipt: {error}"),
                }),
                &format!("MALFORMED: cannot read receipt: {error}"),
            );
            return 2;
        }
    };
    let (receipt, signature_status) = match verify_run_receipt_json(&raw) {
        Ok(result) => result,
        Err(error) => {
            let (status, code) = if error.is_tamper() {
                ("tampered", 1)
            } else {
                ("malformed", 2)
            };
            emit_receipt_verification(
                format,
                &json!({
                    "schema_version": "tracerazor-receipt-verification/v1",
                    "status": status,
                    "receipt": path.to_string_lossy(),
                    "authenticated": false,
                    "error": error.to_string(),
                }),
                &format!("{}: {error}", status.to_ascii_uppercase()),
            );
            return code;
        }
    };

    let expected_key = verify_key
        .map(str::to_string)
        .or_else(|| env_nonempty("TRACERAZOR_VERIFY_KEY"));
    if let Some(expected) = &expected_key {
        if expected.len() != 64
            || !expected
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            emit_receipt_verification(
                format,
                &json!({
                    "schema_version": "tracerazor-receipt-verification/v1",
                    "status": "malformed",
                    "receipt": path.to_string_lossy(),
                    "authenticated": false,
                    "error": "expected verify key must be 64 lowercase hex characters",
                }),
                "MALFORMED: expected verify key must be 64 lowercase hex characters",
            );
            return 2;
        }
    }
    let embedded_key = receipt
        .signature
        .as_ref()
        .map(|signature| signature.public_key.as_str());
    let signer_pinned = signature_status == RunReceiptSignatureStatus::Valid
        && expected_key
            .as_deref()
            .zip(embedded_key)
            .is_some_and(|(expected, actual)| expected == actual);

    let root = path.parent().unwrap_or_else(|| Path::new("."));
    let trace_path = root.join("trace.json");
    let report_path = root.join("report.json");
    let manifest_path = root.join("manifest.json");
    let mut checks = serde_json::Map::new();
    let mut mismatches = Vec::new();
    let mut operational_error = None;
    if signature_status == RunReceiptSignatureStatus::Valid
        && expected_key.is_some()
        && !signer_pinned
    {
        mismatches
            .push("receipt signer does not match the expected Ed25519 public key".to_string());
    }
    for (name, artifact_path, expected) in [
        (
            "persisted_trace",
            &trace_path,
            receipt.persisted_trace_sha256.as_str(),
        ),
        ("report", &report_path, receipt.report_sha256.as_str()),
    ] {
        if !artifact_path.exists() {
            checks.insert(name.to_string(), json!("not_present"));
            continue;
        }
        match fs::read(artifact_path) {
            Ok(bytes) => {
                let actual = sha256_hex(&bytes);
                if actual == expected {
                    checks.insert(name.to_string(), json!("verified"));
                } else {
                    checks.insert(name.to_string(), json!("mismatch"));
                    mismatches.push(format!(
                        "{name} SHA-256 mismatch: receipt {expected}, actual {actual}"
                    ));
                }
            }
            Err(error) => {
                operational_error = Some(format!(
                    "cannot read sibling artifact {}: {error}",
                    artifact_path.display()
                ));
                checks.insert(name.to_string(), json!("unreadable"));
            }
        }
    }
    if manifest_path.exists() {
        match fs::read_to_string(&manifest_path)
            .map_err(anyhow::Error::from)
            .and_then(|raw| serde_json::from_str::<Value>(&raw).map_err(anyhow::Error::from))
        {
            Ok(manifest) => {
                let mut required = vec![
                    ("run_id", receipt.run_id.as_str()),
                    ("audit_trace_sha256", receipt.audit_trace_sha256.as_str()),
                    (
                        "persisted_trace_sha256",
                        receipt.persisted_trace_sha256.as_str(),
                    ),
                ];
                for (field, value) in [
                    ("trace_id", receipt.trace_id.as_deref()),
                    ("session_id", receipt.session_id.as_deref()),
                    ("agent_id", receipt.agent_id.as_deref()),
                    ("parent_agent_id", receipt.parent_agent_id.as_deref()),
                ] {
                    if let Some(value) = value {
                        required.push((field, value));
                    }
                }
                let mut valid = true;
                for (field, expected) in required {
                    match manifest.get(field).and_then(Value::as_str) {
                        Some(actual) if actual == expected => {}
                        Some(actual) => {
                            valid = false;
                            mismatches.push(format!(
                                "manifest {field} mismatch: receipt {expected}, manifest {actual}"
                            ));
                        }
                        None => {
                            operational_error =
                                Some(format!("sibling manifest is missing string field {field}"));
                        }
                    }
                }
                checks.insert(
                    "manifest_identity".to_string(),
                    json!(if valid { "verified" } else { "mismatch" }),
                );
            }
            Err(error) => {
                operational_error = Some(format!("cannot parse sibling manifest: {error:#}"));
                checks.insert("manifest_identity".to_string(), json!("malformed"));
            }
        }
    } else {
        checks.insert("manifest_identity".to_string(), json!("not_present"));
    }

    if let Some(error) = operational_error {
        emit_receipt_verification(
            format,
            &json!({
                "schema_version": "tracerazor-receipt-verification/v1",
                "status": "malformed",
                "receipt": path.to_string_lossy(),
                "authenticated": signature_status == RunReceiptSignatureStatus::Valid,
                "signer_pinned": signer_pinned,
                "signature_status": signature_status,
                "identity_valid": true,
                "hash_checks": checks,
                "error": error,
            }),
            &format!("MALFORMED: {error}"),
        );
        return 2;
    }
    if !mismatches.is_empty() {
        emit_receipt_verification(
            format,
            &json!({
                "schema_version": "tracerazor-receipt-verification/v1",
                "status": "tampered",
                "receipt": path.to_string_lossy(),
                "authenticated": false,
                "signer_pinned": signer_pinned,
                "signature_status": signature_status,
                "identity_valid": true,
                "hash_checks": checks,
                "mismatches": mismatches,
            }),
            &format!("TAMPERED: {}", mismatches.join("; ")),
        );
        return 1;
    }

    let authenticated = signature_status == RunReceiptSignatureStatus::Valid;
    let status = if authenticated { "valid" } else { "unsigned" };
    let warnings = if !authenticated {
        vec![
            "receipt hashes are internally valid but not cryptographically authenticated"
                .to_string(),
        ]
    } else if !signer_pinned {
        vec![
            "signature is valid for its embedded key, but the signer is not pinned; pass --verify-key or set TRACERAZOR_VERIFY_KEY before trusting a remote receipt"
                .to_string(),
        ]
    } else {
        Vec::<String>::new()
    };
    emit_receipt_verification(
        format,
        &json!({
            "schema_version": "tracerazor-receipt-verification/v1",
            "status": status,
            "receipt": path.to_string_lossy(),
            "run_id": receipt.run_id,
            "trace_id": receipt.trace_id,
            "session_id": receipt.session_id,
            "agent_id": receipt.agent_id,
            "parent_agent_id": receipt.parent_agent_id,
            "authenticated": authenticated,
            "signer_pinned": signer_pinned,
            "trusted_offline_receipt": authenticated && signer_pinned,
            "signature_status": signature_status,
            "identity_valid": true,
            "hash_checks": checks,
            "warnings": warnings,
        }),
        if authenticated && signer_pinned {
            "VALID: Ed25519 signer and available sibling artifacts verified"
        } else if authenticated {
            "VALID BUT UNPINNED: signature matches its embedded key; do not trust remotely without --verify-key"
        } else {
            "UNSIGNED: identity and available sibling hashes verified; receipt is not authenticated"
        },
    );
    0
}

fn emit_receipt_verification(format: AgentOutputFormat, value: &Value, message: &str) {
    match format {
        AgentOutputFormat::Json => match serde_json::to_string_pretty(value) {
            Ok(output) => println!("{output}"),
            Err(error) => eprintln!("failed to serialize receipt verification: {error}"),
        },
        AgentOutputFormat::Text => println!("{message}"),
    }
}

fn cmd_doctor(format: AgentOutputFormat) -> Result<()> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let project_root = find_project_root(&cwd);
    let policy_path = effective_policy_path()?;
    let (policy, policy_error) = describe_policy(&policy_path);
    let image_policy = std::env::var("TRACERAZOR_IMAGE_ROOT").ok().map(|root| {
        let path = PathBuf::from(root).join("tracerazor.toml");
        describe_policy(&path).0
    });
    let detections = host_detections(&project_root);
    let selected = resolve_auto_host(&detections);
    let state_path = project_root.join(STATE_RELATIVE_PATH);
    ensure_confined_path(&project_root, &state_path, "agent installation state")?;
    let state_read = read_state(&state_path)?;
    let mut warnings = Vec::new();
    if state_read.invalid {
        warnings.push(format!(
            "installation state is invalid: {}",
            state_path.display()
        ));
    }
    if !policy_path.exists() {
        warnings.push("no project tracerazor.toml; run `tracerazor agent install`".to_string());
    }
    if let Some(error) = &policy_error {
        warnings.push(error.clone());
    }
    let exe = std::env::current_exe().context("cannot resolve TraceRazor executable")?;
    let value = json!({
        "schema_version": STATE_SCHEMA_VERSION,
        "command": "doctor",
        "ok": !state_read.invalid && policy_error.is_none(),
        "version": env!("CARGO_PKG_VERSION"),
        "executable": absolute_path(&exe)?.to_string_lossy(),
        "project_root": project_root.to_string_lossy(),
        "policy": policy,
        "image_policy": image_policy,
        "auto_host": selected.as_str(),
        "hosts": detections,
        "installations": state_read.state.installations,
        "image_root": std::env::var("TRACERAZOR_IMAGE_ROOT").ok(),
        "warnings": warnings,
    });
    emit(
        format,
        &value,
        format!(
            "TraceRazor {}\nExecutable: {}\nProject: {}\nPolicy: {}\nAuto host: {}\nInstallations: {}\nStatus: {}",
            env!("CARGO_PKG_VERSION"),
            exe.display(),
            project_root.display(),
            if policy_path.exists() { "present" } else { "missing" },
            selected.as_str(),
            value["installations"].as_array().map_or(0, Vec::len),
            if value["ok"] == true { "ok" } else { "attention required" },
        ),
    )
}

fn cmd_install(
    requested_host: AgentHost,
    scope: AgentScope,
    mode: AgentMode,
    dry_run: bool,
    format: AgentOutputFormat,
) -> Result<()> {
    let root = scope_root(scope)?;
    let detections = host_detections(&root);
    let host = if requested_host == AgentHost::Auto {
        resolve_auto_host(&detections)
    } else {
        requested_host
    };
    debug_assert_ne!(host, AgentHost::Auto);
    let state_path = root.join(STATE_RELATIVE_PATH);
    ensure_confined_path(&root, &state_path, "agent installation state")?;
    let state_read = read_state(&state_path)?;
    let mut state = state_read.state;
    let before = state.clone();
    let prior = state
        .installations
        .iter()
        .find(|record| record.host == host && record.scope == scope)
        .cloned();
    let mut actions = Vec::new();
    let mut warnings = Vec::new();
    let requested_policy = AgentPolicy::with_mode(mode);
    let override_policy = env_nonempty("TRACERAZOR_POLICY")
        .map(|value| absolute_path(Path::new(&value)))
        .transpose()?;
    let scoped_policy = matches!(scope, AgentScope::Project | AgentScope::Image)
        .then(|| root.join("tracerazor.toml"));
    let policy_path = if let Some(path) = override_policy.clone() {
        if !path.exists() {
            bail!("TRACERAZOR_POLICY does not exist: {}", path.display());
        }
        Some(path)
    } else if let Some(path) = scoped_policy.clone() {
        Some(path)
    } else {
        let discovered = discover_policy()?;
        discovered.exists().then_some(discovered)
    };
    let effective_policy = match policy_path.as_deref().filter(|path| path.exists()) {
        Some(path) => load_policy_file(path)?,
        None => requested_policy.clone(),
    };
    let effective_mode = effective_policy.mode;
    if effective_mode != mode {
        warnings.push(format!(
            "policy mode `{}` overrides requested --mode `{}`",
            effective_mode.as_str(),
            mode.as_str()
        ));
    }

    // Validate every host-controlled destination before the first mutation so
    // an unsafe hook/skill path cannot leave a partial bootstrap behind.
    if let Some(path) = scoped_policy.as_deref() {
        ensure_confined_path(&root, path, "scope policy")?;
    }
    if let Some(skill) = host_skill_path(host, &root) {
        ensure_confined_path(&root, &skill, "agent skill")?;
    }
    if let Some(settings) = host_settings_path(host, scope, &root) {
        ensure_confined_path(&root, &settings, "host hook settings")?;
    }
    if let Some(config) = mcp_config_path(host, scope, &root) {
        ensure_confined_path(&root, &config, "MCP configuration")?;
    }

    if state_read.invalid {
        let message = format!(
            "existing invalid state will be backed up before replacement: {}",
            state_path.display()
        );
        warnings.push(message);
        if !dry_run {
            backup_existing(&state_path, "invalid")?;
        }
    }

    let mut owned_paths = Vec::new();
    let mut modified_paths = Vec::new();
    let mut managed_hooks = Vec::new();
    let mut managed_configs = Vec::new();
    let mut mcp_registered = false;

    if override_policy.is_none() {
        if let Some(policy_path) = scoped_policy {
            ensure_confined_path(&root, &policy_path, "scope policy")?;
            let desired = render_policy(&effective_policy);
            if let Some(owned) = ensure_project_policy(
                &policy_path,
                &desired,
                &state,
                dry_run,
                &mut actions,
                &mut warnings,
            )? {
                owned_paths.push(owned);
            }
        }
    }

    match host {
        AgentHost::Codex | AgentHost::Gemini => {
            let skill = root
                .join(".agents")
                .join("skills")
                .join("tracerazor")
                .join("SKILL.md");
            ensure_confined_path(&root, &skill, "agent skill")?;
            if let Some(owned) = ensure_owned_file(
                &skill,
                TRACERAZOR_SKILL.as_bytes(),
                "agent_skill",
                &state,
                dry_run,
                &mut actions,
                &mut warnings,
            )? {
                owned_paths.push(owned);
            }
        }
        AgentHost::Claude => {
            let skill = root
                .join(".claude")
                .join("skills")
                .join("tracerazor")
                .join("SKILL.md");
            ensure_confined_path(&root, &skill, "agent skill")?;
            if let Some(owned) = ensure_owned_file(
                &skill,
                TRACERAZOR_SKILL.as_bytes(),
                "agent_skill",
                &state,
                dry_run,
                &mut actions,
                &mut warnings,
            )? {
                owned_paths.push(owned);
            }
        }
        AgentHost::Generic => {
            actions.push("record generic policy/state integration".to_string());
        }
        AgentHost::Auto => unreachable!("auto host is resolved before installation"),
    }

    if let Some(settings) = host_settings_path(host, scope, &root) {
        ensure_confined_path(&root, &settings, "host hook settings")?;
        let hook_mode = if effective_policy.automatic_capture() {
            effective_mode
        } else {
            AgentMode::Off
        };
        managed_hooks = install_host_hooks(
            &settings,
            host,
            hook_mode,
            prior.as_ref(),
            dry_run,
            &mut actions,
            &mut warnings,
        )?;
        if !managed_hooks.is_empty() {
            modified_paths.push(path_string(&settings)?);
        }
    }
    if let Some(config) = mcp_config_path(host, scope, &root) {
        let (configs, registered) = install_mcp_registration(
            &config,
            host,
            prior.as_ref(),
            dry_run,
            &mut actions,
            &mut warnings,
        )?;
        managed_configs = configs;
        mcp_registered = registered;
    }

    owned_paths.sort_by(|a, b| a.path.cmp(&b.path));
    owned_paths.dedup_by(|a, b| a.path == b.path);
    modified_paths.sort();
    modified_paths.dedup();
    managed_hooks.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then_with(|| a.event.cmp(&b.event))
            .then_with(|| a.fingerprint.cmp(&b.fingerprint))
    });
    managed_configs.sort_by(|a, b| a.path.cmp(&b.path).then_with(|| a.kind.cmp(&b.kind)));
    let installed_at = prior
        .as_ref()
        .map(|record| record.installed_at.clone())
        .unwrap_or_else(|| Utc::now().to_rfc3339());
    let record = InstallRecord {
        host,
        scope,
        mode: effective_mode,
        root: path_string(&root)?,
        installed_at,
        owned_paths,
        modified_paths,
        managed_hooks: managed_hooks.clone(),
        managed_configs: managed_configs.clone(),
    };
    state
        .installations
        .retain(|existing| !(existing.host == host && existing.scope == scope));
    state.installations.push(record.clone());
    state.installations.sort_by(|a, b| {
        a.scope
            .as_str()
            .cmp(b.scope.as_str())
            .then_with(|| a.host.as_str().cmp(b.host.as_str()))
    });

    let changed = state != before || state_read.invalid;
    if !dry_run && changed {
        write_state(&state_path, &state)?;
        actions.push(format!("write state {}", state_path.display()));
    }
    if actions.is_empty() {
        actions.push("no changes; installation is already current".to_string());
    }

    let value = json!({
        "schema_version": STATE_SCHEMA_VERSION,
        "command": "install",
        "ok": true,
        "changed": changed,
        "dry_run": dry_run,
        "host": host,
        "requested_host": requested_host,
        "scope": scope,
        "requested_mode": mode,
        "mode": effective_mode,
        "policy": {
            "path": policy_path.as_ref().map(|path| path.to_string_lossy().to_string()),
            "capture": effective_policy.capture_str(),
            "hermetic": effective_policy.hermetic,
            "privacy": effective_policy.privacy_str(),
            "persist_raw_content": effective_policy.persist_raw_content,
            "artifact_dir": effective_policy.artifact_dir,
            "min_steps": effective_policy.min_steps,
            "verifier": effective_policy.quality.verifier,
            "enforcement_enabled": effective_policy.enforcement.enabled,
        },
        "automatic_capture": host != AgentHost::Generic && !managed_hooks.is_empty(),
        "capture_status": capture_capability(host, &managed_hooks).0,
        "capture_reason": capture_capability(host, &managed_hooks).1,
        "hook_trust_required": matches!(host, AgentHost::Codex | AgentHost::Gemini),
        "mcp_registered": mcp_registered,
        "root": root.to_string_lossy(),
        "state_path": state_path.to_string_lossy(),
        "record": record,
        "actions": actions,
        "warnings": warnings,
    });
    emit(
        format,
        &value,
        format!(
            "{} TraceRazor for {} at {} scope in {} mode.\n{}",
            if dry_run {
                "Would install"
            } else {
                "Installed"
            },
            host.as_str(),
            scope.as_str(),
            effective_mode.as_str(),
            value["actions"]
                .as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .filter_map(Value::as_str)
                .map(|line| format!("- {line}"))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
    )
}

fn cmd_status(host: AgentHost, scope: AgentScope, format: AgentOutputFormat) -> Result<()> {
    let root = scope_root(scope)?;
    let state_path = root.join(STATE_RELATIVE_PATH);
    ensure_confined_path(&root, &state_path, "agent installation state")?;
    let state_read = read_state(&state_path)?;
    let policy_path = policy_path_for_scope(scope, &root)?;
    let (policy_description, policy_error) = describe_policy(&policy_path);
    let parsed_policy = policy_path
        .exists()
        .then(|| load_policy_file(&policy_path).ok())
        .flatten();
    let records = state_read
        .state
        .installations
        .iter()
        .filter(|record| record.scope == scope && (host == AgentHost::Auto || record.host == host))
        .cloned()
        .collect::<Vec<_>>();
    let statuses = records
        .iter()
        .map(|record| record_status(record, &root, parsed_policy.as_ref()))
        .collect::<Result<Vec<_>>>()?;
    let healthy = !state_read.invalid
        && policy_error.is_none()
        && !statuses.is_empty()
        && statuses
            .iter()
            .all(|status| status["healthy"].as_bool().unwrap_or(false));
    let value = json!({
        "schema_version": STATE_SCHEMA_VERSION,
        "command": "status",
        "ok": !state_read.invalid && policy_error.is_none(),
        "installed": !statuses.is_empty(),
        "healthy": healthy,
        "host": host,
        "scope": scope,
        "root": root.to_string_lossy(),
        "state_path": state_path.to_string_lossy(),
        "policy": policy_description,
        "installations": statuses,
        "warnings": if state_read.invalid { vec!["invalid installation state"] } else { Vec::<&str>::new() },
    });
    emit(
        format,
        &value,
        format!(
            "TraceRazor agent status\nScope: {}\nInstalled: {}\nHealthy: {}\nRecords: {}",
            scope.as_str(),
            value["installed"],
            value["healthy"],
            value["installations"].as_array().map_or(0, Vec::len),
        ),
    )
}

fn cmd_uninstall(
    host: AgentHost,
    scope: AgentScope,
    dry_run: bool,
    format: AgentOutputFormat,
) -> Result<()> {
    let root = scope_root(scope)?;
    let state_path = root.join(STATE_RELATIVE_PATH);
    ensure_confined_path(&root, &state_path, "agent installation state")?;
    let state_read = read_state(&state_path)?;
    if state_read.invalid {
        bail!(
            "cannot safely uninstall from invalid state {}; repair or restore its backup first",
            state_path.display()
        );
    }
    let mut state = state_read.state;
    let targets = state
        .installations
        .iter()
        .filter(|record| record.scope == scope && (host == AgentHost::Auto || record.host == host))
        .cloned()
        .collect::<Vec<_>>();
    let mut actions = Vec::new();
    let mut warnings = Vec::new();
    if targets.is_empty() {
        actions.push("no matching recorded installation".to_string());
    }
    state
        .installations
        .retain(|record| !targets.iter().any(|target| target == record));

    let hook_paths = targets
        .iter()
        .flat_map(|record| record.managed_hooks.iter().map(|hook| hook.path.clone()))
        .collect::<BTreeSet<_>>();
    for path_text in hook_paths {
        let path = PathBuf::from(&path_text);
        let expected = targets.iter().any(|record| {
            record
                .managed_hooks
                .iter()
                .any(|item| item.path == path_text)
                && is_expected_modified_path(record, &path, &root).unwrap_or(false)
        });
        if !expected {
            warnings.push(format!(
                "ignored unexpected settings path from installation state: {}",
                path.display()
            ));
            continue;
        }
        if let Err(error) = ensure_confined_path(&root, &path, "host hook settings") {
            warnings.push(format!("left unsafe hook settings untouched: {error:#}"));
            continue;
        }
        let hooks = targets
            .iter()
            .flat_map(|record| record.managed_hooks.iter())
            .filter(|hook| hook.path == path_text)
            .cloned()
            .collect::<Vec<_>>();
        uninstall_host_hooks(&path, &hooks, dry_run, &mut actions, &mut warnings)?;
    }
    for record in &targets {
        if record.managed_hooks.is_empty() && !record.modified_paths.is_empty() {
            warnings.push(format!(
                "left legacy {} hook settings untouched because no ownership fingerprint was recorded",
                record.host.as_str()
            ));
        }
    }

    for record in &targets {
        for config in &record.managed_configs {
            let Some(expected) = mcp_config_path(record.host, record.scope, &root) else {
                continue;
            };
            if path_string(&expected)? != config.path {
                warnings.push(format!(
                    "ignored unexpected MCP configuration path: {}",
                    config.path
                ));
                continue;
            }
            if let Err(error) = ensure_confined_path(&root, &expected, "MCP configuration") {
                warnings.push(format!(
                    "left unsafe MCP configuration untouched: {error:#}"
                ));
                continue;
            }
            uninstall_mcp_registration(&expected, config, dry_run, &mut actions, &mut warnings)?;
        }
    }

    let mut seen = HashSet::new();
    for owned in targets.iter().flat_map(|record| record.owned_paths.iter()) {
        if !seen.insert(owned.path.clone()) {
            continue;
        }
        let expected = targets.iter().any(|record| {
            record
                .owned_paths
                .iter()
                .any(|item| item.path == owned.path)
                && is_expected_owned_path(record, owned, &root).unwrap_or(false)
        });
        if !expected {
            warnings.push(format!(
                "ignored unexpected owned path from installation state: {}",
                owned.path
            ));
            continue;
        }
        if state
            .installations
            .iter()
            .flat_map(|record| record.owned_paths.iter())
            .any(|remaining| remaining.path == owned.path)
        {
            actions.push(format!("keep shared {}", owned.path));
            continue;
        }
        let path = PathBuf::from(&owned.path);
        if let Err(error) = ensure_confined_path(&root, &path, &owned.kind) {
            warnings.push(format!("left unsafe owned path untouched: {error:#}"));
            continue;
        }
        if !path.exists() {
            continue;
        }
        let content = fs::read(&path)
            .with_context(|| format!("cannot read owned path {}", path.display()))?;
        if sha256_hex(&content) != owned.sha256 {
            warnings.push(format!(
                "left user-modified {} in place: {}",
                owned.kind,
                path.display()
            ));
            continue;
        }
        actions.push(format!("remove {} {}", owned.kind, path.display()));
        if !dry_run {
            fs::remove_file(&path).with_context(|| format!("cannot remove {}", path.display()))?;
            remove_empty_parent(path.parent());
        }
    }

    if !dry_run {
        if state.installations.is_empty() {
            if state_path.exists() {
                fs::remove_file(&state_path)?;
                remove_empty_parent(state_path.parent());
                actions.push(format!("remove state {}", state_path.display()));
            }
        } else {
            write_state(&state_path, &state)?;
            actions.push(format!("update state {}", state_path.display()));
        }
    }
    let value = json!({
        "schema_version": STATE_SCHEMA_VERSION,
        "command": "uninstall",
        "ok": true,
        "dry_run": dry_run,
        "host": host,
        "scope": scope,
        "removed_records": targets.len(),
        "actions": actions,
        "warnings": warnings,
    });
    emit(
        format,
        &value,
        format!(
            "{} {} TraceRazor agent installation record(s).\n{}",
            if dry_run { "Would remove" } else { "Removed" },
            targets.len(),
            value["actions"]
                .as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .filter_map(Value::as_str)
                .map(|line| format!("- {line}"))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
    )
}

fn cmd_run(command: Vec<OsString>) -> Result<i32> {
    let (program, args) = command
        .split_first()
        .context("agent run requires a command after `--`")?;
    let run_id = env_nonempty("TRACERAZOR_RUN_ID").unwrap_or_else(|| generated_id("run"));
    let parent_agent_id = env_nonempty("TRACERAZOR_AGENT_ID")
        .or_else(|| env_nonempty("TRACERAZOR_PARENT_AGENT_ID"))
        .unwrap_or_else(|| generated_id("agent"));
    let child_agent_id = generated_id("agent");
    let policy = env_nonempty("TRACERAZOR_POLICY")
        .map(PathBuf::from)
        .unwrap_or(discover_policy()?);
    let traceparent = env_nonempty("traceparent")
        .or_else(|| env_nonempty("TRACEPARENT"))
        .unwrap_or_else(generate_traceparent);
    let (trace_id, parent_span_id) = traceparent_ids(&traceparent)?;
    let session_id =
        env_nonempty("TRACERAZOR_SESSION_ID").unwrap_or_else(|| generated_id("session"));

    invoke_wrapper_hook(
        AgentHookEvent::SessionStart,
        &run_id,
        &child_agent_id,
        &parent_agent_id,
        &policy,
        &traceparent,
        &trace_id,
        &parent_span_id,
        &session_id,
        None,
    );

    let status = Command::new(program)
        .args(args)
        .env("TRACERAZOR_RUN_ID", &run_id)
        .env("TRACERAZOR_PARENT_AGENT_ID", &parent_agent_id)
        .env("TRACERAZOR_AGENT_ID", &child_agent_id)
        .env("TRACERAZOR_POLICY", &policy)
        .env("TRACERAZOR_SESSION_ID", &session_id)
        .env("TRACERAZOR_TRACE_ID", &trace_id)
        .env("TRACERAZOR_PARENT_SPAN_ID", &parent_span_id)
        .env("TRACEPARENT", &traceparent)
        .env("traceparent", &traceparent)
        .status()
        .with_context(|| format!("failed to start child command {:?}", program))?;
    let code = status.code().unwrap_or(1);
    invoke_wrapper_hook(
        AgentHookEvent::Stop,
        &run_id,
        &child_agent_id,
        &parent_agent_id,
        &policy,
        &traceparent,
        &trace_id,
        &parent_span_id,
        &session_id,
        Some(code),
    );
    Ok(code)
}

#[allow(clippy::too_many_arguments)]
fn invoke_wrapper_hook(
    event: AgentHookEvent,
    run_id: &str,
    agent_id: &str,
    parent_agent_id: &str,
    policy: &Path,
    traceparent: &str,
    trace_id: &str,
    parent_span_id: &str,
    session_id: &str,
    exit_code: Option<i32>,
) {
    let Ok(executable) = std::env::current_exe() else {
        return;
    };
    let mut child = match Command::new(executable)
        .args([
            "agent",
            "hook",
            "--host",
            "generic",
            "--event",
            event.as_cli_str(),
        ])
        .env("TRACERAZOR_RUN_ID", run_id)
        .env("TRACERAZOR_AGENT_ID", agent_id)
        .env("TRACERAZOR_PARENT_AGENT_ID", parent_agent_id)
        .env("TRACERAZOR_POLICY", policy)
        .env("TRACERAZOR_TRACE_ID", trace_id)
        .env("TRACERAZOR_PARENT_SPAN_ID", parent_span_id)
        .env("TRACERAZOR_SESSION_ID", session_id)
        .env("TRACEPARENT", traceparent)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            eprintln!("TraceRazor wrapper hook warning: {error}");
            return;
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        let payload = json!({
            "run_id": run_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "parent_agent_id": parent_agent_id,
            "exit_code": exit_code,
        });
        let _ = stdin.write_all(payload.to_string().as_bytes());
    }
    if let Err(error) = child.wait() {
        eprintln!("TraceRazor wrapper hook warning: {error}");
    }
}

fn record_hook_event(host: AgentHost, event: AgentHookEvent, loaded: &LoadedPolicy) -> Result<()> {
    let mut raw = String::new();
    std::io::stdin().read_to_string(&mut raw)?;
    let payload: Value = if raw.trim().is_empty() {
        json!({})
    } else {
        serde_json::from_str(&raw).context("hook input was not valid JSON")?
    };
    let session_id = first_safe_id(&[
        env_nonempty("TRACERAZOR_SESSION_ID"),
        json_string(&payload, "session_id"),
        json_string(&payload, "sessionId"),
    ])
    .unwrap_or_else(|| generated_id("session"));
    let payload_agent_id = first_safe_id(&[
        json_string(&payload, "agent_id"),
        json_string(&payload, "agentId"),
    ]);
    let subagent = matches!(
        event,
        AgentHookEvent::SubagentStart | AgentHookEvent::SubagentStop
    );
    let explicit_run_id = first_safe_id(&[
        env_nonempty("TRACERAZOR_RUN_ID"),
        json_string(&payload, "run_id"),
        json_string(&payload, "runId"),
    ]);
    let run_id = explicit_run_id.unwrap_or_else(|| {
        if subagent {
            payload_agent_id.as_ref().map_or_else(
                || session_id.clone(),
                |agent| derived_safe_id("run", &format!("{session_id}:{agent}")),
            )
        } else {
            session_id.clone()
        }
    });
    let agent_id = if subagent {
        payload_agent_id
            .clone()
            .or_else(|| env_nonempty("TRACERAZOR_AGENT_ID"))
    } else {
        env_nonempty("TRACERAZOR_AGENT_ID").or(payload_agent_id.clone())
    }
    .filter(|value| is_safe_id(value))
    .unwrap_or_else(|| derived_safe_id("agent", &session_id));
    let parent_agent_id = first_safe_id(&[
        env_nonempty("TRACERAZOR_PARENT_AGENT_ID"),
        json_string(&payload, "parent_agent_id"),
        json_string(&payload, "parentAgentId"),
    ])
    .or_else(|| subagent.then(|| derived_safe_id("agent", &session_id)));
    let inherited_traceparent = env_nonempty("TRACEPARENT").or_else(|| env_nonempty("traceparent"));
    let (trace_from_parent, span_from_parent) = inherited_traceparent
        .as_deref()
        .and_then(|value| traceparent_ids(value).ok())
        .map_or((None, None), |(trace, span)| (Some(trace), Some(span)));
    let trace_id = env_nonempty("TRACERAZOR_TRACE_ID")
        .filter(|value| is_w3c_hex(value, 32))
        .or(trace_from_parent)
        .unwrap_or_else(|| random_hex(16));
    let parent_span_id = env_nonempty("TRACERAZOR_PARENT_SPAN_ID")
        .filter(|value| is_w3c_hex(value, 16))
        .or(span_from_parent);
    let span_id = json_string(&payload, "span_id")
        .filter(|value| is_w3c_hex(value, 16))
        .unwrap_or_else(|| random_hex(8));
    let artifact = artifact_root(loaded)?;
    let run_dir = artifact.join(&run_id);
    ensure_confined_path(&loaded.root, &run_dir, "run artifacts")?;
    fs::create_dir_all(&run_dir)?;
    let events_path = run_dir.join("events.jsonl");
    let now = Utc::now().to_rfc3339();
    let sequence = safe_read_optional(&loaded.root, &events_path, "events spool")?
        .map(|value| value.lines().filter(|line| !line.trim().is_empty()).count())
        .unwrap_or(0);
    // This record represents the lifecycle notification itself. Transcript
    // capture quality is reported separately in the run manifest below.
    let lifecycle_issue = "lifecycle_only";
    let event_value = json!({
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": generated_id("event"),
        "timestamp": now,
        "sequence": sequence,
        "run_id": run_id,
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "parent_agent_id": parent_agent_id,
        "event_type": lifecycle_event_type(event),
        "host": host.as_str(),
        "host_version": null,
        "framework": host.as_str(),
        "framework_version": null,
        "tokens": {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "reasoning": 0,
            "total": 0,
            "provenance": "missing"
        },
        "tool": null,
        "task": null,
        "capture": {
            "quality": "partial",
            "privacy": loaded.policy.privacy_str(),
            "issues": [lifecycle_issue]
        },
        "content": null,
        "content_sha256": null,
        "input_context": null,
        "input_context_sha256": null,
        "output": null,
        "output_sha256": null,
        "metadata": {
            "lifecycle_event": event.as_str(),
            "payload_sha256": sha256_hex(raw.as_bytes())
        }
    });
    safe_append_line(
        &loaded.root,
        &events_path,
        &serde_json::to_string(&event_value)?,
    )?;
    let event_count = safe_read_optional(&loaded.root, &events_path, "events spool")?
        .unwrap_or_default()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    let identity = RunIdentity {
        run_id: &run_id,
        trace_id: &trace_id,
        session_id: &session_id,
        agent_id: &agent_id,
        parent_agent_id: parent_agent_id.as_deref(),
    };
    let capture = capture_terminal_transcript(
        host,
        event,
        &payload,
        &run_dir,
        identity,
        &loaded.policy,
        &loaded.root,
    );
    let manifest_path = run_dir.join("manifest.json");
    let previous_manifest = safe_read_optional(&loaded.root, &manifest_path, "run manifest")?
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok());
    let previously_complete = previous_manifest
        .as_ref()
        .and_then(|value| value.get("status").and_then(Value::as_str))
        .is_some_and(|status| status == "completed");
    let complete = previously_complete || capture.status == "completed";
    let started_at = previous_manifest
        .as_ref()
        .and_then(|value| value.get("started_at"))
        .cloned()
        .unwrap_or_else(|| json!(now));
    let mut issues = capture.issues.clone();
    if let Some(reason) = &capture.reason {
        if !issues.contains(reason) {
            issues.push(reason.clone());
        }
    }
    issues.sort();
    issues.dedup();
    let preserve_completed_capture = previously_complete && capture.status != "completed";
    let (step_count, total_tokens, capture_quality, degraded_ingest, ingest_quality) =
        if preserve_completed_capture {
            let previous = previous_manifest.as_ref().expect("checked above");
            (
                previous
                    .get("step_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize,
                previous
                    .get("total_tokens")
                    .and_then(Value::as_u64)
                    .map(|value| u32::try_from(value).unwrap_or(u32::MAX))
                    .unwrap_or(0),
                previous
                    .get("capture_quality")
                    .and_then(Value::as_str)
                    .unwrap_or("degraded")
                    .to_string(),
                previous
                    .get("degraded_ingest")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                previous.get("ingest_quality").cloned().unwrap_or_else(|| {
                    json!({
                        "status": "degraded",
                        "provider_token_coverage": 0.0,
                        "issues": ["prior_capture_metadata_missing"]
                    })
                }),
            )
        } else {
            (
                capture.step_count,
                capture.total_tokens,
                if complete {
                    capture.ingest_status.to_string()
                } else {
                    "partial".to_string()
                },
                !complete || capture.ingest_status != "complete",
                json!({
                    "status": capture.ingest_status,
                    "provider_token_coverage": capture.provider_token_coverage,
                    "issues": issues,
                }),
            )
        };
    let replayable = if preserve_completed_capture {
        previous_manifest
            .as_ref()
            .and_then(|value| value.get("replayable"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    } else {
        capture.replayable
    };
    let audit_trace_sha256 = if preserve_completed_capture {
        previous_manifest
            .as_ref()
            .and_then(|value| value.get("audit_trace_sha256"))
            .cloned()
            .unwrap_or(Value::Null)
    } else {
        json!(capture.audit_trace_sha256)
    };
    let persisted_trace_sha256 = if preserve_completed_capture {
        previous_manifest
            .as_ref()
            .and_then(|value| value.get("persisted_trace_sha256"))
            .cloned()
            .unwrap_or(Value::Null)
    } else {
        json!(capture.persisted_trace_sha256)
    };
    let lifecycle_issues = if preserve_completed_capture {
        issues
    } else {
        Vec::new()
    };
    let terminal = matches!(
        event,
        AgentHookEvent::SessionEnd
            | AgentHookEvent::AfterAgent
            | AgentHookEvent::SubagentStop
            | AgentHookEvent::Stop
    );
    let mut files = vec!["manifest.json"];
    files.extend(
        [
            "events.jsonl",
            "trace.json",
            "report.json",
            "findings.json",
            "validation.json",
            "run-receipt.json",
        ]
        .into_iter()
        .filter(|name| run_dir.join(name).exists()),
    );
    let manifest = json!({
        "schema_version": "tracerazor-run/v1",
        "status": if complete { "completed" } else { capture.status },
        "run_id": event_value["run_id"],
        "trace_id": event_value["trace_id"],
        "session_id": event_value["session_id"],
        "agent_id": event_value["agent_id"],
        "parent_agent_id": event_value["parent_agent_id"],
        "host": host.as_str(),
        "host_version": null,
        "framework": host.as_str(),
        "framework_version": null,
        "started_at": started_at,
        "ended_at": if terminal { Some(now.clone()) } else { None::<String> },
        "event_count": event_count,
        "step_count": step_count,
        "total_tokens": total_tokens,
        "capture_quality": capture_quality,
        "degraded_ingest": degraded_ingest,
        "ingest_quality": ingest_quality,
        "lifecycle_issues": lifecycle_issues,
        "privacy": loaded.policy.privacy_str(),
        "raw_content_persisted": loaded.policy.persist_raw_content,
        "replayable": replayable,
        "verification_mode": if replayable { "hermetic_replay" } else { "non_replayable_receipt" },
        "audit_trace_sha256": audit_trace_sha256,
        "persisted_trace_sha256": persisted_trace_sha256,
        "policy": {
            "mode": loaded.policy.mode.as_str(),
            "capture": loaded.policy.capture_str(),
            "hermetic": loaded.policy.hermetic,
            "min_steps": loaded.policy.min_steps,
            "path": loaded.path.as_ref().map(|path| path.to_string_lossy().to_string()),
            "verifier": loaded.policy.quality.verifier,
            "enforcement_enabled": loaded.policy.enforcement.enabled,
            "enforcement_executed": false,
        },
        "enforcement_eligible": false,
        "enforcement_ineligible_reasons": ["task_outcome_not_verified", "verifier_not_run"],
        "files": files,
    });
    safe_atomic_write(
        &loaded.root,
        &manifest_path,
        serde_json::to_string_pretty(&manifest)?.as_bytes(),
        "run manifest",
    )?;
    Ok(())
}

fn capture_terminal_transcript(
    host: AgentHost,
    event: AgentHookEvent,
    payload: &Value,
    run_dir: &Path,
    identity: RunIdentity<'_>,
    policy: &AgentPolicy,
    artifact_root: &Path,
) -> CaptureOutcome {
    let terminal = matches!(
        event,
        AgentHookEvent::SessionEnd
            | AgentHookEvent::AfterAgent
            | AgentHookEvent::SubagentStop
            | AgentHookEvent::Stop
    );
    if !terminal {
        return CaptureOutcome {
            status: "running",
            reason: Some("lifecycle_only".to_string()),
            step_count: 0,
            total_tokens: 0,
            ingest_status: "partial",
            provider_token_coverage: 0.0,
            issues: vec!["lifecycle_only".to_string()],
            audit_trace_sha256: None,
            persisted_trace_sha256: None,
            replayable: false,
        };
    }
    if let Some(actual) = json_string(payload, "hook_event_name") {
        let expected = host_hook_event_name(event);
        if actual != expected {
            return partial_capture("hook_event_mismatch");
        }
    }
    if !host_supports_terminal_capture(host, event) {
        return partial_capture("unsupported_capture");
    }
    let transcript_key = if event == AgentHookEvent::SubagentStop {
        "agent_transcript_path"
    } else {
        "transcript_path"
    };
    let transcript = if event == AgentHookEvent::SubagentStop {
        json_string(payload, "agent_transcript_path")
            .or_else(|| json_string(payload, "agentTranscriptPath"))
    } else {
        json_string(payload, "transcript_path").or_else(|| json_string(payload, "transcriptPath"))
    }
    .filter(|value| !value.trim().is_empty())
    .map(PathBuf::from);
    let Some(transcript) = transcript else {
        return partial_capture(&format!("{transcript_key}_missing"));
    };
    match audit_host_transcript(host, &transcript, run_dir, identity, policy, artifact_root) {
        Ok(Some(summary)) => CaptureOutcome {
            status: "completed",
            reason: None,
            step_count: summary.step_count,
            total_tokens: summary.total_tokens,
            ingest_status: summary.ingest_status,
            provider_token_coverage: summary.provider_token_coverage,
            issues: summary.issues,
            audit_trace_sha256: Some(summary.audit_trace_sha256),
            persisted_trace_sha256: Some(summary.persisted_trace_sha256),
            replayable: summary.replayable,
        },
        Ok(None) => partial_capture("below_min_steps"),
        Err(error) => {
            eprintln!("TraceRazor transcript audit warning: {error:#}");
            let message = format!("{error:#}");
            let issue = if message.contains("transcript_read_error") {
                "transcript_read_error"
            } else if message.contains("unsupported_transcript_format") {
                "unsupported_transcript_format"
            } else {
                "audit_error"
            };
            partial_capture(issue)
        }
    }
}

fn partial_capture(reason: &str) -> CaptureOutcome {
    CaptureOutcome {
        status: "partial",
        reason: Some(reason.to_string()),
        step_count: 0,
        total_tokens: 0,
        ingest_status: "partial",
        provider_token_coverage: 0.0,
        issues: vec![reason.to_string()],
        audit_trace_sha256: None,
        persisted_trace_sha256: None,
        replayable: false,
    }
}

fn host_supports_terminal_capture(host: AgentHost, event: AgentHookEvent) -> bool {
    match host {
        AgentHost::Claude => matches!(
            event,
            AgentHookEvent::SessionEnd | AgentHookEvent::SubagentStop
        ),
        AgentHost::Codex => matches!(event, AgentHookEvent::Stop | AgentHookEvent::SubagentStop),
        AgentHost::Gemini => matches!(
            event,
            AgentHookEvent::AfterAgent | AgentHookEvent::SessionEnd
        ),
        AgentHost::Auto | AgentHost::Generic => false,
    }
}

fn host_hook_event_name(event: AgentHookEvent) -> &'static str {
    match event {
        AgentHookEvent::SessionStart => "SessionStart",
        AgentHookEvent::SessionEnd => "SessionEnd",
        AgentHookEvent::AfterAgent => "AfterAgent",
        AgentHookEvent::SubagentStart => "SubagentStart",
        AgentHookEvent::SubagentStop => "SubagentStop",
        AgentHookEvent::PreCompress => "PreCompress",
        AgentHookEvent::Stop => "Stop",
    }
}

fn audit_host_transcript(
    host: AgentHost,
    transcript: &Path,
    run_dir: &Path,
    identity: RunIdentity<'_>,
    policy: &AgentPolicy,
    artifact_root: &Path,
) -> Result<Option<AuditSummary>> {
    let raw = read_trusted_transcript(transcript).context("transcript_read_error")?;
    let ParsedHostTranscript {
        mut trace,
        format,
        mut issues,
    } = parse_host_transcript(host, &raw).context("unsupported_transcript_format")?;
    if trace.steps.len() < policy.min_steps {
        return Ok(None);
    }
    trace.trace_id = identity.run_id.to_string();
    // Bind the report to the exact normalized trace that was scored. In
    // local-redacted mode that raw trace remains memory-only; the persisted
    // trace is deliberately non-replayable and has a distinct receipt hash.
    let audit_trace = serde_json::to_string_pretty(&trace)?;
    let audit_trace_sha256 = sha256_hex(audit_trace.as_bytes());
    let config = ScoringConfig::default();
    let mut report = tracerazor_core::analyse(&mut trace, default_similarity_fn(), &config)?;
    let mut quality = IngestQuality::assess_with_format(&trace, format);
    if !issues.is_empty() {
        quality.degraded = true;
        quality.degraded_ingest = true;
        quality.warnings.append(&mut issues);
        quality.warnings.sort();
        quality.warnings.dedup();
    }
    let ingest_status = if quality.degraded_ingest {
        "degraded"
    } else {
        "complete"
    };
    let provider_token_coverage = quality.token_coverage;
    let issues = quality.warnings.clone();
    let persisted_trace = if policy.persist_raw_content {
        audit_trace.clone()
    } else {
        let mut trace_value: Value = serde_json::from_str(&audit_trace)?;
        redact_trace_value(&mut trace_value);
        serde_json::to_string_pretty(&trace_value)?
    };
    let persisted_trace_sha256 = sha256_hex(persisted_trace.as_bytes());
    let replayable = policy.persist_raw_content && audit_trace_sha256 == persisted_trace_sha256;
    report.manifest = Some(RunManifest::build(
        audit_trace_sha256.clone(),
        env!("CARGO_PKG_VERSION"),
        BOW_BACKEND_ID.to_string(),
        &config,
        policy.min_steps,
        policy.hermetic,
        Some(quality.clone()),
    )?);
    let mut report_value = serde_json::to_value(&report)?;
    if !policy.persist_raw_content {
        redact_report_value(&mut report_value);
    }
    safe_atomic_write(
        artifact_root,
        &run_dir.join("trace.json"),
        persisted_trace.as_bytes(),
        "trace artifact",
    )?;
    let report_json = serde_json::to_string_pretty(&report_value)?;
    safe_atomic_write(
        artifact_root,
        &run_dir.join("report.json"),
        report_json.as_bytes(),
        "report artifact",
    )?;
    let mut findings = serde_json::to_value(&report.fixes)?;
    if !policy.persist_raw_content {
        redact_report_value(&mut findings);
    }
    let findings_json = serde_json::to_string_pretty(&json!({
        "schema_version": "tracerazor-findings/v1",
        "run_id": identity.run_id,
        "findings": findings,
    }))?;
    safe_atomic_write(
        artifact_root,
        &run_dir.join("findings.json"),
        findings_json.as_bytes(),
        "findings artifact",
    )?;
    let validation_json = serde_json::to_string_pretty(&json!({
        "schema_version": "tracerazor-validation/v1",
        "run_id": identity.run_id,
        "status": "not_run",
        "task_quality_verified": false,
        "verifier": if policy.quality.verifier.is_empty() { None } else { Some(policy.quality.verifier.as_str()) },
        "enforcement": {
            "enabled": policy.enforcement.enabled,
            "executed": false
        }
    }))?;
    safe_atomic_write(
        artifact_root,
        &run_dir.join("validation.json"),
        validation_json.as_bytes(),
        "validation artifact",
    )?;
    let mut receipt = RunReceiptV1 {
        schema_version: RunReceiptV1::SCHEMA_VERSION.to_string(),
        run_id: identity.run_id.to_string(),
        trace_id: Some(identity.trace_id.to_string()),
        session_id: Some(identity.session_id.to_string()),
        agent_id: Some(identity.agent_id.to_string()),
        parent_agent_id: identity.parent_agent_id.map(str::to_string),
        created_at: Utc::now().to_rfc3339(),
        privacy: policy.privacy_str().to_string(),
        hermetic: policy.hermetic,
        replayable,
        verification_mode: if replayable {
            "hermetic_replay".to_string()
        } else {
            "non_replayable_receipt".to_string()
        },
        audit_trace_sha256: audit_trace_sha256.clone(),
        persisted_trace_sha256: persisted_trace_sha256.clone(),
        report_sha256: sha256_hex(report_json.as_bytes()),
        signed: false,
        signature: None,
    };
    if let Some(key_hex) = env_nonempty("TRACERAZOR_SIGNING_KEY") {
        match hex_decode_32(&key_hex)
            .context("TRACERAZOR_SIGNING_KEY must be 64 hex chars (32-byte Ed25519 seed)")
            .and_then(|seed| sign_run_receipt(&mut receipt, &seed))
        {
            Ok(()) => {}
            Err(error) => {
                eprintln!(
                    "TraceRazor run receipt signing warning: {error:#}; receipt is explicitly unsigned"
                );
                receipt.signed = false;
                receipt.signature = None;
            }
        }
    }
    let receipt_json = serde_json::to_string_pretty(&receipt)?;
    safe_atomic_write(
        artifact_root,
        &run_dir.join("run-receipt.json"),
        receipt_json.as_bytes(),
        "run receipt",
    )?;
    Ok(Some(AuditSummary {
        step_count: trace.steps.len(),
        total_tokens: trace.effective_total_tokens(),
        ingest_status,
        provider_token_coverage,
        issues,
        audit_trace_sha256,
        persisted_trace_sha256,
        replayable,
    }))
}

fn read_trusted_transcript(path: &Path) -> Result<String> {
    if !path.is_absolute() {
        bail!("transcript path must be absolute");
    }
    reject_symlink(path, "host transcript")?;
    let metadata = fs::metadata(path)
        .with_context(|| format!("cannot inspect transcript {}", path.display()))?;
    if !metadata.is_file() {
        bail!("transcript is not a regular file: {}", path.display());
    }
    if metadata.len() > MAX_HOST_TRANSCRIPT_BYTES {
        bail!(
            "transcript exceeds {} byte limit",
            MAX_HOST_TRANSCRIPT_BYTES
        );
    }
    let mut options = OpenOptions::new();
    options.read(true);
    let file = nofollow_open(&mut options, path)
        .with_context(|| format!("cannot open transcript {}", path.display()))?;
    let mut raw = String::new();
    file.take(MAX_HOST_TRANSCRIPT_BYTES + 1)
        .read_to_string(&mut raw)
        .with_context(|| format!("transcript is not valid UTF-8: {}", path.display()))?;
    if raw.len() as u64 > MAX_HOST_TRANSCRIPT_BYTES {
        bail!(
            "transcript exceeds {} byte limit",
            MAX_HOST_TRANSCRIPT_BYTES
        );
    }
    Ok(raw)
}

fn parse_host_transcript(host: AgentHost, raw: &str) -> Result<ParsedHostTranscript> {
    match host {
        AgentHost::Claude => Ok(ParsedHostTranscript {
            trace: ingest_parse(raw, TraceFormat::ClaudeCode)
                .context("failed to normalize Claude Code transcript")?,
            format: "claude-code",
            issues: Vec::new(),
        }),
        AgentHost::Codex => parse_codex_transcript(raw),
        AgentHost::Gemini => parse_gemini_transcript(raw),
        AgentHost::Auto | AgentHost::Generic => {
            bail!("host does not define a transcript format")
        }
    }
}

fn parse_codex_transcript(raw: &str) -> Result<ParsedHostTranscript> {
    let (records, invalid_lines) = json_records(raw)?;
    let mut steps = Vec::new();
    let mut tool_steps: HashMap<String, usize> = HashMap::new();
    let mut pending_context = String::new();
    let mut first_user_prompt = None;
    let mut fallback_assistant = Vec::new();
    let mut session_id = None;
    let mut model = None;
    let mut reported_total = None;

    for record in records {
        let kind = record
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let payload = record.get("payload").unwrap_or(&record);
        match kind {
            "session_meta" => {
                session_id = session_id.or_else(|| json_string(payload, "id"));
            }
            "turn_context" => {
                model = json_string(payload, "model").or(model);
            }
            "event_msg" => match payload.get("type").and_then(Value::as_str) {
                Some("user_message") => {
                    if let Some(message) = json_string(payload, "message") {
                        if first_user_prompt.is_none() {
                            first_user_prompt = Some(message.clone());
                        }
                        pending_context = message;
                    }
                }
                Some("agent_message") => {
                    if let Some(message) = json_string(payload, "message") {
                        fallback_assistant.push(message);
                    }
                }
                Some("token_count") => {
                    if let Some(usage) = payload
                        .get("info")
                        .and_then(|info| info.get("total_token_usage"))
                        .and_then(usage_total)
                        .or_else(|| {
                            payload
                                .get("info")
                                .and_then(|info| info.get("last_token_usage"))
                                .and_then(usage_total)
                        })
                    {
                        reported_total = Some(reported_total.unwrap_or(0).max(usage));
                    }
                }
                _ => {}
            },
            "response_item" => {
                parse_codex_response_item(
                    payload,
                    &mut steps,
                    &mut tool_steps,
                    &mut pending_context,
                    &mut first_user_prompt,
                );
            }
            _ => {}
        }
    }
    if steps.is_empty() {
        for message in fallback_assistant {
            push_reasoning_step(&mut steps, &message, &pending_context, None);
            pending_context.clear();
        }
    }
    if steps.is_empty() {
        bail!("Codex transcript produced no auditable response items");
    }

    let mut issues = Vec::new();
    if invalid_lines > 0 {
        issues.push("invalid_transcript_records_skipped".to_string());
    }
    let total_tokens = reported_total.unwrap_or(0);
    if total_tokens == 0 {
        issues.push("missing_provider_usage".to_string());
    } else {
        distribute_tokens(&mut steps, 0, total_tokens);
        if steps.len() > 1 {
            issues.push("token_distribution_estimated".to_string());
        }
    }
    let mut metadata = HashMap::new();
    metadata.insert(
        "source".to_string(),
        Value::String("codex-rollout-transcript".to_string()),
    );
    metadata.insert(
        "token_accounting".to_string(),
        Value::String(if total_tokens > 0 {
            "provider-reported session total; per-step distribution estimated".to_string()
        } else {
            "provider usage missing; no character-count estimate used".to_string()
        }),
    );
    if let Some(model) = model {
        metadata.insert("model".to_string(), Value::String(model));
    }
    if let Some(task) = first_user_prompt {
        metadata.insert("task".to_string(), Value::String(task));
    }
    Ok(ParsedHostTranscript {
        trace: Trace {
            trace_id: session_id.unwrap_or_else(|| "codex-transcript".to_string()),
            agent_name: "codex".to_string(),
            framework: "codex".to_string(),
            steps,
            total_tokens,
            task_value_score: 1.0,
            metadata,
        },
        format: "codex-rollout",
        issues,
    })
}

fn parse_codex_response_item(
    item: &Value,
    steps: &mut Vec<TraceStep>,
    tool_steps: &mut HashMap<String, usize>,
    pending_context: &mut String,
    first_user_prompt: &mut Option<String>,
) {
    match item.get("type").and_then(Value::as_str) {
        Some("message") => {
            let text = host_text(item.get("content").unwrap_or(&Value::Null));
            match item.get("role").and_then(Value::as_str) {
                Some("user") => {
                    if first_user_prompt.is_none() && !text.trim().is_empty() {
                        *first_user_prompt = Some(text.clone());
                    }
                    *pending_context = text;
                }
                Some("assistant") if !text.trim().is_empty() => {
                    push_reasoning_step(steps, &text, pending_context, None);
                    pending_context.clear();
                }
                _ => {}
            }
        }
        Some("reasoning") => {
            let mut text = host_text(item.get("summary").unwrap_or(&Value::Null));
            let content = host_text(item.get("content").unwrap_or(&Value::Null));
            if !content.trim().is_empty() {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&content);
            }
            if !text.trim().is_empty() {
                push_reasoning_step(steps, &text, pending_context, None);
                pending_context.clear();
            }
        }
        Some("function_call" | "custom_tool_call" | "local_shell_call" | "web_search_call") => {
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| item.get("type").and_then(Value::as_str))
                .unwrap_or("tool");
            let params = item
                .get("arguments")
                .or_else(|| item.get("input"))
                .or_else(|| item.get("action"))
                .cloned()
                .map(parse_json_string_value);
            let call_id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let index = steps.len();
            steps.push(TraceStep {
                id: index as u32 + 1,
                step_type: StepType::ToolCall,
                content: clip_host_text(&format!("Call {name} with provided arguments"), 2_000),
                tool_name: Some(name.to_string()),
                tool_params: params,
                tool_success: None,
                input_context: (!pending_context.is_empty())
                    .then(|| clip_host_text(pending_context, 4_000)),
                ..Default::default()
            });
            pending_context.clear();
            if let Some(call_id) = call_id {
                tool_steps.insert(call_id, index);
            }
        }
        Some("function_call_output" | "custom_tool_call_output" | "local_shell_call_output") => {
            let Some(call_id) = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(Value::as_str)
            else {
                return;
            };
            let Some(index) = tool_steps.get(call_id).copied() else {
                return;
            };
            let output_value = item.get("output").unwrap_or(&Value::Null);
            let output = host_text_or_json(output_value);
            let failed = output_value
                .get("is_error")
                .or_else(|| output_value.get("isError"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if let Some(step) = steps.get_mut(index) {
                step.tool_success = Some(!failed);
                if failed {
                    step.tool_error = Some(clip_host_text(&output, 2_000));
                } else if !output.is_empty() {
                    step.output = Some(clip_host_text(&output, 2_000));
                }
            }
        }
        _ => {}
    }
}

fn parse_gemini_transcript(raw: &str) -> Result<ParsedHostTranscript> {
    let (records, invalid_lines) = json_records(raw)?;
    let mut messages: Vec<Value> = Vec::new();
    let mut session_id = None;
    for record in records {
        session_id = session_id
            .or_else(|| json_string(&record, "sessionId"))
            .or_else(|| {
                record
                    .get("$set")
                    .and_then(|value| json_string(value, "sessionId"))
            });
        if let Some(rewind_to) = json_string(&record, "$rewindTo") {
            if let Some(index) = messages.iter().position(|message| {
                message.get("id").and_then(Value::as_str) == Some(rewind_to.as_str())
            }) {
                messages.truncate(index);
            } else {
                messages.clear();
            }
            continue;
        }
        if let Some(checkpoint) = record
            .get("$set")
            .and_then(|value| value.get("messages"))
            .and_then(Value::as_array)
        {
            messages.clear();
            for message in checkpoint {
                upsert_message(&mut messages, message.clone());
            }
            continue;
        }
        if let Some(legacy) = record.get("messages").and_then(Value::as_array) {
            for message in legacy {
                upsert_message(&mut messages, message.clone());
            }
        } else if record.get("id").is_some() {
            upsert_message(&mut messages, record);
        }
    }

    let mut steps = Vec::new();
    let mut pending_context = String::new();
    let mut first_user_prompt = None;
    let mut model = None;
    let mut total_tokens = 0_u32;
    let mut issues = Vec::new();
    if invalid_lines > 0 {
        issues.push("invalid_transcript_records_skipped".to_string());
    }
    for message in messages {
        match message.get("type").and_then(Value::as_str) {
            Some("user") => {
                let text = host_text(message.get("content").unwrap_or(&Value::Null));
                if first_user_prompt.is_none() && !text.trim().is_empty() {
                    first_user_prompt = Some(text.clone());
                }
                pending_context = text;
            }
            Some("gemini") => {
                model = json_string(&message, "model").or(model);
                let start = steps.len();
                let mut reasoning = host_text(message.get("thoughts").unwrap_or(&Value::Null));
                let content = host_text(message.get("content").unwrap_or(&Value::Null));
                if !content.trim().is_empty() {
                    if !reasoning.is_empty() {
                        reasoning.push('\n');
                    }
                    reasoning.push_str(&content);
                }
                if !reasoning.trim().is_empty() {
                    push_reasoning_step(&mut steps, &reasoning, &pending_context, None);
                    pending_context.clear();
                }
                if let Some(tool_calls) = message.get("toolCalls").and_then(Value::as_array) {
                    for tool in tool_calls {
                        let name = tool.get("name").and_then(Value::as_str).unwrap_or("tool");
                        let status = tool
                            .get("status")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_ascii_lowercase();
                        let failed = status.contains("error")
                            || status.contains("fail")
                            || status.contains("cancel");
                        let output = host_text_or_json(tool.get("result").unwrap_or(&Value::Null));
                        let index = steps.len();
                        steps.push(TraceStep {
                            id: index as u32 + 1,
                            step_type: StepType::ToolCall,
                            content: clip_host_text(
                                &format!("Call {name} with provided arguments"),
                                2_000,
                            ),
                            tool_name: Some(name.to_string()),
                            tool_params: tool.get("args").cloned(),
                            tool_success: Some(!failed),
                            tool_error: (failed && !output.is_empty())
                                .then(|| clip_host_text(&output, 2_000)),
                            output: (!failed && !output.is_empty())
                                .then(|| clip_host_text(&output, 2_000)),
                            input_context: (!pending_context.is_empty())
                                .then(|| clip_host_text(&pending_context, 4_000)),
                            ..Default::default()
                        });
                        pending_context.clear();
                    }
                }
                let end = steps.len();
                if end > start {
                    if let Some(tokens) = message.get("tokens").and_then(usage_total) {
                        total_tokens = total_tokens.saturating_add(tokens);
                        distribute_tokens(&mut steps, start, tokens);
                        if end - start > 1 {
                            issues.push("token_distribution_estimated".to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if steps.is_empty() {
        bail!("Gemini CLI transcript produced no auditable messages");
    }
    if steps.iter().any(|step| step.tokens == 0) {
        issues.push("missing_provider_usage".to_string());
    }
    issues.sort();
    issues.dedup();
    let mut metadata = HashMap::new();
    metadata.insert(
        "source".to_string(),
        Value::String("gemini-cli-session-transcript".to_string()),
    );
    metadata.insert(
        "token_accounting".to_string(),
        Value::String(
            "provider-reported message totals when present; no character-count estimate used"
                .to_string(),
        ),
    );
    if let Some(model) = model {
        metadata.insert("model".to_string(), Value::String(model));
    }
    if let Some(task) = first_user_prompt {
        metadata.insert("task".to_string(), Value::String(task));
    }
    Ok(ParsedHostTranscript {
        trace: Trace {
            trace_id: session_id.unwrap_or_else(|| "gemini-cli-transcript".to_string()),
            agent_name: "gemini-cli".to_string(),
            framework: "gemini-cli".to_string(),
            steps,
            total_tokens,
            task_value_score: 1.0,
            metadata,
        },
        format: "gemini-cli",
        issues,
    })
}

fn json_records(raw: &str) -> Result<(Vec<Value>, usize)> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("transcript is empty");
    }
    if trimmed.starts_with('[') {
        let records = serde_json::from_str::<Vec<Value>>(trimmed)
            .context("transcript JSON array is invalid")?;
        return Ok((records, 0));
    }
    if !trimmed.contains('\n') {
        if let Ok(record) = serde_json::from_str::<Value>(trimmed) {
            return Ok((vec![record], 0));
        }
    }
    let mut records = Vec::new();
    let mut invalid = 0;
    for line in raw.lines().map(str::trim).filter(|line| !line.is_empty()) {
        match serde_json::from_str::<Value>(line) {
            Ok(record) => records.push(record),
            Err(_) => invalid += 1,
        }
    }
    if records.is_empty() {
        bail!("transcript contains no valid JSON records");
    }
    Ok((records, invalid))
}

fn upsert_message(messages: &mut Vec<Value>, message: Value) {
    let id = message.get("id").and_then(Value::as_str);
    if let Some(index) = id.and_then(|id| {
        messages
            .iter()
            .position(|current| current.get("id").and_then(Value::as_str) == Some(id))
    }) {
        messages[index] = message;
    } else {
        messages.push(message);
    }
}

fn push_reasoning_step(
    steps: &mut Vec<TraceStep>,
    content: &str,
    context: &str,
    agent_id: Option<String>,
) {
    let index = steps.len();
    steps.push(TraceStep {
        id: index as u32 + 1,
        step_type: StepType::Reasoning,
        content: clip_host_text(content, 2_000),
        input_context: (!context.is_empty()).then(|| clip_host_text(context, 4_000)),
        agent_id,
        ..Default::default()
    });
}

fn distribute_tokens(steps: &mut [TraceStep], start: usize, total: u32) {
    let count = steps.len().saturating_sub(start);
    if count == 0 {
        return;
    }
    let share = total / count as u32;
    let mut remainder = total % count as u32;
    for step in &mut steps[start..] {
        step.tokens = share + u32::from(remainder > 0);
        remainder = remainder.saturating_sub(1);
    }
}

fn usage_total(usage: &Value) -> Option<u32> {
    for key in ["total_tokens", "totalTokenCount", "total"] {
        if let Some(value) = usage.get(key).and_then(json_u32) {
            return Some(value);
        }
    }
    let keys = [
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_output_tokens",
        "input",
        "output",
        "cached",
        "thoughts",
        "tool",
    ];
    let mut saw = false;
    let mut total = 0_u32;
    for key in keys {
        if let Some(value) = usage.get(key).and_then(json_u32) {
            saw = true;
            total = total.saturating_add(value);
        }
    }
    saw.then_some(total)
}

fn json_u32(value: &Value) -> Option<u32> {
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
        .map(|value| u32::try_from(value).unwrap_or(u32::MAX))
}

fn parse_json_string_value(value: Value) -> Value {
    if let Some(text) = value.as_str() {
        serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
    } else {
        value
    }
}

fn host_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(values) => values
            .iter()
            .map(host_text)
            .filter(|text| !text.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(values) => {
            let mut parts = Vec::new();
            for key in [
                "text",
                "summary_text",
                "output_text",
                "subject",
                "description",
            ] {
                if let Some(text) = values.get(key).and_then(Value::as_str) {
                    if !text.trim().is_empty() {
                        parts.push(text.to_string());
                    }
                }
            }
            if parts.is_empty() {
                if let Some(content) = values.get("content") {
                    return host_text(content);
                }
            }
            parts.join("\n")
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => String::new(),
    }
}

fn host_text_or_json(value: &Value) -> String {
    let text = host_text(value);
    if !text.is_empty() {
        text
    } else if value.is_null() {
        String::new()
    } else {
        serde_json::to_string(value).unwrap_or_default()
    }
}

fn clip_host_text(text: &str, limit: usize) -> String {
    let count = text.chars().count();
    if count <= limit {
        return text.to_string();
    }
    let clipped = text.chars().take(limit).collect::<String>();
    format!("{clipped}...[+{} chars]", count - limit)
}

fn lifecycle_event_type(event: AgentHookEvent) -> &'static str {
    match event {
        AgentHookEvent::SessionStart | AgentHookEvent::SubagentStart => "run_start",
        AgentHookEvent::SessionEnd
        | AgentHookEvent::AfterAgent
        | AgentHookEvent::SubagentStop
        | AgentHookEvent::PreCompress
        | AgentHookEvent::Stop => "run_end",
    }
}

fn redacted_text(value: &str) -> String {
    format!(
        "[redacted sha256={} chars={}]",
        sha256_hex(value.as_bytes()),
        value.chars().count()
    )
}

fn redact_untrusted(value: &mut Value) {
    match value {
        Value::String(text) => *text = redacted_text(text),
        Value::Array(values) => values.iter_mut().for_each(redact_untrusted),
        Value::Object(values) => {
            let original = std::mem::take(values);
            for (key, mut item) in original {
                redact_untrusted(&mut item);
                values.insert(
                    format!(
                        "[redacted-key sha256={} chars={}]",
                        sha256_hex(key.as_bytes()),
                        key.chars().count()
                    ),
                    item,
                );
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn redact_trace_value(trace: &mut Value) {
    if let Some(metadata) = trace.get_mut("metadata") {
        redact_untrusted(metadata);
    }
    let Some(steps) = trace.get_mut("steps").and_then(Value::as_array_mut) else {
        return;
    };
    const SENSITIVE: &[&str] = &[
        "content",
        "tool_name",
        "tool_params",
        "tool_error",
        "agent_id",
        "input_context",
        "output",
        "flag_details",
    ];
    for step in steps {
        if let Some(object) = step.as_object_mut() {
            for key in SENSITIVE {
                if let Some(value) = object.get_mut(*key) {
                    redact_untrusted(value);
                }
            }
        }
    }
}

fn redact_report_value(value: &mut Value) {
    const SENSITIVE: &[&str] = &[
        "agent_id",
        "description",
        "justification",
        "patch",
        "target",
        "tool_name",
        "tool_params",
        "tool_error",
        "content",
        "input_context",
        "output",
        "flag_details",
        "metadata",
        "error",
        "message",
        "excerpt",
    ];
    match value {
        Value::Array(values) => values.iter_mut().for_each(redact_report_value),
        Value::Object(values) => {
            for (key, item) in values {
                if SENSITIVE.contains(&key.as_str()) {
                    redact_untrusted(item);
                } else {
                    redact_report_value(item);
                }
            }
        }
        _ => {}
    }
}

fn ensure_owned_file(
    path: &Path,
    desired: &[u8],
    kind: &str,
    state: &InstallState,
    dry_run: bool,
    actions: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> Result<Option<OwnedPath>> {
    reject_symlink(path, kind)?;
    let absolute = absolute_path(path)?;
    let path_text = absolute.to_string_lossy().to_string();
    let desired_hash = sha256_hex(desired);
    let prior_owned = state
        .installations
        .iter()
        .flat_map(|record| record.owned_paths.iter())
        .any(|owned| owned.path == path_text);
    if path.exists() {
        let existing =
            fs::read(path).with_context(|| format!("cannot read existing {}", path.display()))?;
        if existing == desired {
            if prior_owned {
                return Ok(Some(OwnedPath {
                    path: path_text,
                    sha256: desired_hash,
                    kind: kind.to_string(),
                }));
            }
            actions.push(format!("reuse existing {} {}", kind, path.display()));
            return Ok(None);
        }
        actions.push(format!(
            "back up and replace existing {} {}",
            kind,
            path.display()
        ));
        if !dry_run {
            backup_existing(path, "bak")?;
        }
        warnings.push(format!(
            "existing {} was preserved in a timestamped backup",
            path.display()
        ));
    } else {
        actions.push(format!("create {} {}", kind, path.display()));
    }
    if !dry_run {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, desired)?;
    }
    Ok(Some(OwnedPath {
        path: path_text,
        sha256: desired_hash,
        kind: kind.to_string(),
    }))
}

fn ensure_project_policy(
    path: &Path,
    desired: &str,
    state: &InstallState,
    dry_run: bool,
    actions: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> Result<Option<OwnedPath>> {
    reject_symlink(path, "policy")?;
    let absolute = absolute_path(path)?;
    let path_text = absolute.to_string_lossy().to_string();
    let prior_owned = state
        .installations
        .iter()
        .flat_map(|record| record.owned_paths.iter())
        .find(|owned| owned.path == path_text && owned.kind == "policy")
        .cloned();
    if !path.exists() {
        actions.push(format!("create policy {}", path.display()));
        if !dry_run {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, desired)?;
        }
        return Ok(Some(OwnedPath {
            path: path_text,
            sha256: sha256_hex(desired.as_bytes()),
            kind: "policy".to_string(),
        }));
    }

    let existing = fs::read(path)?;
    let existing_hash = sha256_hex(&existing);
    if existing != desired.as_bytes() {
        actions.push(format!("preserve existing policy {}", path.display()));
        warnings.push(
            "existing tracerazor.toml was not changed; its policy takes precedence over --mode"
                .to_string(),
        );
    } else {
        actions.push(format!("reuse existing policy {}", path.display()));
    }
    // Continue to own a policy only while its recorded bytes remain unchanged.
    // A user edit relinquishes ownership so uninstall can never delete it.
    Ok(prior_owned
        .filter(|owned| owned.sha256 == existing_hash)
        .map(|owned| OwnedPath {
            path: path_text,
            sha256: existing_hash,
            kind: owned.kind,
        }))
}

fn install_host_hooks(
    path: &Path,
    host: AgentHost,
    mode: AgentMode,
    prior: Option<&InstallRecord>,
    dry_run: bool,
    actions: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> Result<Vec<ManagedHook>> {
    reject_symlink(path, "host hook settings")?;
    if mode == AgentMode::Off
        && prior.is_none_or(|record| {
            record
                .managed_hooks
                .iter()
                .all(|hook| hook.path != path_string(path).unwrap_or_default())
        })
    {
        return Ok(Vec::new());
    }
    let mut settings = read_json_object(path, dry_run, warnings)?;
    if let Some(prior) = prior {
        let prior_hooks = prior
            .managed_hooks
            .iter()
            .filter(|hook| hook.path == path_string(path).unwrap_or_default())
            .cloned()
            .collect::<Vec<_>>();
        let (_, missing) = remove_managed_hooks_value(&mut settings, &prior_hooks);
        if missing > 0 {
            warnings.push(format!(
                "{} previously managed hook(s) were user-modified and left in place",
                missing
            ));
        }
        if prior_hooks.is_empty() && !prior.modified_paths.is_empty() {
            warnings.push(
                "legacy hook state had no exact fingerprints; existing handlers were preserved"
                    .to_string(),
            );
        }
    }

    let mut managed = Vec::new();
    if mode != AgentMode::Off {
        let path_text = path_string(path)?;
        let hooks = ensure_child_object(&mut settings, "hooks");
        for (event, group) in host_hook_groups(host, mode) {
            let fingerprint = hook_fingerprint(&group)?;
            ensure_child_array(hooks, event).push(group);
            managed.push(ManagedHook {
                path: path_text.clone(),
                event: event.to_string(),
                fingerprint,
            });
        }
    }
    let rendered = serde_json::to_string_pretty(&settings)?;
    if path.exists() && fs::read_to_string(path).ok().as_deref() == Some(rendered.as_str()) {
        return Ok(managed);
    }
    actions.push(format!(
        "update {} hooks in {}",
        host.as_str(),
        path.display()
    ));
    if !dry_run {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        if path.exists() {
            backup_existing(path, "bak")?;
        }
        fs::write(path, rendered)?;
    }
    Ok(managed)
}

fn uninstall_host_hooks(
    path: &Path,
    hooks: &[ManagedHook],
    dry_run: bool,
    actions: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    if is_symlink(path)? {
        warnings.push(format!(
            "left symlinked host settings untouched: {}",
            path.display()
        ));
        return Ok(());
    }
    let raw = fs::read_to_string(path)?;
    let mut settings: Value = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(_) => {
            warnings.push(format!(
                "left malformed host settings untouched: {}",
                path.display()
            ));
            return Ok(());
        }
    };
    let (removed, missing) = remove_managed_hooks_value(&mut settings, hooks);
    if missing > 0 {
        warnings.push(format!(
            "left {} user-modified managed hook(s) in place in {}",
            missing,
            path.display()
        ));
    }
    if removed == 0 {
        return Ok(());
    }
    actions.push(format!(
        "remove {} exact TraceRazor hook(s) from {}",
        removed,
        path.display()
    ));
    if !dry_run {
        backup_existing(path, "bak")?;
        fs::write(path, serde_json::to_string_pretty(&settings)?)?;
    }
    Ok(())
}

fn read_json_object(path: &Path, dry_run: bool, warnings: &mut Vec<String>) -> Result<Value> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let raw = fs::read_to_string(path)?;
    match serde_json::from_str(&raw) {
        Ok(Value::Object(map)) => Ok(Value::Object(map)),
        Ok(_) | Err(_) => {
            warnings.push(format!(
                "malformed/non-object settings will be backed up before replacement: {}",
                path.display()
            ));
            if !dry_run {
                backup_existing(path, "invalid")?;
            }
            Ok(json!({}))
        }
    }
}

fn host_hook_groups(host: AgentHost, _mode: AgentMode) -> Vec<(&'static str, Value)> {
    match host {
        AgentHost::Codex => vec![
            codex_hook(
                "SessionStart",
                "session-start",
                10,
                "Loading TraceRazor coach context",
            ),
            codex_hook(
                "SubagentStart",
                "subagent-start",
                10,
                "Linking TraceRazor child run",
            ),
            codex_hook(
                "SubagentStop",
                "subagent-stop",
                30,
                "Recording TraceRazor child run",
            ),
            codex_hook("Stop", "stop", 60, "Finalizing TraceRazor run"),
        ],
        AgentHost::Gemini => vec![
            gemini_hook(
                "SessionStart",
                "tracerazor-session-start",
                "session-start",
                10_000,
                "Load advisory TraceRazor coach context",
            ),
            gemini_hook(
                "AfterAgent",
                "tracerazor-after-agent",
                "after-agent",
                60_000,
                "Audit the completed Gemini turn locally",
            ),
            gemini_hook(
                "SessionEnd",
                "tracerazor-session-end",
                "session-end",
                60_000,
                "Finalize the local TraceRazor run",
            ),
            gemini_hook(
                "PreCompress",
                "tracerazor-pre-compress",
                "pre-compress",
                10_000,
                "Record a context-compression boundary",
            ),
        ],
        AgentHost::Claude => vec![
            claude_hook(
                "SessionStart",
                "session-start",
                10,
                "Loading TraceRazor coach context",
            ),
            claude_hook("SessionEnd", "session-end", 60, "Finalizing TraceRazor run"),
            claude_hook(
                "SubagentStart",
                "subagent-start",
                10,
                "Linking TraceRazor child run",
            ),
            claude_hook(
                "SubagentStop",
                "subagent-stop",
                30,
                "Recording TraceRazor child run",
            ),
        ],
        AgentHost::Auto | AgentHost::Generic => Vec::new(),
    }
}

fn claude_hook(
    event: &'static str,
    event_arg: &str,
    timeout: u64,
    status: &str,
) -> (&'static str, Value) {
    (
        event,
        json!({"hooks": [{
            "type": "command",
            "command": format!("tracerazor agent hook --host claude --event {event_arg}"),
            "timeout": timeout,
            "statusMessage": status
        }]}),
    )
}

fn codex_hook(
    event: &'static str,
    event_arg: &str,
    timeout: u64,
    status: &str,
) -> (&'static str, Value) {
    (
        event,
        json!({"hooks": [{
            "type": "command",
            "command": format!("tracerazor agent hook --host codex --event {event_arg}"),
            "timeout": timeout,
            "statusMessage": status
        }]}),
    )
}

fn gemini_hook(
    event: &'static str,
    name: &str,
    event_arg: &str,
    timeout_ms: u64,
    description: &str,
) -> (&'static str, Value) {
    (
        event,
        json!({
            "hooks": [{
                "name": name,
                "type": "command",
                "command": format!("tracerazor agent hook --host gemini --event {event_arg}"),
                "timeout": timeout_ms,
                "description": description
            }]
        }),
    )
}

fn hook_fingerprint(group: &Value) -> Result<String> {
    Ok(sha256_hex(&serde_json::to_vec(group)?))
}

fn remove_managed_hooks_value(settings: &mut Value, hooks: &[ManagedHook]) -> (usize, usize) {
    let Some(events) = settings.get_mut("hooks").and_then(Value::as_object_mut) else {
        return (0, hooks.len());
    };
    let mut removed = 0;
    let mut missing = 0;
    for hook in hooks {
        let Some(groups) = events.get_mut(&hook.event).and_then(Value::as_array_mut) else {
            missing += 1;
            continue;
        };
        let position = groups.iter().rposition(|group| {
            hook_fingerprint(group)
                .map(|fingerprint| fingerprint == hook.fingerprint)
                .unwrap_or(false)
        });
        if let Some(position) = position {
            groups.remove(position);
            removed += 1;
        } else {
            missing += 1;
        }
    }
    (removed, missing)
}

fn ensure_object(value: &mut Value) {
    if !value.is_object() {
        *value = json!({});
    }
}

fn ensure_child_object<'a>(
    value: &'a mut Value,
    key: &str,
) -> &'a mut serde_json::Map<String, Value> {
    ensure_object(value);
    let child = value
        .as_object_mut()
        .expect("object was ensured")
        .entry(key.to_string())
        .or_insert_with(|| json!({}));
    ensure_object(child);
    child.as_object_mut().expect("child object was ensured")
}

fn ensure_child_array<'a>(
    object: &'a mut serde_json::Map<String, Value>,
    key: &str,
) -> &'a mut Vec<Value> {
    let child = object.entry(key.to_string()).or_insert_with(|| json!([]));
    if !child.is_array() {
        *child = json!([]);
    }
    child.as_array_mut().expect("child array was ensured")
}

fn record_status(
    record: &InstallRecord,
    root: &Path,
    policy: Option<&AgentPolicy>,
) -> Result<Value> {
    let mut files = Vec::new();
    let mut healthy = true;
    for owned in &record.owned_paths {
        let path = PathBuf::from(&owned.path);
        if !is_expected_owned_path(record, owned, root)? {
            healthy = false;
            files.push(json!({
                "path": owned.path,
                "kind": owned.kind,
                "exists": path.exists(),
                "owned_content_matches": false,
                "unexpected_path": true,
            }));
            continue;
        }
        let exists = path.exists();
        let matches = if exists {
            sha256_hex(&fs::read(&path)?) == owned.sha256
        } else {
            false
        };
        healthy &= matches;
        files.push(json!({
            "path": owned.path,
            "kind": owned.kind,
            "exists": exists,
            "owned_content_matches": matches,
        }));
    }
    for modified in &record.modified_paths {
        let path = PathBuf::from(modified);
        if !is_expected_modified_path(record, &path, root)? {
            healthy = false;
            files.push(json!({
                "path": modified,
                "kind": "host_settings",
                "exists": path.exists(),
                "configured": false,
                "unexpected_path": true,
            }));
            continue;
        }
        let configured = path
            .exists()
            .then(|| fs::read_to_string(&path).ok())
            .flatten()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .is_some_and(|settings| {
                let hooks = record
                    .managed_hooks
                    .iter()
                    .filter(|hook| hook.path == *modified)
                    .collect::<Vec<_>>();
                !hooks.is_empty()
                    && hooks
                        .iter()
                        .all(|hook| managed_hook_exists(&settings, hook))
            });
        healthy &= configured;
        files.push(json!({
            "path": modified,
            "kind": "host_settings",
            "exists": path.exists(),
            "configured": configured,
        }));
    }
    let automatic_capture = !record.managed_hooks.is_empty();
    let expected_capture = policy
        .map(AgentPolicy::automatic_capture)
        .unwrap_or(automatic_capture);
    let capture_consistent =
        record.host == AgentHost::Generic || automatic_capture == expected_capture;
    healthy &= capture_consistent;
    let mcp_registered = mcp_config_path(record.host, record.scope, root)
        .map(|path| mcp_registration_present(record.host, &path).unwrap_or(false))
        .unwrap_or(false);
    if record.host != AgentHost::Generic {
        healthy &= mcp_registered;
    }
    Ok(json!({
        "host": record.host,
        "scope": record.scope,
        "mode": record.mode,
        "installed_at": record.installed_at,
        "automatic_capture": automatic_capture,
        "policy_automatic_capture": expected_capture,
        "capture_consistent": capture_consistent,
        "mcp_registered": mcp_registered,
        "capture_status": capture_capability(record.host, &record.managed_hooks).0,
        "capture_reason": capture_capability(record.host, &record.managed_hooks).1,
        "hook_trust_required": matches!(record.host, AgentHost::Codex | AgentHost::Gemini),
        "healthy": healthy,
        "files": files,
    }))
}

fn mcp_registration_present(host: AgentHost, path: &Path) -> Result<bool> {
    if !path.exists() || is_symlink(path)? {
        return Ok(false);
    }
    let raw = fs::read_to_string(path)?;
    if host == AgentHost::Codex {
        let value: toml::Value = match toml::from_str(&raw) {
            Ok(value) => value,
            Err(_) => return Ok(false),
        };
        return Ok(value
            .get("mcp_servers")
            .and_then(|value| value.get("tracerazor"))
            .and_then(|value| value.get("command"))
            .and_then(toml::Value::as_str)
            == Some("tracerazor-mcp"));
    }
    let value: Value = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(_) => return Ok(false),
    };
    Ok(value["mcpServers"]["tracerazor"]["command"] == "tracerazor-mcp")
}

fn capture_capability(
    host: AgentHost,
    managed_hooks: &[ManagedHook],
) -> (&'static str, Option<&'static str>) {
    if managed_hooks.is_empty() {
        return ("disabled", None);
    }
    match host {
        AgentHost::Claude | AgentHost::Codex | AgentHost::Gemini => {
            ("complete_when_transcript_available", None)
        }
        AgentHost::Auto | AgentHost::Generic => ("disabled", None),
    }
}

fn managed_hook_exists(settings: &Value, hook: &ManagedHook) -> bool {
    settings
        .get("hooks")
        .and_then(Value::as_object)
        .and_then(|events| events.get(&hook.event))
        .and_then(Value::as_array)
        .is_some_and(|groups| {
            groups.iter().any(|group| {
                hook_fingerprint(group)
                    .map(|fingerprint| fingerprint == hook.fingerprint)
                    .unwrap_or(false)
            })
        })
}

fn render_policy(policy: &AgentPolicy) -> String {
    format!(
        "schema_version = 1\nmode = \"{}\"\ncapture = \"{}\"\nhermetic = {}\nprivacy = \"{}\"\npersist_raw_content = {}\nartifact_dir = \"{}\"\nmin_steps = {}\n\n[quality]\nverifier = {}\n\n[enforcement]\nenabled = {}\n",
        policy.mode.as_str(),
        policy.capture_str(),
        policy.hermetic,
        policy.privacy_str(),
        policy.persist_raw_content,
        policy.artifact_dir.replace('"', "\\\""),
        policy.min_steps,
        serde_json::to_string(&policy.quality.verifier).unwrap_or_else(|_| "\"\"".to_string()),
        policy.enforcement.enabled,
    )
}

fn load_policy_file(path: &Path) -> Result<AgentPolicy> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("cannot read TraceRazor policy {}", path.display()))?;
    let value: toml::Value =
        toml::from_str(&raw).with_context(|| format!("invalid TOML policy {}", path.display()))?;
    let selected = value.get("tracerazor").cloned().unwrap_or(value);
    let policy: AgentPolicy = selected
        .try_into()
        .with_context(|| format!("invalid TraceRazor policy fields in {}", path.display()))?;
    policy.validate(path)?;
    Ok(policy)
}

fn describe_policy(path: &Path) -> (Value, Option<String>) {
    if !path.exists() {
        return (
            json!({
                "path": path.to_string_lossy(),
                "exists": false,
                "valid": true,
                "automatic_capture": false,
            }),
            None,
        );
    }
    match load_policy_file(path) {
        Ok(policy) => (
            json!({
                "path": path.to_string_lossy(),
                "exists": true,
                "valid": true,
                "mode": policy.mode,
                "capture": policy.capture_str(),
                "automatic_capture": policy.automatic_capture(),
                "hermetic": policy.hermetic,
                "privacy": policy.privacy_str(),
                "persist_raw_content": policy.persist_raw_content,
                "artifact_dir": policy.artifact_dir,
                "min_steps": policy.min_steps,
                "verifier": policy.quality.verifier,
                "enforcement_enabled": policy.enforcement.enabled,
                "enforcement_executed": false,
            }),
            None,
        ),
        Err(error) => {
            let message = format!("invalid policy {}: {error:#}", path.display());
            (
                json!({
                    "path": path.to_string_lossy(),
                    "exists": true,
                    "valid": false,
                    "error": message,
                    "automatic_capture": false,
                }),
                Some(message),
            )
        }
    }
}

fn effective_policy_path() -> Result<PathBuf> {
    if let Some(path) = env_nonempty("TRACERAZOR_POLICY") {
        return absolute_path(Path::new(&path));
    }
    discover_policy()
}

fn policy_path_for_scope(scope: AgentScope, root: &Path) -> Result<PathBuf> {
    if let Some(path) = env_nonempty("TRACERAZOR_POLICY") {
        return absolute_path(Path::new(&path));
    }
    match scope {
        AgentScope::Project | AgentScope::Image => Ok(root.join("tracerazor.toml")),
        AgentScope::User => discover_policy(),
    }
}

fn load_effective_policy() -> Result<LoadedPolicy> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let path = effective_policy_path()?;
    if path.exists() {
        let policy = load_policy_file(&path)?;
        let root = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.clone());
        return Ok(LoadedPolicy {
            policy,
            path: Some(path),
            root,
        });
    }
    Ok(LoadedPolicy {
        policy: AgentPolicy::default(),
        path: None,
        root: find_project_root(&cwd),
    })
}

fn artifact_root(loaded: &LoadedPolicy) -> Result<PathBuf> {
    let root = absolute_path(&loaded.root)?;
    let artifact = root.join(&loaded.policy.artifact_dir);
    ensure_confined_path(&root, &artifact, "artifact_dir")?;
    Ok(artifact)
}

fn latest_coach_context(loaded: &LoadedPolicy) -> Result<Option<String>> {
    let artifact = artifact_root(loaded)?;
    if !artifact.exists() {
        return Ok(None);
    }
    let mut candidates = fs::read_dir(&artifact)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path().join("report.json"))
        .filter(|path| path.is_file() && !is_symlink(path).unwrap_or(true))
        .filter_map(|path| {
            let modified = fs::metadata(&path).ok()?.modified().ok()?;
            Some((modified, path))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.0.cmp(&a.0));
    let Some((_, report_path)) = candidates.into_iter().next() else {
        return Ok(None);
    };
    ensure_confined_path(&loaded.root, &report_path, "coach report")?;
    let report: Value = serde_json::from_str(&fs::read_to_string(&report_path)?)?;
    let score = report["score"]["score"].as_f64();
    let grade = report["score"]["grade"].as_str().unwrap_or("ungraded");
    let tokens = report["total_tokens"].as_u64().unwrap_or(0);
    let run_id = report["trace_id"]
        .as_str()
        .or_else(|| report_path.parent()?.file_name()?.to_str())
        .unwrap_or("previous-run");
    let findings = report["fixes"]
        .as_array()
        .into_iter()
        .flatten()
        .take(3)
        .filter_map(|fix| {
            let kind = fix["fix_type"].as_str()?;
            let risk = fix["risk"].as_str().unwrap_or("needs_review");
            let estimated = fix["estimated_token_savings"].as_u64().unwrap_or(0);
            Some(format!("{kind} ({risk}, estimated {estimated} tokens)"))
        })
        .collect::<Vec<_>>();
    let mut context = if let Some(score) = score {
        format!(
            "TraceRazor coach: previous run {} scored TAS {:.0}/100 ({grade}) across {tokens} tokens.",
            safe_label(run_id),
            score
        )
    } else {
        format!(
            "TraceRazor coach: previous run {} has findings.",
            safe_label(run_id)
        )
    };
    if !findings.is_empty() {
        context.push_str(" Review: ");
        context.push_str(&findings.join("; "));
        context.push('.');
    }
    context.push_str(" Advice is local, advisory, and must be validated by rerun.");
    Ok(Some(context.chars().take(700).collect()))
}

fn safe_label(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
        .take(32)
        .collect()
}

fn host_settings_path(host: AgentHost, scope: AgentScope, root: &Path) -> Option<PathBuf> {
    match host {
        AgentHost::Codex => Some(root.join(".codex").join("hooks.json")),
        AgentHost::Gemini => Some(root.join(".gemini").join("settings.json")),
        AgentHost::Claude => Some(match scope {
            AgentScope::Project => root.join(".claude").join("settings.local.json"),
            AgentScope::User | AgentScope::Image => root.join(".claude").join("settings.json"),
        }),
        AgentHost::Auto | AgentHost::Generic => None,
    }
}

fn host_skill_path(host: AgentHost, root: &Path) -> Option<PathBuf> {
    match host {
        AgentHost::Codex | AgentHost::Gemini => Some(
            root.join(".agents")
                .join("skills")
                .join("tracerazor")
                .join("SKILL.md"),
        ),
        AgentHost::Claude => Some(
            root.join(".claude")
                .join("skills")
                .join("tracerazor")
                .join("SKILL.md"),
        ),
        AgentHost::Auto | AgentHost::Generic => None,
    }
}

fn mcp_config_path(host: AgentHost, scope: AgentScope, root: &Path) -> Option<PathBuf> {
    match host {
        AgentHost::Codex => Some(root.join(".codex").join("config.toml")),
        AgentHost::Claude => Some(match scope {
            AgentScope::Project | AgentScope::Image => root.join(".mcp.json"),
            AgentScope::User => root.join(".claude").join(".mcp.json"),
        }),
        AgentHost::Gemini => Some(root.join(".gemini").join("settings.json")),
        AgentHost::Auto | AgentHost::Generic => None,
    }
}

fn desired_mcp_value() -> Value {
    json!({"command": "tracerazor-mcp", "args": []})
}

fn install_mcp_registration(
    path: &Path,
    host: AgentHost,
    prior: Option<&InstallRecord>,
    dry_run: bool,
    actions: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> Result<(Vec<ManagedConfig>, bool)> {
    reject_symlink(path, "MCP configuration")?;
    let path_text = path_string(path)?;
    let kind = if host == AgentHost::Codex {
        "codex_toml_mcp"
    } else {
        "json_mcp"
    };
    let desired_fingerprint = if host == AgentHost::Codex {
        sha256_hex(CODEX_MCP_BLOCK.as_bytes())
    } else {
        sha256_hex(&serde_json::to_vec(&desired_mcp_value())?)
    };
    let prior_owned = prior.is_some_and(|record| {
        record.managed_configs.iter().any(|entry| {
            entry.path == path_text
                && entry.kind == kind
                && entry.fingerprint == desired_fingerprint
        })
    });

    if host == AgentHost::Codex {
        let mut raw = fs::read_to_string(path).unwrap_or_default();
        if raw.contains(CODEX_MCP_BLOCK) {
            return Ok((
                prior_owned
                    .then(|| ManagedConfig {
                        path: path_text,
                        kind: kind.to_string(),
                        fingerprint: desired_fingerprint,
                    })
                    .into_iter()
                    .collect(),
                true,
            ));
        }
        if let Ok(value) = toml::from_str::<toml::Value>(&raw) {
            if let Some(existing) = value
                .get("mcp_servers")
                .and_then(|value| value.get("tracerazor"))
            {
                let registered =
                    existing.get("command").and_then(toml::Value::as_str) == Some("tracerazor-mcp");
                if registered {
                    return Ok((Vec::new(), true));
                }
                bail!(
                    "existing unmanaged Codex mcp_servers.tracerazor entry in {} was preserved",
                    path.display()
                );
            }
        }
        if !raw.is_empty() && !raw.ends_with('\n') {
            raw.push('\n');
        }
        raw.push_str(CODEX_MCP_BLOCK);
        actions.push(format!("register tracerazor-mcp in {}", path.display()));
        if !dry_run {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            if path.exists() {
                backup_existing(path, "bak")?;
            }
            fs::write(path, raw)?;
        }
        return Ok((
            vec![ManagedConfig {
                path: path_text,
                kind: kind.to_string(),
                fingerprint: desired_fingerprint,
            }],
            true,
        ));
    }

    let mut settings = read_json_object(path, dry_run, warnings)?;
    ensure_object(&mut settings);
    let servers = ensure_child_object(&mut settings, "mcpServers");
    let desired = desired_mcp_value();
    if let Some(existing) = servers.get("tracerazor") {
        if existing == &desired {
            return Ok((
                prior_owned
                    .then(|| ManagedConfig {
                        path: path_text,
                        kind: kind.to_string(),
                        fingerprint: desired_fingerprint,
                    })
                    .into_iter()
                    .collect(),
                true,
            ));
        }
        warnings.push(format!(
            "existing tracerazor MCP entry in {} was backed up and replaced",
            path.display()
        ));
    }
    servers.insert("tracerazor".to_string(), desired);
    actions.push(format!("register tracerazor-mcp in {}", path.display()));
    if !dry_run {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        if path.exists() {
            backup_existing(path, "bak")?;
        }
        fs::write(path, serde_json::to_string_pretty(&settings)?)?;
    }
    Ok((
        vec![ManagedConfig {
            path: path_text,
            kind: kind.to_string(),
            fingerprint: desired_fingerprint,
        }],
        true,
    ))
}

fn uninstall_mcp_registration(
    path: &Path,
    managed: &ManagedConfig,
    dry_run: bool,
    actions: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    reject_symlink(path, "MCP configuration")?;
    if managed.kind == "codex_toml_mcp" {
        let mut raw = fs::read_to_string(path)?;
        if sha256_hex(CODEX_MCP_BLOCK.as_bytes()) != managed.fingerprint
            || !raw.contains(CODEX_MCP_BLOCK)
        {
            warnings.push(format!(
                "left user-modified Codex MCP registration in {}",
                path.display()
            ));
            return Ok(());
        }
        raw = raw.replacen(CODEX_MCP_BLOCK, "", 1);
        actions.push(format!("remove tracerazor-mcp from {}", path.display()));
        if !dry_run {
            backup_existing(path, "bak")?;
            fs::write(path, raw)?;
        }
        return Ok(());
    }

    let raw = fs::read_to_string(path)?;
    let mut settings: Value = serde_json::from_str(&raw)?;
    let Some(servers) = settings
        .get_mut("mcpServers")
        .and_then(Value::as_object_mut)
    else {
        return Ok(());
    };
    let Some(existing) = servers.get("tracerazor") else {
        return Ok(());
    };
    if sha256_hex(&serde_json::to_vec(existing)?) != managed.fingerprint {
        warnings.push(format!(
            "left user-modified MCP registration in {}",
            path.display()
        ));
        return Ok(());
    }
    servers.remove("tracerazor");
    actions.push(format!("remove tracerazor-mcp from {}", path.display()));
    if !dry_run {
        backup_existing(path, "bak")?;
        fs::write(path, serde_json::to_string_pretty(&settings)?)?;
    }
    Ok(())
}

fn scope_root(scope: AgentScope) -> Result<PathBuf> {
    match scope {
        AgentScope::Project => {
            let cwd = std::env::current_dir().context("cannot determine current directory")?;
            Ok(find_project_root(&cwd))
        }
        AgentScope::User => {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .context("HOME/USERPROFILE is not set for user scope")?;
            absolute_path(Path::new(&home))
        }
        AgentScope::Image => {
            let root = std::env::var("TRACERAZOR_IMAGE_ROOT").context(
                "image scope requires TRACERAZOR_IMAGE_ROOT to name the provisioned image root",
            )?;
            let path = PathBuf::from(root);
            if !path.exists() {
                bail!("TRACERAZOR_IMAGE_ROOT does not exist: {}", path.display());
            }
            absolute_path(&path)
        }
    }
}

fn find_project_root(start: &Path) -> PathBuf {
    let mut current = start.to_path_buf();
    loop {
        if current.join("tracerazor.toml").exists() || current.join(".git").exists() {
            return current;
        }
        let Some(parent) = current.parent() else {
            return start.to_path_buf();
        };
        current = parent.to_path_buf();
    }
}

fn read_state(path: &Path) -> Result<StateRead> {
    reject_symlink(path, "agent installation state")?;
    if !path.exists() {
        return Ok(StateRead {
            state: InstallState {
                schema_version: STATE_SCHEMA_VERSION,
                installations: Vec::new(),
            },
            invalid: false,
        });
    }
    let raw = fs::read_to_string(path)?;
    match serde_json::from_str::<InstallState>(&raw) {
        Ok(state) if state.schema_version == STATE_SCHEMA_VERSION => Ok(StateRead {
            state,
            invalid: false,
        }),
        Ok(_) | Err(_) => Ok(StateRead {
            state: InstallState {
                schema_version: STATE_SCHEMA_VERSION,
                installations: Vec::new(),
            },
            invalid: true,
        }),
    }
}

fn write_state(path: &Path, state: &InstallState) -> Result<()> {
    reject_symlink(path, "agent installation state")?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(state)?)?;
    Ok(())
}

fn is_expected_owned_path(record: &InstallRecord, owned: &OwnedPath, root: &Path) -> Result<bool> {
    if matches!(record.scope, AgentScope::Project | AgentScope::Image) && owned.kind == "policy" {
        return Ok(path_string(&root.join("tracerazor.toml"))? == owned.path);
    }
    if owned.kind != "agent_skill" {
        return Ok(false);
    }
    let expected = match record.host {
        AgentHost::Codex | AgentHost::Gemini => root
            .join(".agents")
            .join("skills")
            .join("tracerazor")
            .join("SKILL.md"),
        AgentHost::Claude => root
            .join(".claude")
            .join("skills")
            .join("tracerazor")
            .join("SKILL.md"),
        AgentHost::Auto | AgentHost::Generic => return Ok(false),
    };
    Ok(path_string(&expected)? == owned.path)
}

fn is_expected_modified_path(record: &InstallRecord, path: &Path, root: &Path) -> Result<bool> {
    let Some(expected) = host_settings_path(record.host, record.scope, root) else {
        return Ok(false);
    };
    Ok(path_string(&expected)? == path_string(path)?)
}

fn reject_symlink(path: &Path, kind: &str) -> Result<()> {
    if is_symlink(path)? {
        bail!(
            "refusing to modify symlinked {} at {}",
            kind,
            path.display()
        );
    }
    Ok(())
}

fn ensure_confined_path(root: &Path, path: &Path, kind: &str) -> Result<()> {
    let absolute_root = absolute_path(root)?;
    let absolute_path = absolute_path(path)?;
    let relative = absolute_path
        .strip_prefix(&absolute_root)
        .with_context(|| {
            format!(
                "refusing to access {} outside scope root {}: {}",
                kind,
                absolute_root.display(),
                absolute_path.display()
            )
        })?;
    let canonical_root = fs::canonicalize(&absolute_root)
        .with_context(|| format!("cannot resolve scope root {}", absolute_root.display()))?;
    let mut current = absolute_root;
    for component in relative.components() {
        let std::path::Component::Normal(component) = component else {
            bail!(
                "unsafe {} path component in {}",
                kind,
                absolute_path.display()
            );
        };
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() {
                    bail!(
                        "refusing to access {} through symlink {}",
                        kind,
                        current.display()
                    );
                }
                let resolved = fs::canonicalize(&current)?;
                if !resolved.starts_with(&canonical_root) {
                    bail!(
                        "refusing to access {} outside resolved scope root: {}",
                        kind,
                        resolved.display()
                    );
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn nofollow_open(options: &mut OpenOptions, path: &Path) -> std::io::Result<std::fs::File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // Refuse normal traversal through a reparse-point leaf. The explicit
        // symlink check below remains the primary, portable guard.
        options.custom_flags(0x0020_0000);
    }
    options.open(path)
}

fn safe_read_optional(root: &Path, path: &Path, kind: &str) -> Result<Option<String>> {
    ensure_confined_path(root, path, kind)?;
    reject_symlink(path, kind)?;
    match fs::read_to_string(path) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn safe_append_line(root: &Path, path: &Path, line: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    ensure_confined_path(root, path, "events spool")?;
    reject_symlink(path, "events spool")?;
    let mut options = OpenOptions::new();
    options.create(true).append(true).write(true);
    let mut file = nofollow_open(&mut options, path)?;
    writeln!(file, "{line}")?;
    file.sync_all()?;
    Ok(())
}

fn safe_atomic_write(root: &Path, path: &Path, bytes: &[u8], kind: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    ensure_confined_path(root, path, kind)?;
    reject_symlink(path, kind)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", random_hex(8)));
    ensure_confined_path(root, &temporary, "temporary artifact")?;
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    let mut file = nofollow_open(&mut options, &temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    reject_symlink(path, kind)?;
    if path.exists() {
        fs::remove_file(path)?;
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

fn is_symlink(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(metadata.file_type().is_symlink()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn backup_existing(path: &Path, kind: &str) -> Result<Option<PathBuf>> {
    if !path.exists() {
        return Ok(None);
    }
    let stamp = Utc::now().format("%Y%m%d%H%M%S%3f");
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config");
    let backup = path.with_file_name(format!("{name}.{kind}.{stamp}"));
    fs::copy(path, &backup)?;
    Ok(Some(backup))
}

fn remove_empty_parent(parent: Option<&Path>) {
    let Some(parent) = parent else { return };
    let is_empty = fs::read_dir(parent)
        .ok()
        .is_some_and(|mut entries| entries.next().is_none());
    if is_empty {
        let _ = fs::remove_dir(parent);
    }
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn path_string(path: &Path) -> Result<String> {
    Ok(absolute_path(path)?.to_string_lossy().to_string())
}

fn host_detections(root: &Path) -> Vec<HostDetection> {
    [
        (AgentHost::Codex, "codex", ".codex"),
        (AgentHost::Claude, "claude", ".claude"),
        (AgentHost::Gemini, "gemini", ".gemini"),
    ]
    .into_iter()
    .map(|(host, executable, config)| {
        let resolved = command_on_path(executable);
        let project_config = root.join(config).exists();
        HostDetection {
            host: host.as_str(),
            detected: resolved.is_some() || project_config,
            executable: resolved.map(|path| path.to_string_lossy().to_string()),
            project_config,
        }
    })
    .collect()
}

fn resolve_auto_host(detections: &[HostDetection]) -> AgentHost {
    for desired in [AgentHost::Codex, AgentHost::Claude, AgentHost::Gemini] {
        if detections
            .iter()
            .any(|item| item.host == desired.as_str() && item.detected)
        {
            return desired;
        }
    }
    AgentHost::Generic
}

fn command_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    #[cfg(windows)]
    let suffixes = std::env::var("PATHEXT")
        .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string())
        .split(';')
        .map(|suffix| suffix.to_ascii_lowercase())
        .collect::<Vec<_>>();
    for directory in std::env::split_paths(&path) {
        let direct = directory.join(name);
        if direct.is_file() {
            return Some(direct);
        }
        #[cfg(windows)]
        for suffix in &suffixes {
            let candidate = directory.join(format!("{name}{suffix}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn discover_policy() -> Result<PathBuf> {
    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let mut current = cwd.as_path();
    loop {
        let candidate = current.join("tracerazor.toml");
        if candidate.exists() {
            return Ok(candidate);
        }
        let Some(parent) = current.parent() else {
            return Ok(cwd.join("tracerazor.toml"));
        };
        current = parent;
    }
}

fn generated_id(prefix: &str) -> String {
    let mut random = [0_u8; 8];
    rand::thread_rng().fill_bytes(&mut random);
    format!(
        "{}-{}-{}",
        prefix,
        Utc::now().timestamp_millis(),
        hex(&random)
    )
}

fn derived_safe_id(prefix: &str, seed: &str) -> String {
    let digest = sha256_hex(seed.as_bytes());
    format!("{prefix}-{}", &digest[..24])
}

fn generate_traceparent() -> String {
    let mut trace = [0_u8; 16];
    let mut span = [0_u8; 8];
    rand::thread_rng().fill_bytes(&mut trace);
    rand::thread_rng().fill_bytes(&mut span);
    format!("00-{}-{}-01", hex(&trace), hex(&span))
}

fn random_hex(bytes: usize) -> String {
    let mut value = vec![0_u8; bytes];
    loop {
        rand::thread_rng().fill_bytes(&mut value);
        if value.iter().any(|byte| *byte != 0) {
            return hex(&value);
        }
    }
}

fn is_w3c_hex(value: &str, width: usize) -> bool {
    value.len() == width
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && value.bytes().any(|byte| byte != b'0')
}

fn traceparent_ids(value: &str) -> Result<(String, String)> {
    let parts = value.split('-').collect::<Vec<_>>();
    let valid_hex = |text: &str, width: usize| {
        text.len() == width && text.bytes().all(|byte| byte.is_ascii_hexdigit())
    };
    if parts.len() != 4
        || parts[0] != "00"
        || !valid_hex(parts[1], 32)
        || !valid_hex(parts[2], 16)
        || !valid_hex(parts[3], 2)
        || parts[1].bytes().all(|byte| byte == b'0')
        || parts[2].bytes().all(|byte| byte == b'0')
    {
        bail!("TRACEPARENT must be a valid W3C version 00 trace context");
    }
    Ok((parts[1].to_lowercase(), parts[2].to_lowercase()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn json_string(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

fn first_safe_id(candidates: &[Option<String>]) -> Option<String> {
    candidates
        .iter()
        .filter_map(|candidate| candidate.as_deref())
        .find(|candidate| is_safe_id(candidate))
        .map(str::to_string)
}

fn is_safe_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

fn emit(format: AgentOutputFormat, value: &Value, text: String) -> Result<()> {
    match format {
        AgentOutputFormat::Json => println!("{}", serde_json::to_string_pretty(value)?),
        AgentOutputFormat::Text => println!("{text}"),
    }
    Ok(())
}
