//! Focused end-to-end coverage for the explicit agent bootstrap contract.

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::{json, Value};
use tempfile::TempDir;

const TEST_SIGNING_KEY: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn cli(home: &TempDir) -> Command {
    let mut command = Command::cargo_bin("tracerazor").unwrap();
    command
        .env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .env_remove("TRACERAZOR_IMAGE_ROOT")
        .env_remove("TRACERAZOR_RUN_ID")
        .env_remove("TRACERAZOR_AGENT_ID")
        .env_remove("TRACERAZOR_PARENT_AGENT_ID")
        .env_remove("TRACERAZOR_POLICY")
        .env_remove("TRACERAZOR_TRACE_ID")
        .env_remove("TRACERAZOR_PARENT_SPAN_ID")
        .env_remove("TRACERAZOR_SESSION_ID")
        .env_remove("TRACERAZOR_SIGNING_KEY")
        .env_remove("TRACERAZOR_VERIFY_KEY")
        .env_remove("TRACEPARENT")
        .env_remove("traceparent");
    command
}

fn run_json(home: &TempDir, project: &TempDir, args: &[&str]) -> Value {
    let output = cli(home)
        .current_dir(project.path())
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn assert_event_schema_shape(event: &Value) {
    let schema: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../schemas/tracerazor_event.schema.json"
    )))
    .unwrap();
    let object = event.as_object().unwrap();
    for required in schema["required"].as_array().unwrap() {
        assert!(
            object.contains_key(required.as_str().unwrap()),
            "missing schema field {required}"
        );
    }
    let properties = schema["properties"].as_object().unwrap();
    for key in object.keys() {
        assert!(
            properties.contains_key(key),
            "unexpected schema field {key}"
        );
    }
    let event_types = schema["properties"]["event_type"]["enum"]
        .as_array()
        .unwrap();
    assert!(event_types.contains(&event["event_type"]));
    assert_eq!(event["tokens"]["provenance"], "missing");
    assert_eq!(event["tokens"]["total"], 0);
    assert_eq!(event["tool"], Value::Null);
    assert_eq!(event["content"], Value::Null);
    assert_eq!(event["content_sha256"], Value::Null);
    assert_eq!(event["trace_id"].as_str().unwrap().len(), 32);
    assert_eq!(event["span_id"].as_str().unwrap().len(), 16);
    chrono::DateTime::parse_from_rfc3339(event["timestamp"].as_str().unwrap()).unwrap();
}

#[test]
fn install_dry_run_is_non_mutating() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let result = run_json(
        &home,
        &project,
        &[
            "agent",
            "install",
            "--host",
            "codex",
            "--scope",
            "project",
            "--mode",
            "coach",
            "--dry-run",
            "--format",
            "json",
        ],
    );

    assert_eq!(result["dry_run"], true);
    assert_eq!(result["host"], "codex");
    assert!(!project.path().join("tracerazor.toml").exists());
    assert!(!project.path().join(".agents").exists());
    assert!(!project
        .path()
        .join(".tracerazor")
        .join("agent-install.json")
        .exists());
}

#[test]
fn install_is_idempotent_and_status_is_structured() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let args = [
        "agent", "install", "--host", "codex", "--scope", "project", "--mode", "coach", "--format",
        "json",
    ];
    let first = run_json(&home, &project, &args);
    assert_eq!(first["changed"], true);
    let state_path = project
        .path()
        .join(".tracerazor")
        .join("agent-install.json");
    let first_state = std::fs::read(&state_path).unwrap();

    let second = run_json(&home, &project, &args);
    assert_eq!(second["changed"], false);
    assert_eq!(std::fs::read(&state_path).unwrap(), first_state);

    let status = run_json(
        &home,
        &project,
        &["agent", "status", "--host", "codex", "--format", "json"],
    );
    assert_eq!(status["command"], "status");
    assert_eq!(status["installed"], true);
    assert_eq!(status["healthy"], true);
    assert_eq!(status["installations"].as_array().unwrap().len(), 1);
}

#[test]
fn uninstall_leaves_user_modified_owned_content_in_place() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    run_json(
        &home,
        &project,
        &["agent", "install", "--host", "codex", "--format", "json"],
    );
    let skill = project
        .path()
        .join(".agents")
        .join("skills")
        .join("tracerazor")
        .join("SKILL.md");
    std::fs::write(&skill, "# User-owned replacement\n").unwrap();

    let result = run_json(
        &home,
        &project,
        &["agent", "uninstall", "--host", "codex", "--format", "json"],
    );
    assert_eq!(result["removed_records"], 1);
    assert_eq!(
        std::fs::read_to_string(&skill).unwrap(),
        "# User-owned replacement\n"
    );
    assert!(!project.path().join("tracerazor.toml").exists());
    assert!(result["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|warning| warning.as_str().unwrap().contains("user-modified")));
}

#[test]
fn install_preserves_a_preexisting_project_policy() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let policy = project.path().join("tracerazor.toml");
    let custom = "schema_version = 1\nmode = \"passive\"\n# team-owned\n";
    std::fs::write(&policy, custom).unwrap();

    let result = run_json(
        &home,
        &project,
        &[
            "agent", "install", "--host", "codex", "--mode", "coach", "--format", "json",
        ],
    );
    assert_eq!(std::fs::read_to_string(&policy).unwrap(), custom);
    assert!(result["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|warning| warning.as_str().unwrap().contains("takes precedence")));

    run_json(
        &home,
        &project,
        &["agent", "uninstall", "--host", "codex", "--format", "json"],
    );
    assert_eq!(std::fs::read_to_string(&policy).unwrap(), custom);
}

#[test]
fn committed_off_policy_disables_requested_claude_coach_hooks() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    std::fs::write(
        project.path().join("tracerazor.toml"),
        "schema_version = 1\nmode = \"off\"\n",
    )
    .unwrap();

    let result = run_json(
        &home,
        &project,
        &[
            "agent", "install", "--host", "claude", "--mode", "coach", "--format", "json",
        ],
    );
    assert_eq!(result["requested_mode"], "coach");
    assert_eq!(result["mode"], "off");
    assert_eq!(result["automatic_capture"], false);
    assert!(!project
        .path()
        .join(".claude")
        .join("settings.local.json")
        .exists());
}

#[test]
fn codex_install_adds_capture_hooks_and_uninstall_removes_only_owned_copy() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let settings = project.path().join(".codex").join("hooks.json");
    std::fs::create_dir_all(settings.parent().unwrap()).unwrap();
    let user_group = json!({"hooks": [{
        "type": "command",
        "command": "tracerazor agent hook --host codex --event session-start",
        "timeout": 10,
        "statusMessage": "Loading TraceRazor coach context"
    }]});
    std::fs::write(
        &settings,
        serde_json::to_string_pretty(&json!({
            "owner": "user",
            "hooks": {"SessionStart": [user_group]}
        }))
        .unwrap(),
    )
    .unwrap();

    let installed = run_json(
        &home,
        &project,
        &["agent", "install", "--host", "codex", "--format", "json"],
    );
    assert_eq!(installed["automatic_capture"], true);
    assert_eq!(
        installed["capture_status"],
        "complete_when_transcript_available"
    );
    assert!(installed["capture_reason"].is_null());
    assert_eq!(installed["hook_trust_required"], true);
    assert_eq!(installed["mcp_registered"], true);
    assert_eq!(
        installed["record"]["managed_hooks"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    let value: Value = serde_json::from_str(&std::fs::read_to_string(&settings).unwrap()).unwrap();
    assert_eq!(value["owner"], "user");
    assert_eq!(value["hooks"]["SessionStart"].as_array().unwrap().len(), 2);
    assert_eq!(value["hooks"]["SubagentStart"].as_array().unwrap().len(), 1);
    assert_eq!(value["hooks"]["SubagentStop"].as_array().unwrap().len(), 1);
    assert_eq!(value["hooks"]["Stop"].as_array().unwrap().len(), 1);
    assert!(
        std::fs::read_to_string(project.path().join(".codex").join("config.toml"))
            .unwrap()
            .contains("command = \"tracerazor-mcp\"")
    );

    run_json(
        &home,
        &project,
        &["agent", "uninstall", "--host", "codex", "--format", "json"],
    );
    let value: Value = serde_json::from_str(&std::fs::read_to_string(&settings).unwrap()).unwrap();
    assert_eq!(value["owner"], "user");
    assert_eq!(value["hooks"]["SessionStart"].as_array().unwrap().len(), 1);
    assert!(value["hooks"]["SubagentStart"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(value["hooks"]["SubagentStop"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(value["hooks"]["Stop"].as_array().unwrap().is_empty());
    assert!(
        !std::fs::read_to_string(project.path().join(".codex").join("config.toml"))
            .unwrap()
            .contains("tracerazor-mcp")
    );
}

#[test]
fn gemini_install_uses_millisecond_hook_shape() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let installed = run_json(
        &home,
        &project,
        &["agent", "install", "--host", "gemini", "--format", "json"],
    );
    assert_eq!(installed["automatic_capture"], true);
    assert_eq!(
        installed["capture_status"],
        "complete_when_transcript_available"
    );
    assert!(installed["capture_reason"].is_null());
    assert_eq!(installed["mcp_registered"], true);
    let settings = project.path().join(".gemini").join("settings.json");
    let value: Value = serde_json::from_str(&std::fs::read_to_string(settings).unwrap()).unwrap();
    assert_eq!(
        value["hooks"]["SessionStart"][0]["hooks"][0]["timeout"],
        10_000
    );
    assert_eq!(
        value["hooks"]["SessionEnd"][0]["hooks"][0]["timeout"],
        60_000
    );
    assert_eq!(
        value["hooks"]["AfterAgent"][0]["hooks"][0]["timeout"],
        60_000
    );
    assert_eq!(
        value["hooks"]["PreCompress"][0]["hooks"][0]["name"],
        "tracerazor-pre-compress"
    );
    for event in ["SessionStart", "AfterAgent", "SessionEnd", "PreCompress"] {
        assert!(value["hooks"][event][0].get("matcher").is_none());
    }
    assert_eq!(
        value["mcpServers"]["tracerazor"]["command"],
        "tracerazor-mcp"
    );
}

#[test]
fn shared_agent_skill_survives_until_last_host_is_removed() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    for host in ["codex", "gemini"] {
        run_json(
            &home,
            &project,
            &["agent", "install", "--host", host, "--format", "json"],
        );
    }
    let skill = project
        .path()
        .join(".agents")
        .join("skills")
        .join("tracerazor")
        .join("SKILL.md");
    run_json(
        &home,
        &project,
        &["agent", "uninstall", "--host", "codex", "--format", "json"],
    );
    assert!(skill.exists());
    run_json(
        &home,
        &project,
        &["agent", "uninstall", "--host", "gemini", "--format", "json"],
    );
    assert!(!skill.exists());
}

#[test]
fn uninstall_ignores_tampered_paths_outside_the_scope() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let outside = TempDir::new().unwrap();
    run_json(
        &home,
        &project,
        &["agent", "install", "--host", "generic", "--format", "json"],
    );
    let policy_content = std::fs::read(project.path().join("tracerazor.toml")).unwrap();
    let outside_file = outside.path().join("must-survive.toml");
    std::fs::write(&outside_file, policy_content).unwrap();
    let state_path = project
        .path()
        .join(".tracerazor")
        .join("agent-install.json");
    let mut state: Value =
        serde_json::from_str(&std::fs::read_to_string(&state_path).unwrap()).unwrap();
    state["installations"][0]["owned_paths"][0]["path"] = json!(outside_file.to_string_lossy());
    std::fs::write(&state_path, serde_json::to_string_pretty(&state).unwrap()).unwrap();

    let result = run_json(
        &home,
        &project,
        &[
            "agent",
            "uninstall",
            "--host",
            "generic",
            "--format",
            "json",
        ],
    );
    assert!(outside_file.exists());
    assert!(result["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|warning| warning.as_str().unwrap().contains("unexpected owned path")));
}

#[test]
fn doctor_and_generic_status_emit_valid_json() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    run_json(
        &home,
        &project,
        &["agent", "install", "--host", "generic", "--format", "json"],
    );

    let doctor = run_json(&home, &project, &["agent", "doctor", "--format", "json"]);
    assert_eq!(doctor["command"], "doctor");
    assert!(doctor["hosts"].is_array());
    assert_eq!(doctor["policy"]["exists"], true);

    let status = run_json(
        &home,
        &project,
        &["agent", "status", "--host", "generic", "--format", "json"],
    );
    assert_eq!(status["installed"], true);
    assert_eq!(status["healthy"], true);
}

#[test]
fn claude_uninstall_preserves_unrelated_settings_and_hooks() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let settings = project.path().join(".claude").join("settings.local.json");
    std::fs::create_dir_all(settings.parent().unwrap()).unwrap();
    std::fs::write(
        &settings,
        serde_json::to_string_pretty(&json!({
            "theme": "dark",
            "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "other-tool", "args": []}]}]}
        }))
        .unwrap(),
    )
    .unwrap();

    run_json(
        &home,
        &project,
        &["agent", "install", "--host", "claude", "--format", "json"],
    );
    run_json(
        &home,
        &project,
        &["agent", "uninstall", "--host", "claude", "--format", "json"],
    );
    let value: Value = serde_json::from_str(&std::fs::read_to_string(settings).unwrap()).unwrap();
    assert_eq!(value["theme"], "dark");
    assert_eq!(
        value["hooks"]["Stop"][0]["hooks"][0]["command"],
        "other-tool"
    );
}

#[test]
fn agent_run_propagates_context_and_child_exit_code() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let policy = project.path().join("custom-policy.toml");
    std::fs::write(&policy, "schema_version = 1\n").unwrap();
    let traceparent = "00-11111111111111111111111111111111-2222222222222222-01";
    let mut command = cli(&home);
    command
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "run-fixed")
        .env("TRACERAZOR_AGENT_ID", "parent-fixed")
        .env("TRACERAZOR_POLICY", &policy)
        .env("traceparent", traceparent)
        .args(["agent", "run", "--"]);

    #[cfg(windows)]
    command.args([
        "powershell",
        "-NoProfile",
        "-Command",
        "[Console]::Write(\"$env:TRACERAZOR_RUN_ID|$env:TRACERAZOR_PARENT_AGENT_ID|$env:TRACERAZOR_POLICY|$env:traceparent|$env:TRACEPARENT|$env:TRACERAZOR_TRACE_ID|$env:TRACERAZOR_PARENT_SPAN_ID|$env:TRACERAZOR_SESSION_ID\"); exit 7",
    ]);
    #[cfg(not(windows))]
    command.args([
        "sh",
        "-c",
        "printf '%s|%s|%s|%s|%s|%s|%s|%s' \"$TRACERAZOR_RUN_ID\" \"$TRACERAZOR_PARENT_AGENT_ID\" \"$TRACERAZOR_POLICY\" \"$traceparent\" \"$TRACEPARENT\" \"$TRACERAZOR_TRACE_ID\" \"$TRACERAZOR_PARENT_SPAN_ID\" \"$TRACERAZOR_SESSION_ID\"; exit 7",
    ]);

    command
        .assert()
        .code(7)
        .stdout(predicate::str::contains("run-fixed|parent-fixed|"))
        .stdout(predicate::str::contains(policy.to_string_lossy().as_ref()))
        .stdout(predicate::str::contains(format!(
            "{traceparent}|{traceparent}"
        )))
        .stdout(predicate::str::contains("11111111111111111111111111111111"))
        .stdout(predicate::str::contains("2222222222222222"));
    let events = std::fs::read_to_string(
        project
            .path()
            .join(".tracerazor")
            .join("runs")
            .join("run-fixed")
            .join("events.jsonl"),
    )
    .unwrap();
    let events = events
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["event_type"], "run_start");
    assert_eq!(events[1]["event_type"], "run_end");
    assert_eq!(events[0]["host"], "generic");
}

#[test]
fn gemini_session_hook_records_only_redacted_payload_and_emits_json() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let secret = "sk-do-not-persist-this-secret";
    let output = cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "run-hook")
        .write_stdin(json!({"secret": secret, "cwd": project.path()}).to_string())
        .args([
            "agent",
            "hook",
            "--host",
            "gemini",
            "--event",
            "session-start",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let hook_output: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(hook_output["suppressOutput"], true);
    assert_eq!(
        hook_output["hookSpecificOutput"]["hookEventName"],
        "SessionStart"
    );
    assert!(hook_output["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap()
        .contains("TraceRazor coach"));

    let run_dir = project
        .path()
        .join(".tracerazor")
        .join("runs")
        .join("run-hook");
    let events = std::fs::read_to_string(run_dir.join("events.jsonl")).unwrap();
    assert!(!events.contains(secret));
    let event: Value = serde_json::from_str(events.trim()).unwrap();
    assert_event_schema_shape(&event);
    assert_eq!(event["schema_version"], "tracerazor-event/v1");
    assert_eq!(event["host"], "gemini");
    assert_eq!(event["event_type"], "run_start");
    assert_eq!(event["capture"]["privacy"], "local-redacted");
    assert_eq!(event["capture"]["quality"], "partial");
    assert!(event["metadata"]["payload_sha256"].as_str().unwrap().len() == 64);
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["schema_version"], "tracerazor-run/v1");
    assert_eq!(manifest["status"], "running");
    assert_eq!(manifest["event_count"], 1);

    let terminal = cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "run-hook")
        .write_stdin(json!({"session_id": "ignored"}).to_string())
        .args([
            "agent",
            "hook",
            "--host",
            "gemini",
            "--event",
            "session-end",
        ])
        .output()
        .unwrap();
    assert!(terminal.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&terminal.stdout).unwrap(),
        json!({"suppressOutput": true})
    );
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["status"], "partial");
    assert!(manifest["ingest_quality"]["issues"]
        .as_array()
        .unwrap()
        .contains(&json!("transcript_path_missing")));
    assert_eq!(manifest["event_count"], 2);
}

#[test]
fn claude_agent_terminal_hook_runs_a_hermetic_local_audit() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let transcript = home.path().join("agent-session.jsonl");
    std::fs::write(&transcript, claude_transcript(5)).unwrap();
    cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "run-claude-audit")
        .write_stdin(
            json!({
                "session_id": "session-1",
                "transcript_path": transcript,
                "cwd": project.path()
            })
            .to_string(),
        )
        .args([
            "agent",
            "hook",
            "--host",
            "claude",
            "--event",
            "session-end",
        ])
        .assert()
        .success()
        .stdout(predicate::str::is_empty());

    let run_dir = project
        .path()
        .join(".tracerazor")
        .join("runs")
        .join("run-claude-audit");
    for name in [
        "events.jsonl",
        "manifest.json",
        "trace.json",
        "report.json",
        "findings.json",
        "validation.json",
        "run-receipt.json",
    ] {
        assert!(run_dir.join(name).exists(), "missing {name}");
    }
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["schema_version"], "tracerazor-run/v1");
    assert_eq!(manifest["status"], "completed");
    assert_eq!(manifest["capture_quality"], "complete");
    assert_eq!(manifest["replayable"], false);
    assert_eq!(manifest["verification_mode"], "non_replayable_receipt");
    assert_ne!(
        manifest["audit_trace_sha256"],
        manifest["persisted_trace_sha256"]
    );
    let report: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("report.json")).unwrap())
            .unwrap();
    assert_eq!(report["manifest"]["hermetic"], true);
    cli(&home)
        .current_dir(project.path())
        .args([
            "verify",
            run_dir.join("report.json").to_str().unwrap(),
            run_dir.join("trace.json").to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .code(1);
    cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "next-session")
        .write_stdin("{}")
        .args([
            "agent",
            "hook",
            "--host",
            "codex",
            "--event",
            "session-start",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("previous run"))
        .stdout(predicate::str::contains("TAS"))
        .stdout(predicate::str::contains("local, advisory"));
}

#[test]
fn signed_run_receipt_verifies_and_detects_receipt_and_artifact_tampering() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let transcript = home.path().join("signed-session.jsonl");
    std::fs::write(&transcript, claude_transcript(5)).unwrap();
    cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "signed-run")
        .env("TRACERAZOR_SIGNING_KEY", TEST_SIGNING_KEY)
        .write_stdin(
            json!({
                "session_id": "signed-session",
                "transcript_path": transcript,
                "hook_event_name": "SessionEnd"
            })
            .to_string(),
        )
        .args([
            "agent",
            "hook",
            "--host",
            "claude",
            "--event",
            "session-end",
        ])
        .assert()
        .success();

    let run_dir = project
        .path()
        .join(".tracerazor")
        .join("runs")
        .join("signed-run");
    let receipt_path = run_dir.join("run-receipt.json");
    let receipt: Value =
        serde_json::from_str(&std::fs::read_to_string(&receipt_path).unwrap()).unwrap();
    assert_eq!(receipt["signed"], true);
    assert_eq!(receipt["signature"]["algorithm"], "Ed25519");
    assert_eq!(
        receipt["signature"]["public_key"].as_str().unwrap().len(),
        64
    );
    assert_eq!(
        receipt["signature"]["signature"].as_str().unwrap().len(),
        128
    );

    let public_key = receipt["signature"]["public_key"].as_str().unwrap();
    let unpinned = cli(&home)
        .current_dir(project.path())
        .args([
            "agent",
            "verify-receipt",
            receipt_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(unpinned.status.success());
    let unpinned: Value = serde_json::from_slice(&unpinned.stdout).unwrap();
    assert_eq!(unpinned["status"], "valid");
    assert_eq!(unpinned["authenticated"], true);
    assert_eq!(unpinned["signer_pinned"], false);
    assert_eq!(unpinned["trusted_offline_receipt"], false);

    let verified = cli(&home)
        .current_dir(project.path())
        .args([
            "agent",
            "verify-receipt",
            receipt_path.to_str().unwrap(),
            "--verify-key",
            public_key,
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(verified.status.success());
    let verified: Value = serde_json::from_slice(&verified.stdout).unwrap();
    assert_eq!(verified["status"], "valid");
    assert_eq!(verified["authenticated"], true);
    assert_eq!(verified["signer_pinned"], true);
    assert_eq!(verified["trusted_offline_receipt"], true);
    assert_eq!(verified["hash_checks"]["persisted_trace"], "verified");
    assert_eq!(verified["hash_checks"]["report"], "verified");
    assert_eq!(verified["hash_checks"]["manifest_identity"], "verified");

    cli(&home)
        .current_dir(project.path())
        .args([
            "agent",
            "verify-receipt",
            receipt_path.to_str().unwrap(),
            "--verify-key",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "--format",
            "json",
        ])
        .assert()
        .code(1)
        .stdout(predicate::str::contains("expected Ed25519 public key"));

    let mut tampered_receipt = receipt.clone();
    tampered_receipt["run_id"] = json!("tampered-run");
    let tampered_path = run_dir.join("tampered-receipt.json");
    std::fs::write(
        &tampered_path,
        serde_json::to_string_pretty(&tampered_receipt).unwrap(),
    )
    .unwrap();
    cli(&home)
        .current_dir(project.path())
        .args([
            "agent",
            "verify-receipt",
            tampered_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .code(1)
        .stdout(predicate::str::contains("\"status\": \"tampered\""));

    std::fs::write(run_dir.join("trace.json"), "{}\n").unwrap();
    cli(&home)
        .current_dir(project.path())
        .args([
            "agent",
            "verify-receipt",
            receipt_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .code(1)
        .stdout(predicate::str::contains("persisted_trace"))
        .stdout(predicate::str::contains("mismatch"));
}

#[test]
fn unsigned_run_receipt_is_valid_but_never_trusted() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let transcript = home.path().join("unsigned-session.jsonl");
    std::fs::write(&transcript, claude_transcript(5)).unwrap();
    cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "unsigned-run")
        .write_stdin(
            json!({
                "session_id": "unsigned-session",
                "transcript_path": transcript,
                "hook_event_name": "SessionEnd"
            })
            .to_string(),
        )
        .args([
            "agent",
            "hook",
            "--host",
            "claude",
            "--event",
            "session-end",
        ])
        .assert()
        .success();
    let receipt_path = project
        .path()
        .join(".tracerazor/runs/unsigned-run/run-receipt.json");
    let output = cli(&home)
        .current_dir(project.path())
        .args([
            "agent",
            "verify-receipt",
            receipt_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["status"], "unsigned");
    assert_eq!(result["authenticated"], false);
    assert_eq!(result["trusted_offline_receipt"], false);
    assert!(result["warnings"][0]
        .as_str()
        .unwrap()
        .contains("not cryptographically authenticated"));
}

#[test]
fn malformed_run_receipt_exits_two_with_machine_status() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let receipt = project.path().join("broken-receipt.json");
    std::fs::write(&receipt, "{not-json").unwrap();
    cli(&home)
        .current_dir(project.path())
        .args([
            "agent",
            "verify-receipt",
            receipt.to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .code(2)
        .stdout(predicate::str::contains("\"status\": \"malformed\""));
}

#[test]
fn codex_stop_consumes_current_rollout_jsonl_and_returns_required_json() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let secret = "CODEX_TRANSCRIPT_SECRET_40821";
    let transcript = home.path().join("codex-rollout.jsonl");
    std::fs::write(&transcript, codex_transcript(5, secret)).unwrap();

    let output = cli(&home)
        .current_dir(project.path())
        .write_stdin(
            json!({
                "session_id": "codex-session",
                "transcript_path": transcript,
                "cwd": project.path(),
                "hook_event_name": "Stop",
                "turn_id": "turn-1",
                "last_assistant_message": "done"
            })
            .to_string(),
        )
        .args(["agent", "hook", "--host", "codex", "--event", "stop"])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&output.stdout).unwrap(),
        json!({})
    );

    let run_dir = project
        .path()
        .join(".tracerazor")
        .join("runs")
        .join("codex-session");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["status"], "completed");
    assert_eq!(manifest["capture_quality"], "degraded");
    assert_eq!(manifest["step_count"], 5);
    assert_eq!(manifest["total_tokens"], 500);
    assert!(manifest["ingest_quality"]["issues"]
        .as_array()
        .unwrap()
        .contains(&json!("token_distribution_estimated")));
    assert!(!read_tree_text(&run_dir).contains(secret));
    let trace: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("trace.json")).unwrap())
            .unwrap();
    assert_eq!(trace["framework"], "codex");
}

#[test]
fn codex_unknown_transcript_format_is_an_explicit_partial_receipt() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let transcript = home.path().join("future-codex.jsonl");
    std::fs::write(&transcript, "{\"future_wire_format\":true}\n").unwrap();

    cli(&home)
        .current_dir(project.path())
        .write_stdin(
            json!({
                "session_id": "future-codex",
                "transcript_path": transcript,
                "hook_event_name": "Stop"
            })
            .to_string(),
        )
        .args(["agent", "hook", "--host", "codex", "--event", "stop"])
        .assert()
        .success()
        .stdout(predicate::eq("{}\n"))
        .stderr(predicate::str::contains("unsupported_transcript_format"));

    let run_dir = project
        .path()
        .join(".tracerazor")
        .join("runs")
        .join("future-codex");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["status"], "partial");
    assert!(manifest["ingest_quality"]["issues"]
        .as_array()
        .unwrap()
        .contains(&json!("unsupported_transcript_format")));
    assert!(!run_dir.join("report.json").exists());
}

#[test]
fn claude_subagent_stop_prefers_agent_transcript_and_links_parent() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let main_transcript = home.path().join("main.jsonl");
    let agent_transcript = home.path().join("agent.jsonl");
    std::fs::write(&main_transcript, "not the agent transcript").unwrap();
    std::fs::write(&agent_transcript, claude_transcript(5)).unwrap();

    cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_SIGNING_KEY", TEST_SIGNING_KEY)
        .write_stdin(
            json!({
                "session_id": "claude-parent",
                "transcript_path": main_transcript,
                "agent_transcript_path": agent_transcript,
                "agent_id": "agent-child-1",
                "agent_type": "Explore",
                "hook_event_name": "SubagentStop"
            })
            .to_string(),
        )
        .args([
            "agent",
            "hook",
            "--host",
            "claude",
            "--event",
            "subagent-stop",
        ])
        .assert()
        .success()
        .stdout(predicate::str::is_empty());

    let runs = project.path().join(".tracerazor").join("runs");
    let run_dir = std::fs::read_dir(&runs)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["status"], "completed");
    assert_eq!(manifest["agent_id"], "agent-child-1");
    assert!(manifest["parent_agent_id"]
        .as_str()
        .unwrap()
        .starts_with("agent-"));
    assert_eq!(manifest["step_count"], 5);
    let receipt: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("run-receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt["signed"], true);
    assert_eq!(receipt["agent_id"], "agent-child-1");
    assert_eq!(receipt["parent_agent_id"], manifest["parent_agent_id"]);
    assert_eq!(receipt["session_id"], "claude-parent");
}

#[test]
fn gemini_after_agent_consumes_session_jsonl_with_reported_usage() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let transcript = home.path().join("gemini-session.jsonl");
    std::fs::write(&transcript, gemini_transcript(5)).unwrap();

    let output = cli(&home)
        .current_dir(project.path())
        .write_stdin(
            json!({
                "session_id": "gemini-session",
                "transcript_path": transcript,
                "cwd": project.path(),
                "hook_event_name": "AfterAgent",
                "prompt": "inspect the project",
                "prompt_response": "done"
            })
            .to_string(),
        )
        .args([
            "agent",
            "hook",
            "--host",
            "gemini",
            "--event",
            "after-agent",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&output.stdout).unwrap(),
        json!({"suppressOutput": true})
    );

    let run_dir = project
        .path()
        .join(".tracerazor")
        .join("runs")
        .join("gemini-session");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["status"], "completed");
    assert_eq!(manifest["capture_quality"], "complete");
    assert_eq!(manifest["step_count"], 5);
    assert_eq!(manifest["total_tokens"], 500);
    assert_eq!(manifest["ingest_quality"]["provider_token_coverage"], 1.0);
    let trace: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("trace.json")).unwrap())
            .unwrap();
    assert_eq!(trace["framework"], "gemini-cli");

    cli(&home)
        .current_dir(project.path())
        .write_stdin(
            json!({
                "session_id": "gemini-session",
                "hook_event_name": "SessionEnd",
                "reason": "exit"
            })
            .to_string(),
        )
        .args([
            "agent",
            "hook",
            "--host",
            "gemini",
            "--event",
            "session-end",
        ])
        .assert()
        .success();
    let preserved: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(preserved["status"], "completed");
    assert_eq!(preserved["capture_quality"], "complete");
    assert_eq!(preserved["step_count"], 5);
    assert!(preserved["lifecycle_issues"]
        .as_array()
        .unwrap()
        .contains(&json!("transcript_path_missing")));
}

#[test]
fn terminal_hook_rejects_mismatched_event_before_reading_transcript() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let transcript = home.path().join("must-not-read.jsonl");
    std::fs::write(&transcript, "secret").unwrap();
    cli(&home)
        .current_dir(project.path())
        .write_stdin(
            json!({
                "session_id": "mismatched-event",
                "transcript_path": transcript,
                "hook_event_name": "SessionStart"
            })
            .to_string(),
        )
        .args(["agent", "hook", "--host", "codex", "--event", "stop"])
        .assert()
        .success()
        .stdout(predicate::eq("{}\n"));
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(
            project
                .path()
                .join(".tracerazor/runs/mismatched-event/manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(manifest["ingest_quality"]["issues"]
        .as_array()
        .unwrap()
        .contains(&json!("hook_event_mismatch")));
}

#[test]
fn terminal_hook_rejects_relative_transcript_path_as_partial() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    std::fs::write(
        project.path().join("relative.jsonl"),
        codex_transcript(5, "secret"),
    )
    .unwrap();
    cli(&home)
        .current_dir(project.path())
        .write_stdin(
            json!({
                "session_id": "relative-transcript",
                "transcript_path": "relative.jsonl",
                "hook_event_name": "Stop"
            })
            .to_string(),
        )
        .args(["agent", "hook", "--host", "codex", "--event", "stop"])
        .assert()
        .success()
        .stdout(predicate::eq("{}\n"))
        .stderr(predicate::str::contains("transcript_read_error"));
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(
            project
                .path()
                .join(".tracerazor/runs/relative-transcript/manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["status"], "partial");
    assert!(manifest["ingest_quality"]["issues"]
        .as_array()
        .unwrap()
        .contains(&json!("transcript_read_error")));
}

#[test]
fn capture_off_installs_no_hooks_and_hook_is_a_noop() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    std::fs::write(
        project.path().join("tracerazor.toml"),
        policy_text("off", true, "local-redacted", false, ".tracerazor/runs", 5),
    )
    .unwrap();
    let installed = run_json(
        &home,
        &project,
        &["agent", "install", "--host", "codex", "--format", "json"],
    );
    assert_eq!(installed["automatic_capture"], false);
    assert_eq!(installed["policy"]["capture"], "off");
    assert!(!project.path().join(".codex").join("hooks.json").exists());

    cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "capture-off")
        .write_stdin("{}")
        .args([
            "agent",
            "hook",
            "--host",
            "codex",
            "--event",
            "session-start",
        ])
        .assert()
        .success()
        .stdout(predicate::str::is_empty());
    assert!(!project.path().join(".tracerazor").join("runs").exists());
}

#[test]
fn min_steps_and_custom_artifact_dir_control_terminal_capture() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    std::fs::write(
        project.path().join("tracerazor.toml"),
        policy_text("auto", false, "local-redacted", false, ".audit/runs", 20),
    )
    .unwrap();
    let transcript = home.path().join("short.jsonl");
    std::fs::write(&transcript, claude_transcript(5)).unwrap();
    cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "short-run")
        .write_stdin(json!({"transcript_path": transcript}).to_string())
        .args([
            "agent",
            "hook",
            "--host",
            "claude",
            "--event",
            "session-end",
        ])
        .assert()
        .success();
    let run_dir = project.path().join(".audit").join("runs").join("short-run");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["status"], "partial");
    assert_eq!(manifest["policy"]["min_steps"], 20);
    assert_eq!(manifest["policy"]["hermetic"], false);
    assert!(manifest["ingest_quality"]["issues"]
        .as_array()
        .unwrap()
        .contains(&json!("below_min_steps")));
    assert!(!run_dir.join("trace.json").exists());
}

#[test]
fn invalid_artifact_escape_fails_closed_without_breaking_host() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    std::fs::write(
        project.path().join("tracerazor.toml"),
        policy_text("auto", true, "local-redacted", false, "../escape", 5),
    )
    .unwrap();
    cli(&home)
        .current_dir(project.path())
        .write_stdin("{}")
        .args([
            "agent",
            "hook",
            "--host",
            "gemini",
            "--event",
            "session-start",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"suppressOutput\":true"))
        .stderr(predicate::str::contains("confined relative path"));
    assert!(!project.path().parent().unwrap().join("escape").exists());
}

#[test]
fn default_capture_never_persists_transcript_secret_content() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let secret = "TRACERAZOR_RAW_SECRET_918273";
    let transcript = home.path().join("secret.jsonl");
    std::fs::write(&transcript, claude_transcript_with_text(5, secret)).unwrap();
    cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "redacted-run")
        .write_stdin(json!({"transcript_path": transcript}).to_string())
        .args([
            "agent",
            "hook",
            "--host",
            "claude",
            "--event",
            "session-end",
        ])
        .assert()
        .success();
    let run_dir = project
        .path()
        .join(".tracerazor")
        .join("runs")
        .join("redacted-run");
    assert!(!read_tree_text(&run_dir).contains(secret));
    let findings: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("findings.json")).unwrap())
            .unwrap();
    assert_eq!(findings["schema_version"], "tracerazor-findings/v1");
    assert_eq!(findings["run_id"], "redacted-run");
}

#[test]
fn raw_opt_in_is_replayable_and_persists_content() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let secret = "TRACERAZOR_RAW_OPT_IN_564738";
    std::fs::write(
        project.path().join("tracerazor.toml"),
        policy_text("auto", true, "raw", true, ".tracerazor/runs", 5),
    )
    .unwrap();
    let transcript = home.path().join("raw.jsonl");
    std::fs::write(&transcript, claude_transcript_with_text(5, secret)).unwrap();
    cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "raw-run")
        .write_stdin(json!({"transcript_path": transcript}).to_string())
        .args([
            "agent",
            "hook",
            "--host",
            "claude",
            "--event",
            "session-end",
        ])
        .assert()
        .success();
    let run_dir = project
        .path()
        .join(".tracerazor")
        .join("runs")
        .join("raw-run");
    assert!(read_tree_text(&run_dir).contains(secret));
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["replayable"], true);
    assert_eq!(
        manifest["audit_trace_sha256"],
        manifest["persisted_trace_sha256"]
    );
    cli(&home)
        .current_dir(project.path())
        .args([
            "verify",
            run_dir.join("report.json").to_str().unwrap(),
            run_dir.join("trace.json").to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .success();
}

#[test]
fn claude_installer_uses_agent_native_hook_commands() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    run_json(
        &home,
        &project,
        &["agent", "install", "--host", "claude", "--format", "json"],
    );
    let settings: Value = serde_json::from_str(
        &std::fs::read_to_string(project.path().join(".claude").join("settings.local.json"))
            .unwrap(),
    )
    .unwrap();
    for event in [
        "SessionStart",
        "SessionEnd",
        "SubagentStart",
        "SubagentStop",
    ] {
        let command = settings["hooks"][event][0]["hooks"][0]["command"]
            .as_str()
            .unwrap();
        assert!(command.starts_with("tracerazor agent hook --host claude --event "));
    }
    let mcp: Value =
        serde_json::from_str(&std::fs::read_to_string(project.path().join(".mcp.json")).unwrap())
            .unwrap();
    assert_eq!(mcp["mcpServers"]["tracerazor"]["command"], "tracerazor-mcp");
}

fn claude_transcript(messages: usize) -> String {
    claude_transcript_with_text(messages, "inspect carefully")
}

fn codex_transcript(messages: usize, text: &str) -> String {
    let mut lines = vec![
        json!({
            "type": "session_meta",
            "payload": {"id": "codex-session", "cli_version": "test"}
        })
        .to_string(),
        json!({
            "type": "turn_context",
            "payload": {"model": "gpt-test"}
        })
        .to_string(),
        json!({
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        })
        .to_string(),
    ];
    for index in 0..messages {
        lines.push(
            json!({
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": format!("Step {index} inspects the project state carefully: {text}")
                    }]
                }
            })
            .to_string(),
        );
    }
    lines.push(
        json!({
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 300,
                        "output_tokens": 150,
                        "reasoning_output_tokens": 50,
                        "total_tokens": 500
                    }
                }
            }
        })
        .to_string(),
    );
    lines.join("\n")
}

fn gemini_transcript(messages: usize) -> String {
    let mut lines = vec![
        json!({
            "sessionId": "gemini-session",
            "projectHash": "project-hash",
            "startTime": "2026-01-01T00:00:00Z",
            "lastUpdated": "2026-01-01T00:00:00Z"
        })
        .to_string(),
        json!({
            "id": "user-1",
            "timestamp": "2026-01-01T00:00:01Z",
            "type": "user",
            "content": [{"text": "Inspect the project carefully"}]
        })
        .to_string(),
    ];
    for index in 0..messages {
        lines.push(
            json!({
                "id": format!("gemini-{index}"),
                "timestamp": "2026-01-01T00:00:02Z",
                "type": "gemini",
                "content": [{"text": format!("Step {index} inspects project state carefully")}],
                "tokens": {
                    "input": 70,
                    "output": 30,
                    "cached": 0,
                    "total": 100
                },
                "model": "gemini-test"
            })
            .to_string(),
        );
    }
    lines.join("\n")
}

fn claude_transcript_with_text(messages: usize, text: &str) -> String {
    let mut lines = vec![json!({
        "type": "user",
        "session_id": "session-1",
        "message": {"role": "user", "content": text}
    })
    .to_string()];
    for index in 0..messages {
        lines.push(
            json!({
                "type": "assistant",
                "session_id": "session-1",
                "message": {
                    "id": format!("message-{index}"),
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [{"type": "text", "text": format!("Step {index}: {text}")}],
                    "usage": {
                        "input_tokens": 10,
                        "cache_creation_input_tokens": 20,
                        "cache_read_input_tokens": 30,
                        "output_tokens": 40
                    }
                }
            })
            .to_string(),
        );
    }
    lines.join("\n")
}

fn policy_text(
    capture: &str,
    hermetic: bool,
    privacy: &str,
    persist_raw_content: bool,
    artifact_dir: &str,
    min_steps: usize,
) -> String {
    format!(
        "schema_version = 1\nmode = \"coach\"\ncapture = \"{capture}\"\nhermetic = {hermetic}\nprivacy = \"{privacy}\"\npersist_raw_content = {persist_raw_content}\nartifact_dir = \"{artifact_dir}\"\nmin_steps = {min_steps}\n\n[quality]\nverifier = \"\"\n\n[enforcement]\nenabled = false\n"
    )
}

fn read_tree_text(path: &std::path::Path) -> String {
    let mut output = String::new();
    for entry in std::fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            output.push_str(&read_tree_text(&path));
        } else if let Ok(text) = std::fs::read_to_string(path) {
            output.push_str(&text);
        }
    }
    output
}

#[test]
fn image_scope_fails_clearly_without_root() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    cli(&home)
        .current_dir(project.path())
        .args(["agent", "install", "--host", "generic", "--scope", "image"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("TRACERAZOR_IMAGE_ROOT"));
}

#[test]
fn install_refuses_an_ancestor_symlink_outside_project_scope() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let outside = TempDir::new().unwrap();
    let link = project.path().join(".agents");
    if create_dir_link(outside.path(), &link).is_err() {
        // Windows may not grant symlink creation to an unprivileged test user.
        return;
    }

    cli(&home)
        .current_dir(project.path())
        .args(["agent", "install", "--host", "codex", "--format", "json"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("symlink"));
    assert!(!outside.path().join("skills").exists());
    assert!(!project.path().join("tracerazor.toml").exists());
}

#[test]
fn hook_refuses_artifact_ancestor_symlink() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let outside = TempDir::new().unwrap();
    if create_dir_link(outside.path(), &project.path().join(".tracerazor")).is_err() {
        return;
    }
    cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "unsafe-ancestor")
        .write_stdin("{}")
        .args([
            "agent",
            "hook",
            "--host",
            "codex",
            "--event",
            "session-start",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("symlink"));
    assert!(!outside.path().join("runs").exists());
}

#[test]
fn hook_refuses_events_and_trace_leaf_symlinks() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let outside = TempDir::new().unwrap();
    let run_dir = project
        .path()
        .join(".tracerazor")
        .join("runs")
        .join("unsafe-leaf");
    std::fs::create_dir_all(&run_dir).unwrap();
    let sentinel = outside.path().join("sentinel.txt");
    std::fs::write(&sentinel, "do-not-touch").unwrap();
    if create_file_link(&sentinel, &run_dir.join("events.jsonl")).is_err() {
        return;
    }
    cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "unsafe-leaf")
        .write_stdin("{}")
        .args([
            "agent",
            "hook",
            "--host",
            "codex",
            "--event",
            "session-start",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("symlink"));
    assert_eq!(std::fs::read_to_string(&sentinel).unwrap(), "do-not-touch");

    std::fs::remove_file(run_dir.join("events.jsonl")).unwrap();
    if create_file_link(&sentinel, &run_dir.join("trace.json")).is_err() {
        return;
    }
    let transcript = home.path().join("leaf.jsonl");
    std::fs::write(&transcript, claude_transcript(5)).unwrap();
    cli(&home)
        .current_dir(project.path())
        .env("TRACERAZOR_RUN_ID", "unsafe-leaf")
        .write_stdin(json!({"transcript_path": transcript}).to_string())
        .args([
            "agent",
            "hook",
            "--host",
            "claude",
            "--event",
            "session-end",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("symlink"));
    assert_eq!(std::fs::read_to_string(&sentinel).unwrap(), "do-not-touch");
}

#[cfg(unix)]
fn create_dir_link(target: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(unix)]
fn create_file_link(target: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(windows)]
fn create_file_link(target: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(target, link)
}

#[cfg(windows)]
fn create_dir_link(target: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_dir(target, link)
}
