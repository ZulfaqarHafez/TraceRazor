//! End-to-end tests for the Claude Code SessionStart coach loop and the
//! `--with-skill` install/uninstall behaviour.

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::{json, Value};
use tempfile::TempDir;

fn cli(home: &TempDir) -> Command {
    let mut cmd = Command::cargo_bin("tracerazor").unwrap();
    cmd.env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .env_remove("OPENAI_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("TRACERAZOR_LLM_API_KEY");
    cmd
}

/// Seed a session's on-disk artifacts (summary.json + fixes.json) and the
/// index. `indexed_at` is optional so tests can exercise the mtime fallback
/// (fresh) and an explicit stale timestamp.
fn seed_session(project: &TempDir, trace_id: &str, indexed_at: Option<&str>) {
    let out_dir = project
        .path()
        .join(".tracerazor")
        .join("claude-code")
        .join(trace_id);
    std::fs::create_dir_all(&out_dir).unwrap();

    let summary = json!({
        "trace_id": trace_id,
        "agent_name": "claude-code",
        "framework": "claude-code",
        "tas_score": 42.0,
        "grade": "D",
        "total_tokens": 5000,
        "estimated_tokens_saved": 1234,
        "fix_count": 2,
        "trace": out_dir.join("trace.json"),
        "report": out_dir.join("report.json"),
        "fixes": out_dir.join("fixes.json"),
        "coach": out_dir.join("coach.md"),
        "validated": false,
        "validation_status": "projected_only"
    });
    std::fs::write(
        out_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .unwrap();

    let fixes = json!([
        {"fix_type":"verbosity_reduction","target":"system_prompt","patch":"Be terse.","estimated_token_savings":800,"risk":"safe"},
        {"fix_type":"tool_schema","target":"search","patch":"Mark query required.","estimated_token_savings":434,"risk":"needs_review"}
    ]);
    std::fs::write(
        out_dir.join("fixes.json"),
        serde_json::to_string_pretty(&fixes).unwrap(),
    )
    .unwrap();

    let mut entry = summary;
    if let Some(ts) = indexed_at {
        entry
            .as_object_mut()
            .unwrap()
            .insert("indexed_at".into(), json!(ts));
    }
    let index = json!([entry]);
    std::fs::write(
        project
            .path()
            .join(".tracerazor")
            .join("claude-code")
            .join("index.json"),
        serde_json::to_string_pretty(&index).unwrap(),
    )
    .unwrap();
}

fn session_start(home: &TempDir, project: &TempDir, source: &str) -> Command {
    let mut cmd = cli(home);
    let event = json!({"source": source, "cwd": project.path(), "hook_event_name": "SessionStart"});
    cmd.current_dir(project.path())
        .write_stdin(event.to_string())
        .args(["claude", "hook", "session-start"]);
    cmd
}

#[test]
fn session_start_with_no_artifacts_is_silent() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    session_start(&home, &project, "startup")
        .assert()
        .success()
        .stdout(predicate::str::is_empty());
}

#[test]
fn session_start_emits_advisory_for_fresh_actionable_session() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    // No indexed_at -> freshness falls back to summary.json mtime (just written).
    seed_session(&project, "sess-cafe1234", None);

    session_start(&home, &project, "startup")
        .assert()
        .success()
        .stdout(predicate::str::contains("TraceRazor coach"))
        .stdout(predicate::str::contains("TAS 42/100"))
        .stdout(predicate::str::contains("sess-caf"))
        .stdout(predicate::str::contains("verbosity_reduction (safe)"))
        .stdout(predicate::str::contains(
            "tracerazor apply .tracerazor/claude-code/sess-cafe1234/fixes.json",
        ));
}

#[test]
fn session_start_compact_source_is_silent() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    seed_session(&project, "sess-cafe1234", None);

    session_start(&home, &project, "compact")
        .assert()
        .success()
        .stdout(predicate::str::is_empty());
}

#[test]
fn session_start_stale_artifacts_are_silent() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    // Explicit index timestamp far past the 7-day freshness window.
    seed_session(&project, "sess-cafe1234", Some("2020-01-01T00:00:00+00:00"));

    session_start(&home, &project, "startup")
        .assert()
        .success()
        .stdout(predicate::str::is_empty());
}

fn hook_count(settings: &Value, event: &str) -> usize {
    settings["hooks"][event]
        .as_array()
        .map(|groups| {
            groups
                .iter()
                .map(|g| g["hooks"].as_array().map(|h| h.len()).unwrap_or(0))
                .sum()
        })
        .unwrap_or(0)
}

#[test]
fn install_writes_both_hooks_and_uninstall_removes_both() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();

    cli(&home)
        .current_dir(project.path())
        .args(["claude", "install", "--scope", "local", "--mode", "coach"])
        .assert()
        .success();

    let settings_path = project.path().join(".claude").join("settings.local.json");
    let settings: Value =
        serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
    assert_eq!(hook_count(&settings, "SessionEnd"), 1);
    assert_eq!(hook_count(&settings, "SessionStart"), 1);

    cli(&home)
        .current_dir(project.path())
        .args(["claude", "uninstall", "--scope", "local"])
        .assert()
        .success();

    let settings: Value =
        serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
    assert_eq!(hook_count(&settings, "SessionEnd"), 0);
    assert_eq!(hook_count(&settings, "SessionStart"), 0);
}

#[test]
fn uninstall_removes_legacy_session_end_only_hook() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let settings_dir = project.path().join(".claude");
    std::fs::create_dir_all(&settings_dir).unwrap();
    let settings_path = settings_dir.join("settings.local.json");

    // Simulate settings written by an older version that only installed the
    // SessionEnd hook (no SessionStart).
    let legacy = json!({
        "hooks": {
            "SessionEnd": [{
                "hooks": [{
                    "type": "command",
                    "command": "tracerazor",
                    "args": ["claude", "hook", "session-end", "--mode", "coach"],
                    "timeout": 60,
                    "statusMessage": "TraceRazor auditing Claude Code session"
                }]
            }]
        }
    });
    std::fs::write(
        &settings_path,
        serde_json::to_string_pretty(&legacy).unwrap(),
    )
    .unwrap();

    cli(&home)
        .current_dir(project.path())
        .args(["claude", "uninstall", "--scope", "local"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Removed"));

    let settings: Value =
        serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
    assert_eq!(hook_count(&settings, "SessionEnd"), 0);
}

#[test]
fn install_with_skill_places_skill_at_local_scope() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();

    cli(&home)
        .current_dir(project.path())
        .args([
            "claude",
            "install",
            "--scope",
            "local",
            "--mode",
            "coach",
            "--with-skill",
        ])
        .assert()
        .success();

    let skill_path = project
        .path()
        .join(".claude")
        .join("skills")
        .join("tracerazor")
        .join("SKILL.md");
    let content = std::fs::read_to_string(&skill_path).unwrap();
    assert!(content.contains("name: tracerazor"));
}

#[test]
fn install_with_skill_places_skill_at_user_scope() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();

    cli(&home)
        .current_dir(project.path())
        .args([
            "claude",
            "install",
            "--scope",
            "user",
            "--mode",
            "coach",
            "--with-skill",
        ])
        .assert()
        .success();

    let skill_path = home
        .path()
        .join(".claude")
        .join("skills")
        .join("tracerazor")
        .join("SKILL.md");
    let content = std::fs::read_to_string(&skill_path).unwrap();
    assert!(content.contains("name: tracerazor"));
}
