use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
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

fn claude_transcript(n_messages: usize) -> String {
    let mut lines = vec![
        serde_json::json!({"type":"user","session_id":"sess-1","message":{"role":"user","content":"Fix the failing test"}}).to_string(),
    ];
    for i in 0..n_messages {
        lines.push(serde_json::json!({
            "type":"assistant",
            "session_id":"sess-1",
            "message":{
                "id": format!("msg-{i}"),
                "role":"assistant",
                "model":"claude-haiku-4-5-20251001",
                "content":[{"type":"text","text":format!("Step {i}: inspect and act carefully")}],
                "usage":{
                    "input_tokens":10,
                    "cache_creation_input_tokens":20,
                    "cache_read_input_tokens":200,
                    "output_tokens":30
                }
            }
        }).to_string());
    }
    lines.join("\n")
}

#[test]
fn claude_convert_writes_trace_json() {
    let home = TempDir::new().unwrap();
    let transcript = home.path().join("session.jsonl");
    let out = home.path().join("trace.json");
    std::fs::write(&transcript, claude_transcript(2)).unwrap();

    cli(&home)
        .args([
            "claude",
            "convert",
            transcript.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Wrote"));

    let parsed: Value = serde_json::from_str(&std::fs::read_to_string(out).unwrap()).unwrap();
    assert_eq!(parsed["framework"], "claude-code");
    assert_eq!(parsed["metadata"]["source"], "claude-code-transcript");
}

#[test]
fn claude_install_is_idempotent_and_uninstall_removes_hook() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();

    cli(&home)
        .current_dir(project.path())
        .args(["claude", "install", "--scope", "local", "--mode", "coach"])
        .assert()
        .success();
    cli(&home)
        .current_dir(project.path())
        .args(["claude", "install", "--scope", "local", "--mode", "coach"])
        .assert()
        .success();

    let settings_path = project.path().join(".claude").join("settings.local.json");
    let settings: Value =
        serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
    let groups = settings["hooks"]["SessionEnd"].as_array().unwrap();
    let hook_count: usize = groups
        .iter()
        .map(|g| g["hooks"].as_array().map(|h| h.len()).unwrap_or(0))
        .sum();
    assert_eq!(hook_count, 1);
    assert!(settings_path
        .parent()
        .unwrap()
        .read_dir()
        .unwrap()
        .any(|e| { e.unwrap().file_name().to_string_lossy().contains(".bak.") }));

    cli(&home)
        .current_dir(project.path())
        .args(["claude", "uninstall", "--scope", "local"])
        .assert()
        .success();
    let settings: Value =
        serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
    assert!(settings["hooks"]["SessionEnd"]
        .as_array()
        .map(|a| a.is_empty())
        .unwrap_or(true));
}

#[test]
fn import_langfuse_normalizes_trace() {
    let home = TempDir::new().unwrap();
    let input = home.path().join("langfuse.json");
    let out = home.path().join("trace.json");
    std::fs::write(
        &input,
        r#"{"id":"trace-1","name":"support","observations":[
          {"type":"GENERATION","name":"llm","input":"hello","output":"hi","usageDetails":{"input":10,"output":20}},
          {"type":"SPAN","name":"lookup","input":{"id":"1"},"output":"found","usage":{"total":5}}
        ]}"#,
    )
    .unwrap();

    cli(&home)
        .args([
            "import",
            input.to_str().unwrap(),
            "--from",
            "langfuse",
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();
    let parsed: Value = serde_json::from_str(&std::fs::read_to_string(out).unwrap()).unwrap();
    assert_eq!(parsed["framework"], "langfuse");
    assert_eq!(parsed["total_tokens"], 35);
}

#[test]
fn claude_session_end_hook_writes_artifacts() {
    let home = TempDir::new().unwrap();
    let project = TempDir::new().unwrap();
    let transcript = home.path().join("session.jsonl");
    std::fs::write(&transcript, claude_transcript(5)).unwrap();
    let event = serde_json::json!({
        "session_id":"sess-1",
        "transcript_path": transcript,
        "cwd": project.path()
    });

    cli(&home)
        .write_stdin(event.to_string())
        .args(["claude", "hook", "session-end", "--mode", "coach"])
        .assert()
        .success();

    let out_dir = project
        .path()
        .join(".tracerazor")
        .join("claude-code")
        .join("sess-1");
    assert!(out_dir.join("trace.json").exists());
    assert!(out_dir.join("report.json").exists());
    assert!(out_dir.join("fixes.json").exists());
    assert!(out_dir.join("coach.md").exists());
    assert!(out_dir.join("summary.json").exists());
}
