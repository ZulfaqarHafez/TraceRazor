//! Machine-readable contract tests: `verify --format json`, `list --format
//! json`, the audit skip-status JSON object, and the top-level exit-code help
//! block. These lock the stdout shapes that agent/CI consumers parse.

use std::path::PathBuf;

use assert_cmd::Command;
use serde_json::Value;
use tempfile::TempDir;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("traces")
        .join("support-agent-run-2847.json")
}

fn cli(home: &TempDir) -> Command {
    let mut cmd = Command::cargo_bin("tracerazor").unwrap();
    cmd.env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .env_remove("OPENAI_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("TRACERAZOR_LLM_API_KEY")
        .env_remove("TRACERAZOR_WEIGHTS")
        .env_remove("TRACERAZOR_SIGNING_KEY");
    cmd
}

fn hermetic_audit_json(home: &TempDir, trace: &std::path::Path) -> Value {
    let out = cli(home)
        .args([
            "audit",
            trace.to_str().unwrap(),
            "--format",
            "json",
            "--hermetic",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    serde_json::from_slice(&out.stdout).expect("audit must emit JSON")
}

// ── verify --format json: happy path ─────────────────────────────────────────

#[test]
fn verify_json_verified_object() {
    let home = TempDir::new().unwrap();
    let trace = fixture_path();
    let report = hermetic_audit_json(&home, &trace);

    let dir = TempDir::new().unwrap();
    let report_path = dir.path().join("report.json");
    std::fs::write(&report_path, serde_json::to_string(&report).unwrap()).unwrap();

    let out = cli(&home)
        .args([
            "verify",
            report_path.to_str().unwrap(),
            trace.to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let v: Value =
        serde_json::from_slice(&out.stdout).expect("verify --format json must emit JSON");
    assert_eq!(v["status"], "verified");
    assert_eq!(v["trace_hash"], "ok");
    assert_eq!(v["rescore"], "ok");
    // Unsigned hermetic report: rescore-only, never full.
    assert_eq!(v["signature"], "missing");
    assert_eq!(v["level"], "rescore-only (unsigned)");
    assert!(v["mismatches"].as_array().unwrap().is_empty());
    assert!(v["report_path"].is_string());
    assert!(v["trace_path"].is_string());
}

// ── verify --format json: tampered trace → status tampered + exit 1 ───────────

#[test]
fn verify_json_tampered_exits_1() {
    let home = TempDir::new().unwrap();
    let trace = fixture_path();
    let report = hermetic_audit_json(&home, &trace);

    let dir = TempDir::new().unwrap();
    let report_path = dir.path().join("report.json");
    std::fs::write(&report_path, serde_json::to_string(&report).unwrap()).unwrap();

    // Flip one byte in the trace so its hash no longer matches the manifest.
    let mut tampered_bytes = std::fs::read(&trace).unwrap();
    let last = tampered_bytes.len() - 2;
    tampered_bytes[last] = b' ';
    let tampered = dir.path().join("tampered.json");
    std::fs::write(&tampered, &tampered_bytes).unwrap();

    let out = cli(&home)
        .args([
            "verify",
            report_path.to_str().unwrap(),
            tampered.to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .failure()
        .code(1)
        .get_output()
        .clone();
    let v: Value =
        serde_json::from_slice(&out.stdout).expect("tampered verify must still emit JSON");
    assert_eq!(v["status"], "tampered");
    assert_eq!(v["trace_hash"], "mismatch");
    assert!(!v["mismatches"].as_array().unwrap().is_empty());
}

// ── list --format json ───────────────────────────────────────────────────────

#[test]
fn list_json_empty_and_populated() {
    let home = TempDir::new().unwrap();

    // Empty store yields a valid empty array, not prose.
    let out = cli(&home)
        .args(["list", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .clone();
    let v: Value = serde_json::from_slice(&out.stdout).expect("list --format json must emit JSON");
    assert_eq!(v.as_array().map(|a| a.len()), Some(0));

    // Store a trace (non-hermetic audit persists to the store by default).
    cli(&home)
        .args([
            "audit",
            fixture_path().to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .success();

    let out = cli(&home)
        .args(["list", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .clone();
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    let arr = v.as_array().expect("list json is an array");
    assert!(!arr.is_empty(), "stored trace must appear in list json");
    let first = &arr[0];
    assert!(first["trace_id"].is_string());
    assert!(first.get("agent").is_some());
    assert!(first.get("steps").is_some());
}

// ── audit skip-status JSON on a below-floor trace (exit 0) ───────────────────

#[test]
fn audit_json_skip_status_below_min_steps() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let short = dir.path().join("short.json");
    // Two steps — below the default floor of 5.
    std::fs::write(
        &short,
        r#"{"trace_id":"short-trace","agent_name":"a","framework":"react","steps":[
            {"id":1,"type":"reasoning","content":"think about it","tokens":100},
            {"id":2,"type":"tool_call","content":"do it","tokens":100,"tool_name":"t","tool_success":true}
        ]}"#,
    )
    .unwrap();

    let out = cli(&home)
        .args([
            "audit",
            short.to_str().unwrap(),
            "--format",
            "json",
            "--hermetic",
        ])
        .assert()
        .success() // skip is not an error — exit 0
        .get_output()
        .clone();
    let v: Value = serde_json::from_slice(&out.stdout).expect("skip must emit a JSON object");
    assert_eq!(v["status"], "skipped");
    assert_eq!(v["reason"], "below_min_steps");
    assert_eq!(v["steps_found"], 2);
    assert_eq!(v["min_steps"], 5);
    assert!(v["trace"].is_string());
}

// ── batch audit JSON includes skipped entries in per_file ────────────────────

#[test]
fn audit_batch_json_records_skips() {
    let home = TempDir::new().unwrap();
    let dir = TempDir::new().unwrap();
    let batch = dir.path().join("batch");
    std::fs::create_dir_all(&batch).unwrap();
    // One analysable trace + one below-floor trace.
    std::fs::copy(fixture_path(), batch.join("full.json")).unwrap();
    std::fs::write(
        batch.join("short.json"),
        r#"{"trace_id":"short-trace","agent_name":"a","framework":"react","steps":[
            {"id":1,"type":"reasoning","content":"think about it","tokens":100},
            {"id":2,"type":"tool_call","content":"do it","tokens":100,"tool_name":"t","tool_success":true}
        ]}"#,
    )
    .unwrap();

    let out = cli(&home)
        .args(["audit", batch.to_str().unwrap(), "--format", "json"])
        .assert()
        .success()
        .get_output()
        .clone();
    let v: Value = serde_json::from_slice(&out.stdout).expect("batch must emit JSON");
    assert_eq!(v["mode"], "batch");
    assert_eq!(v["n_skipped"], 1);
    let per_file = v["per_file"].as_array().expect("per_file array");
    let skipped: Vec<&Value> = per_file
        .iter()
        .filter(|e| e["status"] == "skipped")
        .collect();
    assert_eq!(skipped.len(), 1, "skipped trace must appear in per_file");
    assert_eq!(skipped[0]["reason"], "below_min_steps");
}

// ── exit-code contract surfaced in --help ────────────────────────────────────

#[test]
fn help_documents_exit_codes() {
    let home = TempDir::new().unwrap();
    cli(&home)
        .arg("--help")
        .assert()
        .success()
        .stdout(predicates::str::contains("Exit codes"));
}
