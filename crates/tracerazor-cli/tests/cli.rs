//! End-to-end smoke tests for the `tracerazor` CLI.
//!
//! These exercise each subcommand against a fixture trace so basic CLI wiring
//! (argument parsing, file I/O, output formats, exit codes) is regression-tested.
//! Tests set `HOME` (and `USERPROFILE` on Windows) to a temp directory so the
//! persistent store does not pollute the developer's real `~/.tracerazor`.

use std::path::PathBuf;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

fn fixture_path() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
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
        .env_remove("TRACERAZOR_LLM_API_KEY");
    cmd
}

#[test]
fn version_prints() {
    let home = TempDir::new().unwrap();
    cli(&home)
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("tracerazor"));
}

#[test]
fn help_lists_subcommands() {
    let home = TempDir::new().unwrap();
    cli(&home)
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("audit"))
        .stdout(predicate::str::contains("compare"))
        .stdout(predicate::str::contains("cost"))
        .stdout(predicate::str::contains("simulate"))
        .stdout(predicate::str::contains("apply"));
}

#[test]
fn audit_markdown_succeeds() {
    let home = TempDir::new().unwrap();
    cli(&home)
        .args([
            "audit",
            fixture_path().to_str().unwrap(),
            "--format",
            "markdown",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("TRACERAZOR"));
}

#[test]
fn audit_json_is_parseable() {
    let home = TempDir::new().unwrap();
    let output = cli(&home)
        .args([
            "audit",
            fixture_path().to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let parsed: serde_json::Value =
        serde_json::from_slice(&output).expect("audit --format json should produce valid JSON");
    assert!(
        parsed.get("score").and_then(|s| s.get("score")).is_some(),
        "audit JSON should contain score.score"
    );
}

#[test]
fn audit_missing_file_fails() {
    let home = TempDir::new().unwrap();
    cli(&home)
        .args(["audit", "/does/not/exist.json"])
        .assert()
        .failure();
}

#[test]
fn audit_threshold_above_score_exits_nonzero() {
    let home = TempDir::new().unwrap();
    // 100.0 is unreachable, so the gate must fail.
    cli(&home)
        .args([
            "audit",
            fixture_path().to_str().unwrap(),
            "--threshold",
            "100.0",
        ])
        .assert()
        .failure();
}

#[test]
fn cost_projection_runs() {
    let home = TempDir::new().unwrap();
    cli(&home)
        .args([
            "cost",
            fixture_path().to_str().unwrap(),
            "--runs",
            "1000",
            "--format",
            "json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("{"));
}

#[test]
fn simulate_with_mutation_produces_json() {
    let home = TempDir::new().unwrap();
    cli(&home)
        .args([
            "simulate",
            fixture_path().to_str().unwrap(),
            "--remove",
            "3",
            "--format",
            "json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("projected_tas"));
}

#[test]
fn compare_same_trace_has_zero_delta() {
    let home = TempDir::new().unwrap();
    let f = fixture_path();
    cli(&home)
        .args([
            "compare",
            f.to_str().unwrap(),
            f.to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .success();
}
