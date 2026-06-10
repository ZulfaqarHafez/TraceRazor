//! Provenance round-trip: `audit` emits a run manifest binding the report to
//! its inputs, and `verify` reproduces the score from (trace, manifest,
//! version) — or refuses when the trace was tampered with.

use std::path::{Path, PathBuf};

use assert_cmd::Command;
use serde_json::Value;
use tempfile::TempDir;

fn corpus_trace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("traces/external/huggingface/agentinstruct/agentinstruct-os_0.json")
}

fn cli(home: &TempDir) -> Command {
    let mut cmd = Command::cargo_bin("tracerazor").unwrap();
    cmd.env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .env_remove("OPENAI_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("TRACERAZOR_LLM_API_KEY")
        .env_remove("TRACERAZOR_WEIGHTS");
    cmd
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn hermetic_audit_json(home: &TempDir, trace: &Path) -> Value {
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

#[test]
fn audit_emits_binding_run_manifest() {
    let home = TempDir::new().unwrap();
    let trace = corpus_trace();
    let report = hermetic_audit_json(&home, &trace);

    let m = &report["manifest"];
    assert!(!m.is_null(), "report must carry a run manifest");
    assert_eq!(m["hermetic"], true);
    assert_eq!(m["similarity_backend"], "bow");
    assert_eq!(m["n_historical_sequences"], 0);
    assert!(m["baseline_tokens"].is_null());

    // The manifest hash must be the sha256 of the exact on-disk bytes.
    let raw = std::fs::read(&trace).unwrap();
    assert_eq!(m["trace_sha256"].as_str().unwrap(), sha256_hex(&raw));
    assert_eq!(
        m["tool_version"].as_str().unwrap(),
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(m["weights_sha256"].as_str().unwrap().len(), 64);

    // The AGF provenance diagnostic rides along with every report.
    let agf = &report["agf"];
    assert!(!agf.is_null(), "report must carry the AGF diagnostic");
    let score = agf["score"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&score), "AGF in [0,1], got {score}");
}

#[test]
fn verify_round_trip_passes_and_tamper_fails() {
    let home = TempDir::new().unwrap();
    let trace = corpus_trace();
    let report = hermetic_audit_json(&home, &trace);

    let dir = TempDir::new().unwrap();
    let report_path = dir.path().join("report.json");
    std::fs::write(&report_path, serde_json::to_string(&report).unwrap()).unwrap();

    // Round-trip: the report reproduces from (trace, manifest, version).
    // Unsigned reports get "rescore-only (unsigned)" — never "full".
    cli(&home)
        .args([
            "verify",
            report_path.to_str().unwrap(),
            trace.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("re-score        : OK"))
        .stdout(predicates::str::contains("rescore-only (unsigned"));

    // One flipped byte in the trace must fail verification.
    let mut tampered_bytes = std::fs::read(&trace).unwrap();
    let last = tampered_bytes.len() - 2;
    tampered_bytes[last] = b' ';
    let tampered = dir.path().join("tampered.json");
    std::fs::write(&tampered, &tampered_bytes).unwrap();

    cli(&home)
        .args([
            "verify",
            report_path.to_str().unwrap(),
            tampered.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("TAMPERED"));
}

#[test]
fn store_false_flag_is_accepted() {
    // Regression: `--store false` used to be rejected by the arg parser,
    // making the store write-back impossible to disable.
    let home = TempDir::new().unwrap();
    let trace = corpus_trace();
    cli(&home)
        .args([
            "audit",
            trace.to_str().unwrap(),
            "--format",
            "json",
            "--store",
            "false",
        ])
        .assert()
        .success();
}

#[test]
fn exit_code_contract() {
    let trace = corpus_trace();

    // No --threshold: a low score is information, not a failure (exit 0).
    let home = TempDir::new().unwrap();
    cli(&home)
        .args(["audit", trace.to_str().unwrap(), "--format", "json", "--hermetic"])
        .assert()
        .success();

    // Explicit gate that cannot pass: exit 1.
    let home = TempDir::new().unwrap();
    cli(&home)
        .args([
            "audit",
            trace.to_str().unwrap(),
            "--hermetic",
            "--threshold",
            "100.0",
        ])
        .assert()
        .code(1);

    // Broken input: exit 2, distinguishable from a failed gate.
    let home = TempDir::new().unwrap();
    cli(&home)
        .args(["audit", "/does/not/exist.json"])
        .assert()
        .code(2);
}
