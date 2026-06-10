//! Phase 3 signing: Ed25519 report authentication and forgery-attack tests.
//!
//! Acceptance: the four compliance-reviewer forgery attacks all exit 1 (TAMPERED).
//! Also verifies that a legitimately signed, unmodified report exits 0.

use assert_cmd::Command;
use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;
use tempfile::TempDir;

// A deterministic 32-byte Ed25519 seed used throughout the tests
// (hex-encoded: 64 'a' chars = 32 bytes of 0xaa).
const TEST_KEY: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

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
        .env_remove("TRACERAZOR_WEIGHTS")
        .env_remove("TRACERAZOR_SIGNING_KEY");
    cmd
}

/// Produce a signed hermetic JSON report from the corpus trace.
fn signed_audit(home: &TempDir, trace: &std::path::Path) -> Value {
    let out = cli(home)
        .env("TRACERAZOR_SIGNING_KEY", TEST_KEY)
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

/// Write a Value to a temp file and return its path (stays alive via TempDir).
fn write_report(dir: &TempDir, report: &Value) -> PathBuf {
    let path = dir.path().join("report.json");
    let mut f = std::fs::File::create(&path).unwrap();
    write!(f, "{}", serde_json::to_string(report).unwrap()).unwrap();
    path
}

// ── 1. Unmodified signed report must verify OK ────────────────────────────────

#[test]
fn signed_report_round_trip_ok() {
    let home = TempDir::new().unwrap();
    let trace = corpus_trace();
    let report = signed_audit(&home, &trace);

    assert!(
        report["manifest"]["signature"].is_string(),
        "audit with TRACERAZOR_SIGNING_KEY must embed a signature"
    );
    assert!(
        report["manifest"]["signing_key_pub"].is_string(),
        "audit with TRACERAZOR_SIGNING_KEY must embed the public key"
    );

    let dir = TempDir::new().unwrap();
    let report_path = write_report(&dir, &report);

    cli(&home)
        .args(["verify", report_path.to_str().unwrap(), trace.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicates::str::contains("signature       : OK (Ed25519)"))
        .stdout(predicates::str::contains("verified        : full (Ed25519-authenticated"));
}

// ── 2. Attack 1 — TAS edit ────────────────────────────────────────────────────

#[test]
fn attack_tas_edit_exits_1() {
    let home = TempDir::new().unwrap();
    let trace = corpus_trace();
    let mut report = signed_audit(&home, &trace);

    // Forge the TAS score
    report["score"]["score"] = serde_json::json!(99.9);

    let dir = TempDir::new().unwrap();
    let report_path = write_report(&dir, &report);

    cli(&home)
        .args(["verify", report_path.to_str().unwrap(), trace.to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stderr(predicates::str::contains("TAMPERED"));
}

// ── 3. Attack 2 — backend flip ────────────────────────────────────────────────

#[test]
fn attack_backend_flip_exits_1() {
    let home = TempDir::new().unwrap();
    let trace = corpus_trace();
    let mut report = signed_audit(&home, &trace);

    // Flip the similarity_backend to bypass the re-score path
    report["manifest"]["similarity_backend"] =
        serde_json::json!("embeddings:text-embedding-3-small");

    let dir = TempDir::new().unwrap();
    let report_path = write_report(&dir, &report);

    cli(&home)
        .args(["verify", report_path.to_str().unwrap(), trace.to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stderr(predicates::str::contains("TAMPERED"));
}

// ── 4. Attack 3 — AGF edit ────────────────────────────────────────────────────

#[test]
fn attack_agf_edit_exits_1() {
    let home = TempDir::new().unwrap();
    let trace = corpus_trace();
    let mut report = signed_audit(&home, &trace);

    // Forge the AGF score downward (0.0 is clearly wrong and always differs
    // from the real score, which is > 0 for any trace with grounded literals).
    if report["agf"].is_object() {
        report["agf"]["score"] = serde_json::json!(0.0);
    } else {
        // If AGF is null/absent, inject a fake object.
        report["agf"] = serde_json::json!({"score": 0.0, "pass": false, "target": 0.7,
            "checked_literals": 99, "ungrounded": [], "claim_grounding": 0.0,
            "action_grounding": null});
    }

    let dir = TempDir::new().unwrap();
    let report_path = write_report(&dir, &report);

    cli(&home)
        .args(["verify", report_path.to_str().unwrap(), trace.to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stderr(predicates::str::contains("TAMPERED"));
}

// ── 5. Attack 4 — savings edit ────────────────────────────────────────────────

#[test]
fn attack_savings_edit_exits_1() {
    let home = TempDir::new().unwrap();
    let trace = corpus_trace();
    let mut report = signed_audit(&home, &trace);

    // Inflate the savings estimate
    report["savings"]["tokens_saved"] = serde_json::json!(999999);
    report["savings"]["monthly_savings_usd"] = serde_json::json!(99999.99);

    let dir = TempDir::new().unwrap();
    let report_path = write_report(&dir, &report);

    cli(&home)
        .args(["verify", report_path.to_str().unwrap(), trace.to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stderr(predicates::str::contains("TAMPERED"));
}

// ── 6. Unsigned report gets "rescore-only (unsigned)" verdict ─────────────────

#[test]
fn unsigned_report_never_says_full() {
    let home = TempDir::new().unwrap();
    let trace = corpus_trace();

    // Audit WITHOUT signing key
    let out = cli(&home)
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
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(
        report["manifest"]["signature"].is_null(),
        "unsigned audit must not embed a signature"
    );

    let dir = TempDir::new().unwrap();
    let report_path = write_report(&dir, &report);

    let verify_out = cli(&home)
        .args(["verify", report_path.to_str().unwrap(), trace.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8_lossy(&verify_out.stdout);
    assert!(
        !stdout.contains("verified        : full (Ed25519"),
        "unsigned report must not claim full Ed25519 authentication, got: {stdout}"
    );
    assert!(
        stdout.contains("rescore-only (unsigned)") || stdout.contains("hash"),
        "unsigned report must say rescore-only or hash-only, got: {stdout}"
    );
}

// ── 7. keygen produces non-empty output ────────────────────────────────────────

#[test]
fn keygen_prints_key_pair() {
    let home = TempDir::new().unwrap();
    cli(&home)
        .args(["keygen"])
        .assert()
        .success()
        .stdout(predicates::str::contains("TRACERAZOR_SIGNING_KEY="))
        .stdout(predicates::str::contains("TRACERAZOR_VERIFY_KEY="));
}

// ── 8. Bundle round-trip through verify ───────────────────────────────────────

#[test]
fn bundle_round_trip_verify_ok() {
    let home = TempDir::new().unwrap();
    let trace = corpus_trace();
    let dir = TempDir::new().unwrap();
    let bundle_path = dir.path().join("evidence.zip");

    // Export a signed bundle
    cli(&home)
        .env("TRACERAZOR_SIGNING_KEY", TEST_KEY)
        .args([
            "export",
            trace.to_str().unwrap(),
            "--bundle",
            bundle_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(bundle_path.exists(), "export --bundle must create the zip file");

    // Verify the bundle
    cli(&home)
        .args(["verify", bundle_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicates::str::contains("bundle integrity: OK"))
        .stdout(predicates::str::contains("signature       : OK (Ed25519)"));
}
