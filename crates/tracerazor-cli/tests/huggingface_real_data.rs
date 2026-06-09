//! Statistics test: audit the product against **real** agent trajectories
//! sourced from Hugging Face (`zai-org/AgentInstruct`).
//!
//! The corpus under `traces/external/huggingface/agentinstruct/` is converted
//! from the AgentInstruct ReAct dataset (see `tools/convert_agentinstruct.py`
//! and `traces/external/huggingface/agentinstruct/SOURCE.md`). This test runs
//! the real `tracerazor` binary over every trace, aggregates audit statistics,
//! and asserts the auditor behaves sanely on real tool-using agents (every
//! trace parses and scores in-range, structural waste is actually detected,
//! short traces are skipped cleanly). Run with `--nocapture` to print the
//! corpus statistics:
//!
//! ```text
//! cargo test -p tracerazor --test huggingface_real_data -- --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use assert_cmd::Command;
use serde_json::Value;
use tempfile::TempDir;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("traces")
        .join("external")
        .join("huggingface")
        .join("agentinstruct")
}

fn trace_files() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(corpus_dir())
        .expect("HuggingFace AgentInstruct corpus directory must exist")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().map(|x| x == "json").unwrap_or(false)
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("agentinstruct-"))
                    .unwrap_or(false)
        })
        .collect();
    files.sort();
    files
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

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// End-to-end statistics over the real Hugging Face corpus.
#[test]
fn huggingface_agentinstruct_audit_statistics() {
    let home = TempDir::new().unwrap();
    let files = trace_files();
    assert!(
        files.len() >= 8,
        "expected the vendored HuggingFace AgentInstruct corpus (>=8 traces), found {}",
        files.len()
    );

    let mut tas: Vec<f64> = Vec::new();
    let mut srr: Vec<f64> = Vec::new();
    let mut ldi: Vec<f64> = Vec::new();
    let mut obs: Vec<f64> = Vec::new();
    let mut gar: Vec<f64> = Vec::new();
    let mut mvtg: Vec<f64> = Vec::new();
    let mut grades: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_fixes = 0usize;
    let mut analysable = 0usize;
    let mut skipped = 0usize;

    for f in &files {
        let out = cli(&home)
            .args(["audit", f.to_str().unwrap(), "--format", "json"])
            .assert()
            .success()
            .get_output()
            .clone();
        let output = out.stdout;

        match serde_json::from_slice::<Value>(&output) {
            Ok(report) => {
                analysable += 1;
                let score = &report["score"];
                let t = score["score"].as_f64().expect("score.score must be a number");
                assert!(
                    (0.0..=100.0).contains(&t),
                    "TAS for {:?} out of range: {t}",
                    f.file_name().unwrap()
                );
                tas.push(t);

                let grade = score["grade"].as_str().unwrap_or("?").to_string();
                assert!(!grade.is_empty(), "grade must be populated");
                *grades.entry(grade).or_insert(0) += 1;

                let mn = &score["metric_normalised"];
                let get = |k: &str| mn.get(k).and_then(|v| v.as_f64());
                if let Some(v) = get("srr") {
                    srr.push(v);
                }
                if let Some(v) = get("ldi") {
                    ldi.push(v);
                }
                if let Some(v) = get("obs") {
                    obs.push(v);
                }
                if let Some(v) = get("gar") {
                    gar.push(v);
                }
                if let Some(v) = report["mvtg"].as_f64() {
                    mvtg.push(v);
                }
                total_fixes += report["fixes"].as_array().map(|a| a.len()).unwrap_or(0);
            }
            Err(_) => {
                // Sub-floor traces (< MIN_TRACE_STEPS) are skipped with a notice
                // on stderr, not an error. Confirm that is what happened.
                skipped += 1;
                let mut text = String::from_utf8_lossy(&output).into_owned();
                text.push_str(&String::from_utf8_lossy(&out.stderr));
                let low = text.to_lowercase();
                assert!(
                    low.contains("minimum") || low.contains("step"),
                    "non-JSON audit output for {:?} should be a step-count notice, got: {text}",
                    f.file_name().unwrap()
                );
            }
        }
    }

    // ── Invariants that demonstrate the product works on real data ───────────
    assert!(
        analysable >= 7,
        "expected >=7 analysable real traces, got {analysable}"
    );
    assert!(
        !tas.is_empty() && (0.0..=100.0).contains(&mean(&tas)),
        "mean TAS must be a valid score"
    );
    // The auditor must actually *find* structural redundancy on real tool-using
    // agents — otherwise it is blind. SRR is normalised so 1.0 == "no redundancy
    // detected"; at least one real trace must come in below that.
    let min_srr = srr.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        min_srr < 1.0,
        "audit detected no step redundancy anywhere in the real corpus (min normalised SRR = {min_srr})"
    );
    assert!(
        total_fixes > 0,
        "audit produced no fix patches across the entire real corpus"
    );

    // ── Emit the statistics (visible with --nocapture) ───────────────────────
    println!("\n==== Hugging Face AgentInstruct real-data audit ====");
    println!("source            : zai-org/AgentInstruct (ReAct trajectories)");
    println!("traces (total)    : {}", files.len());
    println!("  analysable      : {analysable}");
    println!("  skipped (<5 stp): {skipped}");
    println!("mean TAS          : {:.1}", mean(&tas));
    println!("grade distribution: {grades:?}");
    println!("mean SRR (norm)   : {:.3}", mean(&srr));
    println!("mean LDI (norm)   : {:.3}", mean(&ldi));
    println!("mean GAR (norm)   : {:.3}", mean(&gar));
    println!("mean OBS (norm)   : {:.3}", mean(&obs));
    println!("mean MVTG         : {:.3}", mean(&mvtg));
    println!("fix patches (sum) : {total_fixes}");
    println!("====================================================\n");
}
