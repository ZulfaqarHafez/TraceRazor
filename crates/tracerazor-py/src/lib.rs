//! Native Python bindings for the TraceRazor auditor core.
//!
//! Exposes a single fast, in-process entry point that mirrors what the CLI's
//! `audit ... --format json` does, without spawning a subprocess or writing a
//! temp file. The Python `teacher` package auto-detects and uses this module
//! (`import tracerazor_native`) when it is installed, falling back to the CLI
//! subprocess backend otherwise.
//!
//! Build:  `maturin develop -m crates/tracerazor-py/Cargo.toml`
//! Then:   `python -c "import tracerazor_native, json; \
//!             print(tracerazor_native.audit_json(json.dumps(trace)))"`

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use tracerazor_core::scoring::ScoringConfig;
use tracerazor_ingest::{parse as ingest_parse, TraceFormat};
use tracerazor_semantic::default_similarity_fn;

/// Audit a trace (raw / LangSmith / OTEL JSON, auto-detected) and return the
/// full report as a JSON string -- the exact shape the subprocess backend and
/// the `teacher.Diagnoser` already consume.
#[pyfunction]
fn audit_json(trace_json: &str) -> PyResult<String> {
    let mut trace = ingest_parse(trace_json, TraceFormat::Auto)
        .map_err(|e| PyValueError::new_err(format!("parse failed: {e}")))?;

    let sim_fn = default_similarity_fn();
    let config = ScoringConfig::default();

    let report = tracerazor_core::analyse(&mut trace, sim_fn, &config)
        .map_err(|e| PyValueError::new_err(format!("analyse failed: {e}")))?;

    serde_json::to_string(&report)
        .map_err(|e| PyValueError::new_err(format!("serialise failed: {e}")))
}

/// Audit with an explicit input format: "auto" | "raw" | "langsmith" | "otel".
#[pyfunction]
fn audit_json_with_format(trace_json: &str, fmt: &str) -> PyResult<String> {
    let format = match fmt.to_ascii_lowercase().as_str() {
        "auto" => TraceFormat::Auto,
        "raw" | "rawjson" => TraceFormat::RawJson,
        "langsmith" => TraceFormat::LangSmith,
        "otel" => TraceFormat::Otel,
        other => {
            return Err(PyValueError::new_err(format!("unknown format: {other}")));
        }
    };
    let mut trace = ingest_parse(trace_json, format)
        .map_err(|e| PyValueError::new_err(format!("parse failed: {e}")))?;
    let sim_fn = default_similarity_fn();
    let config = ScoringConfig::default();
    let report = tracerazor_core::analyse(&mut trace, sim_fn, &config)
        .map_err(|e| PyValueError::new_err(format!("analyse failed: {e}")))?;
    serde_json::to_string(&report)
        .map_err(|e| PyValueError::new_err(format!("serialise failed: {e}")))
}

#[pymodule]
fn tracerazor_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(audit_json, m)?)?;
    m.add_function(wrap_pyfunction!(audit_json_with_format, m)?)?;
    m.add("__doc__", "Native TraceRazor auditor bindings (PyO3).")?;
    Ok(())
}
