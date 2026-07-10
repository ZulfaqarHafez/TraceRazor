//! Auditable-run provenance: hashing, Ed25519 report signing, and the full
//! report-verification protocol.
//!
//! Everything `tracerazor verify` checks lives here so that library
//! consumers can produce and verify provenance-bound reports without
//! shelling out to the CLI. The trace parser and similarity backend are
//! injected (mirroring [`crate::analyse`]) so this module stays free of
//! format and network dependencies.

use anyhow::Context;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::report::TraceReport;
use crate::scoring::ScoringConfig;
use crate::types::Trace;

/// SHA-256 of `bytes`, lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Lowercase hex encoding.
pub fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Decode a lowercase/uppercase hex string.
pub fn hex_decode(s: &str) -> anyhow::Result<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        anyhow::bail!("odd-length hex string");
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|_| anyhow::anyhow!("invalid hex character at position {i}"))
        })
        .collect()
}

/// Decode exactly 32 bytes (64 hex chars) — Ed25519 seeds and public keys.
pub fn hex_decode_32(s: &str) -> anyhow::Result<[u8; 32]> {
    let v = hex_decode(s)?;
    v.try_into().map_err(|v: Vec<u8>| {
        anyhow::anyhow!("expected 32 bytes (64 hex chars), got {} bytes", v.len())
    })
}

/// Decode exactly 64 bytes (128 hex chars) — Ed25519 signatures.
pub fn hex_decode_64(s: &str) -> anyhow::Result<[u8; 64]> {
    let v = hex_decode(s)?;
    v.try_into().map_err(|v: Vec<u8>| {
        anyhow::anyhow!("expected 64 bytes (128 hex chars), got {} bytes", v.len())
    })
}

/// Sign the report's canonical bytes with an Ed25519 seed and store the
/// hex-encoded signature + verifying key in the manifest.
///
/// The signature covers every field (including `manifest.similarity_backend`,
/// `agf`, `savings`, `fixes`, `summary`) so any post-audit edit breaks it.
pub fn sign_report(report: &mut TraceReport, seed: &[u8; 32]) -> anyhow::Result<()> {
    let signing_key = SigningKey::from_bytes(seed);
    let verifying_key = signing_key.verifying_key();

    // Normalise f64 values via a JSON round-trip before computing canonical
    // bytes. Without this, some f64s serialise differently depending on whether
    // they were computed in-process (sign time) vs deserialized from a JSON
    // file (verify time), producing last-digit differences such as
    // `0.9333333333333333` vs `0.9333333333333332`. Round-tripping here
    // guarantees both sides use the same parsed-from-JSON f64 values.
    let json = serde_json::to_string(report)
        .context("failed to serialise report for float normalisation")?;
    let normalized: TraceReport = serde_json::from_str(&json)
        .context("failed to deserialise report for float normalisation")?;

    let canonical = normalized
        .canonical_bytes()
        .context("failed to serialise report for signing")?;
    let sig: Signature = signing_key.sign(&canonical);
    if let Some(ref mut m) = report.manifest {
        m.signature = Some(hex_encode(&sig.to_bytes()));
        m.signing_key_pub = Some(hex_encode(verifying_key.as_bytes()));
    }
    Ok(())
}

/// Outcome of checking the embedded Ed25519 signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureStatus {
    /// Signature present and valid over the canonical report bytes.
    Valid,
    /// Neither signature nor public key present.
    Unsigned,
}

/// Verify the embedded Ed25519 signature, if present.
///
/// Returns [`VerifyError::SignatureInvalid`] for a bad signature, a malformed
/// key/signature encoding, or a one-sided manifest (exactly one of the
/// signature / public-key fields present — a signing audit embeds both, so
/// one-sided means a field was stripped after signing).
pub fn check_signature(report: &TraceReport) -> Result<SignatureStatus, VerifyError> {
    let manifest = match report.manifest.as_ref() {
        Some(m) => m,
        None => return Ok(SignatureStatus::Unsigned),
    };

    let (sig_hex, pub_hex) = match (&manifest.signature, &manifest.signing_key_pub) {
        (Some(s), Some(p)) => (s.as_str(), p.as_str()),
        (None, None) => return Ok(SignatureStatus::Unsigned),
        _ => return Err(VerifyError::SignatureInvalid),
    };

    let sig_bytes = match hex_decode_64(sig_hex) {
        Ok(b) => b,
        Err(_) => return Err(VerifyError::SignatureInvalid),
    };
    let pub_bytes = match hex_decode_32(pub_hex) {
        Ok(b) => b,
        Err(_) => return Err(VerifyError::SignatureInvalid),
    };

    let verifying_key = match VerifyingKey::from_bytes(&pub_bytes) {
        Ok(k) => k,
        Err(_) => return Err(VerifyError::SignatureInvalid),
    };
    let sig = Signature::from_bytes(&sig_bytes);
    let canonical = report.canonical_bytes().map_err(|e| {
        VerifyError::Operational(
            anyhow::Error::new(e).context("failed to compute canonical bytes for signature check"),
        )
    })?;

    match verifying_key.verify(&canonical, &sig) {
        Ok(()) => Ok(SignatureStatus::Valid),
        Err(_) => Err(VerifyError::SignatureInvalid),
    }
}

/// Stable contract for an offline agent-run receipt.
///
/// The declaration order is the canonical JSON field order. The optional
/// `signature` envelope is deliberately excluded from [`canonical_bytes`],
/// while `signed` remains covered so stripping the envelope cannot turn an
/// authenticated receipt into an apparently legitimate unsigned receipt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RunReceiptV1 {
    pub schema_version: String,
    pub run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_agent_id: Option<String>,
    pub created_at: String,
    pub privacy: String,
    pub hermetic: bool,
    pub replayable: bool,
    pub verification_mode: String,
    pub audit_trace_sha256: String,
    pub persisted_trace_sha256: String,
    pub report_sha256: String,
    #[serde(default)]
    pub signed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<RunReceiptSignature>,
}

/// Ed25519 authentication envelope for [`RunReceiptV1`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RunReceiptSignature {
    pub algorithm: String,
    pub public_key: String,
    pub signature: String,
}

/// Authentication result for a structurally valid run receipt.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunReceiptSignatureStatus {
    Valid,
    Unsigned,
}

/// Receipt verification separates malformed input from authenticated
/// evidence that failed its signature check. CLI callers map these to exit 2
/// and exit 1 respectively.
#[derive(Debug, thiserror::Error)]
pub enum RunReceiptVerifyError {
    #[error("malformed run receipt: {0}")]
    Malformed(String),
    #[error("run receipt signature is invalid")]
    Tampered,
}

impl RunReceiptVerifyError {
    pub fn is_tamper(&self) -> bool {
        matches!(self, Self::Tampered)
    }
}

impl RunReceiptV1 {
    pub const SCHEMA_VERSION: &'static str = "tracerazor-run-receipt/v1";

    /// Deterministic JSON bytes covered by the receipt signature.
    pub fn canonical_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut payload = self.clone();
        payload.signature = None;
        serde_json::to_vec(&payload).context("failed to canonicalize run receipt")
    }

    /// Validate identity and digest fields independently of signature state.
    pub fn validate_identity(&self) -> Result<(), RunReceiptVerifyError> {
        if self.schema_version != Self::SCHEMA_VERSION {
            return Err(RunReceiptVerifyError::Malformed(format!(
                "unsupported schema_version {:?}",
                self.schema_version
            )));
        }
        if self.run_id.is_empty()
            || self.run_id.len() > 128
            || !self
                .run_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        {
            return Err(RunReceiptVerifyError::Malformed(
                "run_id must be 1-128 ASCII letters, digits, '-' or '_'".to_string(),
            ));
        }
        for (name, value) in [
            ("session_id", self.session_id.as_deref()),
            ("agent_id", self.agent_id.as_deref()),
            ("parent_agent_id", self.parent_agent_id.as_deref()),
        ] {
            if let Some(value) = value {
                if value.is_empty()
                    || value.len() > 128
                    || !value
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
                {
                    return Err(RunReceiptVerifyError::Malformed(format!(
                        "{name} must be 1-128 ASCII letters, digits, '-' or '_'"
                    )));
                }
            }
        }
        if self
            .trace_id
            .as_deref()
            .is_some_and(|value| !is_lower_hex(value, 32) || value.bytes().all(|byte| byte == b'0'))
        {
            return Err(RunReceiptVerifyError::Malformed(
                "trace_id must be 32 lowercase non-zero hex characters".to_string(),
            ));
        }
        chrono::DateTime::parse_from_rfc3339(&self.created_at).map_err(|error| {
            RunReceiptVerifyError::Malformed(format!("created_at is not RFC 3339: {error}"))
        })?;
        if !matches!(self.privacy.as_str(), "local-redacted" | "raw") {
            return Err(RunReceiptVerifyError::Malformed(format!(
                "unsupported privacy mode {:?}",
                self.privacy
            )));
        }
        let expected_mode = if self.replayable {
            "hermetic_replay"
        } else {
            "non_replayable_receipt"
        };
        if self.verification_mode != expected_mode {
            return Err(RunReceiptVerifyError::Malformed(format!(
                "verification_mode must be {expected_mode:?} when replayable is {}",
                self.replayable
            )));
        }
        for (name, digest) in [
            ("audit_trace_sha256", &self.audit_trace_sha256),
            ("persisted_trace_sha256", &self.persisted_trace_sha256),
            ("report_sha256", &self.report_sha256),
        ] {
            if !is_lower_hex(digest, 64) {
                return Err(RunReceiptVerifyError::Malformed(format!(
                    "{name} must be 64 lowercase hex characters"
                )));
            }
        }
        if self.replayable && self.audit_trace_sha256 != self.persisted_trace_sha256 {
            return Err(RunReceiptVerifyError::Malformed(
                "replayable receipt must bind identical audit and persisted trace hashes"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Sign a v1 run receipt using the same 32-byte Ed25519 seed convention as
/// signed TraceRazor reports.
pub fn sign_run_receipt(receipt: &mut RunReceiptV1, seed: &[u8; 32]) -> anyhow::Result<()> {
    receipt
        .validate_identity()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    receipt.signed = true;
    receipt.signature = None;
    let canonical = receipt.canonical_bytes()?;
    let signing_key = SigningKey::from_bytes(seed);
    let signature: Signature = signing_key.sign(&canonical);
    receipt.signature = Some(RunReceiptSignature {
        algorithm: "Ed25519".to_string(),
        public_key: hex_encode(signing_key.verifying_key().as_bytes()),
        signature: hex_encode(&signature.to_bytes()),
    });
    Ok(())
}

/// Parse and validate a v1 run receipt without asserting authenticity.
pub fn parse_run_receipt(raw: &str) -> Result<RunReceiptV1, RunReceiptVerifyError> {
    let receipt: RunReceiptV1 = serde_json::from_str(raw)
        .map_err(|error| RunReceiptVerifyError::Malformed(error.to_string()))?;
    receipt.validate_identity()?;
    match (receipt.signed, receipt.signature.is_some()) {
        (false, false) | (true, true) => Ok(receipt),
        (true, false) => Err(RunReceiptVerifyError::Malformed(
            "signed receipt is missing its signature envelope".to_string(),
        )),
        (false, true) => Err(RunReceiptVerifyError::Malformed(
            "unsigned receipt must not contain a signature envelope".to_string(),
        )),
    }
}

/// Verify a parsed receipt's Ed25519 envelope, or return explicit unsigned
/// status for a well-formed legacy/local receipt.
pub fn verify_run_receipt(
    receipt: &RunReceiptV1,
) -> Result<RunReceiptSignatureStatus, RunReceiptVerifyError> {
    receipt.validate_identity()?;
    let envelope = match (receipt.signed, receipt.signature.as_ref()) {
        (false, None) => return Ok(RunReceiptSignatureStatus::Unsigned),
        (true, Some(envelope)) => envelope,
        (true, None) => {
            return Err(RunReceiptVerifyError::Malformed(
                "signed receipt is missing its signature envelope".to_string(),
            ))
        }
        (false, Some(_)) => {
            return Err(RunReceiptVerifyError::Malformed(
                "unsigned receipt must not contain a signature envelope".to_string(),
            ))
        }
    };
    if envelope.algorithm != "Ed25519" {
        return Err(RunReceiptVerifyError::Malformed(format!(
            "unsupported signature algorithm {:?}",
            envelope.algorithm
        )));
    }
    if !is_lower_hex(&envelope.public_key, 64) {
        return Err(RunReceiptVerifyError::Malformed(
            "signature public_key must be 64 lowercase hex characters".to_string(),
        ));
    }
    if !is_lower_hex(&envelope.signature, 128) {
        return Err(RunReceiptVerifyError::Malformed(
            "signature value must be 128 lowercase hex characters".to_string(),
        ));
    }
    let public_key = VerifyingKey::from_bytes(
        &hex_decode_32(&envelope.public_key)
            .map_err(|error| RunReceiptVerifyError::Malformed(error.to_string()))?,
    )
    .map_err(|error| RunReceiptVerifyError::Malformed(error.to_string()))?;
    let signature = Signature::from_bytes(
        &hex_decode_64(&envelope.signature)
            .map_err(|error| RunReceiptVerifyError::Malformed(error.to_string()))?,
    );
    public_key
        .verify(
            &receipt
                .canonical_bytes()
                .map_err(|error| RunReceiptVerifyError::Malformed(error.to_string()))?,
            &signature,
        )
        .map_err(|_| RunReceiptVerifyError::Tampered)?;
    Ok(RunReceiptSignatureStatus::Valid)
}

/// Parse and verify a receipt in one call.
pub fn verify_run_receipt_json(
    raw: &str,
) -> Result<(RunReceiptV1, RunReceiptSignatureStatus), RunReceiptVerifyError> {
    let receipt = parse_run_receipt(raw)?;
    let status = verify_run_receipt(&receipt)?;
    Ok((receipt, status))
}

fn is_lower_hex(value: &str, width: usize) -> bool {
    value.len() == width
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Why the deterministic re-score step was skipped (the hash/signature
/// checks still ran).
#[derive(Debug, Clone)]
pub enum RescoreStatus {
    /// Re-scored from the trace bytes; every compared field matched.
    Reproduced { tas: f64 },
    /// Report was produced by a different tool version.
    SkippedVersionMismatch { report_version: String },
    /// Embedding-backend scores are not locally reproducible.
    SkippedEmbeddingBackend { backend: String },
    /// The run read store-derived baselines; exact re-score requires that state.
    SkippedStoreInfluenced {
        baseline_tokens: Option<u32>,
        historical_median_steps: Option<f64>,
        n_historical_sequences: usize,
    },
}

/// Successful outcome of [`verify_report`]: what was checked and how far
/// verification could go.
#[derive(Debug, Clone)]
pub struct Verification {
    /// True when the report carried a valid Ed25519 signature.
    pub signed: bool,
    /// SHA-256 of the trace bytes (matched the manifest).
    pub trace_sha256: String,
    /// Whether the deterministic re-score ran, and its result.
    pub rescore: RescoreStatus,
}

/// Verification failure. `is_tamper()` distinguishes evidence of tampering
/// (callers conventionally exit 1) from operational errors (exit 2).
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("report JSON does not match the TraceReport schema: {0}")]
    MalformedReport(String),
    #[error(
        "report carries no run manifest (produced by a pre-provenance \
         TraceRazor version); re-audit with this version to get one"
    )]
    MissingManifest,
    #[error("Ed25519 signature verification failed; report has been modified after signing")]
    SignatureInvalid,
    #[error("trace file hash does not match the manifest (manifest {manifest}, on disk {actual})")]
    TraceHashMismatch {
        signed: bool,
        manifest: String,
        actual: String,
    },
    #[error("re-scored values differ from the report: {}", mismatches.join("; "))]
    RescoreMismatch {
        signed: bool,
        trace_sha256: String,
        recomputed_tas: f64,
        mismatches: Vec<String>,
    },
    #[error(transparent)]
    Operational(#[from] anyhow::Error),
}

impl VerifyError {
    /// True for failures that are evidence of tampering rather than
    /// operational problems.
    pub fn is_tamper(&self) -> bool {
        matches!(
            self,
            VerifyError::SignatureInvalid
                | VerifyError::TraceHashMismatch { .. }
                | VerifyError::RescoreMismatch { .. }
        )
    }
}

/// The full `tracerazor verify` protocol over in-memory bytes.
///
/// Checks, in order: signature (first — it covers every field), trace hash,
/// tool version, backend reproducibility, store influence, then a
/// deterministic re-score compared field-by-field against the report.
///
/// `parse_trace` and `similarity_fn` are injected so core stays independent
/// of `tracerazor-ingest` / `tracerazor-semantic`; pass
/// `tracerazor_ingest::parse` (or equivalent) and the BoW similarity
/// function, with `bow_backend_id` identifying that backend in manifests.
pub fn verify_report(
    report_raw: &str,
    trace_bytes: &[u8],
    current_version: &str,
    bow_backend_id: &str,
    parse_trace: impl FnOnce(&str) -> anyhow::Result<Trace>,
    similarity_fn: impl Fn(&str, &str) -> f64,
) -> Result<Verification, VerifyError> {
    let report_value: serde_json::Value = serde_json::from_str(report_raw)
        .map_err(|e| VerifyError::MalformedReport(format!("report is not valid JSON: {e}")))?;
    let report_struct: TraceReport = serde_json::from_str(report_raw)
        .map_err(|e| VerifyError::MalformedReport(e.to_string()))?;

    let Some(manifest) = report_struct.manifest.clone() else {
        return Err(VerifyError::MissingManifest);
    };

    // ── Signature check — FIRST, before any other check ──────────────────────
    // A valid signature proves the ENTIRE report is authentic: TAS, AGF,
    // savings, fixes, summary, similarity_backend — every field is covered.
    // Any post-audit edit to any field breaks the signature.
    let signed = match check_signature(&report_struct)? {
        SignatureStatus::Valid => true,
        SignatureStatus::Unsigned => false,
    };

    // ── Trace hash ────────────────────────────────────────────────────────────
    let actual_sha = sha256_hex(trace_bytes);
    if actual_sha != manifest.trace_sha256 {
        return Err(VerifyError::TraceHashMismatch {
            signed,
            manifest: manifest.trace_sha256.clone(),
            actual: actual_sha,
        });
    }

    // ── Version gate ──────────────────────────────────────────────────────────
    if manifest.tool_version != current_version {
        return Ok(Verification {
            signed,
            trace_sha256: actual_sha,
            rescore: RescoreStatus::SkippedVersionMismatch {
                report_version: manifest.tool_version.clone(),
            },
        });
    }

    // ── Re-score — only sound for BoW + hermetic ─────────────────────────────
    // Under a valid signature a backend mismatch is TAMPERED (the signature
    // covers similarity_backend), so reaching here with a non-BoW backend
    // means the claim is authentic — it just cannot be locally re-scored.
    if manifest.similarity_backend != bow_backend_id {
        return Ok(Verification {
            signed,
            trace_sha256: actual_sha,
            rescore: RescoreStatus::SkippedEmbeddingBackend {
                backend: manifest.similarity_backend.clone(),
            },
        });
    }
    if manifest.store_influenced() {
        return Ok(Verification {
            signed,
            trace_sha256: actual_sha,
            rescore: RescoreStatus::SkippedStoreInfluenced {
                baseline_tokens: manifest.baseline_tokens,
                historical_median_steps: manifest.historical_median_steps,
                n_historical_sequences: manifest.n_historical_sequences,
            },
        });
    }

    let trace_str = std::str::from_utf8(trace_bytes)
        .context("trace is not valid UTF-8")
        .map_err(VerifyError::Operational)?;
    let mut trace = parse_trace(trace_str)
        .context("failed to parse trace for re-score")
        .map_err(VerifyError::Operational)?;
    if trace.steps.len() < manifest.min_steps.max(2) {
        return Err(VerifyError::Operational(anyhow::anyhow!(
            "trace has {} steps but the manifest floor is {} — nothing to re-score",
            trace.steps.len(),
            manifest.min_steps
        )));
    }

    let config = ScoringConfig {
        weights: manifest.weights.clone(),
        threshold: manifest.threshold,
        cost_per_million_tokens: manifest.cost_per_million_tokens,
        ..Default::default()
    };
    let recomputed =
        crate::analyse(&mut trace, similarity_fn, &config).map_err(VerifyError::Operational)?;

    // ── Compare the WHOLE report — not just TAS + metric_normalised ──────────
    let original_tas = report_value["score"]["score"].as_f64().unwrap_or(f64::NAN);
    let recomputed_tas = recomputed.score.score;
    let mut mismatches: Vec<String> = Vec::new();
    if (original_tas - recomputed_tas).abs() > 1e-9 {
        mismatches.push(format!("TAS {original_tas} -> {recomputed_tas}"));
    }
    let recomputed_score_json =
        serde_json::to_value(&recomputed.score).map_err(|e| VerifyError::Operational(e.into()))?;
    if let (Some(orig), Some(new)) = (
        report_value["score"]["metric_normalised"].as_object(),
        recomputed_score_json["metric_normalised"].as_object(),
    ) {
        for (k, ov) in orig {
            let o = ov.as_f64().unwrap_or(f64::NAN);
            let n = new.get(k).and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
            if (o - n).abs() > 1e-9 {
                mismatches.push(format!("{k} {o} -> {n}"));
            }
        }
    }
    // Compare AGF score
    if let (Some(orig_agf_score), Some(new_agf_score)) = (
        report_value["agf"]["score"].as_f64(),
        recomputed.agf.as_ref().map(|a| a.score),
    ) {
        if (orig_agf_score - new_agf_score).abs() > 1e-9 {
            mismatches.push(format!(
                "agf.score {orig_agf_score:.6} -> {new_agf_score:.6}"
            ));
        }
    }
    // Compare savings.tokens_saved
    if let Some(orig_saved) = report_value["savings"]["tokens_saved"].as_u64() {
        let new_saved = recomputed.savings.tokens_saved as u64;
        if orig_saved != new_saved {
            mismatches.push(format!("savings.tokens_saved {orig_saved} -> {new_saved}"));
        }
    }
    // Compare fix count
    let orig_fix_count = report_value["fixes"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    let new_fix_count = recomputed.fixes.len();
    if orig_fix_count != new_fix_count {
        mismatches.push(format!("fixes count {orig_fix_count} -> {new_fix_count}"));
    }
    // Compare summary (existence; exact text may differ with trailing whitespace)
    let orig_summary = report_value["summary"].as_str().unwrap_or("").trim();
    let new_summary = recomputed.summary.trim();
    if !orig_summary.is_empty() && orig_summary != new_summary {
        mismatches.push("summary text differs".to_string());
    }

    if mismatches.is_empty() {
        Ok(Verification {
            signed,
            trace_sha256: actual_sha,
            rescore: RescoreStatus::Reproduced {
                tas: recomputed_tas,
            },
        })
    } else {
        Err(VerifyError::RescoreMismatch {
            signed,
            trace_sha256: actual_sha,
            recomputed_tas,
            mismatches,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vector() {
        // SHA-256("abc") — FIPS 180-2 test vector.
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hex_round_trip() {
        let bytes = [0u8, 1, 0xab, 0xff];
        assert_eq!(hex_decode(&hex_encode(&bytes)).unwrap(), bytes);
        assert!(hex_decode("abc").is_err()); // odd length
        assert!(hex_decode("zz").is_err()); // invalid chars
    }

    fn receipt() -> RunReceiptV1 {
        RunReceiptV1 {
            schema_version: RunReceiptV1::SCHEMA_VERSION.to_string(),
            run_id: "run-test".to_string(),
            trace_id: Some("1".repeat(32)),
            session_id: Some("session-test".to_string()),
            agent_id: Some("agent-test".to_string()),
            parent_agent_id: None,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            privacy: "local-redacted".to_string(),
            hermetic: true,
            replayable: false,
            verification_mode: "non_replayable_receipt".to_string(),
            audit_trace_sha256: "a".repeat(64),
            persisted_trace_sha256: "b".repeat(64),
            report_sha256: "c".repeat(64),
            signed: false,
            signature: None,
        }
    }

    #[test]
    fn run_receipt_signature_round_trip_and_canonical_bytes() {
        let seed = [0xaa; 32];
        let mut value = receipt();
        sign_run_receipt(&mut value, &seed).unwrap();
        assert!(value.signed);
        assert_eq!(value.signature.as_ref().unwrap().algorithm, "Ed25519");
        assert_eq!(
            verify_run_receipt(&value).unwrap(),
            RunReceiptSignatureStatus::Valid
        );
        let canonical = value.canonical_bytes().unwrap();
        assert!(!String::from_utf8(canonical).unwrap().contains("signature"));

        let encoded = serde_json::to_string(&value).unwrap();
        let (decoded, status) = verify_run_receipt_json(&encoded).unwrap();
        assert_eq!(decoded, value);
        assert_eq!(status, RunReceiptSignatureStatus::Valid);
    }

    #[test]
    fn run_receipt_tamper_is_distinct_from_malformed() {
        let mut value = receipt();
        sign_run_receipt(&mut value, &[0xbb; 32]).unwrap();
        value.run_id = "run-tampered".to_string();
        assert!(matches!(
            verify_run_receipt(&value),
            Err(RunReceiptVerifyError::Tampered)
        ));

        let mut malformed = receipt();
        malformed.report_sha256 = "not-a-digest".to_string();
        assert!(matches!(
            verify_run_receipt(&malformed),
            Err(RunReceiptVerifyError::Malformed(_))
        ));
    }

    #[test]
    fn legacy_unsigned_run_receipt_is_explicit() {
        assert_eq!(
            verify_run_receipt(&receipt()).unwrap(),
            RunReceiptSignatureStatus::Unsigned
        );
    }
}
