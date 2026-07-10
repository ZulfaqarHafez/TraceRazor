//! Local OTLP/HTTP ingest endpoint.
//!
//! The endpoint accepts the standard JSON and binary Protobuf transports at
//! `/v1/traces`. Both use the same OTel normalizer, durable local-redacted
//! receipt, and commit ordering so transport choice cannot weaken privacy or
//! silently drop a trace from a multi-trace batch.

use anyhow::{bail, Context, Result};
use axum::{
    body::Bytes,
    extract::{rejection::BytesRejection, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use chrono::Utc;
use flate2::read::MultiGzDecoder;
use opentelemetry_proto::tonic::collector::trace::v1::{
    ExportTracePartialSuccess, ExportTraceServiceResponse,
};
use prost::Message;
use serde::Serialize;
use serde_json::json;
#[cfg(unix)]
use std::fs::File;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};
use tracerazor_core::{provenance::sha256_hex, types::Trace};
use uuid::Uuid;

use crate::{state::AppState, MAX_BODY_BYTES};

const MAX_TRACE_STEPS: usize = 50_000;
const STATUS_INVALID_ARGUMENT: i32 = 3;
const STATUS_RESOURCE_EXHAUSTED: i32 = 8;
const STATUS_INTERNAL: i32 = 13;
const STATUS_UNAVAILABLE: i32 = 14;
pub(crate) const STATUS_UNAUTHENTICATED: i32 = 16;

const JSON_CONTENT_TYPE: &str = "application/json";
const PROTOBUF_CONTENT_TYPE: &str = "application/x-protobuf";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WireFormat {
    Json,
    Protobuf,
}

impl WireFormat {
    fn transport(self) -> &'static str {
        match self {
            Self::Json => "otlp_http_json",
            Self::Protobuf => "otlp_http_protobuf",
        }
    }
}

/// Wire-compatible subset of `google.rpc.Status` used for binary OTLP errors.
#[derive(Clone, PartialEq, Message)]
struct ProtobufStatus {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
}

#[derive(Serialize)]
struct SpoolReceipt<'a> {
    schema_version: &'static str,
    received_at: String,
    transport: &'static str,
    content_encoding: &'static str,
    privacy: &'static str,
    source_payload_sha256: &'a str,
    traces: &'a [Trace],
}

/// POST `/v1/traces` — receive an OTLP/HTTP `ExportTraceServiceRequest`.
pub(crate) async fn export_traces(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let response_format = wire_format(&headers).unwrap_or(WireFormat::Json);
    let body = match body {
        Ok(body) => body,
        Err(rejection) => {
            let status = rejection.status();
            let (status, code, message) = if status == StatusCode::PAYLOAD_TOO_LARGE {
                (
                    StatusCode::PAYLOAD_TOO_LARGE,
                    STATUS_RESOURCE_EXHAUSTED,
                    format!("OTLP payload exceeds the {MAX_BODY_BYTES}-byte request limit"),
                )
            } else {
                (
                    StatusCode::BAD_REQUEST,
                    STATUS_INVALID_ARGUMENT,
                    "unable to read OTLP request body".to_string(),
                )
            };
            return error_response_with_format(status, code, message, response_format);
        }
    };

    let wire_format =
        match wire_format(&headers) {
            Some(format) => format,
            None => return error_response(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                STATUS_INVALID_ARGUMENT,
                "OTLP receiver supports Content-Type: application/json or application/x-protobuf",
            ),
        };

    let is_gzip = match content_encoding(&headers) {
        Some(is_gzip) => is_gzip,
        None => {
            return error_response_with_format(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                STATUS_INVALID_ARGUMENT,
                "OTLP receiver supports Content-Encoding: gzip or identity",
                wire_format,
            )
        }
    };

    let decompressed;
    let payload_bytes: &[u8] = if is_gzip {
        let compressed = body.clone();
        decompressed = match tokio::task::spawn_blocking(move || decode_gzip(&compressed)).await {
            Ok(Ok(decoded)) => decoded,
            Ok(Err(GzipError::TooLarge)) => {
                return error_response_with_format(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    STATUS_RESOURCE_EXHAUSTED,
                    format!(
                        "decompressed OTLP payload exceeds the {MAX_BODY_BYTES}-byte request limit"
                    ),
                    wire_format,
                )
            }
            Ok(Err(GzipError::Invalid(error))) => {
                eprintln!("OTLP gzip decode failed: {error}");
                return error_response_with_format(
                    StatusCode::BAD_REQUEST,
                    STATUS_INVALID_ARGUMENT,
                    "invalid gzip-compressed OTLP/HTTP payload",
                    wire_format,
                );
            }
            Err(error) => {
                eprintln!("OTLP gzip worker failed: {error:#}");
                return error_response_with_format(
                    StatusCode::SERVICE_UNAVAILABLE,
                    STATUS_UNAVAILABLE,
                    "OTLP decompression is temporarily unavailable",
                    wire_format,
                );
            }
        };
        &decompressed
    } else {
        &body
    };

    let traces = match parse_payload(payload_bytes, wire_format) {
        Ok(traces) => traces,
        Err(error) => {
            eprintln!("OTLP ingest rejected malformed payload: {error:#}");
            return error_response_with_format(
                StatusCode::BAD_REQUEST,
                STATUS_INVALID_ARGUMENT,
                match wire_format {
                    WireFormat::Json => "invalid OTLP/HTTP JSON trace payload",
                    WireFormat::Protobuf => "invalid OTLP/HTTP protobuf trace payload",
                },
                wire_format,
            );
        }
    };
    if traces.is_empty() {
        return success_response(wire_format, None);
    }
    let ingest_warning = batch_ingest_warning(&traces);

    let step_count = traces
        .iter()
        .map(|trace| trace.steps.len())
        .fold(0usize, usize::saturating_add);
    if step_count > MAX_TRACE_STEPS {
        return error_response_with_format(
            StatusCode::PAYLOAD_TOO_LARGE,
            STATUS_RESOURCE_EXHAUSTED,
            format!("normalized batch has {step_count} steps; maximum is {MAX_TRACE_STEPS}"),
            wire_format,
        );
    }

    // Raw prompt, completion, parameters, and error content remain memory-only
    // by default. The durable receipt and SQLite record retain hashes, token
    // counts, and structural metadata so local-redacted is the safe default.
    let payload_sha256 = sha256_hex(&body);
    let persisted_traces = traces
        .into_iter()
        .map(|trace| redact_trace(trace, &payload_sha256))
        .collect::<Vec<_>>();
    let receipt = SpoolReceipt {
        schema_version: "tracerazor-otlp-spool/v1",
        received_at: Utc::now().to_rfc3339(),
        transport: wire_format.transport(),
        content_encoding: if is_gzip { "gzip" } else { "identity" },
        privacy: "local-redacted",
        source_payload_sha256: &payload_sha256,
        traces: &persisted_traces,
    };
    let receipt_bytes = match serde_json::to_vec(&receipt) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("OTLP receipt serialization failed: {error:#}");
            return error_response_with_format(
                StatusCode::INTERNAL_SERVER_ERROR,
                STATUS_INTERNAL,
                "failed to prepare durable OTLP receipt",
                wire_format,
            );
        }
    };

    let spool_dir = (*state.otlp_spool_dir).clone();
    let spool_result =
        tokio::task::spawn_blocking(move || write_spool_receipt(&spool_dir, &receipt_bytes)).await;
    match spool_result {
        Ok(Ok(_)) => {}
        Ok(Err(error)) => {
            eprintln!("OTLP spool write failed: {error:#}");
            return error_response_with_format(
                StatusCode::SERVICE_UNAVAILABLE,
                STATUS_UNAVAILABLE,
                "durable OTLP spool is unavailable",
                wire_format,
            );
        }
        Err(error) => {
            eprintln!("OTLP spool worker failed: {error:#}");
            return error_response_with_format(
                StatusCode::SERVICE_UNAVAILABLE,
                STATUS_UNAVAILABLE,
                "durable OTLP spool is unavailable",
                wire_format,
            );
        }
    }

    for trace in &persisted_traces {
        if let Err(error) = state.store.save_trace(trace, None).await {
            // The complete batch receipt was already fsynced and atomically
            // renamed, so a later store outage cannot silently discard any
            // trace ID from the accepted export. A retry safely upserts IDs.
            eprintln!("OTLP trace store commit failed after spooling: {error:#}");
            return error_response_with_format(
                StatusCode::SERVICE_UNAVAILABLE,
                STATUS_UNAVAILABLE,
                "OTLP batch was durably spooled but local ingestion failed",
                wire_format,
            );
        }
    }

    success_response(wire_format, ingest_warning.as_deref())
}

fn parse_payload(payload: &[u8], format: WireFormat) -> Result<Vec<Trace>> {
    match format {
        WireFormat::Json => {
            let payload = std::str::from_utf8(payload)
                .context("OTLP/HTTP JSON payload must be valid UTF-8")?;
            if is_empty_export(payload) {
                return Ok(Vec::new());
            }
            tracerazor_ingest::otel::parse_many(payload)
        }
        WireFormat::Protobuf => tracerazor_ingest::otel::parse_many_protobuf(payload),
    }
}

fn batch_ingest_warning(traces: &[Trace]) -> Option<String> {
    let mut issues = BTreeSet::new();
    let mut degraded = false;
    for trace in traces {
        let Some(otlp) = trace.metadata.get("otlp") else {
            continue;
        };
        degraded |= otlp
            .get("degraded_ingest")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if let Some(values) = otlp.get("issues").and_then(serde_json::Value::as_array) {
            issues.extend(values.iter().filter_map(serde_json::Value::as_str));
        }
    }
    if !degraded {
        return None;
    }
    let issue_summary = issues.into_iter().take(8).collect::<Vec<_>>().join(", ");
    Some(if issue_summary.is_empty() {
        "TraceRazor accepted the batch with degraded ingest; hard enforcement is disabled"
            .to_string()
    } else {
        format!(
            "TraceRazor accepted the batch with degraded ingest; hard enforcement is disabled: {issue_summary}"
        )
    })
}

fn success_response(format: WireFormat, warning: Option<&str>) -> Response {
    match format {
        WireFormat::Json => match warning {
            Some(warning) => (
                StatusCode::OK,
                Json(json!({
                    "partialSuccess": {
                        "rejectedSpans": "0",
                        "errorMessage": warning,
                    }
                })),
            )
                .into_response(),
            // ExportTraceServiceResponse has no required fields. `{}` is the
            // canonical proto-JSON full-success response.
            None => (StatusCode::OK, Json(json!({}))).into_response(),
        },
        WireFormat::Protobuf => {
            let response = ExportTraceServiceResponse {
                partial_success: warning.map(|warning| ExportTracePartialSuccess {
                    rejected_spans: 0,
                    error_message: warning.to_string(),
                }),
            };
            binary_response(StatusCode::OK, response.encode_to_vec())
        }
    }
}

pub(crate) fn error_response(
    status: StatusCode,
    code: i32,
    message: impl Into<String>,
) -> Response {
    error_response_with_format(status, code, message, WireFormat::Json)
}

/// Match authentication failures to the request's OTLP wire format. Unknown
/// content types receive a JSON status because no valid response format was
/// negotiated.
pub(crate) fn error_response_for_request(
    headers: &HeaderMap,
    status: StatusCode,
    code: i32,
    message: impl Into<String>,
) -> Response {
    error_response_with_format(
        status,
        code,
        message,
        wire_format(headers).unwrap_or(WireFormat::Json),
    )
}

fn error_response_with_format(
    status: StatusCode,
    code: i32,
    message: impl Into<String>,
    format: WireFormat,
) -> Response {
    let message = message.into();
    let mut response = match format {
        WireFormat::Json => {
            // OTLP/HTTP JSON errors use the proto-JSON representation of
            // google.rpc.Status.
            (status, Json(json!({ "code": code, "message": message }))).into_response()
        }
        WireFormat::Protobuf => {
            binary_response(status, ProtobufStatus { code, message }.encode_to_vec())
        }
    };
    if status == StatusCode::UNAUTHORIZED {
        response
            .headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    }
    response
}

fn binary_response(status: StatusCode, body: Vec<u8>) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, PROTOBUF_CONTENT_TYPE)],
        body,
    )
        .into_response()
}

fn wire_format(headers: &HeaderMap) -> Option<WireFormat> {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)?;
    if content_type.eq_ignore_ascii_case(JSON_CONTENT_TYPE) {
        Some(WireFormat::Json)
    } else if content_type.eq_ignore_ascii_case(PROTOBUF_CONTENT_TYPE) {
        Some(WireFormat::Protobuf)
    } else {
        None
    }
}

fn content_encoding(headers: &HeaderMap) -> Option<bool> {
    let Some(value) = headers.get(header::CONTENT_ENCODING) else {
        return Some(false);
    };
    let Ok(value) = value.to_str() else {
        return None;
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "identity" => Some(false),
        "gzip" => Some(true),
        _ => None,
    }
}

enum GzipError {
    Invalid(std::io::Error),
    TooLarge,
}

fn decode_gzip(compressed: &[u8]) -> std::result::Result<Vec<u8>, GzipError> {
    let decoder = MultiGzDecoder::new(compressed);
    let mut limited = decoder.take((MAX_BODY_BYTES + 1) as u64);
    let mut decoded = Vec::new();
    limited
        .read_to_end(&mut decoded)
        .map_err(GzipError::Invalid)?;
    if decoded.len() > MAX_BODY_BYTES {
        return Err(GzipError::TooLarge);
    }
    Ok(decoded)
}

fn is_empty_export(payload: &str) -> bool {
    let Ok(serde_json::Value::Object(root)) = serde_json::from_str(payload) else {
        return false;
    };
    if root.is_empty() {
        return true;
    }
    let Some(serde_json::Value::Array(resource_spans)) = root
        .get("resourceSpans")
        .or_else(|| root.get("resource_spans"))
    else {
        return false;
    };
    resource_spans.iter().all(|resource_span| {
        let Some(resource_span) = resource_span.as_object() else {
            return false;
        };
        let scope_spans = resource_span
            .get("scopeSpans")
            .or_else(|| resource_span.get("scope_spans"));
        match scope_spans {
            None => true,
            Some(serde_json::Value::Array(scope_spans)) => scope_spans.iter().all(|scope_span| {
                scope_span
                    .get("spans")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(Vec::is_empty)
            }),
            Some(_) => false,
        }
    })
}

fn redact_trace(mut trace: Trace, payload_sha256: &str) -> Trace {
    trace.agent_name = redacted_identifier(&trace.agent_name);
    for step in &mut trace.steps {
        step.content = redacted_text(&step.content);
        if let Some(params) = step.tool_params.as_mut() {
            let encoded = serde_json::to_vec(params).unwrap_or_default();
            *params = json!({
                "redacted_sha256": sha256_hex(&encoded),
                "bytes": encoded.len(),
            });
        }
        if let Some(error) = step.tool_error.as_mut() {
            *error = redacted_text(error);
        }
        if let Some(context) = step.input_context.as_mut() {
            *context = redacted_text(context);
        }
        if let Some(output) = step.output.as_mut() {
            *output = redacted_text(output);
        }
        if let Some(agent_id) = step.agent_id.as_mut() {
            *agent_id = redacted_identifier(agent_id);
        }
        step.flag_details.clear();
    }
    // The normalizer owns this namespaced metadata and guarantees it contains
    // only structural IDs, exact token counts/provenance, booleans, and issue
    // codes. Retain it so local-redacted receipts do not erase the evidence
    // needed to distinguish a real zero from missing usage.
    let safe_otlp_metadata = trace.metadata.get("otlp").cloned();
    let canonical_metadata = trace.metadata.iter().collect::<BTreeMap<_, _>>();
    let source_metadata = serde_json::to_vec(&canonical_metadata).unwrap_or_default();
    trace.metadata.clear();
    trace.metadata.insert(
        "source_metadata_sha256".to_string(),
        serde_json::Value::String(sha256_hex(&source_metadata)),
    );
    trace.metadata.insert(
        "source_metadata_bytes".to_string(),
        serde_json::Value::from(source_metadata.len() as u64),
    );
    trace.metadata.insert(
        "privacy".to_string(),
        serde_json::Value::String("local-redacted".to_string()),
    );
    trace.metadata.insert(
        "source_payload_sha256".to_string(),
        serde_json::Value::String(payload_sha256.to_string()),
    );
    if let Some(metadata) = safe_otlp_metadata {
        trace.metadata.insert("otlp".to_string(), metadata);
    }
    trace
}

fn redacted_identifier(value: &str) -> String {
    format!("[redacted-id sha256={}]", sha256_hex(value.as_bytes()))
}

fn redacted_text(value: &str) -> String {
    format!(
        "[redacted sha256={} chars={}]",
        sha256_hex(value.as_bytes()),
        value.chars().count()
    )
}

fn write_spool_receipt(configured_dir: &Path, contents: &[u8]) -> Result<PathBuf> {
    let spool_dir = prepare_spool_dir(configured_dir)?;
    let id = Uuid::new_v4();
    let final_path = spool_dir.join(format!("otlp-{id}.json"));
    let temporary_path = spool_dir.join(format!(".otlp-{id}.tmp"));

    let result = (|| -> Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&temporary_path)
            .with_context(|| format!("cannot create spool file {}", temporary_path.display()))?;
        file.write_all(contents)
            .context("cannot write OTLP spool receipt")?;
        file.sync_all().context("cannot fsync OTLP spool receipt")?;
        drop(file);

        // Both paths are generated leaves in the same verified directory, so
        // rename is atomic and no request-controlled path reaches the filesystem.
        fs::rename(&temporary_path, &final_path)
            .context("cannot atomically publish OTLP spool receipt")?;
        sync_directory(&spool_dir)?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result?;
    Ok(final_path)
}

fn prepare_spool_dir(configured_dir: &Path) -> Result<PathBuf> {
    if configured_dir.as_os_str().is_empty() {
        bail!("OTLP spool directory cannot be empty");
    }
    if configured_dir
        .components()
        .any(|component| component == Component::ParentDir)
    {
        bail!("OTLP spool directory must not contain '..'");
    }

    let absolute = if configured_dir.is_absolute() {
        configured_dir.to_path_buf()
    } else {
        std::env::current_dir()
            .context("cannot resolve current directory for OTLP spool")?
            .join(configured_dir)
    };
    reject_existing_links(&absolute)?;
    #[cfg(unix)]
    let existed = absolute.exists();
    fs::create_dir_all(&absolute)
        .with_context(|| format!("cannot create OTLP spool directory {}", absolute.display()))?;
    reject_existing_links(&absolute)?;
    if !absolute.is_dir() {
        bail!("OTLP spool path is not a directory");
    }

    #[cfg(unix)]
    if !existed {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&absolute, fs::Permissions::from_mode(0o700))
            .context("cannot restrict OTLP spool directory permissions")?;
    }

    Ok(absolute)
}

fn reject_existing_links(path: &Path) -> Result<()> {
    for component in path.ancestors().collect::<Vec<_>>().into_iter().rev() {
        if component.as_os_str().is_empty() {
            continue;
        }
        match fs::symlink_metadata(component) {
            Ok(metadata) => {
                if is_link_or_reparse_point(&metadata) {
                    bail!(
                        "OTLP spool path contains a symlink or reparse point: {}",
                        component.display()
                    );
                }
                if !metadata.is_dir() && component != path {
                    bail!(
                        "OTLP spool path contains a non-directory component: {}",
                        component.display()
                    );
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("cannot inspect OTLP spool path {}", component.display())
                })
            }
        }
    }
    Ok(())
}

#[cfg(windows)]
fn is_link_or_reparse_point(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
    metadata.file_type().is_symlink()
        || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn is_link_or_reparse_point(metadata: &fs::Metadata) -> bool {
    metadata.file_type().is_symlink()
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .context("cannot fsync OTLP spool directory")
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    // The file contents are flushed before the same-directory atomic rename.
    // Opening directories for FlushFileBuffers requires platform-specific
    // flags on Windows, which std does not expose.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;
    use flate2::{write::GzEncoder, Compression};
    use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
    use std::sync::Arc;
    use tracerazor_store::TraceStore;

    const SAMPLE: &str = include_str!("../../tracerazor-ingest/tests/fixtures/otel_protojson.json");
    const PROTO_TRACE_A: &str = "11111111111111111111111111111111";
    const PROTO_TRACE_B: &str = "22222222222222222222222222222222";

    fn protobuf_sample(mixed_trace_ids: bool) -> Vec<u8> {
        let mut sample: serde_json::Value = serde_json::from_str(SAMPLE).unwrap();
        let spans = sample["resourceSpans"][0]["scopeSpans"][0]["spans"]
            .as_array_mut()
            .unwrap();
        spans[0]["traceId"] = json!(PROTO_TRACE_A);
        spans[0]["spanId"] = json!("1111111111111111");
        spans[1]["traceId"] = json!(if mixed_trace_ids {
            PROTO_TRACE_B
        } else {
            PROTO_TRACE_A
        });
        spans[1]["spanId"] = json!("2222222222222222");
        let request: ExportTraceServiceRequest = serde_json::from_value(sample).unwrap();
        request.encode_to_vec()
    }

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("tracerazor-otlp-{label}-{}", Uuid::new_v4()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    async fn server(spool_dir: PathBuf, token: Option<&str>) -> (TestServer, Arc<TraceStore>) {
        let state = AppState::new_with_spool_dir(":mem:", spool_dir)
            .await
            .unwrap();
        let store = state.store.clone();
        let app = crate::build_app_with_token(state, token.map(str::to_string));
        (TestServer::new(app).unwrap(), store)
    }

    #[tokio::test]
    async fn successful_export_is_spooled_atomically_and_ingested() {
        let temp = TestDir::new("success");
        let spool = temp.path().join("spool");
        let (server, store) = server(spool.clone(), None).await;

        let response = server
            .post("/v1/traces")
            .text(SAMPLE)
            .content_type("application/json")
            .await;
        response.assert_status_ok();
        assert_eq!(response.json::<serde_json::Value>(), json!({}));

        let entries = fs::read_dir(&spool)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].extension().and_then(|value| value.to_str()),
            Some("json")
        );
        assert!(entries
            .iter()
            .all(|path| path.extension().and_then(|value| value.to_str()) != Some("tmp")));

        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(&entries[0]).unwrap()).unwrap();
        assert_eq!(receipt["schema_version"], "tracerazor-otlp-spool/v1");
        assert_eq!(receipt["traces"][0]["trace_id"], "abc1");
        assert_eq!(receipt["privacy"], "local-redacted");
        let otlp = &receipt["traces"][0]["metadata"]["otlp"];
        assert_eq!(otlp["enforcement_eligible"], true);
        assert_eq!(otlp["spans"][0]["span_id"], "s1");
        assert_eq!(otlp["spans"][0]["token_usage"]["input_tokens"], 310);
        assert_eq!(otlp["spans"][0]["token_usage"]["output_tokens"], 42);
        assert_eq!(otlp["spans"][0]["token_usage"]["total_tokens"], 352);
        assert_eq!(
            otlp["spans"][0]["token_usage"]["total_source"],
            "derived_from_reported_input_output"
        );
        let persisted = fs::read_to_string(&entries[0]).unwrap();
        for secret in ["Find the failing test", "test_refund_flow", "otel-agent"] {
            assert!(!persisted.contains(secret), "receipt leaked {secret}");
        }

        let stored = store.get_trace("abc1").await.unwrap().unwrap();
        assert_eq!(stored.trace.steps.len(), 2);
        assert_eq!(stored.trace.total_tokens, 542);
        assert!(stored.trace.steps[0]
            .content
            .starts_with("[redacted sha256="));
        assert!(stored.trace.agent_name.starts_with("[redacted-id sha256="));
    }

    #[tokio::test]
    async fn degraded_content_and_missing_usage_return_partial_success_and_persist_ledger() {
        let temp = TestDir::new("degraded-ledger");
        let spool = temp.path().join("spool");
        let (server, store) = server(spool.clone(), None).await;
        let payload = json!({
            "resourceSpans": [{"scopeSpans": [{"spans": [{
                "traceId": "trace-degraded",
                "spanId": "span-child",
                "parentSpanId": "span-parent",
                "name": "chat fallback",
                "attributes": [
                    {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                    {"key": "gen_ai.input.messages", "value": {"bytesValue": "c2VjcmV0"}}
                ]
            }]}]}]
        });

        let response = server.post("/v1/traces").json(&payload).await;
        response.assert_status_ok();
        let response_body: serde_json::Value = response.json();
        assert_eq!(response_body["partialSuccess"]["rejectedSpans"], "0");
        assert!(response_body["partialSuccess"]["errorMessage"]
            .as_str()
            .unwrap()
            .contains("hard enforcement is disabled"));

        let receipt_path = fs::read_dir(&spool)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let persisted = fs::read_to_string(receipt_path).unwrap();
        assert!(!persisted.contains("c2VjcmV0"));
        let receipt: serde_json::Value = serde_json::from_str(&persisted).unwrap();
        let otlp = &receipt["traces"][0]["metadata"]["otlp"];
        assert_eq!(otlp["degraded_ingest"], true);
        assert_eq!(otlp["enforcement_eligible"], false);
        assert_eq!(otlp["spans"][0]["span_id"], "span-child");
        assert_eq!(otlp["spans"][0]["parent_span_id"], "span-parent");
        assert!(otlp["spans"][0]["token_usage"]["total_tokens"].is_null());
        assert_eq!(
            otlp["spans"][0]["token_usage"]["provenance"]["total_tokens"],
            "missing"
        );

        let stored = store.get_trace("trace-degraded").await.unwrap().unwrap();
        assert_eq!(stored.trace.metadata["otlp"]["enforcement_eligible"], false);
    }

    #[tokio::test]
    async fn protobuf_missing_usage_returns_binary_partial_success() {
        let temp = TestDir::new("protobuf-partial");
        let spool = temp.path().join("spool");
        let (server, _) = server(spool, None).await;
        let mut payload: serde_json::Value = serde_json::from_str(SAMPLE).unwrap();
        let spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
            .as_array_mut()
            .unwrap();
        spans.truncate(1);
        spans[0]["traceId"] = json!(PROTO_TRACE_A);
        spans[0]["spanId"] = json!("1111111111111111");
        spans[0]["attributes"]
            .as_array_mut()
            .unwrap()
            .retain(|attribute| {
                !attribute["key"]
                    .as_str()
                    .is_some_and(|key| key.starts_with("gen_ai.usage."))
            });
        let request: ExportTraceServiceRequest = serde_json::from_value(payload).unwrap();

        let response = server
            .post("/v1/traces")
            .bytes(request.encode_to_vec().into())
            .content_type(PROTOBUF_CONTENT_TYPE)
            .await;
        response.assert_status_ok();
        let response = ExportTraceServiceResponse::decode(response.as_bytes().clone()).unwrap();
        let partial = response.partial_success.expect("partial success warning");
        assert_eq!(partial.rejected_spans, 0);
        assert!(partial.error_message.contains("missing_total_token_usage"));
    }

    #[tokio::test]
    async fn protobuf_export_matches_otlp_response_redaction_and_batch_durability() {
        let temp = TestDir::new("protobuf-success");
        let spool = temp.path().join("spool");
        let (server, store) = server(spool.clone(), None).await;

        let response = server
            .post("/v1/traces")
            .bytes(protobuf_sample(true).into())
            .content_type(PROTOBUF_CONTENT_TYPE)
            .await;
        response.assert_status_ok();
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            HeaderValue::from_static(PROTOBUF_CONTENT_TYPE)
        );
        let success = ExportTraceServiceResponse::decode(response.as_bytes().clone()).unwrap();
        assert!(success.partial_success.is_none());

        let entries = fs::read_dir(&spool)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 1);
        let persisted = fs::read_to_string(&entries[0]).unwrap();
        for secret in ["Find the failing test", "test_refund_flow", "otel-agent"] {
            assert!(!persisted.contains(secret), "receipt leaked {secret}");
        }
        let receipt: serde_json::Value = serde_json::from_str(&persisted).unwrap();
        assert_eq!(receipt["transport"], "otlp_http_protobuf");
        assert_eq!(receipt["privacy"], "local-redacted");
        let mut trace_ids = receipt["traces"]
            .as_array()
            .unwrap()
            .iter()
            .map(|trace| trace["trace_id"].as_str().unwrap())
            .collect::<Vec<_>>();
        trace_ids.sort_unstable();
        assert_eq!(trace_ids, [PROTO_TRACE_A, PROTO_TRACE_B]);

        let first = store.get_trace(PROTO_TRACE_A).await.unwrap().unwrap();
        let second = store.get_trace(PROTO_TRACE_B).await.unwrap().unwrap();
        assert_eq!(first.trace.total_tokens, 352);
        assert_eq!(second.trace.total_tokens, 190);
        assert!(first.trace.steps[0]
            .content
            .starts_with("[redacted sha256="));
    }

    #[test]
    fn local_redaction_removes_arbitrary_metadata_and_sensitive_step_fields() {
        let mut trace = tracerazor_ingest::otel::parse_many(SAMPLE)
            .unwrap()
            .remove(0);
        trace.metadata.insert(
            "secret-resource-key".to_string(),
            serde_json::Value::String("secret-resource-value".to_string()),
        );
        trace.steps[0].agent_id = Some("secret-agent-id".to_string());
        trace.steps[0].tool_params = Some(json!({"api_key": "sk-secret"}));
        trace.steps[0].tool_error = Some("secret-tool-error".to_string());

        let redacted = redact_trace(trace, "payload-hash");
        let serialized = serde_json::to_string(&redacted).unwrap();
        for secret in [
            "otel-agent",
            "secret-resource-key",
            "secret-resource-value",
            "secret-agent-id",
            "sk-secret",
            "secret-tool-error",
            "Find the failing test",
        ] {
            assert!(!serialized.contains(secret), "redaction leaked {secret}");
        }
        assert_eq!(redacted.metadata["privacy"], "local-redacted");
        assert!(redacted.metadata.contains_key("source_metadata_sha256"));
    }

    #[tokio::test]
    async fn mixed_trace_batch_spools_and_stores_every_trace_id() {
        let temp = TestDir::new("mixed");
        let spool = temp.path().join("spool");
        let (server, store) = server(spool.clone(), None).await;
        let mut batch: serde_json::Value = serde_json::from_str(SAMPLE).unwrap();
        batch["resourceSpans"][0]["scopeSpans"][0]["spans"][1]["traceId"] =
            serde_json::Value::String("def2".to_string());

        let response = server.post("/v1/traces").json(&batch).await;
        response.assert_status_ok();

        let receipt_path = fs::read_dir(&spool)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(receipt_path).unwrap()).unwrap();
        let mut trace_ids = receipt["traces"]
            .as_array()
            .unwrap()
            .iter()
            .map(|trace| trace["trace_id"].as_str().unwrap())
            .collect::<Vec<_>>();
        trace_ids.sort_unstable();
        assert_eq!(trace_ids, ["abc1", "def2"]);

        let first = store.get_trace("abc1").await.unwrap().unwrap();
        let second = store.get_trace("def2").await.unwrap().unwrap();
        assert_eq!(first.trace.steps.len(), 1);
        assert_eq!(second.trace.steps.len(), 1);
        assert_eq!(first.trace.total_tokens, 352);
        assert_eq!(second.trace.total_tokens, 190);
    }

    #[tokio::test]
    async fn gzip_export_is_supported_with_a_decompressed_size_limit() {
        let temp = TestDir::new("gzip");
        let spool = temp.path().join("spool");
        let (server, store) = server(spool.clone(), None).await;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(SAMPLE.as_bytes()).unwrap();
        let compressed = encoder.finish().unwrap();
        let accepted = server
            .post("/v1/traces")
            .bytes(compressed.into())
            .content_type("application/json")
            .add_header(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"))
            .await;
        accepted.assert_status_ok();
        assert!(store.get_trace("abc1").await.unwrap().is_some());

        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&vec![b' '; MAX_BODY_BYTES + 1]).unwrap();
        let compressed_bomb = encoder.finish().unwrap();
        let rejected = server
            .post("/v1/traces")
            .bytes(compressed_bomb.into())
            .content_type("application/json")
            .add_header(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"))
            .await;
        rejected.assert_status(StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            rejected.json::<serde_json::Value>()["code"],
            STATUS_RESOURCE_EXHAUSTED
        );
    }

    #[tokio::test]
    async fn protobuf_gzip_and_errors_keep_binary_contract_and_size_limit() {
        let temp = TestDir::new("protobuf-gzip-errors");
        let spool = temp.path().join("spool");
        let (server, store) = server(spool.clone(), None).await;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&protobuf_sample(false)).unwrap();
        let accepted = server
            .post("/v1/traces")
            .bytes(encoder.finish().unwrap().into())
            .content_type(PROTOBUF_CONTENT_TYPE)
            .add_header(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"))
            .await;
        accepted.assert_status_ok();
        assert_eq!(
            accepted.headers()[header::CONTENT_TYPE],
            HeaderValue::from_static(PROTOBUF_CONTENT_TYPE)
        );
        assert!(store.get_trace(PROTO_TRACE_A).await.unwrap().is_some());

        let malformed = server
            .post("/v1/traces")
            .bytes(vec![0xff].into())
            .content_type(PROTOBUF_CONTENT_TYPE)
            .await;
        malformed.assert_status(StatusCode::BAD_REQUEST);
        assert_eq!(
            malformed.headers()[header::CONTENT_TYPE],
            HeaderValue::from_static(PROTOBUF_CONTENT_TYPE)
        );
        let status = ProtobufStatus::decode(malformed.as_bytes().clone()).unwrap();
        assert_eq!(status.code, STATUS_INVALID_ARGUMENT);
        assert!(status.message.contains("protobuf"));

        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&vec![0u8; MAX_BODY_BYTES + 1]).unwrap();
        let oversized = server
            .post("/v1/traces")
            .bytes(encoder.finish().unwrap().into())
            .content_type(PROTOBUF_CONTENT_TYPE)
            .add_header(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"))
            .await;
        oversized.assert_status(StatusCode::PAYLOAD_TOO_LARGE);
        let status = ProtobufStatus::decode(oversized.as_bytes().clone()).unwrap();
        assert_eq!(status.code, STATUS_RESOURCE_EXHAUSTED);

        let oversized_wire = server
            .post("/v1/traces")
            .bytes(vec![0u8; MAX_BODY_BYTES + 1].into())
            .content_type(PROTOBUF_CONTENT_TYPE)
            .await;
        oversized_wire.assert_status(StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            oversized_wire.headers()[header::CONTENT_TYPE],
            HeaderValue::from_static(PROTOBUF_CONTENT_TYPE)
        );
        let status = ProtobufStatus::decode(oversized_wire.as_bytes().clone()).unwrap();
        assert_eq!(status.code, STATUS_RESOURCE_EXHAUSTED);

        // Only the first, valid batch reached the durable spool.
        assert_eq!(fs::read_dir(&spool).unwrap().count(), 1);
    }

    #[tokio::test]
    async fn missing_and_wrong_bearer_tokens_are_otlp_errors() {
        let temp = TestDir::new("auth");
        let spool = temp.path().join("spool");
        let (server, _) = server(spool.clone(), Some("correct-token")).await;

        for authorization in [None, Some("Bearer wrong-token")] {
            let mut request = server
                .post("/v1/traces")
                .text(SAMPLE)
                .content_type("application/json");
            if let Some(value) = authorization {
                request = request
                    .add_header(header::AUTHORIZATION, HeaderValue::from_str(value).unwrap());
            }
            let response = request.await;
            response.assert_status(StatusCode::UNAUTHORIZED);
            let body: serde_json::Value = response.json();
            assert_eq!(body["code"], STATUS_UNAUTHENTICATED);
        }
        assert!(!spool.exists());

        let accepted = server
            .post("/v1/traces")
            .text(SAMPLE)
            .content_type("application/json")
            .add_header(
                header::AUTHORIZATION,
                HeaderValue::from_static("Bearer correct-token"),
            )
            .await;
        accepted.assert_status_ok();
        assert_eq!(fs::read_dir(&spool).unwrap().count(), 1);
    }

    #[tokio::test]
    async fn protobuf_auth_failure_is_binary_and_never_spooled() {
        let temp = TestDir::new("protobuf-auth");
        let spool = temp.path().join("spool");
        let (server, _) = server(spool.clone(), Some("correct-token")).await;

        let response = server
            .post("/v1/traces")
            .bytes(protobuf_sample(false).into())
            .content_type(PROTOBUF_CONTENT_TYPE)
            .add_header(
                header::AUTHORIZATION,
                HeaderValue::from_static("Bearer wrong-token"),
            )
            .await;
        response.assert_status(StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            HeaderValue::from_static(PROTOBUF_CONTENT_TYPE)
        );
        assert_eq!(
            response.headers()[header::WWW_AUTHENTICATE],
            HeaderValue::from_static("Bearer")
        );
        let status = ProtobufStatus::decode(response.as_bytes().clone()).unwrap();
        assert_eq!(status.code, STATUS_UNAUTHENTICATED);
        assert!(!spool.exists());
    }

    #[tokio::test]
    async fn oversized_and_malformed_payloads_are_rejected_without_spooling() {
        let temp = TestDir::new("invalid");
        let spool = temp.path().join("spool");
        let (server, _) = server(spool.clone(), None).await;

        let oversized = server
            .post("/v1/traces")
            .bytes(vec![b' '; MAX_BODY_BYTES + 1].into())
            .content_type("application/json")
            .await;
        oversized.assert_status(StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            oversized.json::<serde_json::Value>()["code"],
            STATUS_RESOURCE_EXHAUSTED
        );

        let malformed = server
            .post("/v1/traces")
            .text("{")
            .content_type("application/json")
            .await;
        malformed.assert_status(StatusCode::BAD_REQUEST);
        assert_eq!(
            malformed.json::<serde_json::Value>()["code"],
            STATUS_INVALID_ARGUMENT
        );
        assert!(!spool.exists());
    }

    #[tokio::test]
    async fn empty_otlp_export_is_a_successful_noop() {
        let temp = TestDir::new("empty");
        let spool = temp.path().join("spool");
        let (server, _) = server(spool.clone(), None).await;

        for payload in [
            "{}",
            r#"{"resourceSpans": []}"#,
            r#"{"resourceSpans": [{"scopeSpans": [{}]}]}"#,
        ] {
            let response = server
                .post("/v1/traces")
                .text(payload)
                .content_type("application/json")
                .await;
            response.assert_status_ok();
            assert_eq!(response.json::<serde_json::Value>(), json!({}));
        }
        let protobuf = server
            .post("/v1/traces")
            .bytes(Vec::new().into())
            .content_type(PROTOBUF_CONTENT_TYPE)
            .await;
        protobuf.assert_status_ok();
        assert_eq!(
            protobuf.headers()[header::CONTENT_TYPE],
            HeaderValue::from_static(PROTOBUF_CONTENT_TYPE)
        );
        let success = ExportTraceServiceResponse::decode(protobuf.as_bytes().clone()).unwrap();
        assert!(success.partial_success.is_none());
        assert!(!spool.exists());
    }

    #[tokio::test]
    async fn spool_path_traversal_is_rejected() {
        let temp = TestDir::new("path-safety");
        let unsafe_spool = temp.path().join("spool").join("..").join("escape");
        let (server, store) = server(unsafe_spool, None).await;

        let response = server
            .post("/v1/traces")
            .text(SAMPLE)
            .content_type("application/json")
            .await;
        response.assert_status(StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.json::<serde_json::Value>()["code"],
            STATUS_UNAVAILABLE
        );
        assert!(!temp.path().join("escape").exists());
        assert!(store.get_trace("abc1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn trace_id_never_controls_the_spool_path() {
        let temp = TestDir::new("trace-id-path");
        let spool = temp.path().join("spool");
        let (server, store) = server(spool.clone(), None).await;
        let hostile = SAMPLE.replace("\"abc1\"", "\"../../escape\"");

        let response = server
            .post("/v1/traces")
            .text(hostile)
            .content_type("application/json")
            .await;
        response.assert_status_ok();

        let entries = fs::read_dir(&spool)
            .unwrap()
            .map(|entry| entry.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 1);
        assert!(entries[0]
            .file_name()
            .to_string_lossy()
            .starts_with("otlp-"));
        assert!(!temp.path().join("escape").exists());
        assert!(store.get_trace("../../escape").await.unwrap().is_some());
    }
}
