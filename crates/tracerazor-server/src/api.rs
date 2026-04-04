/// REST API routes for TraceRazor server.
///
/// All routes are mounted at `/api/`:
///   POST   /api/audit         — ingest + analyse a trace JSON
///   GET    /api/traces        — list all stored traces
///   GET    /api/traces/:id    — get full stored trace + report
///   DELETE /api/traces/:id    — delete a trace
///   GET    /api/dashboard     — aggregate dashboard data
///   GET    /api/agents        — per-agent statistics
///   GET    /api/agents/:name  — stats for a single agent
use axum::{
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracerazor_core::{analyse, scoring::ScoringConfig};
use tracerazor_ingest::{parse, TraceFormat};
use tracerazor_semantic::{default_similarity_fn, BowSimilarity, Similarity};
use tracerazor_store::{KGP_CAPTURE_THRESHOLD, build_kb_entry};

use crate::state::{AppState, WsEvent};

/// Returns the API sub-router. State is injected by the caller via `with_state`.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(index))
        .route("/audit", post(audit))
        .route("/traces", get(list_traces))
        .route("/traces/:id", get(get_trace).delete(delete_trace))
        .route("/dashboard", get(dashboard))
        .route("/agents", get(list_agents))
        .route("/agents/:name", get(get_agent))
        .route("/compare", get(compare))
        .route("/metrics", get(metrics))
        // Known-Good-Paths knowledge base
        .route("/kb", get(list_kb))
        .route("/kb/:id", get(get_kb_entry).delete(delete_kb_entry))
}

async fn index() -> impl IntoResponse {
    Json(json!({
        "service": "TraceRazor API",
        "version": env!("CARGO_PKG_VERSION"),
        "endpoints": [
            "POST /api/audit",
            "GET  /api/traces",
            "GET  /api/traces/:id",
            "DELETE /api/traces/:id",
            "GET  /api/dashboard",
            "GET  /api/agents",
            "GET  /api/agents/:name"
        ]
    }))
}

#[derive(Deserialize)]
pub struct AuditRequest {
    /// Raw trace JSON (same schema as the CLI).
    pub trace: serde_json::Value,
}

#[derive(Serialize)]
pub struct AuditResponse {
    pub trace_id: String,
    pub agent_name: String,
    pub tas_score: f64,
    pub grade: String,
    pub tokens_saved: u32,
    pub report_markdown: String,
    /// Whether this trace was auto-captured into the KB (TAS ≥ threshold).
    pub captured_to_kb: bool,
    /// Closest matching KB entry for this agent (if similarity ≥ 0.45).
    pub kb_match: Option<tracerazor_store::KgpMatch>,
}

/// POST /api/audit
async fn audit(
    State(state): State<AppState>,
    Json(req): Json<AuditRequest>,
) -> Result<impl IntoResponse, AppError> {
    let trace_str = serde_json::to_string(&req.trace)
        .map_err(|e| AppError::bad_request(format!("Invalid trace JSON: {e}")))?;

    let mut trace = parse(&trace_str, TraceFormat::Auto)
        .map_err(|e| AppError::bad_request(format!("Ingest error: {e}")))?;

    let sim_fn = default_similarity_fn();
    let config = ScoringConfig::default();
    let report = analyse(&mut trace, sim_fn, &config)
        .map_err(AppError::internal)?;

    state
        .store
        .save_trace(&trace, Some(&report))
        .await
        .map_err(AppError::internal)?;

    let tokens_saved = report.savings.tokens_saved;
    let tas_score = report.score.score;
    let grade = report.score.grade.to_string();

    // ── KB: find similar prior runs before potentially adding this one ────────
    let kb_match = find_kb_match(&state, &trace, &report).await;

    // ── KB: auto-capture if this trace scores above the threshold ─────────────
    let captured_to_kb = if tas_score >= KGP_CAPTURE_THRESHOLD {
        let entry = build_kb_entry(&trace, &report);
        state.store.save_kb_entry(&entry).await.map_err(AppError::internal)?;
        true
    } else {
        false
    };

    let _ = state.events.send(WsEvent::TraceAnalysed {
        trace_id: trace.trace_id.clone(),
        agent_name: trace.agent_name.clone(),
        tas_score,
        grade: grade.clone(),
        tokens_saved,
    });

    Ok((
        StatusCode::OK,
        Json(AuditResponse {
            trace_id: trace.trace_id.clone(),
            agent_name: trace.agent_name.clone(),
            tas_score,
            grade,
            tokens_saved,
            report_markdown: report.to_markdown(),
            captured_to_kb,
            kb_match,
        }),
    ))
}

/// Find the best matching KB entry for the incoming trace using BoW similarity.
async fn find_kb_match(
    state: &AppState,
    trace: &tracerazor_core::types::Trace,
    report: &tracerazor_core::report::TraceReport,
) -> Option<tracerazor_store::KgpMatch> {
    const MATCH_THRESHOLD: f64 = 0.45;

    let kb_entries = state.store.list_kb_for_agent(&trace.agent_name).await.ok()?;
    if kb_entries.is_empty() {
        return None;
    }

    // Use the same task_hint derivation as build_kb_entry.
    let incoming_hint = trace
        .steps
        .iter()
        .find(|s| matches!(s.step_type, tracerazor_core::types::StepType::Reasoning))
        .map(|s| s.content.chars().take(300).collect::<String>())
        .unwrap_or_else(|| trace.agent_name.clone());

    let bow = BowSimilarity::new();
    let best = kb_entries
        .into_iter()
        // Don't match against the trace itself if it was just captured.
        .filter(|e| e.source_trace_id != trace.trace_id)
        .map(|e| {
            let sim = bow.similarity(&incoming_hint, &e.task_hint);
            (e, sim)
        })
        .filter(|(_, sim)| *sim >= MATCH_THRESHOLD)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

    best.map(|(entry, similarity)| tracerazor_store::KgpMatch { entry, similarity })
}

/// GET /api/traces
async fn list_traces(State(state): State<AppState>) -> Result<impl IntoResponse, AppError> {
    let traces = state.store.list_traces().await.map_err(AppError::internal)?;
    Ok(Json(traces))
}

/// GET /api/traces/:id
async fn get_trace(
    Path(id): Path<String>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    match state.store.get_trace(&id).await.map_err(AppError::internal)? {
        Some(stored) => Ok(Json(stored)),
        None => Err(AppError::not_found(format!("Trace '{id}' not found"))),
    }
}

/// DELETE /api/traces/:id
async fn delete_trace(
    Path(id): Path<String>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    state
        .store
        .delete_trace(&id)
        .await
        .map_err(AppError::internal)?;
    Ok(StatusCode::NO_CONTENT)
}

/// GET /api/dashboard
async fn dashboard(State(state): State<AppState>) -> Result<impl IntoResponse, AppError> {
    let data = state
        .store
        .dashboard_data()
        .await
        .map_err(AppError::internal)?;
    Ok(Json(data))
}

/// GET /api/agents
async fn list_agents(State(state): State<AppState>) -> Result<impl IntoResponse, AppError> {
    let stats = state
        .store
        .all_agent_stats()
        .await
        .map_err(AppError::internal)?;
    Ok(Json(stats))
}

/// GET /api/agents/:name
async fn get_agent(
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    match state
        .store
        .agent_stats(&name)
        .await
        .map_err(AppError::internal)?
    {
        Some(stats) => Ok(Json(stats)),
        None => Err(AppError::not_found(format!("Agent '{name}' not found"))),
    }
}

// ── Error helper ──────────────────────────────────────────────────────────────

struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(msg: impl Into<String>) -> Self {
        AppError { status: StatusCode::BAD_REQUEST, message: msg.into() }
    }
    fn not_found(msg: impl Into<String>) -> Self {
        AppError { status: StatusCode::NOT_FOUND, message: msg.into() }
    }
    fn internal(e: impl std::fmt::Display) -> Self {
        AppError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: e.to_string(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

// ── Known-Good-Paths KB ───────────────────────────────────────────────────────

/// GET /api/kb
async fn list_kb(State(state): State<AppState>) -> Result<impl IntoResponse, AppError> {
    let entries = state.store.list_kb_entries().await.map_err(AppError::internal)?;
    Ok(Json(entries))
}

/// GET /api/kb/:id
async fn get_kb_entry(
    Path(id): Path<String>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    match state.store.get_kb_entry(&id).await.map_err(AppError::internal)? {
        Some(e) => Ok(Json(e)),
        None => Err(AppError::not_found(format!("KB entry '{id}' not found"))),
    }
}

/// DELETE /api/kb/:id
async fn delete_kb_entry(
    Path(id): Path<String>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    state.store.delete_kb_entry(&id).await.map_err(AppError::internal)?;
    Ok(StatusCode::NO_CONTENT)
}

// ── Compare ───────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct CompareParams {
    a: String,
    b: String,
}

#[derive(Serialize)]
pub struct CompareResponse {
    pub a: tracerazor_store::TraceSummary,
    pub b: tracerazor_store::TraceSummary,
    /// b.tas_score − a.tas_score (positive = b improved)
    pub tas_diff: f64,
    /// b.tokens_saved − a.tokens_saved
    pub tokens_saved_diff: i64,
    pub verdict: String,
}

/// GET /api/compare?a=trace-id-1&b=trace-id-2
async fn compare(
    Query(params): Query<CompareParams>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    let sa = state
        .store
        .get_trace(&params.a)
        .await
        .map_err(AppError::internal)?
        .ok_or_else(|| AppError::not_found(format!("Trace '{}' not found", params.a)))?;

    let sb = state
        .store
        .get_trace(&params.b)
        .await
        .map_err(AppError::internal)?
        .ok_or_else(|| AppError::not_found(format!("Trace '{}' not found", params.b)))?;

    // Build summaries inline (mirrors store::to_summary logic).
    let sum_a = tracerazor_store::TraceSummary {
        trace_id: sa.trace.trace_id.clone(),
        agent_name: sa.trace.agent_name.clone(),
        framework: sa.trace.framework.clone(),
        total_steps: sa.trace.steps.len(),
        total_tokens: sa.trace.effective_total_tokens(),
        tas_score: sa.report.as_ref().map(|r| r.score.score),
        grade: sa.report.as_ref().map(|r| r.score.grade.to_string()),
        stored_at: sa.stored_at.clone(),
        tokens_saved: sa.report.as_ref().map(|r| r.savings.tokens_saved),
    };

    let sum_b = tracerazor_store::TraceSummary {
        trace_id: sb.trace.trace_id.clone(),
        agent_name: sb.trace.agent_name.clone(),
        framework: sb.trace.framework.clone(),
        total_steps: sb.trace.steps.len(),
        total_tokens: sb.trace.effective_total_tokens(),
        tas_score: sb.report.as_ref().map(|r| r.score.score),
        grade: sb.report.as_ref().map(|r| r.score.grade.to_string()),
        stored_at: sb.stored_at.clone(),
        tokens_saved: sb.report.as_ref().map(|r| r.savings.tokens_saved),
    };

    let tas_a = sum_a.tas_score.unwrap_or(0.0);
    let tas_b = sum_b.tas_score.unwrap_or(0.0);
    let tas_diff = ((tas_b - tas_a) * 10.0).round() / 10.0;

    let saved_a = sum_a.tokens_saved.unwrap_or(0) as i64;
    let saved_b = sum_b.tokens_saved.unwrap_or(0) as i64;
    let tokens_saved_diff = saved_b - saved_a;

    let verdict = match tas_diff {
        d if d > 5.0 => format!("B improved by {:.1} TAS points", d),
        d if d < -5.0 => format!("B regressed by {:.1} TAS points", d.abs()),
        d => format!("No significant change ({:+.1} TAS points)", d),
    };

    Ok(Json(CompareResponse { a: sum_a, b: sum_b, tas_diff, tokens_saved_diff, verdict }))
}

// ── Prometheus Metrics ────────────────────────────────────────────────────────

/// GET /api/metrics  — Prometheus text exposition format (no external crate).
async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    let data = match state.store.dashboard_data().await {
        Ok(d) => d,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(header::CONTENT_TYPE, "text/plain")],
                format!("# error fetching metrics: {e}\n"),
            );
        }
    };

    let out = format!(
        "# HELP tracerazor_traces_total Total number of traces stored\n\
         # TYPE tracerazor_traces_total gauge\n\
         tracerazor_traces_total {traces}\n\
         # HELP tracerazor_agents_total Number of distinct agents seen\n\
         # TYPE tracerazor_agents_total gauge\n\
         tracerazor_agents_total {agents}\n\
         # HELP tracerazor_avg_tas_score Average TAS score across all traces\n\
         # TYPE tracerazor_avg_tas_score gauge\n\
         tracerazor_avg_tas_score {avg_tas}\n\
         # HELP tracerazor_tokens_saved_total Cumulative tokens saved across all traces\n\
         # TYPE tracerazor_tokens_saved_total counter\n\
         tracerazor_tokens_saved_total {tokens_saved}\n\
         # HELP tracerazor_cost_saved_usd_total Cumulative USD saved (rough estimate at $3/M tokens)\n\
         # TYPE tracerazor_cost_saved_usd_total counter\n\
         tracerazor_cost_saved_usd_total {cost_saved}\n",
        traces = data.total_traces,
        agents = data.total_agents,
        avg_tas = data.avg_tas,
        tokens_saved = data.total_tokens_saved,
        cost_saved = data.total_cost_saved_usd,
    );

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        out,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;
    use serde_json::json;

    async fn test_app() -> TestServer {
        let state = crate::state::AppState::new(":mem:").await.unwrap();
        let app = crate::build_app(state);
        TestServer::new(app).unwrap()
    }

    #[tokio::test]
    async fn test_index() {
        let server = test_app().await;
        let resp = server.get("/api").await;
        resp.assert_status_ok();
    }

    #[tokio::test]
    async fn test_audit_and_list() {
        let server = test_app().await;

        let trace = json!({
            "trace_id": "api-test-001",
            "agent_name": "test-agent",
            "framework": "raw",
            "total_tokens": 2500,
            "task_value_score": 1.0,
            "steps": [
                {"id": 1, "step_type": "reasoning", "content": "Analyse the user request about order refund", "tokens": 500},
                {"id": 2, "step_type": "tool_call", "content": "Fetch order details", "tokens": 400,
                 "tool_name": "get_order", "tool_success": true},
                {"id": 3, "step_type": "reasoning", "content": "Analyse the user request about order refund again", "tokens": 500},
                {"id": 4, "step_type": "tool_call", "content": "Check eligibility", "tokens": 400,
                 "tool_name": "check_eligibility", "tool_success": false,
                 "tool_error": "missing param"},
                {"id": 5, "step_type": "tool_call", "content": "Check eligibility retry", "tokens": 400,
                 "tool_name": "check_eligibility", "tool_success": true},
                {"id": 6, "step_type": "tool_call", "content": "Process refund", "tokens": 300,
                 "tool_name": "process_refund", "tool_success": true}
            ]
        });

        let resp = server.post("/api/audit").json(&json!({"trace": trace})).await;
        resp.assert_status_ok();
        let body: serde_json::Value = resp.json();
        assert_eq!(body["trace_id"], "api-test-001");
        assert!(body["tas_score"].as_f64().unwrap() >= 0.0);

        let list = server.get("/api/traces").await;
        list.assert_status_ok();
        let items: Vec<serde_json::Value> = list.json();
        assert_eq!(items.len(), 1);
    }

    #[tokio::test]
    async fn test_dashboard_empty() {
        let server = test_app().await;
        let resp = server.get("/api/dashboard").await;
        resp.assert_status_ok();
        let body: serde_json::Value = resp.json();
        assert_eq!(body["total_traces"], 0);
    }

    #[tokio::test]
    async fn test_get_trace_not_found() {
        let server = test_app().await;
        let resp = server.get("/api/traces/nonexistent").await;
        resp.assert_status(StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_metrics() {
        let server = test_app().await;
        let resp = server.get("/api/metrics").await;
        resp.assert_status_ok();
        let body = resp.text();
        assert!(body.contains("tracerazor_traces_total"));
        assert!(body.contains("tracerazor_avg_tas_score"));
    }

    #[tokio::test]
    async fn test_compare_not_found() {
        let server = test_app().await;
        let resp = server.get("/api/compare?a=x&b=y").await;
        resp.assert_status(StatusCode::NOT_FOUND);
    }
}
