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
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracerazor_core::{analyse, scoring::ScoringConfig};
use tracerazor_ingest::{parse, TraceFormat};
use tracerazor_semantic::default_similarity_fn;

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

    let _ = state.events.send(WsEvent::TraceAnalysed {
        trace_id: trace.trace_id.clone(),
        agent_name: trace.agent_name.clone(),
        tas_score: report.score.score,
        grade: report.score.grade.to_string(),
        tokens_saved,
    });

    Ok((
        StatusCode::OK,
        Json(AuditResponse {
            trace_id: trace.trace_id.clone(),
            agent_name: trace.agent_name.clone(),
            tas_score: report.score.score,
            grade: report.score.grade.to_string(),
            tokens_saved,
            report_markdown: report.to_markdown(),
        }),
    ))
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
}
