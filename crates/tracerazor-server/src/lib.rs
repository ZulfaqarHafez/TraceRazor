//! TraceRazor HTTP server library.
//!
//! Exposes the Axum application builder and a [`run_server`] entry point so
//! the server can be started either via the `tracerazor-server` binary or the
//! `tracerazor serve` CLI alias.
//!
//! ## Authentication
//!
//! Setting `TRACERAZOR_API_TOKEN` requires `Authorization: Bearer <token>` on
//! every `/api/*` route and on `/ws`. Without the env var the API is open —
//! suitable only for loopback/dev use, and [`run_server`] warns loudly when
//! binding a non-loopback address unauthenticated.

pub mod api;
pub mod state;
pub mod ws;

use anyhow::Result;
use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::{header, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::services::ServeDir;

use state::AppState;

/// Maximum accepted request body size (16 MiB). Prevents memory-exhaustion DoS
/// via an unbounded `POST /api/audit` payload while leaving ample room for
/// legitimately large traces.
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Lightweight Alpine.js + Chart.js dashboard — embedded in the binary,
/// no build step required. Served at `/`.
static DASHBOARD_HTML: &str = include_str!("dashboard.html");

async fn dashboard_handler() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        DASHBOARD_HTML,
    )
}

/// Parse TRACERAZOR_CORS_ORIGINS into an AllowOrigin policy.
/// Accepts a comma-separated list of origins (e.g. "https://app.example.com,http://localhost:3000").
/// Falls back to permissive `Any` when unset or empty — suitable for local dev.
fn cors_origins() -> AllowOrigin {
    match std::env::var("TRACERAZOR_CORS_ORIGINS") {
        Ok(val) if !val.is_empty() && val != "*" => {
            let mut origins = Vec::new();
            for raw in val.split(',') {
                let raw = raw.trim();
                if raw.is_empty() {
                    continue;
                }
                // Don't panic on a malformed value — skip it and warn so a typo
                // in the env var can't take the whole server down at startup.
                match raw.parse() {
                    Ok(origin) => origins.push(origin),
                    Err(e) => eprintln!(
                        "warning: ignoring invalid origin '{raw}' in TRACERAZOR_CORS_ORIGINS: {e}"
                    ),
                }
            }
            if origins.is_empty() {
                AllowOrigin::any()
            } else {
                AllowOrigin::list(origins)
            }
        }
        _ => AllowOrigin::any(),
    }
}

/// Liveness probe (`GET /healthz`). Reports that the process is up and the
/// async runtime is responsive. Deliberately does NOT touch the database — a
/// transient DB issue should not make an orchestrator kill an otherwise-healthy
/// process (that's what readiness is for).
async fn healthz() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({ "status": "ok", "service": "tracerazor", "version": env!("CARGO_PKG_VERSION") })),
    )
}

/// Readiness probe (`GET /readyz`). Reports whether the service can actually
/// serve requests — i.e. the SQLite store is reachable. Returns 503 when not
/// ready so load balancers / orchestrators stop routing traffic.
async fn readyz(State(state): State<AppState>) -> impl IntoResponse {
    match state.store.health_check().await {
        Ok(()) => (StatusCode::OK, Json(json!({ "status": "ready" }))),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "unavailable", "error": e.to_string() })),
        ),
    }
}

/// Expected bearer token, threaded into the auth middleware as its own state.
#[derive(Clone)]
struct ApiToken(Arc<str>);

/// Constant-time byte comparison — a naive `==` short-circuits on the first
/// differing byte and leaks token prefixes through response timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Bearer-token gate applied to `/api/*` and `/ws` when a token is configured.
/// Health probes and the static dashboard page stay open (the dashboard's API
/// calls are themselves gated).
async fn require_bearer(State(expected): State<ApiToken>, req: Request, next: Next) -> Response {
    let provided = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match provided {
        Some(tok) if constant_time_eq(tok.as_bytes(), expected.0.as_bytes()) => {
            next.run(req).await
        }
        _ => (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            Json(json!({
                "error": "unauthorized: this server requires `Authorization: Bearer <token>` (TRACERAZOR_API_TOKEN)"
            })),
        )
            .into_response(),
    }
}

/// Build the Axum application router, reading the bearer token from the
/// `TRACERAZOR_API_TOKEN` environment variable.
pub fn build_app(state: AppState) -> Router {
    let token = std::env::var("TRACERAZOR_API_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());
    build_app_with_token(state, token)
}

/// Build the Axum application router with an explicit bearer token
/// (`None` = unauthenticated). Extracted for testability.
pub fn build_app_with_token(state: AppState, api_token: Option<String>) -> Router {
    // Restrict to the methods/headers the API actually uses rather than `Any`,
    // shrinking the cross-origin attack surface.
    let cors = CorsLayer::new()
        .allow_origin(cors_origins())
        .allow_methods([Method::GET, Method::POST, Method::DELETE])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    // Everything that exposes data or accepts writes sits behind the token.
    let mut protected = Router::new()
        .nest("/api", api::router())
        .route("/ws", axum::routing::get(ws::handler));
    if let Some(token) = api_token {
        protected = protected.layer(axum::middleware::from_fn_with_state(
            ApiToken(token.into()),
            require_bearer,
        ));
    }

    Router::new()
        .merge(protected)
        // Kubernetes-style health probes (also used by the container HEALTHCHECK).
        // Always open: orchestrators don't carry credentials.
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        // Lightweight dashboard embedded in binary (always available).
        .route("/", axum::routing::get(dashboard_handler))
        // React build served at /app (optional — run `npm run build` in dashboard/).
        .nest_service("/app", ServeDir::new("dashboard/dist"))
        .layer(cors)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

/// Self-contained health probe used by the container `HEALTHCHECK` so the
/// runtime image needs no `curl`/`wget`. Hits the local liveness endpoint and
/// returns an error (non-zero exit) if it's not healthy.
pub async fn run_health_probe(port: u16) -> Result<()> {
    let url = format!("http://127.0.0.1:{port}/healthz");
    let resp = reqwest::Client::new()
        .get(&url)
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("health probe request to {url} failed: {e}"))?;
    if resp.status().is_success() {
        Ok(())
    } else {
        Err(anyhow::anyhow!("health probe got HTTP {}", resp.status()))
    }
}

/// Options for [`run_server`].
pub struct ServeOptions {
    /// Port to listen on.
    pub port: u16,
    /// Bind address. Loopback by default so the server is never unintentionally
    /// exposed on all interfaces.
    pub bind: String,
    /// SQLite database path (`:mem:` for in-memory).
    pub db_path: String,
}

impl Default for ServeOptions {
    fn default() -> Self {
        ServeOptions {
            port: 8080,
            bind: "127.0.0.1".to_string(),
            db_path: "./tracerazor.db".to_string(),
        }
    }
}

fn is_loopback(bind: &str) -> bool {
    matches!(bind, "127.0.0.1" | "localhost" | "::1" | "[::1]")
}

/// Start the server and block until it exits. Shared by the
/// `tracerazor-server` binary and the `tracerazor serve` CLI alias.
pub async fn run_server(opts: ServeOptions) -> Result<()> {
    let state = AppState::new(&opts.db_path).await?;

    let token = std::env::var("TRACERAZOR_API_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());
    let authed = token.is_some();
    if !authed && !is_loopback(&opts.bind) {
        eprintln!(
            "WARNING: binding {} without TRACERAZOR_API_TOKEN — the API is \
             unauthenticated. Set TRACERAZOR_API_TOKEN to require \
             `Authorization: Bearer <token>` on /api routes.",
            opts.bind
        );
    }
    let app = build_app_with_token(state, token);

    let addr: SocketAddr = format!("{}:{}", opts.bind, opts.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid bind address '{}:{}': {e}", opts.bind, opts.port))?;
    println!("TraceRazor server listening on http://{}", addr);
    println!(
        "API auth:           {}",
        if authed {
            "Bearer token required (TRACERAZOR_API_TOKEN)"
        } else {
            "none (set TRACERAZOR_API_TOKEN to enable)"
        }
    );
    println!("Dashboard (Alpine): http://localhost:{}/", opts.port);
    println!(
        "Dashboard (React):  http://localhost:{}/app  (requires: cd dashboard && npm run build)",
        opts.port
    );
    println!("Metrics:            http://localhost:{}/api/metrics", opts.port);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod auth_tests {
    use super::*;
    use axum_test::TestServer;

    async fn server_with_token(token: Option<&str>) -> TestServer {
        let state = AppState::new(":mem:").await.unwrap();
        let app = build_app_with_token(state, token.map(String::from));
        TestServer::new(app).unwrap()
    }

    #[tokio::test]
    async fn unauthenticated_request_gets_401_when_token_set() {
        let server = server_with_token(Some("s3cret-token")).await;
        let resp = server.get("/api/traces").await;
        resp.assert_status(StatusCode::UNAUTHORIZED);
        let body: serde_json::Value = resp.json();
        assert!(body["error"].as_str().unwrap().contains("unauthorized"));
    }

    #[tokio::test]
    async fn wrong_token_gets_401() {
        let server = server_with_token(Some("s3cret-token")).await;
        let resp = server
            .get("/api/traces")
            .add_header(
                header::AUTHORIZATION,
                axum::http::HeaderValue::from_static("Bearer wrong-token"),
            )
            .await;
        resp.assert_status(StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn correct_token_passes() {
        let server = server_with_token(Some("s3cret-token")).await;
        let resp = server
            .get("/api/traces")
            .add_header(
                header::AUTHORIZATION,
                axum::http::HeaderValue::from_static("Bearer s3cret-token"),
            )
            .await;
        resp.assert_status_ok();
    }

    #[tokio::test]
    async fn audit_post_requires_token() {
        let server = server_with_token(Some("s3cret-token")).await;
        let resp = server
            .post("/api/audit")
            .json(&serde_json::json!({"trace": {}}))
            .await;
        resp.assert_status(StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn health_probes_stay_open_with_token_set() {
        let server = server_with_token(Some("s3cret-token")).await;
        server.get("/healthz").await.assert_status_ok();
        server.get("/readyz").await.assert_status_ok();
    }

    #[tokio::test]
    async fn no_token_configured_means_open_api() {
        let server = server_with_token(None).await;
        server.get("/api/traces").await.assert_status_ok();
    }

    #[test]
    fn constant_time_eq_basics() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(constant_time_eq(b"", b""));
    }
}
