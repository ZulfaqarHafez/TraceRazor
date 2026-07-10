//! TraceRazor HTTP server library.
//!
//! Exposes the Axum application builder and a [`run_server`] entry point so
//! the server can be started either via the `tracerazor-server` binary or the
//! `tracerazor serve` CLI alias.
//!
//! ## Authentication
//!
//! Setting `TRACERAZOR_API_TOKEN` requires `Authorization: Bearer <token>` on
//! every `/api/*` route, the OTLP `/v1/traces` receiver, and `/ws`. Without the
//! env var these surfaces are open — suitable only for loopback/dev use.
//! [`run_server`] refuses every non-loopback bind unless bearer auth is enabled
//! and `TRACERAZOR_TLS_TERMINATED=true` explicitly asserts a trusted reverse
//! proxy TLS boundary. TraceRazor does not terminate TLS itself.

pub mod api;
mod otlp;
pub mod state;
pub mod ws;

use anyhow::Result;
use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::{header, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde_json::json;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::services::ServeDir;

use state::AppState;

/// Maximum accepted request body size (16 MiB). Prevents memory-exhaustion DoS
/// via unbounded audit or OTLP ingest payloads while leaving ample room for
/// legitimately large traces.
pub(crate) const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Lightweight Alpine.js + Chart.js dashboard — embedded in the binary,
/// no build step required. Served at `/`.
static DASHBOARD_HTML: &str = include_str!("dashboard.html");

async fn dashboard_handler() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (
                header::CONTENT_SECURITY_POLICY,
                "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            ),
        ],
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
                default_cors_origins()
            } else {
                AllowOrigin::list(origins)
            }
        }
        Ok(val) if val == "*" => {
            eprintln!(
                "warning: TRACERAZOR_CORS_ORIGINS='*' allows any browser origin; use an explicit origin list for shared deployments"
            );
            AllowOrigin::any()
        }
        _ => default_cors_origins(),
    }
}

fn default_cors_origins() -> AllowOrigin {
    AllowOrigin::predicate(|origin, _| is_loopback_origin(origin))
}

fn is_loopback_origin(origin: &HeaderValue) -> bool {
    let Ok(origin) = origin.to_str() else {
        return false;
    };
    let Ok(url) = reqwest::Url::parse(origin) else {
        return false;
    };
    matches!(url.scheme(), "http" | "https")
        && url.host_str().is_some_and(|host| {
            host.eq_ignore_ascii_case("localhost")
                || host
                    .trim_matches(['[', ']'])
                    .parse::<IpAddr>()
                    .is_ok_and(|ip| ip.is_loopback())
        })
}

/// Liveness probe (`GET /healthz`). Reports that the process is up and the
/// async runtime is responsive. Deliberately does NOT touch the database — a
/// transient DB issue should not make an orchestrator kill an otherwise-healthy
/// process (that's what readiness is for).
async fn healthz() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(
            json!({ "status": "ok", "service": "tracerazor", "version": env!("CARGO_PKG_VERSION") }),
        ),
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

/// Bearer-token gate applied to `/api/*`, `/v1/traces`, and `/ws` when a token
/// is configured. Health probes and the static dashboard page stay open (the
/// dashboard's API calls are themselves gated).
async fn require_bearer(State(expected): State<ApiToken>, req: Request, next: Next) -> Response {
    let is_otlp_export = req.uri().path() == "/v1/traces";
    let provided = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match provided {
        Some(tok) if constant_time_eq(tok.as_bytes(), expected.0.as_bytes()) => {
            next.run(req).await
        }
        _ if is_otlp_export => otlp::error_response_for_request(
            req.headers(),
            StatusCode::UNAUTHORIZED,
            otlp::STATUS_UNAUTHENTICATED,
            "unauthorized: this server requires `Authorization: Bearer <token>`",
        ),
        _ => {
            (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer")],
                Json(json!({
                    "error": "unauthorized: this server requires `Authorization: Bearer <token>` (TRACERAZOR_API_TOKEN)"
                })),
            )
                .into_response()
        }
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
        .route("/v1/traces", axum::routing::post(otlp::export_traces))
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
    let bind = bind.trim();
    if bind.eq_ignore_ascii_case("localhost") {
        return true;
    }
    bind.trim_matches(['[', ']'])
        .parse::<IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

fn validate_bind_security(bind: &str, authed: bool, tls_terminated: bool) -> Result<()> {
    if is_loopback(bind) {
        return Ok(());
    }
    if !authed {
        return Err(anyhow::anyhow!(
            "refusing non-loopback bind {bind} without TRACERAZOR_API_TOKEN; use loopback or configure both bearer auth and a TLS reverse proxy"
        ));
    }
    if !tls_terminated {
        return Err(anyhow::anyhow!(
            "refusing plaintext non-loopback bind {bind}; terminate TLS at a trusted reverse proxy and set TRACERAZOR_TLS_TERMINATED=true only when that boundary is active"
        ));
    }
    Ok(())
}

/// Start the server and block until it exits. Shared by the
/// `tracerazor-server` binary and the `tracerazor serve` CLI alias.
pub async fn run_server(opts: ServeOptions) -> Result<()> {
    let token = std::env::var("TRACERAZOR_API_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());
    let authed = token.is_some();
    let tls_terminated = std::env::var("TRACERAZOR_TLS_TERMINATED")
        .is_ok_and(|value| value.eq_ignore_ascii_case("true"));
    validate_bind_security(&opts.bind, authed, tls_terminated)?;

    let spool_dir = std::env::var_os("TRACERAZOR_OTLP_SPOOL_DIR")
        .filter(|value| !value.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(state::DEFAULT_OTLP_SPOOL_DIR));
    let state = AppState::new_with_spool_dir(&opts.db_path, &spool_dir).await?;
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
    println!(
        "Transport boundary: {}",
        if is_loopback(&opts.bind) {
            "loopback-only"
        } else {
            "TLS terminated by trusted reverse proxy (asserted by TRACERAZOR_TLS_TERMINATED=true)"
        }
    );
    println!("Dashboard (Alpine): http://localhost:{}/", opts.port);
    println!(
        "Dashboard (React):  http://localhost:{}/app  (requires: cd dashboard && npm run build)",
        opts.port
    );
    println!(
        "Metrics:            http://localhost:{}/api/metrics",
        opts.port
    );
    println!(
        "OTLP/HTTP JSON+PB:  http://localhost:{}/v1/traces",
        opts.port
    );
    println!("OTLP spool:         {}", spool_dir.display());

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

    #[test]
    fn default_cors_origin_policy_is_loopback_only() {
        assert!(is_loopback_origin(&HeaderValue::from_static(
            "http://127.0.0.1:5173"
        )));
        assert!(is_loopback_origin(&HeaderValue::from_static(
            "http://localhost:3000"
        )));
        assert!(is_loopback_origin(&HeaderValue::from_static(
            "http://[::1]:5173"
        )));
        assert!(!is_loopback_origin(&HeaderValue::from_static(
            "https://evil.example"
        )));
        assert!(!is_loopback_origin(&HeaderValue::from_static(
            "not an origin"
        )));
    }

    #[test]
    fn non_loopback_bind_requires_auth_and_tls_boundary() {
        assert!(validate_bind_security("127.0.0.1", false, false).is_ok());
        assert!(validate_bind_security("[::1]", false, false).is_ok());
        assert!(validate_bind_security("0.0.0.0", false, false).is_err());
        assert!(validate_bind_security("192.168.1.25", true, false).is_err());
        assert!(validate_bind_security("0.0.0.0", false, true).is_err());
        assert!(validate_bind_security("0.0.0.0", true, true).is_ok());
    }
}
