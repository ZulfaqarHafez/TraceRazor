pub mod api;
pub mod state;
pub mod ws;

use anyhow::Result;
use axum::{
    extract::{DefaultBodyLimit, State},
    http::{header, Method, StatusCode},
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use serde_json::json;
use std::net::SocketAddr;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::services::ServeDir;

/// Maximum accepted request body size (16 MiB). Prevents memory-exhaustion DoS
/// via an unbounded `POST /api/audit` payload while leaving ample room for
/// legitimately large traces.
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

use state::AppState;

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

/// Build the Axum application router. Extracted for testability.
pub fn build_app(state: AppState) -> Router {
    // Restrict to the methods/headers the API actually uses rather than `Any`,
    // shrinking the cross-origin attack surface.
    let cors = CorsLayer::new()
        .allow_origin(cors_origins())
        .allow_methods([Method::GET, Method::POST, Method::DELETE])
        .allow_headers([header::CONTENT_TYPE]);

    Router::new()
        .nest("/api", api::router())
        // Kubernetes-style health probes (also used by the container HEALTHCHECK).
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/ws", axum::routing::get(ws::handler))
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
async fn run_health_probe(port: u16) -> Result<()> {
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

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    // Health-probe mode: `tracerazor-server --health-check` checks a running
    // instance and exits 0 (healthy) / 1 (unhealthy). Used by the container
    // HEALTHCHECK. Exit explicitly so the output stays terse and the code is
    // deterministic (no anyhow backtrace dump).
    if std::env::args().skip(1).any(|a| a == "--health-check") {
        match run_health_probe(port).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                eprintln!("unhealthy: {e}");
                std::process::exit(1);
            }
        }
    }

    let db_path = std::env::var("TRACERAZOR_DB_PATH")
        .unwrap_or_else(|_| "./tracerazor.db".to_string());

    let state = AppState::new(&db_path).await?;
    let app = build_app(state);

    // Bind to loopback by default so the server is not unintentionally exposed on
    // all interfaces. Set TRACERAZOR_BIND_ADDR=0.0.0.0 to expose it deliberately
    // (do so only behind auth / a trusted network — there is no built-in auth).
    let bind_host =
        std::env::var("TRACERAZOR_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1".to_string());
    let addr: SocketAddr = format!("{bind_host}:{port}")
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid bind address '{bind_host}:{port}': {e}"))?;
    println!("TraceRazor server listening on http://{}", addr);
    println!("Dashboard (Alpine): http://localhost:{}/", port);
    println!("Dashboard (React):  http://localhost:{}/app  (requires: cd dashboard && npm run build)", port);
    println!("Metrics:            http://localhost:{}/api/metrics", port);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
