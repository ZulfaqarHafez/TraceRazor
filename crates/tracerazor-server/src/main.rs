pub mod api;
pub mod state;
pub mod ws;

use anyhow::Result;
use axum::{
    extract::DefaultBodyLimit,
    http::{header, Method},
    response::IntoResponse,
    Router,
};
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
        .route("/ws", axum::routing::get(ws::handler))
        // Lightweight dashboard embedded in binary (always available).
        .route("/", axum::routing::get(dashboard_handler))
        // React build served at /app (optional — run `npm run build` in dashboard/).
        .nest_service("/app", ServeDir::new("dashboard/dist"))
        .layer(cors)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    let db_path = std::env::var("TRACERAZOR_DB_PATH")
        .unwrap_or_else(|_| "./tracerazor.db".to_string());

    let state = AppState::new(&db_path).await?;
    let app = build_app(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

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
