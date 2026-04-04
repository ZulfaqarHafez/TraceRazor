pub mod api;
pub mod state;
pub mod ws;

use anyhow::Result;
use axum::{
    http::header,
    response::IntoResponse,
    Router,
};
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

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

/// Build the Axum application router. Extracted for testability.
pub fn build_app(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .nest("/api", api::router())
        .route("/ws", axum::routing::get(ws::handler))
        // Lightweight dashboard embedded in binary (always available).
        .route("/", axum::routing::get(dashboard_handler))
        // React build served at /app (optional — run `npm run build` in dashboard/).
        .nest_service("/app", ServeDir::new("dashboard/dist"))
        .layer(cors)
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

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("TraceRazor server listening on http://localhost:{}", port);
    println!("Dashboard (Alpine): http://localhost:{}/", port);
    println!("Dashboard (React):  http://localhost:{}/app  (requires: cd dashboard && npm run build)", port);
    println!("Metrics:            http://localhost:{}/api/metrics", port);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
