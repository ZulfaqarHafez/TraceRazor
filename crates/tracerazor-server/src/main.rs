pub mod api;
pub mod state;
pub mod ws;

use anyhow::Result;
use axum::Router;
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

use state::AppState;

/// Build the Axum application router. Extracted for testability.
pub fn build_app(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // State is set once at the top level so both /api and /ws share it.
    Router::new()
        .nest("/api", api::router())
        .route("/ws", axum::routing::get(ws::handler))
        .nest_service("/", ServeDir::new("dashboard/dist"))
        .layer(cors)
        .with_state(state)
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    // Database path from env, default to ./tracerazor.db
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
    println!("Dashboard: http://localhost:{}/", port);
    println!("API docs:  http://localhost:{}/api/", port);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
