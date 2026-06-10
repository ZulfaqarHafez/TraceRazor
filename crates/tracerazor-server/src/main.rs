//! `tracerazor-server` binary — thin wrapper over the server library.
//! The same server is also reachable as `tracerazor serve`.

use anyhow::Result;
use tracerazor_server::{run_health_probe, run_server, ServeOptions};

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

    let db_path =
        std::env::var("TRACERAZOR_DB_PATH").unwrap_or_else(|_| "./tracerazor.db".to_string());

    // Bind to loopback by default so the server is not unintentionally exposed
    // on all interfaces. Set TRACERAZOR_BIND_ADDR=0.0.0.0 to expose it
    // deliberately — and set TRACERAZOR_API_TOKEN when you do.
    let bind =
        std::env::var("TRACERAZOR_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1".to_string());

    run_server(ServeOptions { port, bind, db_path }).await
}
