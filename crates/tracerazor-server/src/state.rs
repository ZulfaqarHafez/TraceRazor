use anyhow::Result;
use std::{path::PathBuf, sync::Arc};
use tokio::sync::broadcast;
use tracerazor_store::TraceStore;

pub const DEFAULT_OTLP_SPOOL_DIR: &str = ".tracerazor/otlp-spool";

/// Shared server state threaded through Axum handlers.
#[derive(Clone)]
pub struct AppState {
    pub store: Arc<TraceStore>,
    /// Directory containing durable OTLP/HTTP ingest receipts.
    pub otlp_spool_dir: Arc<PathBuf>,
    /// Broadcast channel for real-time WebSocket events.
    pub events: broadcast::Sender<WsEvent>,
}

/// Events pushed to WebSocket subscribers.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsEvent {
    /// A new trace was analysed and stored.
    TraceAnalysed {
        trace_id: String,
        agent_name: String,
        tas_score: f64,
        grade: String,
        tokens_saved: u32,
    },
    /// A real-time loop was detected in a streaming trace.
    LoopDetected {
        trace_id: String,
        step_id: u32,
        cycle: String,
    },
}

impl AppState {
    pub async fn new(db_path: &str) -> Result<Self> {
        Self::new_with_spool_dir(db_path, DEFAULT_OTLP_SPOOL_DIR).await
    }

    pub async fn new_with_spool_dir(
        db_path: &str,
        otlp_spool_dir: impl Into<PathBuf>,
    ) -> Result<Self> {
        let store = if db_path == ":mem:" {
            TraceStore::connect_mem().await?
        } else {
            TraceStore::connect_file(db_path).await?
        };
        let (tx, _) = broadcast::channel(256);
        Ok(AppState {
            store: Arc::new(store),
            otlp_spool_dir: Arc::new(otlp_spool_dir.into()),
            events: tx,
        })
    }
}
