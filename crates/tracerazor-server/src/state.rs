use anyhow::Result;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracerazor_store::TraceStore;

/// Shared server state threaded through Axum handlers.
#[derive(Clone)]
pub struct AppState {
    pub store: Arc<TraceStore>,
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
        let store = if db_path == ":mem:" {
            TraceStore::connect_mem().await?
        } else {
            TraceStore::connect_file(db_path).await?
        };
        let (tx, _) = broadcast::channel(256);
        Ok(AppState {
            store: Arc::new(store),
            events: tx,
        })
    }
}
