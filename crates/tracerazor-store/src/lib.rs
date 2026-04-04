/// Persistent trace storage using SurrealDB in-memory engine.
///
/// Phase 1: SurrealDB in-memory (within-session historical benchmarking).
/// Phase 2: Switch to `kv-surrealkv` for cross-session persistence with a
///          local file path (e.g., `surrealdb://./tracerazor.db`).
///
/// The store persists:
///   - Raw traces (for re-analysis)
///   - Analysis reports (for historical comparison)
///   - Aggregate stats per agent (for baseline computation)
use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use surrealdb::{Surreal, engine::local::Mem};
use tracerazor_core::{report::TraceReport, types::Trace};

/// A stored trace entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredTrace {
    pub stored_at: String,
    pub trace: Trace,
    pub report: Option<TraceReport>,
}

/// Aggregate statistics for an agent across all analysed traces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    pub agent_name: String,
    pub trace_count: u64,
    pub avg_tas: f64,
    pub avg_tokens: f64,
    pub min_tas: f64,
    pub max_tas: f64,
}

/// TraceRazor store wrapping SurrealDB.
pub struct TraceStore {
    db: Surreal<surrealdb::engine::local::Db>,
}

impl TraceStore {
    /// Connect to an in-memory SurrealDB instance.
    pub async fn connect_mem() -> Result<Self> {
        let db = Surreal::new::<Mem>(()).await?;
        db.use_ns("tracerazor").use_db("traces").await?;
        Ok(TraceStore { db })
    }

    /// Store a trace (and optionally its report).
    pub async fn save_trace(&self, trace: &Trace, report: Option<&TraceReport>) -> Result<()> {
        let entry = StoredTrace {
            stored_at: Utc::now().to_rfc3339(),
            trace: trace.clone(),
            report: report.cloned(),
        };

        let _: Option<StoredTrace> = self
            .db
            .upsert(("traces", trace.trace_id.as_str()))
            .content(entry)
            .await?;

        Ok(())
    }

    /// Retrieve a stored trace by ID.
    pub async fn get_trace(&self, trace_id: &str) -> Result<Option<StoredTrace>> {
        let result: Option<StoredTrace> =
            self.db.select(("traces", trace_id)).await?;
        Ok(result)
    }

    /// List all stored trace IDs with their TAS scores.
    pub async fn list_traces(&self) -> Result<Vec<TraceSummary>> {
        let stored: Vec<StoredTrace> = self.db.select("traces").await?;
        Ok(stored
            .into_iter()
            .map(|st| TraceSummary {
                trace_id: st.trace.trace_id.clone(),
                agent_name: st.trace.agent_name.clone(),
                framework: st.trace.framework.clone(),
                total_steps: st.trace.steps.len(),
                total_tokens: st.trace.effective_total_tokens(),
                tas_score: st.report.as_ref().map(|r| r.score.score),
                grade: st
                    .report
                    .as_ref()
                    .map(|r| r.score.grade.to_string()),
                stored_at: st.stored_at,
            })
            .collect())
    }

    /// Compute aggregate stats for a given agent.
    pub async fn agent_stats(&self, agent_name: &str) -> Result<Option<AgentStats>> {
        let summaries = self.list_traces().await?;
        let agent_summaries: Vec<&TraceSummary> = summaries
            .iter()
            .filter(|s| s.agent_name == agent_name)
            .collect();

        if agent_summaries.is_empty() {
            return Ok(None);
        }

        let scores: Vec<f64> = agent_summaries
            .iter()
            .filter_map(|s| s.tas_score)
            .collect();

        let avg_tas = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        let avg_tokens = agent_summaries
            .iter()
            .map(|s| s.total_tokens as f64)
            .sum::<f64>()
            / agent_summaries.len() as f64;

        Ok(Some(AgentStats {
            agent_name: agent_name.to_string(),
            trace_count: agent_summaries.len() as u64,
            avg_tas,
            avg_tokens,
            min_tas: scores.iter().cloned().fold(f64::INFINITY, f64::min),
            max_tas: scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }))
    }

    /// Get the baseline (median) token count for an agent.
    /// Used for VAE normalised_token_cost computation.
    pub async fn baseline_tokens(&self, agent_name: &str) -> Result<Option<u32>> {
        let summaries = self.list_traces().await?;
        let mut tokens: Vec<u32> = summaries
            .iter()
            .filter(|s| s.agent_name == agent_name)
            .map(|s| s.total_tokens)
            .collect();

        if tokens.is_empty() {
            return Ok(None);
        }

        tokens.sort();
        let median = tokens[tokens.len() / 2];
        Ok(Some(median))
    }
}

/// A brief summary of a stored trace for listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSummary {
    pub trace_id: String,
    pub agent_name: String,
    pub framework: String,
    pub total_steps: usize,
    pub total_tokens: u32,
    pub tas_score: Option<f64>,
    pub grade: Option<String>,
    pub stored_at: String,
}
