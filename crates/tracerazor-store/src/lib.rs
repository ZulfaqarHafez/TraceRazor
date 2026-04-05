/// Persistent trace storage using SurrealDB.
///
/// Two modes:
///   `TraceStore::connect_mem()`       — in-memory, for CLI sessions and tests.
///   `TraceStore::connect_file(path)`  — persistent (kv-surrealkv), for the server.
pub mod kb;
pub use kb::{KgpEntry, KgpMatch, KgpStep, KGP_CAPTURE_THRESHOLD, build_kb_entry};

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use tracerazor_core::{
    metrics::dbo::HistoricalSequence,
    report::{Anomaly, TraceReport},
    types::Trace,
};

/// A stored trace entry with its analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredTrace {
    pub stored_at: String,
    pub trace: Trace,
    pub report: Option<TraceReport>,
}

/// Aggregate statistics for one agent across all analysed traces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    pub agent_name: String,
    pub trace_count: u64,
    pub avg_tas: f64,
    pub avg_tokens: f64,
    pub min_tas: f64,
    pub max_tas: f64,
    pub total_tokens_saved: u32,
    pub total_cost_saved_usd: f64,
}

/// Summary row for the trace list.
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
    pub tokens_saved: Option<u32>,
}

/// Dashboard aggregate data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub total_traces: usize,
    pub total_agents: usize,
    pub avg_tas: f64,
    pub total_tokens_saved: u32,
    pub total_cost_saved_usd: f64,
    pub agent_rankings: Vec<AgentStats>,
    pub recent_traces: Vec<TraceSummary>,
    pub tas_trend: Vec<TasTrendPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TasTrendPoint {
    pub timestamp: String,
    pub agent_name: String,
    pub tas_score: f64,
    pub tokens: u32,
}

/// Rolling baseline statistics for an agent, used for anomaly detection (E-04).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBaseline {
    pub agent_name: String,
    /// Rolling mean TAS score.
    pub mean_tas: f64,
    /// Rolling standard deviation of TAS scores.
    pub std_dev_tas: f64,
    /// Number of traces used to compute this baseline.
    pub sample_count: usize,
}

/// The TraceRazor store — wraps SurrealDB with a stable API.
pub struct TraceStore {
    db: Surreal<surrealdb::engine::local::Db>,
}

impl TraceStore {
    /// In-memory store (no persistence across restarts). Used by the CLI and tests.
    pub async fn connect_mem() -> Result<Self> {
        use surrealdb::engine::local::Mem;
        let db = Surreal::new::<Mem>(()).await?;
        db.use_ns("tracerazor").use_db("traces").await?;
        Ok(TraceStore { db })
    }

    /// Persistent file-backed store using SurrealKV. Used by the server.
    pub async fn connect_file(path: &str) -> Result<Self> {
        use surrealdb::engine::local::SurrealKv;
        let db = Surreal::new::<SurrealKv>(path).await?;
        db.use_ns("tracerazor").use_db("traces").await?;
        Ok(TraceStore { db })
    }

    // ── Write ──────────────────────────────────────────────────────────────

    /// Store a trace and optionally its report.
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

    // ── Read ───────────────────────────────────────────────────────────────

    /// Retrieve a stored trace by ID.
    pub async fn get_trace(&self, trace_id: &str) -> Result<Option<StoredTrace>> {
        Ok(self.db.select(("traces", trace_id)).await?)
    }

    /// List all stored trace summaries.
    pub async fn list_traces(&self) -> Result<Vec<TraceSummary>> {
        let stored: Vec<StoredTrace> = self.db.select("traces").await?;
        Ok(stored.into_iter().map(Self::to_summary).collect())
    }

    /// Aggregate statistics for a specific agent.
    pub async fn agent_stats(&self, agent_name: &str) -> Result<Option<AgentStats>> {
        let summaries = self.list_traces().await?;
        Ok(Self::compute_agent_stats(agent_name, &summaries))
    }

    /// Statistics for all agents.
    pub async fn all_agent_stats(&self) -> Result<Vec<AgentStats>> {
        let summaries = self.list_traces().await?;
        let mut agents: std::collections::HashSet<String> = std::collections::HashSet::new();
        for s in &summaries {
            agents.insert(s.agent_name.clone());
        }
        let mut stats: Vec<AgentStats> = agents
            .iter()
            .filter_map(|a| Self::compute_agent_stats(a, &summaries))
            .collect();
        // Sort by avg_tas ascending (worst offenders first).
        stats.sort_by(|a, b| a.avg_tas.partial_cmp(&b.avg_tas).unwrap());
        Ok(stats)
    }

    /// Median token count for an agent (used for VAE baseline).
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
        Ok(Some(tokens[tokens.len() / 2]))
    }

    /// Historical median step count for an agent (used for RDA accuracy).
    pub async fn historical_median_steps(&self, agent_name: &str) -> Result<Option<f64>> {
        let stored: Vec<StoredTrace> = self.db.select("traces").await?;
        let mut step_counts: Vec<usize> = stored
            .iter()
            .filter(|st| st.trace.agent_name == agent_name)
            .map(|st| st.trace.steps.len())
            .collect();
        if step_counts.len() < 3 {
            return Ok(None);
        }
        step_counts.sort();
        let median = step_counts[step_counts.len() / 2] as f64;
        Ok(Some(median))
    }

    /// Historical tool-call sequences for an agent (used for DBO computation).
    ///
    /// Returns sequences for all stored traces by this agent.
    /// DBO cold-starts when fewer than 10 similar sequences are found.
    pub async fn historical_sequences(
        &self,
        agent_name: &str,
    ) -> Result<Vec<HistoricalSequence>> {
        let stored: Vec<StoredTrace> = self.db.select("traces").await?;
        let sequences = stored
            .into_iter()
            .filter(|st| st.trace.agent_name == agent_name)
            .map(|st| {
                let tool_sequence: Vec<String> = st
                    .trace
                    .steps
                    .iter()
                    .filter_map(|s| s.tool_name.clone())
                    .collect();
                HistoricalSequence {
                    tool_sequence,
                    total_tokens: st.trace.effective_total_tokens(),
                }
            })
            .collect();
        Ok(sequences)
    }

    /// Delete a trace by ID.
    pub async fn delete_trace(&self, trace_id: &str) -> Result<()> {
        let _: Option<StoredTrace> = self.db.delete(("traces", trace_id)).await?;
        Ok(())
    }

    /// Build the full dashboard aggregate payload.
    pub async fn dashboard_data(&self) -> Result<DashboardData> {
        let summaries = self.list_traces().await?;
        let agent_stats = self.all_agent_stats().await?;

        let total_traces = summaries.len();
        let total_agents = agent_stats.len();
        let scores: Vec<f64> = summaries.iter().filter_map(|s| s.tas_score).collect();
        let avg_tas = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };
        let total_tokens_saved: u32 = summaries.iter().filter_map(|s| s.tokens_saved).sum();
        let total_cost_saved_usd = total_tokens_saved as f64 * 3.0 / 1_000_000.0;

        let mut trend: Vec<TasTrendPoint> = summaries
            .iter()
            .filter_map(|s| {
                s.tas_score.map(|t| TasTrendPoint {
                    timestamp: s.stored_at.clone(),
                    agent_name: s.agent_name.clone(),
                    tas_score: t,
                    tokens: s.total_tokens,
                })
            })
            .collect();
        trend.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        let mut recent = summaries;
        recent.sort_by(|a, b| b.stored_at.cmp(&a.stored_at));
        recent.truncate(20);

        Ok(DashboardData {
            total_traces,
            total_agents,
            avg_tas: (avg_tas * 10.0).round() / 10.0,
            total_tokens_saved,
            total_cost_saved_usd: (total_cost_saved_usd * 100.0).round() / 100.0,
            agent_rankings: agent_stats,
            recent_traces: recent,
            tas_trend: trend,
        })
    }

    // ── Anomaly Detection (E-04) ───────────────────────────────────────────

    /// Detect anomalies across all 8 metrics + TAS composite for a newly computed report.
    ///
    /// Checks every metric independently against its own rolling baseline.
    /// Returns one `Anomaly` entry per metric that deviates by more than 2σ.
    /// Requires ≥ 5 prior traces with reports for the agent.
    pub async fn detect_all_anomalies(
        &self,
        agent_name: &str,
        report: &tracerazor_core::report::TraceReport,
    ) -> Result<Vec<Anomaly>> {
        let stored: Vec<StoredTrace> = self.db.select("traces").await?;

        // Collect historical normalised-metric vectors for this agent.
        let history: Vec<[f64; 9]> = stored
            .iter()
            .filter(|st| st.trace.agent_name == agent_name)
            .filter_map(|st| st.report.as_ref())
            .map(|r| {
                [
                    r.score.score,                  // idx 0: TAS
                    r.score.srr.normalised(),        // idx 1
                    r.score.ldi.normalised(),        // idx 2
                    r.score.tca.normalised(),        // idx 3
                    r.score.tur.normalised(),        // idx 4
                    r.score.cce.normalised(),        // idx 5
                    r.score.rda.normalised(),        // idx 6
                    r.score.isr.normalised(),        // idx 7
                    r.score.dbo.normalised(),        // idx 8
                ]
            })
            .collect();

        if history.len() < 5 {
            return Ok(vec![]);
        }

        let names = ["tas_score", "srr", "ldi", "tca", "tur", "cce", "rda", "isr", "dbo"];
        let current = [
            report.score.score,
            report.score.srr.normalised(),
            report.score.ldi.normalised(),
            report.score.tca.normalised(),
            report.score.tur.normalised(),
            report.score.cce.normalised(),
            report.score.rda.normalised(),
            report.score.isr.normalised(),
            report.score.dbo.normalised(),
        ];

        let mut anomalies = Vec::new();

        for (i, &metric_name) in names.iter().enumerate() {
            let values: Vec<f64> = history.iter().map(|row| row[i]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt().max(0.01); // floor to avoid division by near-zero σ

            let z = (current[i] - mean) / std_dev;
            if z.abs() > 2.0 {
                anomalies.push(Anomaly {
                    metric: metric_name.to_string(),
                    value: (current[i] * 1000.0).round() / 1000.0,
                    z_score: (z * 100.0).round() / 100.0,
                    baseline_mean: (mean * 1000.0).round() / 1000.0,
                    baseline_std: (std_dev * 1000.0).round() / 1000.0,
                });
            }
        }

        Ok(anomalies)
    }

    /// Compute the rolling baseline for an agent's TAS score.
    ///
    /// Returns `None` if fewer than 5 traces exist (insufficient for statistics).
    pub async fn agent_baseline(&self, agent_name: &str) -> Result<Option<AgentBaseline>> {
        let summaries = self.list_traces().await?;
        let scores: Vec<f64> = summaries
            .iter()
            .filter(|s| s.agent_name == agent_name)
            .filter_map(|s| s.tas_score)
            .collect();

        if scores.len() < 5 {
            return Ok(None);
        }

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance =
            scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std_dev = variance.sqrt();

        Ok(Some(AgentBaseline {
            agent_name: agent_name.to_string(),
            mean_tas: (mean * 10.0).round() / 10.0,
            std_dev_tas: (std_dev * 10.0).round() / 10.0,
            sample_count: scores.len(),
        }))
    }

    /// Detect anomalies for a newly computed TAS score against the agent's baseline.
    ///
    /// Returns anomaly entries if the score deviates by more than 2 standard
    /// deviations from the rolling mean. Pass the result into `report.anomalies`.
    pub async fn detect_anomalies(
        &self,
        agent_name: &str,
        tas_score: f64,
    ) -> Result<Vec<Anomaly>> {
        let Some(baseline) = self.agent_baseline(agent_name).await? else {
            return Ok(vec![]);
        };

        let std_dev = baseline.std_dev_tas.max(1.0); // floor to avoid division by tiny σ
        let z_score = (tas_score - baseline.mean_tas) / std_dev;

        if z_score.abs() > 2.0 {
            Ok(vec![Anomaly {
                metric: "tas_score".into(),
                value: tas_score,
                z_score: (z_score * 100.0).round() / 100.0,
                baseline_mean: baseline.mean_tas,
                baseline_std: baseline.std_dev_tas,
            }])
        } else {
            Ok(vec![])
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    fn to_summary(st: StoredTrace) -> TraceSummary {
        let tokens_saved = st.report.as_ref().map(|r| r.savings.tokens_saved);
        TraceSummary {
            trace_id: st.trace.trace_id.clone(),
            agent_name: st.trace.agent_name.clone(),
            framework: st.trace.framework.clone(),
            total_steps: st.trace.steps.len(),
            total_tokens: st.trace.effective_total_tokens(),
            tas_score: st.report.as_ref().map(|r| r.score.score),
            grade: st.report.as_ref().map(|r| r.score.grade.to_string()),
            stored_at: st.stored_at,
            tokens_saved,
        }
    }

    fn compute_agent_stats(agent_name: &str, summaries: &[TraceSummary]) -> Option<AgentStats> {
        let agent: Vec<&TraceSummary> = summaries
            .iter()
            .filter(|s| s.agent_name == agent_name)
            .collect();
        if agent.is_empty() {
            return None;
        }
        let scores: Vec<f64> = agent.iter().filter_map(|s| s.tas_score).collect();
        let avg_tas = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };
        let total_tokens_saved: u32 = agent.iter().filter_map(|s| s.tokens_saved).sum();
        Some(AgentStats {
            agent_name: agent_name.to_string(),
            trace_count: agent.len() as u64,
            avg_tas: (avg_tas * 10.0).round() / 10.0,
            avg_tokens: agent.iter().map(|s| s.total_tokens as f64).sum::<f64>()
                / agent.len() as f64,
            min_tas: scores.iter().cloned().fold(f64::INFINITY, f64::min),
            max_tas: scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            total_tokens_saved,
            total_cost_saved_usd: total_tokens_saved as f64 * 3.0 / 1_000_000.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracerazor_core::types::{StepType, Trace, TraceStep};

    fn dummy_trace(id: &str, agent: &str) -> Trace {
        Trace {
            trace_id: id.to_string(),
            agent_name: agent.to_string(),
            framework: "raw".to_string(),
            steps: vec![TraceStep {
                id: 1,
                step_type: StepType::Reasoning,
                content: "test".into(),
                tokens: 100,
                tool_name: None,
                tool_params: None,
                tool_success: None,
                tool_error: None,
                agent_id: None,
                input_context: None,
                output: None,
                flags: vec![],
                flag_details: vec![],
            }],
            total_tokens: 100,
            task_value_score: 1.0,
            metadata: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_save_and_retrieve() {
        let store = TraceStore::connect_mem().await.unwrap();
        let trace = dummy_trace("t1", "agent-a");
        store.save_trace(&trace, None).await.unwrap();
        let retrieved = store.get_trace("t1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().trace.agent_name, "agent-a");
    }

    #[tokio::test]
    async fn test_list_traces() {
        let store = TraceStore::connect_mem().await.unwrap();
        store.save_trace(&dummy_trace("t1", "agent-a"), None).await.unwrap();
        store.save_trace(&dummy_trace("t2", "agent-b"), None).await.unwrap();
        let list = store.list_traces().await.unwrap();
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn test_baseline_tokens() {
        let store = TraceStore::connect_mem().await.unwrap();
        let mut t1 = dummy_trace("t1", "agent-a");
        t1.total_tokens = 1000;
        let mut t2 = dummy_trace("t2", "agent-a");
        t2.total_tokens = 2000;
        store.save_trace(&t1, None).await.unwrap();
        store.save_trace(&t2, None).await.unwrap();
        let baseline = store.baseline_tokens("agent-a").await.unwrap();
        assert!(baseline.is_some());
    }

    #[tokio::test]
    async fn test_dashboard_data() {
        let store = TraceStore::connect_mem().await.unwrap();
        store.save_trace(&dummy_trace("t1", "agent-a"), None).await.unwrap();
        let data = store.dashboard_data().await.unwrap();
        assert_eq!(data.total_traces, 1);
    }

    #[tokio::test]
    async fn test_delete_trace() {
        let store = TraceStore::connect_mem().await.unwrap();
        store.save_trace(&dummy_trace("t1", "agent-a"), None).await.unwrap();
        store.delete_trace("t1").await.unwrap();
        let retrieved = store.get_trace("t1").await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_historical_sequences_empty() {
        let store = TraceStore::connect_mem().await.unwrap();
        let sequences = store.historical_sequences("agent-a").await.unwrap();
        assert!(sequences.is_empty());
    }

    #[tokio::test]
    async fn test_anomaly_detection_insufficient_data() {
        let store = TraceStore::connect_mem().await.unwrap();
        // Only 2 traces — below the 5-trace minimum.
        store.save_trace(&dummy_trace("t1", "agent-a"), None).await.unwrap();
        store.save_trace(&dummy_trace("t2", "agent-a"), None).await.unwrap();
        let anomalies = store.detect_anomalies("agent-a", 30.0).await.unwrap();
        assert!(anomalies.is_empty());
    }

    #[tokio::test]
    async fn test_detect_all_anomalies_insufficient_data() {
        let store = TraceStore::connect_mem().await.unwrap();
        // Only 3 traces — below the 5-trace minimum.
        for i in 1..=3 {
            store.save_trace(&dummy_trace(&format!("t{i}"), "agent-b"), None).await.unwrap();
        }
        // Build a minimal report with defaults.
        use tracerazor_core::{analyse, scoring::ScoringConfig};
        use tracerazor_core::types::{StepType, TraceStep};
        let mut trace = dummy_trace("probe", "agent-b");
        // Add enough steps for analysis.
        trace.steps = (1..=6).map(|id| TraceStep {
            id,
            step_type: StepType::Reasoning,
            content: format!("step {id}"),
            tokens: 200,
            tool_name: None,
            tool_params: None,
            tool_success: None,
            tool_error: None,
            agent_id: None,
            input_context: None,
            output: None,
            flags: vec![],
            flag_details: vec![],
        }).collect();
        trace.total_tokens = 1200;
        let config = ScoringConfig::default();
        let report = analyse(&mut trace, |_, _| 0.0_f64, &config).unwrap();
        let anomalies = store.detect_all_anomalies("agent-b", &report).await.unwrap();
        assert!(anomalies.is_empty(), "should be empty with < 5 historical reports");
    }
}
