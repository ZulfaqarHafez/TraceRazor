//! Known-Good-Paths (KGP) Knowledge Base.
//!
//! When a trace scores ≥ `KGP_CAPTURE_THRESHOLD` (default 85), its optimal
//! execution path is stored here. Future traces by the same agent are matched
//! against the KB and shown a "this looks like run X that scored 94 — here's
//! the path it took" hint.
//!
//! Storage: a `kb_entries` table in the same SQLite database as `traces`.

use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};

use super::TraceStore;

/// Minimum TAS score for a trace to be auto-captured into the KB.
pub const KGP_CAPTURE_THRESHOLD: f64 = 85.0;

/// A single step in an optimal execution path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgpStep {
    pub step_id: u32,
    pub step_type: String,
    /// Truncated content (first 200 chars) — enough for similarity matching.
    pub content_summary: String,
    pub tool_name: Option<String>,
    pub tokens_actual: u32,
    /// Suggested token count after trimming (None = keep as-is).
    pub tokens_optimal: Option<u32>,
}

/// One KB entry — a high-scoring trace's optimal path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgpEntry {
    /// UUID assigned at capture time. Used as the row primary key.
    pub kb_id: String,
    pub source_trace_id: String,
    pub agent_name: String,
    /// Text used for similarity matching (first reasoning step content).
    pub task_hint: String,
    pub tas_score: f64,
    pub grade: String,
    pub total_steps: usize,
    pub optimal_steps: usize,
    pub total_tokens: u32,
    pub optimal_tokens: u32,
    /// The KEEP/TRIM steps from the optimal path diff.
    pub path: Vec<KgpStep>,
    pub captured_at: String,
}

/// A KB match returned alongside an audit result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgpMatch {
    pub entry: KgpEntry,
    /// BoW cosine similarity between the incoming trace and this KB entry.
    pub similarity: f64,
}

impl TraceStore {
    // ── Write ──────────────────────────────────────────────────────────────

    /// Store a KGP entry. Uses the entry's `kb_id` as the row primary key.
    pub async fn save_kb_entry(&self, entry: &KgpEntry) -> Result<()> {
        let id = entry.kb_id.clone();
        let json = serde_json::to_string(entry).context("serialise KgpEntry")?;
        self.conn
            .call(move |c| {
                c.execute(
                    "INSERT INTO kb_entries (id, data) VALUES (?1, ?2) \
                     ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                    rusqlite::params![id, json],
                )?;
                Ok(())
            })
            .await
            .context("save_kb_entry")?;
        Ok(())
    }

    /// Delete a KB entry by its UUID.
    pub async fn delete_kb_entry(&self, id: &str) -> Result<()> {
        let id = id.to_string();
        self.conn
            .call(move |c| {
                c.execute("DELETE FROM kb_entries WHERE id = ?1", rusqlite::params![id])?;
                Ok(())
            })
            .await
            .context("delete_kb_entry")?;
        Ok(())
    }

    // ── Read ───────────────────────────────────────────────────────────────

    /// List all KB entries.
    pub async fn list_kb_entries(&self) -> Result<Vec<KgpEntry>> {
        let raws: Vec<String> = self
            .conn
            .call(|c| {
                let mut stmt = c.prepare("SELECT data FROM kb_entries")?;
                let rows = stmt
                    .query_map([], |r| r.get::<_, String>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .context("list_kb_entries")?;
        raws.into_iter()
            .map(|s| serde_json::from_str::<KgpEntry>(&s).context("deserialise KgpEntry"))
            .collect()
    }

    /// List KB entries for a specific agent.
    pub async fn list_kb_for_agent(&self, agent_name: &str) -> Result<Vec<KgpEntry>> {
        let all = self.list_kb_entries().await?;
        Ok(all.into_iter().filter(|e| e.agent_name == agent_name).collect())
    }

    /// Get a single KB entry by its UUID.
    pub async fn get_kb_entry(&self, id: &str) -> Result<Option<KgpEntry>> {
        let id = id.to_string();
        let raw: Option<String> = self
            .conn
            .call(move |c| {
                let row = c
                    .query_row(
                        "SELECT data FROM kb_entries WHERE id = ?1",
                        rusqlite::params![id],
                        |r| r.get::<_, String>(0),
                    )
                    .optional()?;
                Ok(row)
            })
            .await
            .context("get_kb_entry")?;
        match raw {
            Some(s) => Ok(Some(
                serde_json::from_str(&s).context("deserialise KgpEntry")?,
            )),
            None => Ok(None),
        }
    }
}

/// Build a `KgpEntry` from a high-scoring trace and its optimal path diff.
///
/// Called by the server's audit handler after analysis completes.
pub fn build_kb_entry(
    trace: &tracerazor_core::types::Trace,
    report: &tracerazor_core::report::TraceReport,
) -> KgpEntry {
    use tracerazor_core::report::DiffAction;
    use uuid::Uuid;

    // Use the first reasoning step's content as the task hint for matching.
    let task_hint = trace
        .steps
        .iter()
        .find(|s| matches!(s.step_type, tracerazor_core::types::StepType::Reasoning))
        .map(|s| s.content.chars().take(300).collect::<String>())
        .unwrap_or_else(|| trace.agent_name.clone());

    // Extract KEEP/TRIM steps from the diff as the optimal path.
    let path: Vec<KgpStep> = report
        .diff
        .iter()
        .filter(|d| matches!(d.action, DiffAction::Keep | DiffAction::Trim))
        .map(|d| KgpStep {
            step_id: d.step_id,
            step_type: d.step_type.clone(),
            content_summary: d.description.chars().take(200).collect(),
            tool_name: None,
            tokens_actual: d.tokens_actual,
            tokens_optimal: d.tokens_suggested,
        })
        .collect();

    let optimal_tokens: u32 = path
        .iter()
        .map(|s| s.tokens_optimal.unwrap_or(s.tokens_actual))
        .sum();

    KgpEntry {
        kb_id: Uuid::new_v4().to_string(),
        source_trace_id: trace.trace_id.clone(),
        agent_name: trace.agent_name.clone(),
        task_hint,
        tas_score: report.score.score,
        grade: report.score.grade.to_string(),
        total_steps: report.total_steps,
        optimal_steps: path.len(),
        total_tokens: report.total_tokens,
        optimal_tokens,
        path,
        captured_at: Utc::now().to_rfc3339(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kb_save_list_delete() {
        let store = TraceStore::connect_mem().await.unwrap();

        let entry = KgpEntry {
            kb_id: "test-kb-001".to_string(),
            source_trace_id: "trace-001".to_string(),
            agent_name: "agent-a".to_string(),
            task_hint: "Process refund for customer order".to_string(),
            tas_score: 92.0,
            grade: "Excellent".to_string(),
            total_steps: 8,
            optimal_steps: 5,
            total_tokens: 4000,
            optimal_tokens: 2200,
            path: vec![],
            captured_at: "2026-01-01T00:00:00Z".to_string(),
        };

        store.save_kb_entry(&entry).await.unwrap();

        let list = store.list_kb_entries().await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].source_trace_id, "trace-001");
        assert_eq!(list[0].kb_id, "test-kb-001");

        let by_agent = store.list_kb_for_agent("agent-a").await.unwrap();
        assert_eq!(by_agent.len(), 1);

        let empty = store.list_kb_for_agent("other-agent").await.unwrap();
        assert!(empty.is_empty());

        store.delete_kb_entry("test-kb-001").await.unwrap();
        let after_delete = store.list_kb_entries().await.unwrap();
        assert!(after_delete.is_empty());
    }
}
