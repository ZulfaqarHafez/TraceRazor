//! Observation Token Share (OBS)
//!
//! Fraction of a trace's tokens spent on tool calls/observations (tool I/O)
//! rather than reasoning. Promoted into the composite after it was the one
//! candidate feature that predicted real recoverable token waste and replicated
//! across two independent datasets (tau-bench, tau2-bench): higher observation
//! share correlated with *less* recoverable waste (reasoning-heavy trajectories
//! carry more removable fluff). See `crate::features` and the paper.
//!
//! Caveat: the direction was validated on conversational tool-agent datasets.
//! On agents with very large tool dumps (e.g. coding agents) the relationship
//! may differ, which is why OBS carries a modest weight and should be
//! re-checked per domain via the calibration tool.
//!
//! `normalised()` returns the share directly (higher = better, per the data).

use serde::{Deserialize, Serialize};

use crate::types::{StepType, Trace};

const TARGET: f64 = 0.30;

/// Result of the OBS metric computation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObsResult {
    /// Observation token share (0.0–1.0). Higher = more tokens on tool I/O.
    pub score: f64,
    pub observation_tokens: u32,
    pub total_tokens: u32,
    pub pass: bool,
    pub target: f64,
}

impl ObsResult {
    pub fn normalised(&self) -> f64 {
        self.score.clamp(0.0, 1.0)
    }
}

/// Compute the observation token share for a trace.
pub fn compute(trace: &Trace) -> ObsResult {
    let total: u32 = trace.steps.iter().map(|s| s.tokens).fold(0u32, u32::saturating_add);
    let obs: u32 = trace
        .steps
        .iter()
        .filter(|s| s.step_type == StepType::ToolCall || s.tool_name.is_some())
        .map(|s| s.tokens)
        .fold(0u32, u32::saturating_add);
    let score = if total > 0 {
        (obs as f64 / total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    ObsResult {
        score: (score * 1000.0).round() / 1000.0,
        observation_tokens: obs,
        total_tokens: total,
        pass: score >= TARGET,
        target: TARGET,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, Trace, TraceStep};
    use std::collections::HashMap;

    fn step(id: u32, st: StepType, tokens: u32) -> TraceStep {
        let tool = st == StepType::ToolCall;
        TraceStep {
            id,
            step_type: st,
            content: "x".into(),
            tokens,
            tool_name: if tool { Some("t".into()) } else { None },
            tool_params: None,
            tool_success: if tool { Some(true) } else { None },
            tool_error: None,
            agent_id: None,
            input_context: None,
            output: None,
            flags: vec![],
            flag_details: vec![],
        }
    }

    fn trace(steps: Vec<TraceStep>) -> Trace {
        Trace {
            trace_id: "t".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps,
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn obs_share_even_split() {
        let r = compute(&trace(vec![
            step(1, StepType::Reasoning, 100),
            step(2, StepType::ToolCall, 100),
        ]));
        assert!((r.score - 0.5).abs() < 1e-9);
        assert!((r.normalised() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn obs_share_zero_for_all_reasoning() {
        let r = compute(&trace(vec![
            step(1, StepType::Reasoning, 100),
            step(2, StepType::Reasoning, 100),
        ]));
        assert_eq!(r.score, 0.0);
    }
}
