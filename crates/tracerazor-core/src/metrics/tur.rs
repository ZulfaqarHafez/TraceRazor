/// Token Utilisation Ratio (TUR)
///
/// Ratio of tokens that directly contributed to the final output versus
/// total tokens consumed across all steps.
///
/// Formula: TUR = useful_output_tokens / total_tokens_consumed
/// Target: > 0.35 for complex agents. > 0.7 for simple single-call agents.
use serde::{Deserialize, Serialize};

use crate::types::{StepType, Trace};

/// Result of the TUR metric computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurResult {
    /// TUR ratio (0.0–1.0). Higher is better.
    pub score: f64,
    pub useful_tokens: u32,
    pub total_tokens: u32,
    pub wasted_tokens: u32,
    pub pass: bool,
    pub target: f64,
}

impl TurResult {
    /// Normalised score for TAS composite (0.0–1.0, higher = better).
    pub fn normalised(&self) -> f64 {
        // Normalise against the complex-agent target of 0.35
        // so a score at target = 0.5 in composite terms.
        (self.score / 0.70).min(1.0)
    }
}

const TARGET_COMPLEX: f64 = 0.35;

/// Compute the TUR metric.
///
/// Heuristic approach (no ground-truth output tracing needed):
/// - Steps that are flagged as REDUNDANT, LOOP, or MISFIRE contribute 0 useful tokens.
/// - All other steps contribute their full tokens as "useful".
/// - Steps with OverDepth contribute only a fraction (estimated 30%) of their tokens.
///
/// This heuristic gives a reasonable approximation without requiring
/// token-level attribution.
pub fn compute(trace: &Trace) -> TurResult {
    use crate::types::StepFlag;

    let total = trace.effective_total_tokens();
    if total == 0 {
        return TurResult {
            score: 1.0,
            useful_tokens: 0,
            total_tokens: 0,
            wasted_tokens: 0,
            pass: true,
            target: TARGET_COMPLEX,
        };
    }

    let mut useful: u32 = 0;
    for step in &trace.steps {
        if step.flags.contains(&StepFlag::Redundant)
            || step.flags.contains(&StepFlag::Loop)
            || step.flags.contains(&StepFlag::Misfire)
        {
            // Wasted — contributes 0 useful tokens.
            continue;
        }
        if step.flags.contains(&StepFlag::OverDepth) {
            // Partially useful — contribute 30%.
            useful += (step.tokens as f64 * 0.30) as u32;
            continue;
        }
        if step.flags.contains(&StepFlag::ContextBloat) {
            // Count only non-duplicate portion. We approximate 50% waste.
            useful += (step.tokens as f64 * 0.50) as u32;
            continue;
        }
        useful += step.tokens;
    }

    let ratio = useful as f64 / total as f64;
    let wasted = total.saturating_sub(useful);

    // Determine target: use stricter target for simple traces.
    let tool_call_count = trace
        .steps
        .iter()
        .filter(|s| s.step_type == StepType::ToolCall)
        .count();
    let target = if tool_call_count <= 1 { 0.70 } else { TARGET_COMPLEX };

    TurResult {
        score: (ratio * 1000.0).round() / 1000.0,
        useful_tokens: useful,
        total_tokens: total,
        wasted_tokens: wasted,
        pass: ratio >= target,
        target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, TraceStep};
    use std::collections::HashMap;

    fn clean_step(id: u32, tokens: u32) -> TraceStep {
        TraceStep {
            id,
            step_type: StepType::Reasoning,
            content: "content".into(),
            tokens,
            tool_name: None,
            tool_params: None,
            tool_success: None,
            tool_error: None,
            agent_id: None,
            input_context: None,
            output: None,
            flags: vec![],
            flag_details: vec![],
        }
    }

    #[test]
    fn test_tur_clean_trace() {
        let trace = Trace {
            trace_id: "t1".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                clean_step(1, 100),
                clean_step(2, 200),
                clean_step(3, 300),
                clean_step(4, 200),
                clean_step(5, 200),
            ],
            total_tokens: 1000,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let result = compute(&trace);
        assert_eq!(result.score, 1.0);
        assert!(result.pass);
    }

    #[test]
    fn test_tur_with_waste() {
        use crate::types::StepFlag;
        let mut trace = Trace {
            trace_id: "t2".into(),
            agent_name: "a".into(),
            framework: "raw".into(),
            steps: vec![
                clean_step(1, 200),
                clean_step(2, 200),
                {
                    let mut s = clean_step(3, 500);
                    s.flags.push(StepFlag::Redundant);
                    s
                },
                clean_step(4, 100),
                clean_step(5, 100),
            ],
            total_tokens: 1100,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        };
        let result = compute(&mut trace);
        // 600 useful out of 1100 total
        let expected = 600.0 / 1100.0;
        assert!((result.score - expected).abs() < 0.01);
    }
}
