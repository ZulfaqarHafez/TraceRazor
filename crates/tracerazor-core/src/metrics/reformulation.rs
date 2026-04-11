/// Reformulation Detection
///
/// Detects when a reasoning step's opening sentence is a near-paraphrase of
/// the step's input_context, adding no new information. Detection uses bigram
/// Jaccard overlap ≥ 0.70 as the threshold.
///
/// Flagged steps receive `StepFlag::Reformulation` and are eligible for a
/// `ReformulationGuard` fix.
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::types::{StepFlag, Trace, TraceStep};

/// Minimum bigram Jaccard overlap to flag a step as reformulation.
pub const REFORMULATION_THRESHOLD: f64 = 0.70;

/// A step flagged as reformulating its input context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReformulationStep {
    pub step_id: u32,
    /// Bigram Jaccard overlap with input_context (0.0–1.0).
    pub overlap: f64,
}

/// Detect steps whose opening sentence overlaps significantly with input_context.
///
/// Only steps that have a non-empty `input_context` field are examined.
/// Returns an empty vec when no steps are flagged.
pub fn detect(trace: &Trace) -> Vec<ReformulationStep> {
    trace
        .steps
        .iter()
        .filter_map(|step| {
            let ctx = step.input_context.as_deref()?;
            let first = first_sentence(&step.content);
            if first.is_empty() || ctx.is_empty() {
                return None;
            }
            let overlap = bigram_jaccard(first, ctx);
            if overlap >= REFORMULATION_THRESHOLD {
                Some(ReformulationStep {
                    step_id: step.id,
                    overlap,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Annotate steps with `StepFlag::Reformulation` for all detected reformulations.
pub fn annotate_steps(steps: &mut [TraceStep], detected: &[ReformulationStep]) {
    for step in steps.iter_mut() {
        if let Some(r) = detected.iter().find(|r| r.step_id == step.id) {
            if !step.flags.contains(&StepFlag::Reformulation) {
                step.flags.push(StepFlag::Reformulation);
                step.flag_details.push(format!(
                    "reformulates input context ({:.0}% bigram overlap)",
                    r.overlap * 100.0
                ));
            }
        }
    }
}

/// Extract the first sentence from text (up to the first `.`, `!`, or `?`).
fn first_sentence(text: &str) -> &str {
    text.split(['.', '!', '?'])
        .next()
        .unwrap_or(text)
        .trim()
}

/// Build a set of consecutive word bigrams from text.
fn bigrams(text: &str) -> HashSet<(String, String)> {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphabetic())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();
    words
        .windows(2)
        .map(|pair| (pair[0].clone(), pair[1].clone()))
        .collect()
}

/// Bigram Jaccard similarity between two text strings.
fn bigram_jaccard(a: &str, b: &str) -> f64 {
    let set_a = bigrams(a);
    let set_b = bigrams(b);
    if set_a.is_empty() && set_b.is_empty() {
        return 0.0;
    }
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, Trace, TraceStep};
    use std::collections::HashMap;

    fn make_step(id: u32, content: &str, input_context: Option<&str>) -> TraceStep {
        TraceStep {
            id,
            step_type: StepType::Reasoning,
            content: content.into(),
            tokens: 200,
            tool_name: None,
            tool_params: None,
            tool_success: None,
            tool_error: None,
            agent_id: None,
            input_context: input_context.map(|s| s.into()),
            output: None,
            flags: vec![],
            flag_details: vec![],
        }
    }

    fn make_trace(steps: Vec<TraceStep>) -> Trace {
        Trace {
            trace_id: "ref-test".into(),
            agent_name: "agent".into(),
            framework: "raw".into(),
            steps,
            total_tokens: 0,
            task_value_score: 1.0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_no_reformulation_without_context() {
        let trace = make_trace(vec![make_step(
            1,
            "Process the refund for order ORD-9182.",
            None,
        )]);
        let detected = detect(&trace);
        assert!(detected.is_empty(), "no context = no reformulation");
    }

    #[test]
    fn test_high_overlap_flagged() {
        // Step opening sentence is nearly identical to input_context.
        let ctx =
            "The user wants to process a refund for order ORD-9182 placed on Monday";
        let content = "The user wants to process a refund for order ORD-9182.";
        let trace = make_trace(vec![make_step(1, content, Some(ctx))]);
        let detected = detect(&trace);
        assert!(!detected.is_empty(), "high overlap should be flagged");
        assert!(
            detected[0].overlap >= REFORMULATION_THRESHOLD,
            "overlap {:.2} should be >= {:.2}",
            detected[0].overlap,
            REFORMULATION_THRESHOLD
        );
    }

    #[test]
    fn test_low_overlap_not_flagged() {
        let ctx = "User: please check order status";
        let content = "Fetching order details via get_order_details tool.";
        let trace = make_trace(vec![make_step(1, content, Some(ctx))]);
        let detected = detect(&trace);
        assert!(detected.is_empty(), "low overlap should not be flagged");
    }

    #[test]
    fn test_empty_context_skipped() {
        let trace = make_trace(vec![make_step(1, "Process refund.", Some(""))]);
        let detected = detect(&trace);
        assert!(detected.is_empty(), "empty context should not flag");
    }

    #[test]
    fn test_annotate_steps_adds_flag() {
        let mut trace = make_trace(vec![make_step(
            1,
            "The user wants to process a refund for order ORD-9182.",
            Some("The user wants to process a refund for order ORD-9182 placed on Monday"),
        )]);
        let detected = detect(&trace);
        annotate_steps(&mut trace.steps, &detected);
        assert!(
            trace.steps[0].flags.contains(&StepFlag::Reformulation),
            "step should be flagged as Reformulation"
        );
        assert!(
            !trace.steps[0].flag_details.is_empty(),
            "flag detail should be populated"
        );
    }

    #[test]
    fn test_annotate_steps_no_duplicate_flag() {
        let mut trace = make_trace(vec![make_step(
            1,
            "The user wants to process a refund for order ORD-9182.",
            Some("The user wants to process a refund for order ORD-9182 placed on Monday"),
        )]);
        let detected = detect(&trace);
        // Annotate twice — should not double-flag.
        annotate_steps(&mut trace.steps, &detected);
        annotate_steps(&mut trace.steps, &detected);
        assert_eq!(
            trace.steps[0]
                .flags
                .iter()
                .filter(|f| **f == StepFlag::Reformulation)
                .count(),
            1,
            "should not add duplicate Reformulation flags"
        );
    }
}