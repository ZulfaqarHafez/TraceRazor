pub mod agf;
pub mod cce;
pub mod ccr;
pub mod csd;
pub mod dbo;
pub mod gar;
pub mod isr;
pub mod ldi;
pub mod obs;
pub mod rda;
pub mod reformulation;
pub mod shl;
pub mod srr;
pub mod tca;
pub mod tpe;
pub mod tur;
pub mod vdi;
mod verbosity_data;

pub use agf::{AgfResult, UngroundedLiteral};
pub use cce::{CceResult, ContextBloatStep};
pub use ccr::{CcrResult, CcrStepResult};
pub use csd::{CsdResult, CsdStepResult};
pub use dbo::{BranchDecision, DboResult, HistoricalSequence};
pub use gar::{GarResult, GarStepResult};
pub use isr::{IsrResult, LowNoveltyStep};
pub use ldi::{DetectedLoop, LdiResult};
pub use obs::ObsResult;
pub use rda::{RdaResult, TaskComplexity};
pub use reformulation::ReformulationStep;
pub use shl::ShlResult;
pub use srr::{SrrRedundantPair, SrrResult};
pub use tca::{TcaResult, ToolMisfire};
pub use tpe::{GoalOrigin, TpeResult};
pub use tur::TurResult;
pub use vdi::{VdiResult, VdiStepResult};

use crate::types::{StepType, TraceStep};

/// Minimum words of prose for a tool-call step to count as reasoning-bearing.
/// ReAct agents fuse the thought and the action into one turn ("Think: … Act:
/// bash …"); a bare invocation ("Calling get_order …") stays below this bar.
pub(crate) const MIN_TOOL_REASONING_WORDS: usize = 12;

/// Whether a step carries natural-language reasoning worth scoring for the
/// goal/continuity metrics (GAR, CSD): any reasoning step, plus ReAct tool-call
/// turns whose content embeds a substantive thought rather than a bare tool
/// invocation. Without this, those metrics see only the sparse final-answer
/// steps of a tool-using agent and collapse toward zero.
pub(crate) fn carries_reasoning(step: &TraceStep) -> bool {
    match step.step_type {
        StepType::Reasoning => true,
        StepType::ToolCall => step.content.split_whitespace().count() >= MIN_TOOL_REASONING_WORDS,
        _ => false,
    }
}

/// The semantic text of a step, for goal/continuity scoring (GAR, CSD).
///
/// ReAct turns fuse prose and code in one step ("Think: count the files …
/// Act: bash ```grep -c 'Linux' /home/user/notes.txt```"). Inside the code,
/// the *argument literals* (paths, quoted search strings, filenames, numbers)
/// are task-grounded — they are exactly the tokens shared with the
/// natural-language goal and with neighbouring thoughts — while the *syntax*
/// (command names, flags, operators) is generic vocabulary that dilutes
/// lexical similarity. Fenced code blocks are therefore reduced to their
/// literals; prose passes through unchanged. When the reduction would leave
/// nothing at all, the original content is returned so the step stays
/// comparable.
pub(crate) fn reasoning_text(step: &TraceStep) -> String {
    let content = step.content.as_str();
    if !content.contains("```") {
        return content.to_string();
    }
    let mut out = String::with_capacity(content.len());
    let mut in_fence = false;
    for line in content.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            for lit in code_literals(line) {
                out.push_str(&lit);
                out.push(' ');
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if out.split_whitespace().next().is_none() {
        content.to_string()
    } else {
        out
    }
}

/// Task-grounded literals of one code line: quoted spans, paths, globs,
/// filenames, and bare numbers. Command names, flags, and operators are
/// dropped — they are syntax shared by unrelated commands, not task content.
fn code_literals(line: &str) -> Vec<String> {
    let mut lits = Vec::new();
    let mut unquoted = String::new();
    let mut in_quote: Option<char> = None;
    let mut span = String::new();
    for c in line.chars() {
        match in_quote {
            Some(q) if c == q => {
                if !span.trim().is_empty() {
                    lits.push(span.clone());
                }
                span.clear();
                in_quote = None;
            }
            Some(_) => span.push(c),
            None if c == '\'' || c == '"' => in_quote = Some(c),
            None => unquoted.push(c),
        }
    }
    if !span.trim().is_empty() {
        // Unterminated quote — keep what we saw.
        lits.push(span);
    }
    for tok in unquoted.split_whitespace() {
        if is_unquoted_literal(tok) {
            lits.push(tok.to_string());
        }
    }
    lits
}

/// Unquoted code tokens that name task entities rather than syntax.
fn is_unquoted_literal(tok: &str) -> bool {
    if tok.starts_with('-') {
        return false; // flag
    }
    if tok.contains('/') || tok.contains('*') || tok.contains('?') {
        return true; // path or glob
    }
    if tok.contains('.') && tok.len() > 2 {
        return true; // file.ext style
    }
    !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod shared_tests {
    use super::*;

    fn tool_step(content: &str) -> TraceStep {
        TraceStep {
            id: 1,
            step_type: StepType::ToolCall,
            content: content.to_string(),
            tokens: 50,
            tool_name: Some("bash".into()),
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
    fn reasoning_text_reduces_fences_to_literals() {
        let s = tool_step(
            "Think: I need to count the files.\n\nAct: bash\n\n\
             ```bash\ngrep -c 'Linux' /home/user/notes.txt | wc -l\n```",
        );
        let text = reasoning_text(&s);
        assert!(text.contains("count the files"), "prose kept: {text}");
        assert!(text.contains("Linux"), "quoted literal kept: {text}");
        assert!(
            text.contains("/home/user/notes.txt"),
            "path literal kept: {text}"
        );
        assert!(!text.contains("grep"), "command syntax dropped: {text}");
        assert!(!text.contains("-c"), "flags dropped: {text}");
        assert!(!text.contains("wc"), "command syntax dropped: {text}");
    }

    #[test]
    fn reasoning_text_passthrough_without_fence() {
        let s = tool_step("Think: plain thought, no code block here.");
        assert_eq!(reasoning_text(&s), s.content);
    }

    #[test]
    fn reasoning_text_falls_back_when_nothing_remains() {
        // Fence with no prose and no literals (pure syntax) → fall back to the
        // original content so the step stays comparable.
        let s = tool_step("```bash\npwd\n```");
        assert_eq!(reasoning_text(&s), s.content);
    }

    #[test]
    fn reasoning_text_keeps_sql_string_literals() {
        let s = tool_step(
            "Think: filter the orders table down to the European region rows.\n\n\
             ```sql\nSELECT count(*) FROM orders WHERE region = 'EU';\n```",
        );
        let text = reasoning_text(&s);
        assert!(text.contains("EU"), "SQL string literal kept: {text}");
        assert!(!text.contains("SELECT"), "SQL keywords dropped: {text}");
        assert!(!text.contains("WHERE"), "SQL keywords dropped: {text}");
    }
}
