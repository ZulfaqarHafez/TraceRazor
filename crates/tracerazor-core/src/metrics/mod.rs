pub mod srr;
pub mod ldi;
pub mod tca;
pub mod tur;
pub mod cce;
pub mod rda;
pub mod isr;
pub mod dbo;
mod verbosity_data;
pub mod vdi;
pub mod shl;
pub mod ccr;
pub mod reformulation;
pub mod gar;
pub mod csd;
pub mod obs;
pub mod tpe;

pub use srr::{SrrResult, SrrRedundantPair};
pub use ldi::{LdiResult, DetectedLoop};
pub use tca::{TcaResult, ToolMisfire};
pub use tur::TurResult;
pub use cce::{CceResult, ContextBloatStep};
pub use rda::{RdaResult, TaskComplexity};
pub use isr::{IsrResult, LowNoveltyStep};
pub use dbo::{DboResult, BranchDecision, HistoricalSequence};
pub use vdi::{VdiResult, VdiStepResult};
pub use shl::ShlResult;
pub use ccr::{CcrResult, CcrStepResult};
pub use reformulation::ReformulationStep;
pub use gar::{GarResult, GarStepResult};
pub use csd::{CsdResult, CsdStepResult};
pub use obs::ObsResult;
pub use tpe::{GoalOrigin, TpeResult};

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
        StepType::ToolCall => {
            step.content.split_whitespace().count() >= MIN_TOOL_REASONING_WORDS
        }
        _ => false,
    }
}