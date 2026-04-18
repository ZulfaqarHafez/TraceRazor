"""ConsensusReport: structured output of an adaptive-K run."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenCounts:
    input: int = 0
    output: int = 0
    cached: int = 0

    @property
    def fresh(self) -> int:
        """Input tokens that were NOT served from cache."""
        return self.input - self.cached

    @property
    def total(self) -> int:
        return self.input + self.output

    def __iadd__(self, other: "TokenCounts") -> "TokenCounts":
        self.input += other.input
        self.output += other.output
        self.cached += other.cached
        return self

    def as_dict(self) -> Dict[str, int]:
        return {
            "input": self.input,
            "output": self.output,
            "cached": self.cached,
            "fresh": self.fresh,
            "total": self.total,
        }


@dataclass
class StepRecord:
    """Per-step detail for post-run analysis."""
    step_number: int
    outcome: str                      # Outcome.value string
    consensus_rate: float
    k_used: int
    executed_tool: Optional[str]
    executed_args: Optional[dict]
    final_answer: Optional[str]
    is_mutating: bool
    tokens: TokenCounts
    divergent_alternatives: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConsensusReport:
    """Complete record of one adaptive-K agent run.

    Attributes
    ----------
    trajectory:
        Ordered list of executed actions. Each entry is a dict with a "type"
        key ("tool_call" or "final_answer") plus action-specific fields.
    consensus_rate:
        Per-step consensus rate (float in [0, 1]). Index aligns with
        trajectory entries.
    divergences:
        Step indices (0-based) where outcome was PARTIAL or DIVERGENT.
    tokens:
        Aggregate token accounting across all K*steps LLM calls.
    steps:
        Detailed per-step records for debugging and benchmarking.
    """
    trajectory: List[Dict[str, Any]]
    consensus_rate: List[float]
    divergences: List[int]
    tokens: TokenCounts
    steps: List[StepRecord] = field(default_factory=list)

    # ── convenience ────────────────────────────────────────────────────────

    def avg_consensus(self) -> float:
        if not self.consensus_rate:
            return 0.0
        return sum(self.consensus_rate) / len(self.consensus_rate)

    def summary(self) -> str:
        return (
            f"steps={len(self.trajectory)}  "
            f"avg_consensus={self.avg_consensus():.1%}  "
            f"divergences={len(self.divergences)}  "
            f"tokens={self.tokens.total:,} "
            f"(cached={self.tokens.cached:,}  fresh={self.tokens.fresh:,})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory": self.trajectory,
            "consensus_rate": self.consensus_rate,
            "divergences": self.divergences,
            "tokens": self.tokens.as_dict(),
            "avg_consensus": self.avg_consensus(),
            "step_count": len(self.trajectory),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)
