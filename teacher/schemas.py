"""Typed data contracts for the TraceRazor Teacher / Orchestrator.

These mirror the designs from the v2 planning departments:

* ``WastePattern`` / ``Diagnosis``      -- what the auditor tells us (P2-A, P1).
* ``Intervention`` / ``AgentConfig``    -- the *applicable, reversible* edits (P2-A).
* ``EvalResult`` / ``QualityGate``      -- the closed-loop verify gate (P2-B).
* ``PlaybookEntry`` / ``Outcome``       -- cross-agent memory (P3-A).

Everything here is plain ``dataclasses`` + stdlib so the package has zero hard
dependencies and runs fully offline.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Waste taxonomy -- 1:1 with the auditor's metric codes.
# --------------------------------------------------------------------------- #
class WasteKind(str, Enum):
    REDUNDANT_STEP = "srr"      # near-duplicate steps
    LOOP = "ldi"                # repeated tool/action loop
    TOOL_MISFIRE = "tca"        # tool called with bad/missing params
    OVER_DEPTH = "rda"          # more reasoning steps than the task needs
    CONTEXT_BLOAT = "cce"       # duplicate context carried forward / reformulation
    VERBOSITY = "vdi"           # filler-heavy, low-density reasoning
    HEDGING = "shl"             # sycophancy / preamble / hedging


@dataclass(frozen=True)
class WastePattern:
    """A fingerprint of *why* tokens are wasted -- the unit memory keys on."""
    kind: WasteKind
    severity: float             # 0..1
    step_ids: tuple[int, ...]
    est_token_waste: int

    @property
    def signature(self) -> str:
        # Stable across traces/agents so a lesson learned on one agent
        # transfers to another with the same pattern.
        bucket = "hi" if self.severity >= 0.5 else "lo"
        return f"{self.kind.value}|{bucket}"


@dataclass
class Diagnosis:
    trace_id: str
    agent_name: str
    framework: str
    tas_score: float            # auditor TAS (ordinal); higher == leaner
    total_tokens: int
    patterns: list[WastePattern]
    source: str = "builtin"     # "auditor" if produced by the Rust binary
    backend: str = "builtin"    # "native" | "subprocess" | "builtin"
    auditor_fixes: list[dict] = field(default_factory=list)   # raw fixes[] from the auditor
    savings: dict[str, Any] = field(default_factory=dict)      # raw savings{} from the auditor
    raw: dict[str, Any] = field(default_factory=dict)

    def pattern(self, kind: "WasteKind") -> Optional["WastePattern"]:
        for p in self.patterns:
            if p.kind is kind:
                return p
        return None


# --------------------------------------------------------------------------- #
# Interventions -- typed, idempotent, reversible edits (not raw strings).
# --------------------------------------------------------------------------- #
class Target(str, Enum):
    SYSTEM_PROMPT = "system_prompt"
    TOOL_DEF = "tool_def"
    RUNTIME_POLICY = "runtime_policy"   # proxy / middleware config
    DECODING = "decoding"


class Tier(int, Enum):
    """Curriculum order: cheap & safe -> deep & risky (P3-A)."""
    INLINE = 0      # runtime policy nudge, reversible per call
    PROMPT = 1      # system-prompt section edit
    TOOL = 2        # tool-schema edit
    STRUCT = 3      # control-flow / budget caps


@dataclass(frozen=True)
class Intervention:
    waste_pattern: WasteKind
    target: Target
    tier: Tier
    key: str                    # section name / tool name / policy key (idempotency key)
    payload: dict[str, Any]
    predicted_savings: int
    predicted_risk: float       # 0..1
    rationale: str = ""

    @property
    def id(self) -> str:
        raw = f"{self.waste_pattern.value}|{self.target.value}|{self.key}"
        return "iv_" + hashlib.sha1(raw.encode()).hexdigest()[:10]


@dataclass
class AgentConfig:
    """The thing interventions mutate. ``policies`` drives the mock runtime."""
    system_prompt_sections: dict[str, str] = field(default_factory=dict)
    tool_required_params: dict[str, list[str]] = field(default_factory=dict)
    policies: dict[str, Any] = field(default_factory=dict)
    version: int = 0

    def clone(self) -> "AgentConfig":
        return copy.deepcopy(self)

    def render_prompt(self) -> str:
        # Deterministic, stable section order -> reproducible.
        return "\n".join(
            f"## {k}\n{v}" for k, v in sorted(self.system_prompt_sections.items())
        )


# --------------------------------------------------------------------------- #
# Verification / quality gate (P2-B).
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    tokens: list[int]
    success: list[bool]
    tas: float = 0.0

    @property
    def mean_tokens(self) -> float:
        return sum(self.tokens) / max(len(self.tokens), 1)

    @property
    def success_rate(self) -> float:
        return sum(self.success) / max(len(self.success), 1)


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT_QUALITY = "REJECT_QUALITY"   # task success regressed -> rollback
    REJECT_NO_GAIN = "REJECT_NO_GAIN"   # safe but no real savings


@dataclass
class VerifiedResult:
    intervention: Intervention
    decision: Decision
    tokens_before: float
    tokens_after: float
    success_before: float
    success_after: float
    tas_before: float
    tas_after: float

    @property
    def token_delta_pct(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return 100.0 * (self.tokens_after - self.tokens_before) / self.tokens_before

    @property
    def accepted(self) -> bool:
        return self.decision is Decision.ACCEPT


# --------------------------------------------------------------------------- #
# Memory / playbook (P3-A).
# --------------------------------------------------------------------------- #
@dataclass
class Outcome:
    pattern_signature: str
    intervention_id: str
    accepted: bool
    token_delta_pct: float
    tas_delta: float


@dataclass
class PlaybookEntry:
    pattern_signature: str
    waste_kind: WasteKind
    framework: str
    intervention_key: str
    trials: int = 0
    wins: int = 0
    mean_token_saving_pct: float = 0.0
    mean_tas_delta: float = 0.0

    @property
    def winrate(self) -> float:
        return self.wins / max(self.trials, 1)

    def record(self, outcome: Outcome) -> None:
        n = self.trials
        self.trials += 1
        if outcome.accepted:
            self.wins += 1
        # running means
        self.mean_token_saving_pct = (
            (self.mean_token_saving_pct * n) + (-outcome.token_delta_pct)
        ) / self.trials
        self.mean_tas_delta = (
            (self.mean_tas_delta * n) + outcome.tas_delta
        ) / self.trials
