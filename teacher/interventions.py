"""Intervention synthesis + application.

``propose`` maps a ``Diagnosis`` to typed, applicable ``Intervention`` objects
(the taxonomy from P2-A). ``apply`` performs a deterministic, idempotent,
reversible edit on an ``AgentConfig``.

Each waste pattern is paired with the *cheapest effective* intervention tier;
the Teacher's curriculum tries tiers in order. One deliberately over-aggressive
candidate (a STRUCT step-cap that can starve real tool calls) is included so
the quality gate has something genuine to reject.

Typed subclasses (SystemPromptIntervention, ToolDefIntervention,
RuntimePolicyIntervention, MemoryPolicyIntervention, DecodingIntervention)
extend the base Intervention dataclass with apply()/rollback() semantics and
are available for callers that need richer introspection than the free-function
``apply()`` provides.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from .schemas import (
    AgentConfig,
    Diagnosis,
    Intervention,
    Target,
    Tier,
    WasteKind,
)


# --------------------------------------------------------------------------- #
# Typed intervention risk levels.
# --------------------------------------------------------------------------- #
class InterventionRisk(str, Enum):
    SAFE = "safe"
    NEEDS_REVIEW = "needs_review"
    DANGEROUS = "dangerous"


# --------------------------------------------------------------------------- #
# Typed intervention subclasses (OO interface over the flat Intervention DTO).
# These are separate from the frozen Intervention dataclass in schemas.py and
# are used by callers that need apply/rollback/validate semantics.
# --------------------------------------------------------------------------- #
@dataclass
class TypedIntervention:
    """Abstract base for typed interventions with apply/rollback semantics."""
    target: str            # system_prompt | tool_def | runtime_policy | memory_policy | decoding
    kind: str
    risk: InterventionRisk = InterventionRisk.SAFE
    rationale: str = ""
    estimated_savings: int = 0

    def apply(self, config: AgentConfig) -> AgentConfig:
        raise NotImplementedError

    def rollback(self, config: AgentConfig) -> AgentConfig:
        raise NotImplementedError

    def validate(self) -> bool:
        return True

    def is_idempotent(self) -> bool:
        return True

    def to_diff(self) -> str:
        return f"[{self.target}] {self.kind}: {self.rationale}"

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "kind": self.kind,
            "risk": self.risk.value,
            "rationale": self.rationale,
            "estimated_savings": self.estimated_savings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TypedIntervention":
        return cls(
            target=d["target"],
            kind=d["kind"],
            risk=InterventionRisk(d.get("risk", "safe")),
            rationale=d.get("rationale", ""),
            estimated_savings=d.get("estimated_savings", 0),
        )


@dataclass
class SystemPromptIntervention(TypedIntervention):
    """Edit a system-prompt section: append, prepend, or replace."""
    section: str = ""    # section name, e.g. "efficiency_rules"
    content: str = ""    # text to add/replace
    operation: str = "append"   # "append" | "prepend" | "replace_section"
    target: str = field(default="system_prompt", init=False)  # type: ignore[assignment]
    kind: str = field(default="system_prompt_edit", init=False)
    risk: InterventionRisk = InterventionRisk.SAFE

    def __post_init__(self):
        self.target = "system_prompt"

    def apply(self, config: AgentConfig) -> AgentConfig:
        new = copy.deepcopy(config)
        if self.operation == "append":
            existing = new.system_prompt_sections.get(self.section, "")
            new.system_prompt_sections[self.section] = (
                (existing + "\n\n" + self.content).strip() if existing
                else self.content
            )
        elif self.operation == "prepend":
            existing = new.system_prompt_sections.get(self.section, "")
            new.system_prompt_sections[self.section] = (
                (self.content + "\n\n" + existing).strip() if existing
                else self.content
            )
        elif self.operation == "replace_section":
            new.system_prompt_sections[self.section] = self.content
        new.version += 1
        return new

    def rollback(self, config: AgentConfig) -> AgentConfig:
        new = copy.deepcopy(config)
        if self.operation in ("append", "prepend", "replace_section"):
            new.system_prompt_sections.pop(self.section, None)
        new.version += 1
        return new

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"section": self.section, "content": self.content,
                  "operation": self.operation})
        return d


@dataclass
class ToolDefIntervention(TypedIntervention):
    """Modify a tool's JSON schema (mark required, add description/enum)."""
    tool_name: str = ""
    parameter: str = ""
    change_type: str = "mark_required"   # "mark_required" | "add_description" | "add_enum"
    details: str = ""
    target: str = field(default="tool_def", init=False)  # type: ignore[assignment]
    kind: str = field(default="tool_def_edit", init=False)
    risk: InterventionRisk = InterventionRisk.NEEDS_REVIEW

    def __post_init__(self):
        self.target = "tool_def"

    def apply(self, config: AgentConfig) -> AgentConfig:
        new = copy.deepcopy(config)
        if self.change_type == "mark_required":
            existing = set(new.tool_required_params.get(self.tool_name, []))
            existing.add(self.parameter)
            new.tool_required_params[self.tool_name] = sorted(existing)
        new.version += 1
        return new

    def rollback(self, config: AgentConfig) -> AgentConfig:
        new = copy.deepcopy(config)
        if self.change_type == "mark_required":
            existing = set(new.tool_required_params.get(self.tool_name, []))
            existing.discard(self.parameter)
            new.tool_required_params[self.tool_name] = sorted(existing)
        new.version += 1
        return new

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"tool_name": self.tool_name, "parameter": self.parameter,
                  "change_type": self.change_type, "details": self.details})
        return d


@dataclass
class RuntimePolicyIntervention(TypedIntervention):
    """Set a runtime policy key (loop_breaker, step_cap, retry_limit, etc.)."""
    policy_name: str = ""   # "loop_breaker" | "step_cap" | "retry_limit"
    value: Any = None
    target: str = field(default="runtime_policy", init=False)  # type: ignore[assignment]
    kind: str = field(default="runtime_policy_edit", init=False)
    risk: InterventionRisk = InterventionRisk.SAFE

    def __post_init__(self):
        self.target = "runtime_policy"

    def apply(self, config: AgentConfig) -> AgentConfig:
        new = copy.deepcopy(config)
        new.policies[self.policy_name] = self.value
        new.version += 1
        return new

    def rollback(self, config: AgentConfig) -> AgentConfig:
        new = copy.deepcopy(config)
        new.policies.pop(self.policy_name, None)
        new.version += 1
        return new

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"policy_name": self.policy_name, "value": self.value})
        return d


@dataclass
class MemoryPolicyIntervention(TypedIntervention):
    """Constrain context-window or history to reduce carried-forward tokens."""
    policy: str = "limit_history"    # "summarize_context" | "limit_history"
    window_size: Optional[int] = None
    target: str = field(default="memory_policy", init=False)  # type: ignore[assignment]
    kind: str = field(default="memory_policy_edit", init=False)
    risk: InterventionRisk = InterventionRisk.SAFE

    def __post_init__(self):
        self.target = "memory_policy"

    def apply(self, config: AgentConfig) -> AgentConfig:
        new = copy.deepcopy(config)
        entry: dict[str, Any] = {"policy": self.policy}
        if self.window_size is not None:
            entry["window_size"] = self.window_size
        new.policies[f"memory:{self.policy}"] = entry
        new.version += 1
        return new

    def rollback(self, config: AgentConfig) -> AgentConfig:
        new = copy.deepcopy(config)
        new.policies.pop(f"memory:{self.policy}", None)
        new.version += 1
        return new

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"policy": self.policy, "window_size": self.window_size})
        return d


@dataclass
class DecodingIntervention(TypedIntervention):
    """Adjust LLM decoding parameters (temperature, max_tokens, stop sequences)."""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    target: str = field(default="decoding", init=False)  # type: ignore[assignment]
    kind: str = field(default="decoding_edit", init=False)
    risk: InterventionRisk = InterventionRisk.SAFE

    def __post_init__(self):
        self.target = "decoding"

    def apply(self, config: AgentConfig) -> AgentConfig:
        new = copy.deepcopy(config)
        if self.temperature is not None:
            new.policies["decoding:temperature"] = self.temperature
        if self.max_tokens is not None:
            new.policies["decoding:max_tokens"] = self.max_tokens
        if self.stop_sequences is not None:
            new.policies["decoding:stop_sequences"] = self.stop_sequences
        new.version += 1
        return new

    def rollback(self, config: AgentConfig) -> AgentConfig:
        new = copy.deepcopy(config)
        if self.temperature is not None:
            new.policies.pop("decoding:temperature", None)
        if self.max_tokens is not None:
            new.policies.pop("decoding:max_tokens", None)
        if self.stop_sequences is not None:
            new.policies.pop("decoding:stop_sequences", None)
        new.version += 1
        return new

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop_sequences": self.stop_sequences,
        })
        return d


@dataclass
class InterventionBundle:
    """Ordered collection of TypedInterventions applied atomically."""
    interventions: List[TypedIntervention] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)   # author, timestamp, rationale

    def apply(self, config: AgentConfig) -> AgentConfig:
        for iv in self.interventions:
            config = iv.apply(config)
        return config

    def rollback(self, config: AgentConfig) -> AgentConfig:
        for iv in reversed(self.interventions):
            config = iv.rollback(config)
        return config

    def risk_summary(self) -> str:
        risks = [iv.risk for iv in self.interventions]
        if InterventionRisk.DANGEROUS in risks:
            return "dangerous"
        if InterventionRisk.NEEDS_REVIEW in risks:
            return "needs_review"
        return "safe"

    def to_dict(self) -> dict:
        return {
            "interventions": [iv.to_dict() for iv in self.interventions],
            "metadata": self.metadata,
        }


# --------------------------------------------------------------------------- #
# Auditor-fix bridge (unchanged from original).
# --------------------------------------------------------------------------- #
def _auditor_fix_spec(fix: dict):
    ft = fix.get("fix_type", "")
    patch = fix.get("patch", "")
    target = fix.get("target", "")
    table = {
        "hedge_reduction": (WasteKind.HEDGING, Target.SYSTEM_PROMPT, Tier.PROMPT,
                            "NO_HEDGING", {"body": patch}),
        "reformulation_guard": (WasteKind.REDUNDANT_STEP, Target.SYSTEM_PROMPT, Tier.PROMPT,
                                "NO_REFORMULATION", {"body": patch}),
        "context_compression": (WasteKind.CONTEXT_BLOAT, Target.SYSTEM_PROMPT, Tier.PROMPT,
                                "CONTEXT_COMPRESSION", {"body": patch}),
        "verbosity_reduction": (WasteKind.VERBOSITY, Target.SYSTEM_PROMPT, Tier.PROMPT,
                                "EFFICIENCY_RULES", {"body": patch}),
        "caveman_prompt_insert": (WasteKind.VERBOSITY, Target.SYSTEM_PROMPT, Tier.PROMPT,
                                  "EFFICIENCY_RULES", {"body": patch}),
        "termination_guard": (WasteKind.LOOP, Target.RUNTIME_POLICY, Tier.INLINE,
                              "loop_breaker", {"value": {"max_repeats": 1}}),
        "tool_schema": (WasteKind.TOOL_MISFIRE, Target.TOOL_DEF, Tier.TOOL,
                        target or "tool", {"params": _params_from_patch(patch)}),
        "prompt_insert": (WasteKind.OVER_DEPTH, Target.SYSTEM_PROMPT, Tier.PROMPT,
                          "STEP_BUDGET", {"body": patch}),
    }
    return table.get(ft)


def _params_from_patch(patch: str) -> list[str]:
    # Tool-schema patches name the missing parameter, e.g.
    # "missing required parameter: order_id". Best-effort extraction.
    found = re.findall(r"parameter[s]?:?\s*([a-z_][a-z0-9_]*)", patch, re.IGNORECASE)
    found += re.findall(r'"([a-z_][a-z0-9_]*)"', patch)
    seen, out = set(), []
    for f in found:
        if f not in seen and f not in ("required",):
            seen.add(f)
            out.append(f)
    return out[:4]


def from_auditor_fixes(diagnosis: Diagnosis) -> list[Intervention]:
    """Turn the auditor's emitted ``fixes[]`` into applicable interventions."""
    out: list[Intervention] = []
    for fix in diagnosis.auditor_fixes:
        spec = _auditor_fix_spec(fix)
        if spec is None:
            continue
        kind, target, tier, key, payload = spec
        out.append(Intervention(
            kind, target, tier, key, payload,
            predicted_savings=int(fix.get("estimated_token_savings", 0)),
            predicted_risk=0.15 if target is Target.TOOL_DEF else 0.05,
            rationale=f"auditor fix: {fix.get('fix_type')} ({fix.get('target','')})".strip()))
    return out


def propose(diagnosis: Diagnosis) -> list[Intervention]:
    """Diagnosis -> candidate interventions, deduped by id.

    Combines the auditor's own ``fixes[]`` (when present, e.g. from the real
    Rust backend) with the built-in taxonomy below, keeping the higher
    predicted-savings estimate on collision.

    Covers all five intervention targets: SYSTEM_PROMPT, TOOL_DEF,
    RUNTIME_POLICY, MEMORY_POLICY, DECODING.
    """
    out: dict[str, Intervention] = {}

    def add(iv: Intervention) -> None:
        existing = out.get(iv.id)
        if existing is None or iv.predicted_savings > existing.predicted_savings:
            out[iv.id] = iv

    for iv in from_auditor_fixes(diagnosis):
        add(iv)

    kinds = {p.kind: p for p in diagnosis.patterns}

    if WasteKind.HEDGING in kinds:
        p = kinds[WasteKind.HEDGING]
        add(Intervention(
            WasteKind.HEDGING, Target.SYSTEM_PROMPT, Tier.PROMPT, "NO_HEDGING",
            {"body": "Do not begin responses with preamble or hedging "
                     "(certainly, let me, I'd be happy to, I think, possibly)."},
            predicted_savings=p.est_token_waste, predicted_risk=0.05,
            rationale="Strip sycophantic / hedging preamble (SHL)."))

    if WasteKind.VERBOSITY in kinds:
        p = kinds[WasteKind.VERBOSITY]
        add(Intervention(
            WasteKind.VERBOSITY, Target.SYSTEM_PROMPT, Tier.PROMPT, "EFFICIENCY_RULES",
            {"body": "Be terse. No filler (basically, essentially, to be honest). "
                     "State only task-relevant facts."},
            predicted_savings=p.est_token_waste, predicted_risk=0.05,
            rationale="Raise information density (VDI)."))
        # DECODING: temperature=0.0 forces greedy decoding, reducing verbosity.
        add(Intervention(
            WasteKind.VERBOSITY, Target.DECODING, Tier.INLINE, "decoding_temperature",
            {"value": {"temperature": 0.0, "top_p": 0.9}},
            predicted_savings=max(p.est_token_waste // 4, 1), predicted_risk=0.1,
            rationale="Greedy decoding (temp=0) reduces verbose sampling output (VDI)."))

    if WasteKind.REDUNDANT_STEP in kinds:
        p = kinds[WasteKind.REDUNDANT_STEP]
        add(Intervention(
            WasteKind.REDUNDANT_STEP, Target.SYSTEM_PROMPT, Tier.PROMPT, "NO_REFORMULATION",
            {"body": "Do not restate or re-read the request. Use context already in memory."},
            predicted_savings=p.est_token_waste, predicted_risk=0.1,
            rationale="Remove reformulation / near-duplicate steps (SRR/CCE)."))
        # MEMORY_POLICY: limit history window to avoid carrying redundant context.
        add(Intervention(
            WasteKind.REDUNDANT_STEP, Target.MEMORY_POLICY, Tier.INLINE, "history_limit",
            {"value": {"max_turns": 10}},
            predicted_savings=max(p.est_token_waste // 3, 1), predicted_risk=0.05,
            rationale="Limit history window to prune redundant context (CCE)."))

    if WasteKind.CONTEXT_BLOAT in kinds:
        p = kinds[WasteKind.CONTEXT_BLOAT]
        add(Intervention(
            WasteKind.CONTEXT_BLOAT, Target.SYSTEM_PROMPT, Tier.PROMPT, "CONTEXT_COMPRESSION",
            {"body": "Summarise previous context in one sentence before each new step."},
            predicted_savings=p.est_token_waste, predicted_risk=0.1,
            rationale="Compress redundant context (CCE)."))
        # MEMORY_POLICY: summarise context instead of carrying it verbatim.
        add(Intervention(
            WasteKind.CONTEXT_BLOAT, Target.MEMORY_POLICY, Tier.INLINE, "summarize_context",
            {"value": {"summarize": True, "max_summary_tokens": 200}},
            predicted_savings=max(p.est_token_waste // 2, 1), predicted_risk=0.05,
            rationale="Summarise context window to reduce bloat (CCE)."))

    if WasteKind.LOOP in kinds:
        p = kinds[WasteKind.LOOP]
        add(Intervention(
            WasteKind.LOOP, Target.RUNTIME_POLICY, Tier.INLINE, "loop_breaker",
            {"value": {"max_repeats": 1}},
            predicted_savings=p.est_token_waste, predicted_risk=0.05,
            rationale="Runtime guard: block repeating an identical tool call (LDI)."))

    if WasteKind.OVER_DEPTH in kinds:
        p = kinds[WasteKind.OVER_DEPTH]
        # Safe prompt-level budget.
        add(Intervention(
            WasteKind.OVER_DEPTH, Target.SYSTEM_PROMPT, Tier.PROMPT, "STEP_BUDGET",
            {"body": "Take the minimum steps needed. Do not explore unlikely edge cases."},
            predicted_savings=p.est_token_waste, predicted_risk=0.15,
            rationale="Discourage over-deep reasoning (RDA)."))
        # Deliberately risky STRUCT cap -- the gate should reject it when it
        # starves required tool calls. Demonstrates quality preservation.
        add(Intervention(
            WasteKind.OVER_DEPTH, Target.RUNTIME_POLICY, Tier.STRUCT, "step_cap",
            {"value": 2},
            predicted_savings=p.est_token_waste * 3, predicted_risk=0.8,
            rationale="Hard step cap (aggressive; may break multi-tool tasks)."))

    return list(out.values())


def apply(iv: Intervention, cfg: AgentConfig) -> AgentConfig:
    """Pure, deterministic, idempotent edit -> a NEW config."""
    new = cfg.clone()
    if iv.target is Target.SYSTEM_PROMPT:
        new.system_prompt_sections[iv.key] = iv.payload["body"]
    elif iv.target is Target.TOOL_DEF:
        req = set(new.tool_required_params.get(iv.key, []))
        req |= set(iv.payload.get("params", []))
        new.tool_required_params[iv.key] = sorted(req)
    elif iv.target in (Target.RUNTIME_POLICY, Target.DECODING):
        new.policies[iv.key] = iv.payload["value"]
    elif iv.target is Target.MEMORY_POLICY:
        new.policies[iv.key] = iv.payload["value"]
    new.version += 1
    return new
