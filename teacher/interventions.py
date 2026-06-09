"""Intervention synthesis + application.

``propose`` maps a ``Diagnosis`` to typed, applicable ``Intervention`` objects
(the taxonomy from P2-A). ``apply`` performs a deterministic, idempotent,
reversible edit on an ``AgentConfig``.

Each waste pattern is paired with the *cheapest effective* intervention tier;
the Teacher's curriculum tries tiers in order. One deliberately over-aggressive
candidate (a STRUCT step-cap that can starve real tool calls) is included so
the quality gate has something genuine to reject.
"""
from __future__ import annotations

import re

from .schemas import (
    AgentConfig,
    Diagnosis,
    Intervention,
    Target,
    Tier,
    WasteKind,
)


# Map the auditor's own fix_type strings onto applicable interventions.
# fix_type -> (WasteKind, Target, Tier, section/policy key, payload-builder)
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

    if WasteKind.REDUNDANT_STEP in kinds:
        p = kinds[WasteKind.REDUNDANT_STEP]
        add(Intervention(
            WasteKind.REDUNDANT_STEP, Target.SYSTEM_PROMPT, Tier.PROMPT, "NO_REFORMULATION",
            {"body": "Do not restate or re-read the request. Use context already in memory."},
            predicted_savings=p.est_token_waste, predicted_risk=0.1,
            rationale="Remove reformulation / near-duplicate steps (SRR/CCE)."))

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
    new.version += 1
    return new
