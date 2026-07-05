"""Render TRICE policies and compressed context artifacts."""

from __future__ import annotations

import json
from typing import Any

from .policy import ContextPolicy
from .segment import Segment


def render_policy_json(policy: ContextPolicy) -> str:
    return json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n"


def render_context(policy: ContextPolicy, segments: list[Segment]) -> str:
    by_id = {s.segment_id: s for s in segments}
    lines = [
        "TRICE_CONTEXT_POLICY",
        f"algorithm={policy.algorithm}",
        f"projected_input_savings_pct={policy.projected_input_savings_pct:.2f}",
        "",
    ]
    for decision in policy.decisions:
        segment = by_id[decision.segment_id]
        lines.append(f"[{decision.segment_id} step={decision.step_id} action={decision.action} state={decision.state}]")
        lines.append(_render_segment_text(segment, decision.action))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def policy_from_dict(data: dict[str, Any]) -> ContextPolicy:
    from .policy import PolicyDecision

    return ContextPolicy(
        algorithm=data["algorithm"],
        budget_ratio=float(data["budget_ratio"]),
        bucket_size=int(data.get("bucket_size", 128)),
        baseline_input_tokens=int(data["baseline_input_tokens"]),
        budget_tokens=int(data["budget_tokens"]),
        policy_tokens=int(data["policy_tokens"]),
        projected_input_savings_pct=float(data["projected_input_savings_pct"]),
        budget_exceeded=bool(data.get("budget_exceeded", False)),
        constraints=dict(data.get("constraints") or {}),
        decisions=tuple(PolicyDecision(**d) for d in data.get("decisions", [])),
    )


def _render_segment_text(segment: Segment, action: str) -> str:
    if action in {"keep", "anchor_prefix"}:
        return segment.text
    if action == "extract":
        first = segment.text.splitlines()[0][:400]
        ids = ", ".join(segment.identifiers[:8]) if segment.identifiers else "none"
        return f"extract: {first}\nidentifiers: {ids}"
    if action == "summarize":
        words = segment.text.split()
        summary = " ".join(words[:28])
        if len(words) > 28:
            summary += " ..."
        ids = ", ".join(segment.identifiers[:8]) if segment.identifiers else "none"
        return f"summary: {summary}\nidentifiers: {ids}"
    if action == "lazy_recall":
        return (
            f"lazy_recall receipt={segment.receipt[:16]} "
            f"pointer={segment.rehydrate_pointer} state={segment.state.value}"
        )
    if action == "mask_with_receipt":
        return f"masked receipt={segment.receipt[:16]} pointer={segment.rehydrate_pointer}"
    raise ValueError(f"unknown TRICE action: {action}")
