"""Trace segmentation for TRICE.

The segmenter deliberately starts simple: one segment per agent step, enriched
with stable receipts, identifiers, and a conservative state label. The labels
are policy hints, not truth claims; replay and rollout evidence can update the
policy weights later.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SegmentState(str, Enum):
    ESSENTIAL = "essential"
    REHYDRATABLE = "rehydratable"
    EXPIRED = "expired"
    REDUNDANT = "redundant"
    DISTRACTOR = "distractor"
    UNKNOWN = "unknown"


IDENT_RE = re.compile(
    r"\b(?:[A-Z][A-Z0-9_]{2,}|[A-Za-z_][A-Za-z0-9_]*\.(?:py|rs|ts|js|json|md)|"
    r"[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*|"
    r"[A-Za-z_][A-Za-z0-9_]*\([^)]{0,40}\)|REF-[0-9]+|ORD-[0-9]+)\b"
)

FILLER_TERMS = {
    "let me",
    "basically",
    "essentially",
    "to be honest",
    "actually",
    "think deeply",
    "re-evaluate",
    "double check",
}

MUTATING_TOOL_PARTS = {
    "write",
    "edit",
    "update",
    "delete",
    "remove",
    "create",
    "process",
    "send",
    "commit",
    "refund",
    "book",
    "cancel",
}


@dataclass(frozen=True)
class Segment:
    segment_id: str
    step_id: int
    kind: str
    state: SegmentState
    text: str
    tokens: int
    locked: bool
    receipt: str
    identifiers: tuple[str, ...] = field(default_factory=tuple)
    rehydrate_pointer: str | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "step_id": self.step_id,
            "kind": self.kind,
            "state": self.state.value,
            "tokens": self.tokens,
            "locked": self.locked,
            "receipt": self.receipt,
            "identifiers": list(self.identifiers),
            "rehydrate_pointer": self.rehydrate_pointer,
            "rationale": self.rationale,
        }


def load_trace(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def segments_from_trace(trace: dict[str, Any]) -> list[Segment]:
    steps = trace.get("steps") or []
    retried_successes = _retried_successes(steps)
    goal = _trace_goal(trace)
    out: list[Segment] = []
    for idx, step in enumerate(steps):
        step_id = int(step.get("id") or idx + 1)
        kind = str(step.get("type") or step.get("step_type") or "unknown")
        text = _step_text(step)
        receipt = hashlib.sha256(text.encode("utf-8")).hexdigest()
        identifiers = tuple(sorted(set(IDENT_RE.findall(text))))
        state, rationale = _classify_step(step, idx, len(steps), retried_successes, goal, text)
        locked = state is SegmentState.ESSENTIAL
        pointer = f"trace:{trace.get('trace_id', 'trace')}:step:{step_id}"
        out.append(
            Segment(
                segment_id=f"s{step_id}",
                step_id=step_id,
                kind=kind,
                state=state,
                text=text,
                tokens=max(int(step.get("tokens") or _estimate_tokens(text)), 1),
                locked=locked,
                receipt=receipt,
                identifiers=identifiers,
                rehydrate_pointer=pointer,
                rationale=rationale,
            )
        )
    return out


def _trace_goal(trace: dict[str, Any]) -> str:
    metadata = trace.get("metadata") or {}
    for key in ("task", "goal", "objective", "user_request", "query", "question", "instruction"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if trace.get("steps"):
        return str(trace["steps"][0].get("input_context") or trace["steps"][0].get("content") or "")
    return ""


def _step_text(step: dict[str, Any]) -> str:
    parts = [
        str(step.get("content") or ""),
        str(step.get("output") or ""),
        str(step.get("input_context") or ""),
    ]
    if step.get("tool_name"):
        parts.append(f"tool:{step['tool_name']}")
    if step.get("tool_error"):
        parts.append(f"error:{step['tool_error']}")
    if step.get("tool_params"):
        parts.append(json.dumps(step["tool_params"], sort_keys=True))
    return "\n".join(p for p in parts if p).strip()


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.35))


def _retried_successes(steps: list[dict[str, Any]]) -> set[tuple[int, str]]:
    successes: dict[str, list[int]] = {}
    failures: list[tuple[int, str]] = []
    for idx, step in enumerate(steps):
        tool = str(step.get("tool_name") or "")
        if not tool:
            continue
        if step.get("tool_success") is False:
            failures.append((idx, tool))
        elif step.get("tool_success") is True:
            successes.setdefault(tool, []).append(idx)
    expired = set()
    for idx, tool in failures:
        if any(j > idx for j in successes.get(tool, [])):
            expired.add((idx, tool))
    return expired


def _classify_step(
    step: dict[str, Any],
    idx: int,
    n_steps: int,
    retried_successes: set[tuple[int, str]],
    goal: str,
    text: str,
) -> tuple[SegmentState, str]:
    flags = {str(f).upper() for f in step.get("flags", [])}
    tool = str(step.get("tool_name") or "")
    tool_l = tool.lower()
    text_l = text.lower()
    is_mutating = any(part in tool_l for part in MUTATING_TOOL_PARTS)

    if (idx, tool) in retried_successes:
        return SegmentState.EXPIRED, "failed tool call followed by a later successful retry"
    if step.get("tool_success") is False and (idx, tool) not in retried_successes:
        return SegmentState.ESSENTIAL, "unresolved failure/error evidence"
    if idx == 0:
        return SegmentState.ESSENTIAL, "first step anchors the task"
    if idx == n_steps - 1:
        return SegmentState.ESSENTIAL, "final state/answer is a quality anchor"
    if is_mutating and step.get("tool_success") is True:
        return SegmentState.ESSENTIAL, "successful mutating tool call"
    if flags & {"REDUNDANT", "LOOP", "REFORMULATION"}:
        return SegmentState.REDUNDANT, "TraceRazor flagged redundant or looping behavior"
    if tool and step.get("tool_success") is True:
        return SegmentState.REHYDRATABLE, "successful read/tool observation can be re-fetched"
    if _looks_redundant(text_l):
        return SegmentState.REDUNDANT, "lexical reformulation or repeated meta-reasoning"
    if _looks_distracting(text_l, goal):
        return SegmentState.DISTRACTOR, "low task overlap with filler-heavy language"
    return SegmentState.UNKNOWN, "no conservative state rule matched"


def _looks_redundant(text_l: str) -> bool:
    return (
        "parse the user request again" in text_l
        or "re-evaluating whether" in text_l
        or "re-evaluate whether" in text_l
        or text_l.count("refund") > 8 and "again" in text_l
    )


def _looks_distracting(text_l: str, goal: str) -> bool:
    filler_hits = sum(1 for term in FILLER_TERMS if term in text_l)
    if filler_hits >= 2:
        return True
    if not goal:
        return False
    goal_words = {w for w in re.findall(r"[a-z0-9_]{4,}", goal.lower()) if len(w) > 3}
    text_words = set(re.findall(r"[a-z0-9_]{4,}", text_l))
    if not goal_words:
        return False
    overlap = len(goal_words & text_words) / len(goal_words)
    return overlap < 0.15 and len(text_words) > 20
