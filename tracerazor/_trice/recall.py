"""Deterministic evidence-recall accounting for TRICE context policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .policy import ContextPolicy, PolicyDecision

RECALL_SCHEMA_VERSION = "trice-evidence-recall/v1"
EVIDENCE_STATES = {"essential"}
INLINE_RECALL_ACTIONS = {"keep", "anchor_prefix", "extract", "summarize"}
POINTER_RECALL_ACTIONS = {"lazy_recall", "mask_with_receipt"}


@dataclass(frozen=True)
class EvidenceRecallReport:
    schema_version: str
    evidence_recall: float
    required_min: float
    passed: bool
    obligation_count: int
    obligation_tokens: int
    recalled_count: int
    recalled_tokens: int
    missing: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_recall": self.evidence_recall,
            "required_min": self.required_min,
            "passed": self.passed,
            "obligation_count": self.obligation_count,
            "obligation_tokens": self.obligation_tokens,
            "recalled_count": self.recalled_count,
            "recalled_tokens": self.recalled_tokens,
            "missing": list(self.missing),
        }


def evidence_recall_from_policy(policy: ContextPolicy | dict[str, Any]) -> EvidenceRecallReport:
    """Measure whether policy output preserves essential evidence obligations."""

    data = policy.to_dict() if isinstance(policy, ContextPolicy) else policy
    constraints = dict(data.get("constraints") or {})
    required_min = float(constraints.get("evidence_recall_min", 0.95))
    decisions = [_decision_from_raw(raw) for raw in data.get("decisions", [])]
    obligations = [decision for decision in decisions if _is_evidence_obligation(decision)]
    obligation_tokens = sum(max(0, int(decision.get("original_tokens") or 0)) for decision in obligations)
    recalled = [decision for decision in obligations if _is_recalled(decision)]
    recalled_tokens = sum(max(0, int(decision.get("original_tokens") or 0)) for decision in recalled)
    missing = tuple(_missing_row(decision) for decision in obligations if not _is_recalled(decision))
    recall = 1.0 if obligation_tokens == 0 else recalled_tokens / obligation_tokens
    recall = round(recall, 6)
    return EvidenceRecallReport(
        schema_version=RECALL_SCHEMA_VERSION,
        evidence_recall=recall,
        required_min=required_min,
        passed=recall + 1e-12 >= required_min,
        obligation_count=len(obligations),
        obligation_tokens=obligation_tokens,
        recalled_count=len(recalled),
        recalled_tokens=recalled_tokens,
        missing=missing,
    )


def _decision_from_raw(raw: PolicyDecision | dict[str, Any]) -> dict[str, Any]:
    return raw.to_dict() if isinstance(raw, PolicyDecision) else dict(raw)


def _is_evidence_obligation(decision: dict[str, Any]) -> bool:
    return bool(decision.get("locked")) or str(decision.get("state") or "") in EVIDENCE_STATES


def _is_recalled(decision: dict[str, Any]) -> bool:
    action = str(decision.get("action") or "")
    if action in INLINE_RECALL_ACTIONS:
        return True
    if action in POINTER_RECALL_ACTIONS:
        return bool(decision.get("receipt")) and bool(decision.get("rehydrate_pointer"))
    return False


def _missing_row(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": decision.get("segment_id"),
        "step_id": decision.get("step_id"),
        "state": decision.get("state"),
        "action": decision.get("action"),
        "locked": bool(decision.get("locked")),
        "original_tokens": int(decision.get("original_tokens") or 0),
        "reason": "essential evidence was neither inline-retained nor recallable by receipt and pointer",
    }
