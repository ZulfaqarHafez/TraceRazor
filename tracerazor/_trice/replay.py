"""Recorded-trace replay metrics for TRICE policies."""

from __future__ import annotations

from dataclasses import dataclass

from .policy import ContextPolicy
from .segment import Segment


@dataclass(frozen=True)
class ReplayMetrics:
    evidence_recall: float
    action_divergence: float
    expired_info_retention: float
    rehydration_success: float
    compression_overhead: float
    pass_noninferior: bool

    def to_dict(self) -> dict[str, float | bool]:
        return self.__dict__.copy()


def evaluate_policy(segments: list[Segment], policy: ContextPolicy) -> ReplayMetrics:
    by_id = {s.segment_id: s for s in segments}
    decisions = list(policy.decisions)
    required = {i for d in decisions for i in by_id[d.segment_id].identifiers if d.locked}
    available = set()
    lazy_receipts = set()
    destructive_changes = 0
    expired_original = 0
    expired_kept = 0
    lazy_total = 0
    lazy_valid = 0

    for d in decisions:
        segment = by_id[d.segment_id]
        if d.action in {"keep", "anchor_prefix", "extract", "summarize"}:
            available.update(segment.identifiers)
        elif d.action == "lazy_recall":
            lazy_receipts.add(d.receipt)
            lazy_total += 1
            if segment.receipt == d.receipt and segment.rehydrate_pointer:
                lazy_valid += 1
        if d.locked and d.action not in {"keep", "anchor_prefix"}:
            destructive_changes += 1
        if d.state == "expired":
            expired_original += d.original_tokens
            if d.action in {"keep", "extract", "summarize", "anchor_prefix"}:
                expired_kept += d.policy_tokens

    recall = 1.0 if not required else len(required & available) / len(required)
    divergence = destructive_changes / max(1, len([d for d in decisions if d.locked]))
    expired_retention = 0.0 if expired_original == 0 else expired_kept / expired_original
    rehydration = 1.0 if lazy_total == 0 else lazy_valid / lazy_total
    overhead = policy.policy_tokens / max(1, policy.baseline_input_tokens)
    return ReplayMetrics(
        evidence_recall=round(recall, 4),
        action_divergence=round(divergence, 4),
        expired_info_retention=round(expired_retention, 4),
        rehydration_success=round(rehydration, 4),
        compression_overhead=round(overhead, 4),
        pass_noninferior=recall >= 0.95 and divergence == 0.0,
    )
