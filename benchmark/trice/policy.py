"""Budgeted policy solver for TRICE."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .score import ActionCandidate, ScoreWeights, action_candidates
from .segment import Segment

BUCKET_SIZE = 128


@dataclass(frozen=True)
class PolicyDecision:
    segment_id: str
    step_id: int
    state: str
    action: str
    original_tokens: int
    policy_tokens: int
    locked: bool
    receipt: str
    rehydrate_pointer: str | None
    value: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ContextPolicy:
    algorithm: str
    budget_ratio: float
    bucket_size: int
    baseline_input_tokens: int
    budget_tokens: int
    policy_tokens: int
    projected_input_savings_pct: float
    budget_exceeded: bool
    decisions: tuple[PolicyDecision, ...]
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "budget_ratio": self.budget_ratio,
            "bucket_size": self.bucket_size,
            "baseline_input_tokens": self.baseline_input_tokens,
            "budget_tokens": self.budget_tokens,
            "policy_tokens": self.policy_tokens,
            "projected_input_savings_pct": self.projected_input_savings_pct,
            "budget_exceeded": self.budget_exceeded,
            "constraints": self.constraints,
            "decisions": [d.to_dict() for d in self.decisions],
        }


def solve_policy(
    segments: list[Segment],
    budget_ratio: float = 0.40,
    weights: ScoreWeights | None = None,
) -> ContextPolicy:
    weights = weights or ScoreWeights()
    baseline_tokens = sum(s.tokens for s in segments)
    requested_budget = max(1, int(round(baseline_tokens * budget_ratio)))
    candidates = [action_candidates(s, weights) for s in segments]
    min_required = sum(min(c.tokens for c in cs) for cs in candidates)
    effective_budget = max(requested_budget, min_required)
    budget_buckets = _buckets(effective_budget)

    # Multi-choice knapsack: exactly one action per segment.
    dp: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
    for cs in candidates:
        ndp: dict[int, tuple[float, list[int]]] = {}
        for used, (value, picks) in dp.items():
            for i, cand in enumerate(cs):
                nxt = used + _buckets(cand.tokens)
                if nxt > budget_buckets:
                    continue
                score = value + cand.value
                if nxt not in ndp or score > ndp[nxt][0]:
                    ndp[nxt] = (score, picks + [i])
        dp = ndp

    if not dp:
        picks = [min(range(len(cs)), key=lambda i: cs[i].tokens) for cs in candidates]
    else:
        _, picks = max(dp.values(), key=lambda item: item[0])

    decisions = tuple(_decision(segment, candidates[i][pick]) for i, (segment, pick) in enumerate(zip(segments, picks)))
    policy_tokens = sum(d.policy_tokens for d in decisions)
    savings_pct = 0.0 if baseline_tokens == 0 else 100.0 * (baseline_tokens - policy_tokens) / baseline_tokens
    return ContextPolicy(
        algorithm="trice-v0.1-multi-choice-knapsack",
        budget_ratio=budget_ratio,
        bucket_size=BUCKET_SIZE,
        baseline_input_tokens=baseline_tokens,
        budget_tokens=requested_budget,
        policy_tokens=policy_tokens,
        projected_input_savings_pct=round(savings_pct, 2),
        budget_exceeded=policy_tokens > requested_budget,
        decisions=decisions,
        constraints={
            "evidence_recall_min": 0.95,
            "pass_rate_noninferiority_pp": -2,
            "locked_anchors_unchanged": True,
        },
    )


def _decision(segment: Segment, candidate: ActionCandidate) -> PolicyDecision:
    return PolicyDecision(
        segment_id=segment.segment_id,
        step_id=segment.step_id,
        state=segment.state.value,
        action=candidate.action,
        original_tokens=segment.tokens,
        policy_tokens=candidate.tokens,
        locked=segment.locked,
        receipt=segment.receipt,
        rehydrate_pointer=segment.rehydrate_pointer,
        value=round(candidate.value, 6),
        rationale=candidate.rationale,
    )


def _buckets(tokens: int) -> int:
    return max(1, int(math.ceil(tokens / BUCKET_SIZE)))
