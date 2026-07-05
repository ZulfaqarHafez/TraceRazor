"""TRICE segment and action scoring."""

from __future__ import annotations

from dataclasses import dataclass

from .segment import Segment, SegmentState


@dataclass(frozen=True)
class ScoreWeights:
    risk_lambda: float = 1.4
    cost_mu: float = 0.8
    cache_rho: float = 0.5
    hallucination_gamma: float = 1.1
    input_price_per_1k: float = 0.003
    prefill_alpha: float = 0.000002
    kv_beta: float = 0.000001


@dataclass(frozen=True)
class SegmentScore:
    utility: float
    risk: float
    cost: float
    cache: float
    hallucination: float

    def value(self, weights: ScoreWeights) -> float:
        return (
            self.utility
            - weights.risk_lambda * self.risk
            - weights.cost_mu * self.cost
            + weights.cache_rho * self.cache
            - weights.hallucination_gamma * self.hallucination
        )


@dataclass(frozen=True)
class ActionCandidate:
    action: str
    tokens: int
    score: SegmentScore
    value: float
    rationale: str


def score_segment(segment: Segment, weights: ScoreWeights | None = None) -> SegmentScore:
    weights = weights or ScoreWeights()
    utility = {
        SegmentState.ESSENTIAL: 1.0,
        SegmentState.REHYDRATABLE: 0.72,
        SegmentState.UNKNOWN: 0.58,
        SegmentState.EXPIRED: 0.24,
        SegmentState.REDUNDANT: 0.18,
        SegmentState.DISTRACTOR: 0.12,
    }[segment.state]
    if segment.identifiers:
        utility += min(0.18, 0.03 * len(segment.identifiers))

    risk = {
        SegmentState.ESSENTIAL: 1.0,
        SegmentState.UNKNOWN: 0.55,
        SegmentState.REHYDRATABLE: 0.30,
        SegmentState.EXPIRED: 0.16,
        SegmentState.REDUNDANT: 0.12,
        SegmentState.DISTRACTOR: 0.10,
    }[segment.state]
    cache = 0.35 if segment.step_id <= 2 or segment.locked else 0.08
    hallucination = 0.75 if segment.state in {SegmentState.ESSENTIAL, SegmentState.UNKNOWN} else 0.25
    return SegmentScore(
        utility=min(1.2, utility),
        risk=risk,
        cost=_cost(segment.tokens, weights),
        cache=cache,
        hallucination=hallucination,
    )


def action_candidates(segment: Segment, weights: ScoreWeights | None = None) -> list[ActionCandidate]:
    weights = weights or ScoreWeights()
    base = score_segment(segment, weights)
    actions: list[tuple[str, int, SegmentScore, str]] = []

    if segment.locked:
        action = "anchor_prefix" if segment.step_id <= 2 else "keep"
        boosted = SegmentScore(base.utility, 0.0, _cost(segment.tokens, weights), base.cache + 0.35, 0.0)
        actions.append((action, segment.tokens, boosted, "locked anchor kept byte-for-byte"))
    else:
        actions.append(("keep", segment.tokens, base, "full segment retained"))
        actions.append(("extract", _ratio_tokens(segment.tokens, 0.45, 32), _transform(base, 0.88, 0.55, 0.55, 0.42, weights), "extractive compression"))
        actions.append(("summarize", _ratio_tokens(segment.tokens, 0.25, 24), _transform(base, 0.70, 0.42, 0.40, 0.70, weights), "short natural-language state summary"))
        actions.append(("lazy_recall", _ratio_tokens(segment.tokens, 0.12, 20), _transform(base, 0.55, 0.35, 0.85, 0.45, weights), "receipt plus rehydration pointer"))
        actions.append(("mask_with_receipt", _ratio_tokens(segment.tokens, 0.07, 12), _transform(base, 0.28, 0.18, 0.55, 0.38, weights), "drop text, retain cryptographic receipt"))

    out: list[ActionCandidate] = []
    for action, tokens, score, rationale in actions:
        out.append(ActionCandidate(action, tokens, score, score.value(weights), rationale))
    return out


def _ratio_tokens(tokens: int, ratio: float, floor: int) -> int:
    if tokens <= floor:
        return max(1, tokens)
    return max(floor, int(tokens * ratio))


def _cost(tokens: int, weights: ScoreWeights) -> float:
    return (
        (tokens / 1000.0) * weights.input_price_per_1k
        + weights.prefill_alpha * tokens
        + weights.kv_beta * tokens
    )


def _transform(
    base: SegmentScore,
    utility_mult: float,
    risk_mult: float,
    cache_mult: float,
    hallucination_mult: float,
    weights: ScoreWeights,
) -> SegmentScore:
    return SegmentScore(
        utility=base.utility * utility_mult,
        risk=base.risk * risk_mult,
        cost=base.cost,
        cache=base.cache * cache_mult,
        hallucination=base.hallucination * hallucination_mult,
    )
