"""TRICE online policy-weight update."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LearningWeights:
    utility: float = 1.0
    risk: float = 1.4
    cost: float = 0.8
    cache: float = 0.5
    hallucination: float = 1.1

    def as_vector(self) -> tuple[float, float, float, float, float]:
        return (self.utility, self.risk, self.cost, self.cache, self.hallucination)


@dataclass(frozen=True)
class PolicyUpdate:
    reward: float
    prediction: float
    error: float
    weights: LearningWeights


def update_weights(
    weights: LearningWeights,
    features: tuple[float, float, float, float, float],
    measured_input_savings: float,
    quality_drop: float,
    evidence_recall_failure: float,
    compression_overhead: float,
    learning_rate: float = 0.08,
    noninferiority_delta: float = 0.02,
) -> PolicyUpdate:
    reward = (
        measured_input_savings
        - 2.0 * max(0.0, quality_drop - noninferiority_delta)
        - 0.35 * compression_overhead
        - 1.5 * evidence_recall_failure
    )
    vector = weights.as_vector()
    prediction = sum(w * x for w, x in zip(vector, features))
    error = reward - prediction
    updated = tuple(max(0.0, w + learning_rate * error * x) for w, x in zip(vector, features))
    return PolicyUpdate(
        reward=round(reward, 6),
        prediction=round(prediction, 6),
        error=round(error, 6),
        weights=LearningWeights(*updated),
    )
