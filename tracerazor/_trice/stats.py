"""Deterministic statistics and claim gates for TRICE evidence."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable


DEFAULT_BOOTSTRAP_SEED = 20260621


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    mean: float
    high: float
    level: float = 0.95

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ClaimGate:
    scope: str
    target_savings: float
    mean_savings: float
    savings_ci: ConfidenceInterval
    baseline_pass_rate: float
    trice_pass_rate: float
    trice_pass_ci: ConfidenceInterval
    pass_regressions: int
    evidence_recall_minimum: float
    evidence_recall_required: float
    evidence_recall_failures: int
    accepted_rounds: int
    total_rounds: int
    smoke_gate_passed: bool
    broad_claim_allowed: bool
    rationale: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["savings_ci"] = self.savings_ci.to_dict()
        data["trice_pass_ci"] = self.trice_pass_ci.to_dict()
        return data


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    iterations: int = 5000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    level: float = 0.95,
) -> ConfidenceInterval:
    xs = list(values)
    if not xs:
        return ConfidenceInterval(0.0, 0.0, 0.0, level)
    if len(xs) == 1:
        return ConfidenceInterval(xs[0], xs[0], xs[0], level)
    rng = random.Random(seed)
    n = len(xs)
    samples = []
    for _ in range(iterations):
        samples.append(mean(xs[rng.randrange(n)] for _ in range(n)))
    samples.sort()
    alpha = (1.0 - level) / 2.0
    return ConfidenceInterval(
        low=round(_quantile(samples, alpha), 6),
        mean=round(mean(xs), 6),
        high=round(_quantile(samples, 1.0 - alpha), 6),
        level=level,
    )


def clustered_bootstrap_mean_ci(
    clusters: dict[str, Iterable[float]],
    *,
    iterations: int = 5000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    level: float = 0.95,
) -> ConfidenceInterval:
    """Bootstrap a mean by resampling task/repo clusters, not individual runs."""

    grouped: dict[str, list[float]] = {}
    for key, values in clusters.items():
        xs = list(values)
        if xs:
            grouped[key] = xs
    flat = [value for values in grouped.values() for value in values]
    if not flat:
        return ConfidenceInterval(0.0, 0.0, 0.0, level)
    if len(grouped) <= 1:
        return ConfidenceInterval(round(mean(flat), 6), round(mean(flat), 6), round(mean(flat), 6), level)
    keys = sorted(grouped)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        selected: list[float] = []
        for _ in keys:
            selected.extend(grouped[rng.choice(keys)])
        samples.append(mean(selected))
    samples.sort()
    alpha = (1.0 - level) / 2.0
    return ConfidenceInterval(
        low=round(_quantile(samples, alpha), 6),
        mean=round(mean(flat), 6),
        high=round(_quantile(samples, 1.0 - alpha), 6),
        level=level,
    )


def wilson_ci(successes: int, total: int, *, level: float = 0.95) -> ConfidenceInterval:
    if total <= 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, level)
    z = 1.959963984540054 if abs(level - 0.95) < 1e-9 else 1.959963984540054
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    return ConfidenceInterval(
        low=round(max(0.0, centre - radius), 6),
        mean=round(phat, 6),
        high=round(min(1.0, centre + radius), 6),
        level=level,
    )


def claim_gate_from_rounds(rounds: list, target_savings: float = 0.60) -> ClaimGate:
    savings = [float(_field(r, "measured_input_savings")) for r in rounds]
    baseline_pass = [_condition(_field(r, "baseline")).passed for r in rounds]
    trice_pass = [_condition(_field(r, "optimized")).passed for r in rounds]
    accepted = [bool(_field(r, "accepted")) for r in rounds]
    evidence_recalls = [_evidence_recall(_field(r, "optimized")) for r in rounds]
    evidence_recall_required = 0.95
    evidence_recall_failures = sum(1 for value in evidence_recalls if value + 1e-12 < evidence_recall_required)
    regressions = sum(1 for b, t in zip(baseline_pass, trice_pass) if b and not t)
    savings_ci = bootstrap_mean_ci(savings)
    trice_ci = wilson_ci(sum(trice_pass), len(trice_pass))
    smoke_gate = (
        bool(rounds)
        and savings_ci.low >= target_savings
        and regressions == 0
        and all(accepted)
        and all(trice_pass)
        and evidence_recall_failures == 0
    )
    broad = False
    if smoke_gate:
        rationale = (
            "local deterministic smoke passed; broad claim still requires held-out provider runs "
            "with repeated trials and clustered confidence intervals"
        )
    else:
        rationale = "local deterministic smoke did not clear the savings/pass preservation gate"
    return ClaimGate(
        scope="local_deterministic_smoke",
        target_savings=target_savings,
        mean_savings=round(mean(savings) if savings else 0.0, 6),
        savings_ci=savings_ci,
        baseline_pass_rate=round(sum(baseline_pass) / len(baseline_pass), 6) if baseline_pass else 0.0,
        trice_pass_rate=round(sum(trice_pass) / len(trice_pass), 6) if trice_pass else 0.0,
        trice_pass_ci=trice_ci,
        pass_regressions=regressions,
        evidence_recall_minimum=round(min(evidence_recalls), 6) if evidence_recalls else 1.0,
        evidence_recall_required=evidence_recall_required,
        evidence_recall_failures=evidence_recall_failures,
        accepted_rounds=sum(accepted),
        total_rounds=len(rounds),
        smoke_gate_passed=smoke_gate,
        broad_claim_allowed=broad,
        rationale=rationale,
    )


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


@dataclass(frozen=True)
class _ConditionView:
    passed: bool


def _condition(obj) -> _ConditionView:
    if isinstance(obj, dict):
        return _ConditionView(bool(obj["passed"]))
    return _ConditionView(bool(obj.passed))


def _field(obj, key: str):
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def _optional_field(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _evidence_recall(condition) -> float:
    value = _optional_field(condition, "evidence_recall", None)
    if value is not None:
        return float(value)
    report = _optional_field(condition, "evidence_recall_report", None)
    if isinstance(report, dict) and report.get("evidence_recall") is not None:
        return float(report["evidence_recall"])
    return 1.0
