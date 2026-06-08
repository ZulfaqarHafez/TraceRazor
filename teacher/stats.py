"""Statistical quality-preservation gate for online verification (P2-B).

Upgrades the lightweight mean-comparison gate to a real test:

  * token savings  -- bootstrap a 90% CI for the mean reduction; require its
    lower bound to clear a minimum effect size.
  * task success   -- a one-sided non-inferiority test on the success
    proportion; require the lower CI bound of (p_trial - p_base) to stay above
    a margin -delta.

Accept iff BOTH hold. A token win that the data can't distinguish from noise,
or that erodes success beyond delta, is rejected. Pure stdlib (no numpy/scipy)
so it runs anywhere.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .schemas import Decision, EvalResult


@dataclass
class GateEvidence:
    n_base: int
    n_trial: int
    mean_tokens_base: float
    mean_tokens_trial: float
    savings_pct_point: float
    savings_pct_lo90: float        # lower bound of the 90% CI (the headline)
    success_base: float
    success_trial: float
    success_delta_lo90: float      # lower bound of (p_trial - p_base)
    decision: Decision

    def summary(self) -> str:
        return (
            f"tokens {self.mean_tokens_base:.0f}->{self.mean_tokens_trial:.0f} "
            f"(save {self.savings_pct_point:.1f}%, 90% CI lo {self.savings_pct_lo90:.1f}%); "
            f"success {self.success_base*100:.0f}%->{self.success_trial*100:.0f}% "
            f"(delta lo90 {self.success_delta_lo90*100:+.1f}pp) => {self.decision.value}")


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _bootstrap_savings_lo(base, trial, conf=0.90, iters=3000, rng=None) -> float:
    """Lower bound of the 90% CI for mean % token savings.

    When base/trial are aligned run-for-run (same holdout, same order) this uses
    a PAIRED bootstrap -- resampling run indices and taking matched (base, trial)
    pairs -- which controls for task heterogeneity and tightens the interval.
    Falls back to an unpaired bootstrap for unequal-length samples.
    """
    rng = rng or random.Random(12345)
    if not base or not trial:
        return 0.0
    samples = []
    if len(base) == len(trial):
        n = len(base)
        for _ in range(iters):
            idx = [rng.randrange(n) for _ in range(n)]
            mb = _mean([base[i] for i in idx])
            mt = _mean([trial[i] for i in idx])
            samples.append(100.0 * (mb - mt) / mb if mb else 0.0)
    else:
        nb, nt = len(base), len(trial)
        for _ in range(iters):
            mb = _mean([base[rng.randrange(nb)] for _ in range(nb)])
            mt = _mean([trial[rng.randrange(nt)] for _ in range(nt)])
            samples.append(100.0 * (mb - mt) / mb if mb else 0.0)
    samples.sort()
    lo_idx = int((1 - conf) / 2 * iters)
    return samples[lo_idx]


def _noninferiority_lo(succ_trial, succ_base, z=1.645) -> float:
    """One-sided lower bound (90%) of (p_trial - p_base), normal approximation."""
    if not succ_trial or not succ_base:
        return 0.0
    p1, n1 = _mean([1.0 if s else 0.0 for s in succ_trial]), len(succ_trial)
    p0, n0 = _mean([1.0 if s else 0.0 for s in succ_base]), len(succ_base)
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return (p1 - p0) - z * se


@dataclass
class StatGate:
    """Statistical quality-preservation gate."""
    min_savings_pct: float = 3.0   # required lower-CI savings
    success_delta: float = 0.05    # non-inferiority margin (5pp)
    bootstrap_iters: int = 3000
    seed: int = 12345

    def evaluate(self, base: EvalResult, trial: EvalResult) -> GateEvidence:
        rng = random.Random(self.seed)
        sav_lo = _bootstrap_savings_lo(base.tokens, trial.tokens,
                                       iters=self.bootstrap_iters, rng=rng)
        sav_pt = (100.0 * (base.mean_tokens - trial.mean_tokens) / base.mean_tokens
                  if base.mean_tokens else 0.0)
        succ_lo = _noninferiority_lo(trial.success, base.success)

        if succ_lo <= -self.success_delta:
            decision = Decision.REJECT_QUALITY
        elif sav_lo < self.min_savings_pct:
            decision = Decision.REJECT_NO_GAIN
        else:
            decision = Decision.ACCEPT

        return GateEvidence(
            n_base=len(base.tokens), n_trial=len(trial.tokens),
            mean_tokens_base=base.mean_tokens, mean_tokens_trial=trial.mean_tokens,
            savings_pct_point=sav_pt, savings_pct_lo90=sav_lo,
            success_base=base.success_rate, success_trial=trial.success_rate,
            success_delta_lo90=succ_lo, decision=decision)

    def decide(self, base: EvalResult, trial: EvalResult) -> Decision:
        return self.evaluate(base, trial).decision
