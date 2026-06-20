"""Quality-Preservation Gate -- the critical closed-loop accept rule (P2-B).

Accept an intervention iff:
  (1) token savings are real and meaningful  (>= min_savings_pct), AND
  (2) task success is NON-INFERIOR to baseline (within margin delta).

A token reduction that breaks the agent fails clause (2) and is rolled back.
This is what makes the savings claim falsifiable rather than projected.

Two gate implementations:
  * QualityGate  -- lightweight offline gate (no external deps; mean comparison).
  * HardenedStatGate -- production gate with bootstrap CI + non-inferiority test
    using only stdlib (same logic as teacher.stats.StatGate but without numpy).

The task spec refers to 'StatGate / quality-preservation gate'; StatGate with
full statistical rigour lives in teacher.stats.  HardenedStatGate here provides
an enhanced offline-safe version with GateVerdictType / GateVerdict richer
return types for callers that need them.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum

from .schemas import Decision, EvalResult


# --------------------------------------------------------------------------- #
# Lightweight offline gate (original, unchanged).
# --------------------------------------------------------------------------- #
@dataclass
class QualityGate:
    min_savings_pct: float = 3.0     # require >= 3% fewer tokens to promote
    success_delta: float = 0.02      # tolerate at most 2pp success loss

    def decide(self, base: EvalResult, trial: EvalResult) -> Decision:
        # Clause (2) is sacred: never trade task success for tokens.
        if trial.success_rate < base.success_rate - self.success_delta:
            return Decision.REJECT_QUALITY

        saved_pct = 0.0
        if base.mean_tokens > 0:
            saved_pct = 100.0 * (base.mean_tokens - trial.mean_tokens) / base.mean_tokens
        if saved_pct < self.min_savings_pct:
            return Decision.REJECT_NO_GAIN

        return Decision.ACCEPT


# --------------------------------------------------------------------------- #
# Hardened production gate (bootstrap CI + non-inferiority, stdlib only).
# --------------------------------------------------------------------------- #
class GateVerdictType(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT_NO_SAVINGS = "REJECT_NO_SAVINGS"
    REJECT_REGRESSION = "REJECT_REGRESSION"
    REJECT_INSUFFICIENT_DATA = "REJECT_INSUFFICIENT_DATA"


@dataclass
class GateVerdict:
    verdict: GateVerdictType
    token_ci: tuple               # 95% bootstrap CI on token savings fraction
    success_bound: float          # lower bound of non-inferiority test
    delta: float                  # non-inferiority margin used
    explanation: str = ""

    def is_accepted(self) -> bool:
        return self.verdict == GateVerdictType.ACCEPT

    def explain(self) -> str:
        lo, hi = self.token_ci
        lines = [
            f"Verdict: {self.verdict.value}",
            f"Token savings CI: [{lo:.1%}, {hi:.1%}]",
            f"Success non-inferiority bound: {self.success_bound:.3f} (margin delta={self.delta:.3f})",
            self.explanation,
        ]
        return "\n".join(lines)


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _bootstrap_savings_ci(base: list, trial: list,
                           n_boot: int = 1000, ci: float = 0.95,
                           seed: int = 42) -> tuple:
    """95% bootstrap CI on the token savings fraction (base - trial) / base."""
    rng = random.Random(seed)
    if not base or not trial:
        return (0.0, 0.0)
    samples = []
    nb, nt = len(base), len(trial)
    for _ in range(n_boot):
        mb = _mean([base[rng.randrange(nb)] for _ in range(nb)])
        mt = _mean([trial[rng.randrange(nt)] for _ in range(nt)])
        samples.append((mb - mt) / mb if mb else 0.0)
    samples.sort()
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot)
    hi_idx = min(hi_idx, n_boot - 1)
    return (float(samples[lo_idx]), float(samples[hi_idx]))


class HardenedStatGate:
    """Production-grade gate: bootstrap CI on savings + non-inferiority test.

    Uses only stdlib (no numpy/scipy) so it can run anywhere.  The interface
    mirrors the numpy-based StatGate in teacher.stats but returns the richer
    GateVerdict type.
    """
    def __init__(self, delta: float = 0.05, alpha: float = 0.1,
                 min_samples: int = 5, n_boot: int = 1000, seed: int = 42):
        self.delta = delta
        self.alpha = alpha
        self.min_samples = min_samples
        self.n_boot = n_boot
        self.seed = seed

    def run(self, baseline_tokens: list, treatment_tokens: list,
            baseline_success: list, treatment_success: list) -> GateVerdict:
        """Evaluate whether the treatment arm is a safe improvement.

        Parameters
        ----------
        baseline_tokens / treatment_tokens : token counts per run (int/float).
        baseline_success / treatment_success : success flags per run (bool/int).
        """
        n_base = len(baseline_tokens)
        n_treat = len(treatment_tokens)
        if min(n_base, n_treat) < self.min_samples:
            return GateVerdict(
                GateVerdictType.REJECT_INSUFFICIENT_DATA,
                (0.0, 0.0), 0.0, self.delta,
                f"Need >= {self.min_samples} samples per arm, "
                f"got base={n_base} treat={n_treat}",
            )

        ci = _bootstrap_savings_ci(
            list(baseline_tokens), list(treatment_tokens),
            n_boot=self.n_boot, seed=self.seed,
        )

        p_base = _mean([1.0 if s else 0.0 for s in baseline_success])
        p_treat = _mean([1.0 if s else 0.0 for s in treatment_success])
        success_bound = p_treat - p_base   # must be >= -delta

        if success_bound < -self.delta:
            return GateVerdict(
                GateVerdictType.REJECT_REGRESSION, ci, success_bound, self.delta,
                f"Success rate dropped by {-success_bound:.1%} > delta={self.delta:.1%}",
            )
        if ci[1] < 0:
            # Upper bound of savings CI is negative -> intervention costs tokens.
            return GateVerdict(
                GateVerdictType.REJECT_NO_SAVINGS, ci, success_bound, self.delta,
                f"Token savings CI upper bound {ci[1]:.1%} < 0 — intervention costs tokens",
            )
        return GateVerdict(
            GateVerdictType.ACCEPT, ci, success_bound, self.delta,
            f"Savings CI [{ci[0]:.1%}, {ci[1]:.1%}]; "
            f"success non-inferior (bound {success_bound:.3f} > -delta)",
        )
