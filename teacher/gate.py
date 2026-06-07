"""Quality-Preservation Gate -- the critical closed-loop accept rule (P2-B).

Accept an intervention iff:
  (1) token savings are real and meaningful  (>= min_savings_pct), AND
  (2) task success is NON-INFERIOR to baseline (within margin delta).

A token reduction that breaks the agent fails clause (2) and is rolled back.
This is what makes the savings claim falsifiable rather than projected.

The statistics here are intentionally lightweight (means + a margin) so the
package stays dependency-free; the design spec calls for a paired Wilcoxon +
proportion non-inferiority CI in production, which slots in behind the same
``decide`` interface.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schemas import Decision, EvalResult


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
