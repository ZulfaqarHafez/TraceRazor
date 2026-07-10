"""User-conditioned learning for TRICE V2.

The profile is intentionally small and auditable. It learns only runtime
preferences and safety constraints, not private task content. The live rollout
runner uses it to choose the next budget ratio and to decide whether a result
is allowed to count as verified.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .evidence import write_text_lf


@dataclass
class UserPreferenceProfile:
    user_id: str = "local"
    target_savings: float = 0.60
    budget_ratio: float = 0.40
    pass_noninferiority_margin_pp: float = -2.0
    require_live_rollout: bool = True
    prefer_receipts: bool = True
    allow_test_edits: bool = False
    max_rounds: int = 3
    rounds_seen: int = 0
    lessons: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "UserPreferenceProfile":
        p = Path(path)
        if not p.is_file():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**data)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        write_text_lf(p, json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    def ingest_feedback(self, feedback: str | None) -> "UserPreferenceProfile":
        text = (feedback or "").strip()
        if not text:
            return self
        low = text.lower()

        pct = _extract_percentage(low)
        if pct is not None and ("saving" in low or "token" in low or "s tier" in low):
            self.target_savings = _clamp(pct, 0.05, 0.90)
            self.budget_ratio = _clamp(1.0 - self.target_savings, 0.10, 0.85)
            self.lessons.append(f"user target set to {self.target_savings:.0%} input-token savings")

        if "real" in low and ("run" in low or "repo" in low or "not replay" in low):
            self.require_live_rollout = True
            self.lessons.append("user requires live rollout evidence, not replay-only acceptance")
        if "replay" in low and ("not" in low or "no " in low):
            self.require_live_rollout = True
            self.lessons.append("replay is allowed only as a preflight, not as final proof")
        if "learn from" in low or "user" in low:
            self.lessons.append("adapt budget and safety from user feedback before acting")
        if "aggressive" in low or "s tier" in low or "60" in low:
            self.budget_ratio = min(self.budget_ratio, 0.40)
            self.lessons.append("prefer aggressive compression when live pass preservation holds")
        if "safe" in low or "don't break" in low or "do not break" in low:
            self.budget_ratio = max(self.budget_ratio, 0.50)
            self.lessons.append("relax compression when user explicitly prioritizes safety")
        if "tests" in low and ("do not modify" in low or "don't modify" in low):
            self.allow_test_edits = False
            self.lessons.append("never edit tests during managed rollouts")

        self.lessons = _dedupe_tail(self.lessons, keep=24)
        return self

    def adapt_from_outcome(self, savings: float, pass_noninferior: bool) -> "UserPreferenceProfile":
        self.rounds_seen += 1
        if not pass_noninferior:
            self.budget_ratio = _clamp(self.budget_ratio + 0.10, 0.15, 0.90)
            self.lessons.append(
                f"round {self.rounds_seen}: pass gate failed; relaxing budget to {self.budget_ratio:.0%}"
            )
        elif savings + 0.01 < self.target_savings:
            self.budget_ratio = _clamp(self.budget_ratio - 0.05, 0.10, 0.90)
            self.lessons.append(
                f"round {self.rounds_seen}: savings {savings:.0%} below target; tightening budget to {self.budget_ratio:.0%}"
            )
        else:
            self.lessons.append(
                f"round {self.rounds_seen}: accepted {savings:.0%} savings with pass preservation"
            )
        self.lessons = _dedupe_tail(self.lessons, keep=24)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_percentage(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if not match:
        return None
    return float(match.group(1)) / 100.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def _dedupe_tail(items: list[str], keep: int) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out[-keep:]
