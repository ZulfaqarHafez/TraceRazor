"""Playbook -- cross-session, cross-agent experience store (P3-A).

Keys lessons on a ``WastePattern.signature`` (kind + severity bucket) rather
than a specific agent, so an intervention proven on one agent is ranked first
when the same waste pattern shows up on another. Persisted as JSON so it
survives across runs and can be shared between teams.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .schemas import Outcome, PlaybookEntry, WasteKind


class Playbook:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.entries: dict[str, PlaybookEntry] = {}
        if path and os.path.exists(path):
            self._load()

    @staticmethod
    def _key(signature: str, intervention_key: str) -> str:
        return f"{signature}::{intervention_key}"

    def record(self, signature: str, waste_kind: WasteKind, framework: str,
               intervention_key: str, outcome: Outcome) -> None:
        k = self._key(signature, intervention_key)
        entry = self.entries.get(k)
        if entry is None:
            entry = PlaybookEntry(signature, waste_kind, framework, intervention_key)
            self.entries[k] = entry
        entry.record(outcome)

    def prior_winrate(self, signature: str, intervention_key: str) -> float:
        entry = self.entries.get(self._key(signature, intervention_key))
        return entry.winrate if entry else 0.5  # neutral prior

    def best_for(self, signature: str) -> Optional[PlaybookEntry]:
        cands = [e for e in self.entries.values()
                 if e.pattern_signature == signature and e.trials > 0]
        return max(cands, key=lambda e: e.winrate, default=None)

    # -- persistence -------------------------------------------------------- #
    def save(self) -> None:
        if not self.path:
            return
        payload = {
            k: {
                "pattern_signature": e.pattern_signature,
                "waste_kind": e.waste_kind.value,
                "framework": e.framework,
                "intervention_key": e.intervention_key,
                "trials": e.trials, "wins": e.wins,
                "mean_token_saving_pct": round(e.mean_token_saving_pct, 2),
                "mean_tas_delta": round(e.mean_tas_delta, 2),
            }
            for k, e in self.entries.items()
        }
        with open(self.path, "w") as fh:
            json.dump(payload, fh, indent=2)

    def _load(self) -> None:
        with open(self.path) as fh:
            payload = json.load(fh)
        for k, d in payload.items():
            self.entries[k] = PlaybookEntry(
                d["pattern_signature"], WasteKind(d["waste_kind"]), d["framework"],
                d["intervention_key"], d["trials"], d["wins"],
                d["mean_token_saving_pct"], d["mean_tas_delta"])

    def summary(self) -> str:
        if not self.entries:
            return "(empty playbook)"
        rows = sorted(self.entries.values(),
                      key=lambda e: e.mean_token_saving_pct, reverse=True)
        lines = ["  pattern        intervention        winrate  ~saving"]
        for e in rows:
            lines.append(
                f"  {e.pattern_signature:<14} {e.intervention_key:<19} "
                f"{e.winrate*100:5.0f}%  {e.mean_token_saving_pct:5.1f}%")
        return "\n".join(lines)
