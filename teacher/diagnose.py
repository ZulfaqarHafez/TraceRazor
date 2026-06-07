"""Diagnoser -- turns a trace into a ``Diagnosis``.

Prefers the **real Rust auditor** (``tracerazor audit ... --format json``) when
the binary is available, so the Teacher is driven by the genuine TraceRazor
metrics. Falls back to a transparent built-in heuristic so the package always
runs offline / in CI with no build step.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

from .schemas import Diagnosis, WasteKind, WastePattern


# Map auditor metric codes -> our waste taxonomy.
_AUDITOR_METRIC_TO_KIND = {
    "srr": WasteKind.REDUNDANT_STEP,
    "ldi": WasteKind.LOOP,
    "tca": WasteKind.TOOL_MISFIRE,
    "rda": WasteKind.OVER_DEPTH,
    "cce": WasteKind.CONTEXT_BLOAT,
    "vdi": WasteKind.VERBOSITY,
    "shl": WasteKind.HEDGING,
}

_HEDGE_RE = re.compile(
    r"\b(certainly|i'd be happy|let me|i think|generally speaking|possibly|"
    r"basically|to be honest|at the end of the day|essentially)\b",
    re.IGNORECASE,
)


def _find_binary() -> Optional[str]:
    env = os.environ.get("TRACERAZOR_BIN")
    if env and os.path.exists(env):
        return env
    for cand in ("target/release/tracerazor", "target/debug/tracerazor"):
        if os.path.exists(cand):
            return cand
    return shutil.which("tracerazor")


class Diagnoser:
    """Diagnose a trace. ``prefer_auditor`` controls whether to shell out."""

    def __init__(self, prefer_auditor: bool = True):
        self.binary = _find_binary() if prefer_auditor else None

    # -- public ------------------------------------------------------------- #
    def diagnose(self, trace: dict) -> Diagnosis:
        if self.binary:
            try:
                return self._diagnose_auditor(trace)
            except Exception:
                pass  # fall through to builtin
        return self._diagnose_builtin(trace)

    # -- real Rust auditor -------------------------------------------------- #
    def _diagnose_auditor(self, trace: dict) -> Diagnosis:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as fh:
            json.dump(trace, fh)
            path = fh.name
        try:
            proc = subprocess.run(
                [self.binary, "audit", path, "--format", "json"],
                capture_output=True, text=True, timeout=30,
            )
            data = json.loads(proc.stdout)
        finally:
            os.unlink(path)

        score = data.get("score", {})
        tas = float(score.get("score", 0.0))
        patterns: list[WastePattern] = []
        # The auditor reports each metric's normalised health; turn failing
        # metrics into waste patterns.
        for code, kind in _AUDITOR_METRIC_TO_KIND.items():
            m = score.get(code)
            sev = self._metric_severity(code, m)
            if sev > 0.0:
                patterns.append(
                    WastePattern(kind=kind, severity=sev, step_ids=(),
                                 est_token_waste=self._waste_for(code, data))
                )
        return Diagnosis(
            trace_id=data.get("trace_id", trace.get("trace_id", "?")),
            agent_name=data.get("agent_name", trace.get("agent_name", "?")),
            framework=data.get("framework", trace.get("framework", "?")),
            tas_score=tas, total_tokens=int(data.get("total_tokens", 0)),
            patterns=patterns, source="auditor", raw=data,
        )

    @staticmethod
    def _metric_severity(code: str, m) -> float:
        """Best-effort: pull a 0..1 'badness' out of a metric blob."""
        if not isinstance(m, dict):
            return 0.0
        if m.get("pass") is False:
            return 0.6
        # Some metrics expose a score where lower == worse.
        for k in ("score", "value", "rate"):
            if k in m and isinstance(m[k], (int, float)):
                return 0.0  # passed metric -> no pattern
        return 0.0

    @staticmethod
    def _waste_for(code: str, data: dict) -> int:
        sav = data.get("savings", {})
        return int(sav.get("tokens_saved", 0)) // max(len(_AUDITOR_METRIC_TO_KIND), 1)

    # -- transparent built-in heuristic ------------------------------------ #
    def _diagnose_builtin(self, trace: dict) -> Diagnosis:
        steps = trace.get("steps", [])
        total = sum(s.get("tokens", 0) for s in steps)
        patterns: list[WastePattern] = []

        # Hedging / verbosity.
        hedge_steps, hedge_tokens = [], 0
        for s in steps:
            if s.get("type") == "reasoning" and _HEDGE_RE.search(s.get("content", "")):
                hedge_steps.append(s["id"])
                hedge_tokens += s.get("tokens", 0) // 3
        if hedge_steps:
            patterns.append(WastePattern(
                WasteKind.HEDGING, min(1.0, len(hedge_steps) / max(len(steps), 1) + 0.2),
                tuple(hedge_steps), hedge_tokens))

        # Redundant / reformulation steps (token-overlap with a prior step).
        redundant_ids, red_tokens = self._redundant_steps(steps)
        if redundant_ids:
            patterns.append(WastePattern(
                WasteKind.REDUNDANT_STEP, 0.6, tuple(redundant_ids), red_tokens))

        # Loops: same tool + params called consecutively.
        loop_ids, loop_tokens = self._loops(steps)
        if loop_ids:
            patterns.append(WastePattern(
                WasteKind.LOOP, 0.6, tuple(loop_ids), loop_tokens))

        # Over-depth: many reasoning steps relative to tool calls.
        reasoning = [s for s in steps if s.get("type") == "reasoning"]
        tools = [s for s in steps if s.get("type") == "tool_call"]
        if len(reasoning) > max(len(tools), 1) + 1:
            patterns.append(WastePattern(
                WasteKind.OVER_DEPTH, 0.5,
                tuple(s["id"] for s in reasoning[1:]), reasoning[-1].get("tokens", 0)))

        tas = self._builtin_tas(total, patterns)
        return Diagnosis(
            trace_id=trace.get("trace_id", "?"),
            agent_name=trace.get("agent_name", "?"),
            framework=trace.get("framework", "?"),
            tas_score=tas, total_tokens=total, patterns=patterns, source="builtin",
        )

    @staticmethod
    def _redundant_steps(steps) -> tuple[list[int], int]:
        ids, tokens = [], 0
        seen: list[set] = []
        for s in steps:
            words = set(re.findall(r"\w+", s.get("content", "").lower()))
            for prev in seen:
                if words and len(words & prev) / len(words | prev) >= 0.6:
                    ids.append(s["id"])
                    tokens += s.get("tokens", 0)
                    break
            seen.append(words)
        return ids, tokens

    @staticmethod
    def _loops(steps) -> tuple[list[int], int]:
        ids, tokens = [], 0
        last = None
        for s in steps:
            if s.get("type") == "tool_call":
                sig = (s.get("tool_name"), json.dumps(s.get("tool_params", {}), sort_keys=True))
                if sig == last:
                    ids.append(s["id"])
                    tokens += s.get("tokens", 0)
                last = sig
        return ids, tokens

    @staticmethod
    def _builtin_tas(total: int, patterns) -> float:
        """A simple cardinal-ish efficiency score: 100 minus attributed waste %."""
        if total == 0:
            return 100.0
        waste = sum(p.est_token_waste for p in patterns)
        return round(max(0.0, min(100.0, 100.0 * (1 - waste / total))), 1)
