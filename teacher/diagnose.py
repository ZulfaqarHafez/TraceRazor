"""Diagnoser -- turns a trace into a rich ``Diagnosis``.

Backends, tried in order:

  1. **native**     -- ``import tracerazor_native`` (PyO3 binding; see
                       ``crates/tracerazor-py``). Calls the Rust core in-process,
                       no subprocess, no temp files.
  2. **subprocess** -- shells the ``tracerazor`` binary (auto-detected). Robust,
                       works against any installed build.
  3. **builtin**    -- a transparent pure-Python heuristic so the package always
                       runs offline / in CI with no Rust at all.

The auditor (native or subprocess) emits a rich report: a TAS score, all 13
metric blobs (each with ``pass``/``target`` and metric-specific detail such as
``srr.redundant_steps`` or ``tca.misfires``), a step-level ``diff``, ``savings``,
and ready-made ``fixes``. ``_diagnose_auditor`` parses *all* of that into
``WastePattern`` objects with real severity, step-id attribution, and token
attribution -- and carries the auditor's own ``fixes`` through so the Teacher
can act on them directly.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Optional

from .schemas import Diagnosis, WasteKind, WastePattern


# --------------------------------------------------------------------------- #
# Backends.
# --------------------------------------------------------------------------- #
def _find_binary() -> Optional[str]:
    env = os.environ.get("TRACERAZOR_BIN")
    if env and os.path.exists(env):
        return env
    for cand in ("target/release/tracerazor", "target/debug/tracerazor"):
        if os.path.exists(cand):
            return cand
    return shutil.which("tracerazor")


def _native_backend() -> Optional[Callable[[dict], dict]]:
    """Return an audit fn backed by the PyO3 binding, if importable."""
    try:
        import tracerazor_native  # type: ignore
    except Exception:
        return None

    def audit(trace: dict) -> dict:
        return json.loads(tracerazor_native.audit_json(json.dumps(trace)))

    return audit


def _subprocess_backend(binary: str) -> Callable[[dict], dict]:
    def audit(trace: dict) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(trace, fh)
            path = fh.name
        try:
            proc = subprocess.run(
                [binary, "audit", path, "--format", "json"],
                capture_output=True, text=True, timeout=30,
            )
            return json.loads(proc.stdout)
        finally:
            os.unlink(path)

    return audit


# --------------------------------------------------------------------------- #
# Metric -> waste-pattern extraction specs.
#
# Each spec turns one auditor metric blob (+ the trace) into a magnitude in
# [0,1] (used for severity), a set of implicated step ids, and a token-waste
# estimate. Directions differ per metric, so each spec encodes its own
# waste-proportional signal explicitly rather than relying on the raw score.
# --------------------------------------------------------------------------- #
def _ints(obj: Any) -> list[int]:
    """Best-effort: pull step ids out of an arbitrary detail structure."""
    out: list[int] = []

    def walk(o: Any) -> None:
        if isinstance(o, bool):
            return
        if isinstance(o, int):
            out.append(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                if k in ("step_id", "step", "step_a", "step_b", "id", "goal_step_id"):
                    walk(v)
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return out


def _step_tokens(trace: dict, ids: list[int]) -> int:
    by_id = {s.get("id"): s.get("tokens", 0) for s in trace.get("steps", [])}
    return sum(by_id.get(i, 0) for i in set(ids))


def _frac(n: float, d: float) -> float:
    return (n / d) if d else 0.0


def _extract(code: str, m: dict, trace: dict) -> tuple[float, tuple[int, ...], int]:
    """Return (magnitude 0..1, step_ids, token_waste) for one metric blob."""
    total_steps = m.get("total_steps") or len(trace.get("steps", [])) or 1
    total_tok = trace.get("_total_tokens") or sum(
        s.get("tokens", 0) for s in trace.get("steps", [])) or 1

    if code == "srr":
        pairs = m.get("redundant_steps", [])
        later = [p.get("step_b") for p in pairs if isinstance(p, dict)]
        return (_frac(m.get("redundant_count", len(pairs)), total_steps),
                tuple(later), _step_tokens(trace, later))
    if code == "ldi":
        loops = m.get("loops", [])
        ids = _ints(loops)
        return (min(1.0, len(loops) * 0.4), tuple(ids), _step_tokens(trace, ids))
    if code == "tca":
        mis = m.get("misfires", [])
        ids = _ints(mis)
        return (_frac(len(mis), m.get("total_tool_calls", 1)), tuple(ids),
                _step_tokens(trace, ids))
    if code == "tur":
        waste = m.get("wasted_tokens", 0)
        return (_frac(waste, m.get("total_tokens", total_tok)), (), waste)
    if code == "cce":
        dup = m.get("duplicate_tokens", 0)
        ids = _ints(m.get("bloated_steps", []))
        return (_frac(dup, m.get("total_input_tokens", total_tok)), tuple(ids), dup)
    if code == "rda":
        exp, act = m.get("expected_steps", 0), m.get("actual_steps", total_steps)
        over = max(0, act - exp)
        return (_frac(over, max(exp, 1)), (), 0)
    if code == "isr":
        low = m.get("low_novelty_steps", [])
        ids = _ints(low)
        return (_frac(len(low), total_steps), tuple(ids), _step_tokens(trace, ids))
    if code == "cce_dup":  # unused alias
        return (0.0, (), 0)
    if code == "vdi":
        low = m.get("low_density_steps", []) or m.get("entropy_low_steps", [])
        ids = _ints(low)
        return (_frac(len(ids), total_steps) + 0.2, tuple(ids), _step_tokens(trace, ids) // 3)
    if code == "shl":
        flagged = m.get("flagged_sentences", 0)
        return (_frac(flagged, max(m.get("total_sentences", 1), 1)), (), 0)
    if code == "ccr":
        cut = m.get("total_cuttable_tokens", 0)
        return (_frac(cut, total_tok), (), cut)
    if code == "gar":
        low = m.get("low_advancement_steps", [])
        ids = _ints(low)
        return (_frac(len(ids), total_steps), tuple(ids), 0)
    if code == "csd":
        pairs = m.get("high_drift_pairs", [])
        ids = _ints(pairs)
        return (min(1.0, len(pairs) * 0.3), tuple(ids), 0)
    return (0.0, (), 0)


_METRIC_KIND = {
    "srr": WasteKind.REDUNDANT_STEP, "isr": WasteKind.REDUNDANT_STEP,
    "cce": WasteKind.CONTEXT_BLOAT, "ldi": WasteKind.LOOP,
    "tca": WasteKind.TOOL_MISFIRE, "rda": WasteKind.OVER_DEPTH,
    "tur": WasteKind.OVER_DEPTH, "vdi": WasteKind.VERBOSITY,
    "ccr": WasteKind.VERBOSITY, "shl": WasteKind.HEDGING,
    "gar": WasteKind.OVER_DEPTH, "csd": WasteKind.CONTEXT_BLOAT,
}

_HEDGE_RE = re.compile(
    r"\b(certainly|i'd be happy|let me|i think|generally speaking|possibly|"
    r"basically|to be honest|at the end of the day|essentially)\b", re.IGNORECASE)


class Diagnoser:
    def __init__(self, prefer_auditor: bool = True):
        self.binary = _find_binary() if prefer_auditor else None
        self._audit = None
        self.backend = "builtin"
        if prefer_auditor:
            native = _native_backend()
            if native is not None:
                self._audit, self.backend = native, "native"
            elif self.binary:
                self._audit, self.backend = _subprocess_backend(self.binary), "subprocess"

    # -- public ------------------------------------------------------------- #
    def diagnose(self, trace: dict) -> Diagnosis:
        if self._audit is not None:
            try:
                return self._diagnose_auditor(trace)
            except Exception:
                pass  # fall through to builtin on any backend hiccup
        return self._diagnose_builtin(trace)

    # -- real auditor (native or subprocess), rich structured parse --------- #
    def _diagnose_auditor(self, trace: dict) -> Diagnosis:
        data = self._audit(trace)
        score = data.get("score", {})
        tas = float(score.get("score", 0.0))

        # Merge per-metric extractions into one WastePattern per WasteKind,
        # taking the max severity and unioning step ids / token waste.
        agg: dict[WasteKind, dict] = {}
        for code, kind in _METRIC_KIND.items():
            m = score.get(code)
            if not isinstance(m, dict):
                continue
            failing = m.get("pass") is False
            mag, ids, tokens = _extract(code, m, trace)
            if not failing and mag < 0.25:
                continue  # passing & low-magnitude -> not material
            sev = min(1.0, (0.4 if failing else 0.0) + mag)
            cur = agg.setdefault(kind, {"sev": 0.0, "ids": set(), "tok": 0})
            cur["sev"] = max(cur["sev"], sev)
            cur["ids"].update(ids)
            cur["tok"] += tokens

        patterns = [
            WastePattern(kind=kind, severity=round(v["sev"], 2),
                         step_ids=tuple(sorted(v["ids"])), est_token_waste=int(v["tok"]))
            for kind, v in agg.items() if v["sev"] > 0.0
        ]
        patterns.sort(key=lambda p: p.severity, reverse=True)

        return Diagnosis(
            trace_id=data.get("trace_id", trace.get("trace_id", "?")),
            agent_name=data.get("agent_name", trace.get("agent_name", "?")),
            framework=data.get("framework", trace.get("framework", "?")),
            tas_score=tas, total_tokens=int(data.get("total_tokens", 0)),
            patterns=patterns, source="auditor", backend=self.backend,
            auditor_fixes=list(data.get("fixes", []) or []),
            savings=dict(data.get("savings", {}) or {}), raw=data,
        )

    # -- transparent built-in heuristic ------------------------------------- #
    def _diagnose_builtin(self, trace: dict) -> Diagnosis:
        steps = trace.get("steps", [])
        total = sum(s.get("tokens", 0) for s in steps)
        patterns: list[WastePattern] = []

        hedge_steps, hedge_tokens = [], 0
        for s in steps:
            if s.get("type") == "reasoning" and _HEDGE_RE.search(s.get("content", "")):
                hedge_steps.append(s["id"])
                hedge_tokens += s.get("tokens", 0) // 3
        if hedge_steps:
            patterns.append(WastePattern(
                WasteKind.HEDGING, min(1.0, len(hedge_steps) / max(len(steps), 1) + 0.2),
                tuple(hedge_steps), hedge_tokens))

        red_ids, red_tok = self._redundant_steps(steps)
        if red_ids:
            patterns.append(WastePattern(
                WasteKind.REDUNDANT_STEP, 0.6, tuple(red_ids), red_tok))

        loop_ids, loop_tok = self._loops(steps)
        if loop_ids:
            patterns.append(WastePattern(WasteKind.LOOP, 0.6, tuple(loop_ids), loop_tok))

        reasoning = [s for s in steps if s.get("type") == "reasoning"]
        tools = [s for s in steps if s.get("type") == "tool_call"]
        if len(reasoning) > max(len(tools), 1) + 1:
            patterns.append(WastePattern(
                WasteKind.OVER_DEPTH, 0.5, tuple(s["id"] for s in reasoning[1:]),
                reasoning[-1].get("tokens", 0)))

        tas = self._builtin_tas(total, patterns)
        return Diagnosis(
            trace_id=trace.get("trace_id", "?"), agent_name=trace.get("agent_name", "?"),
            framework=trace.get("framework", "?"), tas_score=tas, total_tokens=total,
            patterns=patterns, source="builtin", backend="builtin")

    @staticmethod
    def _redundant_steps(steps) -> tuple[list[int], int]:
        ids, tokens, seen = [], 0, []
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
        ids, tokens, last = [], 0, None
        for s in steps:
            if s.get("type") == "tool_call":
                sig = (s.get("tool_name"),
                       json.dumps(s.get("tool_params", {}), sort_keys=True))
                if sig == last:
                    ids.append(s["id"])
                    tokens += s.get("tokens", 0)
                last = sig
        return ids, tokens

    @staticmethod
    def _builtin_tas(total: int, patterns) -> float:
        if total == 0:
            return 100.0
        waste = sum(p.est_token_waste for p in patterns)
        return round(max(0.0, min(100.0, 100.0 * (1 - waste / total))), 1)
