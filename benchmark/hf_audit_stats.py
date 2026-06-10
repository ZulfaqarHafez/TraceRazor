#!/usr/bin/env python3
"""Audit the real Hugging Face AgentInstruct corpus and emit aggregate statistics.

Runs the `tracerazor` binary over every trace in
``traces/external/huggingface/agentinstruct/`` and writes:

* ``traces/external/huggingface/agentinstruct/STATS.json`` — machine-readable.
* ``docs/huggingface_agentinstruct_audit.md`` — the human report cited by the paper.

Usage::

    cargo build --release -p tracerazor
    python -m benchmark.hf_audit_stats            # or: python benchmark/hf_audit_stats.py
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "traces" / "external" / "huggingface" / "agentinstruct"
STATS_JSON = CORPUS_DIR / "STATS.json"
REPORT_MD = REPO_ROOT / "docs" / "huggingface_agentinstruct_audit.md"

_METRIC_CODES = ["srr", "ldi", "tca", "rda", "isr", "tur", "cce", "dbo",
                 "vdi", "shl", "ccr", "gar", "csd", "obs"]


def _find_binary() -> str:
    for cand in ("release", "debug"):
        p = REPO_ROOT / "target" / cand / "tracerazor"
        if p.exists():
            return str(p)
    raise SystemExit("tracerazor binary not found; run `cargo build --release -p tracerazor`")


def _audit(binary: str, trace: Path,
           min_steps: Optional[int] = None) -> Optional[Dict[str, Any]]:
    # Fresh HOME per audit: each measurement is independent (no cross-trace
    # history effects on DBO/RDA baselines, no audit-order dependence).
    env = dict(os.environ, HOME=tempfile.mkdtemp())
    env.pop("OPENAI_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    cmd = [binary, "audit", str(trace), "--format", "json"]
    if min_steps is not None:
        cmd += ["--min-steps", str(min_steps)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None  # sub-floor trace, skipped with a notice


def collect() -> Dict[str, Any]:
    binary = _find_binary()
    files = sorted(CORPUS_DIR.glob("agentinstruct-*.json"))
    if not files:
        raise SystemExit(f"no traces in {CORPUS_DIR}; run tools/convert_agentinstruct.py --bundled")

    per_trace: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for f in files:
        report = _audit(binary, f)
        if report is None:
            skipped.append(f.stem)
            continue
        score = report["score"]
        mn = score.get("metric_normalised", {})
        per_trace.append({
            "trace_id": report["trace_id"],
            "steps": report["total_steps"],
            "tokens": report["total_tokens"],
            "tas": round(score["score"], 1),
            "grade": str(score["grade"]),
            "mvtg": round(report.get("mvtg", 0.0), 3),
            "fixes": len(report.get("fixes", [])),
            "agf": round((report.get("agf") or {}).get("score", float("nan")), 3),
            "metrics": {k: round(mn.get(k, float("nan")), 3) for k in _METRIC_CODES},
        })

    tas = [t["tas"] for t in per_trace]
    grades: Dict[str, int] = {}
    for t in per_trace:
        grades[t["grade"]] = grades.get(t["grade"], 0) + 1
    metric_means = {
        k: round(statistics.fmean([t["metrics"][k] for t in per_trace]), 3)
        for k in _METRIC_CODES
    } if per_trace else {}

    # Second pass: the full corpus with the short-trace opt-in (--min-steps 2).
    # Real ReAct task runs are mostly 3–4 steps; this measures the product on
    # the entire corpus rather than only the traces above the default floor.
    full_corpus: List[Dict[str, Any]] = []
    for f in files:
        report = _audit(binary, f, min_steps=2)
        if report is not None:
            full_corpus.append({
                "trace_id": report["trace_id"],
                "steps": report["total_steps"],
                "tas": round(report["score"]["score"], 1),
                "grade": str(report["score"]["grade"]),
                "fixes": len(report.get("fixes", [])),
            })
    fc_tas = [t["tas"] for t in full_corpus]

    return {
        "dataset": "zai-org/AgentInstruct (Hugging Face)",
        "n_traces": len(files),
        "n_analysable": len(per_trace),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "mean_tas": round(statistics.fmean(tas), 1) if tas else None,
        "median_tas": round(statistics.median(tas), 1) if tas else None,
        "grade_distribution": grades,
        "mean_mvtg": round(statistics.fmean([t["mvtg"] for t in per_trace]), 3) if per_trace else None,
        "total_fixes": sum(t["fixes"] for t in per_trace),
        "mean_agf": round(statistics.fmean([t["agf"] for t in per_trace]), 3) if per_trace else None,
        "metric_means_normalised": metric_means,
        "per_trace": per_trace,
        "full_corpus_min_steps_2": {
            "n_audited": len(full_corpus),
            "mean_tas": round(statistics.fmean(fc_tas), 1) if fc_tas else None,
            "total_fixes": sum(t["fixes"] for t in full_corpus),
            "per_trace": full_corpus,
        },
    }


def render_markdown(stats: Dict[str, Any]) -> str:
    lines = [
        "# TraceRazor on Real Hugging Face Agent Trajectories",
        "",
        "Audit statistics for the product run over real ReAct agent trajectories",
        "from the Hugging Face dataset [`zai-org/AgentInstruct`]"
        "(https://huggingface.co/datasets/zai-org/AgentInstruct) "
        "(formerly `THUDM/AgentInstruct`; arXiv:2310.12823). The corpus is the",
        "vendored real sample converted by `tools/convert_agentinstruct.py`; see",
        "`traces/external/huggingface/agentinstruct/SOURCE.md` for provenance and",
        "the live-fetch path. Reproduce with `python -m benchmark.hf_audit_stats`.",
        "",
        "## Corpus",
        "",
        f"- Traces: **{stats['n_traces']}** "
        f"({stats['n_analysable']} analysable, {stats['n_skipped']} skipped <5 steps)",
        f"- Mean TAS: **{stats['mean_tas']}** (median {stats['median_tas']})",
        f"- Grade distribution: {stats['grade_distribution']}",
        f"- Mean MVTG (structural waste): **{stats['mean_mvtg']}**",
        f"- Fix patches emitted: **{stats['total_fixes']}**",
        f"- Mean AGF (grounding-fidelity diagnostic): **{stats.get('mean_agf')}**",
        "",
        "## Mean normalised metric scores (1.0 = no waste detected)",
        "",
        "| Metric | Mean (normalised) |",
        "|---|---:|",
    ]
    for k in _METRIC_CODES:
        v = stats["metric_means_normalised"].get(k)
        lines.append(f"| {k.upper()} | {v} |")
    lines += [
        "",
        "## Per-trace",
        "",
        "| Trace | Steps | Tokens | TAS | Grade | SRR | LDI | GAR | OBS | Fixes |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for t in stats["per_trace"]:
        m = t["metrics"]
        lines.append(
            f"| {t['trace_id']} | {t['steps']} | {t['tokens']} | {t['tas']} | "
            f"{t['grade']} | {m['srr']} | {m['ldi']} | {m['gar']} | {m['obs']} | {t['fixes']} |"
        )
    fc = stats.get("full_corpus_min_steps_2") or {}
    if fc:
        lines += [
            "",
            "## Full corpus with the short-trace opt-in (`--min-steps 2`)",
            "",
            "Most real ReAct task runs finish in 3-4 steps — below the default",
            "5-step floor. With `--min-steps 2` the audit covers the entire corpus:",
            "",
            f"- Audited: **{fc['n_audited']}/{stats['n_traces']}**",
            f"- Mean TAS: **{fc['mean_tas']}**",
            f"- Fix patches: **{fc['total_fixes']}**",
            "",
            "| Trace | Steps | TAS | Grade | Fixes |",
            "|---|---:|---:|---|---:|",
        ]
        for t in fc["per_trace"]:
            lines.append(
                f"| {t['trace_id']} | {t['steps']} | {t['tas']} | {t['grade']} | {t['fixes']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    stats = collect()
    STATS_JSON.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_markdown(stats), encoding="utf-8")
    print(f"Wrote {STATS_JSON.relative_to(REPO_ROOT)} and {REPORT_MD.relative_to(REPO_ROOT)}")
    print(f"  n={stats['n_analysable']} analysable, mean TAS={stats['mean_tas']}, "
          f"grades={stats['grade_distribution']}, fixes={stats['total_fixes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
