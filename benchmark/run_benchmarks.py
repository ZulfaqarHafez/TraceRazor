#!/usr/bin/env python3
"""
Run every real public agent trace under `traces/external/` through
`tracerazor audit` and produce a markdown table of measured TAS scores, grades,
tokens, waste, and estimated savings. Output is written to benchmark/RESULTS.md.

These are real trajectories (tau-bench airline/retail; SWE-agent edit-format
variants), not synthetic scenarios. Reproduce with:

    cargo build --release -p tracerazor
    python -m benchmark.run_benchmarks

Requires the `tracerazor` binary on PATH or in target/{release,debug}/.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    from benchmark._binary import find_tracerazor_binary
except ModuleNotFoundError:  # support `python benchmark/run_benchmarks.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmark._binary import find_tracerazor_binary


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRACES = REPO / "traces" / "external"
RESULTS = HERE / "RESULTS.md"
NON_TRACE_JSON = {"STATS.json"}


def find_binary() -> str:
    try:
        return find_tracerazor_binary(REPO)
    except RuntimeError as exc:
        sys.exit(str(exc))


def audit(binary: str, trace_path: Path) -> dict | None:
    # Hermetic + fresh HOME per audit: results are a pure function of
    # (trace, binary), independent of audit order and local store history —
    # required for the CI drift check on RESULTS.md to be meaningful.
    import os
    import tempfile

    env = dict(os.environ, HOME=tempfile.mkdtemp())
    env.pop("OPENAI_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    result = subprocess.run(
        [binary, "audit", str(trace_path), "--format", "json", "--hermetic"],
        capture_output=True, text=True, check=False, env=env,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"audit failed for {trace_path} (exit {result.returncode}): "
            f"{result.stderr.strip()[:500]}"
        )
    if not result.stdout.strip():
        return None  # e.g. fewer than the minimum steps
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"audit for {trace_path} did not emit valid JSON: {result.stdout[:500]}"
        ) from exc


def main() -> None:
    binary = find_binary()
    traces = sorted(t for t in TRACES.rglob("*.json") if t.name not in NON_TRACE_JSON)
    if not traces:
        sys.exit(f"No trace files found under {TRACES}")

    rows = []
    for t in traces:
        report = audit(binary, t)
        if report is None:
            continue
        score = report["score"]
        savings = report.get("savings", {})
        total_tokens = report.get("total_tokens", 0)
        tokens_saved = savings.get("tokens_saved", 0)
        waste_pct = (tokens_saved / total_tokens * 100.0) if total_tokens else 0.0
        rows.append({
            "source": t.parent.name,
            "trace": t.stem,
            "tas": score["score"],
            "grade": score.get("grade", "?"),
            "tokens": total_tokens,
            "savings": tokens_saved,
            "waste_pct": waste_pct,
            "n_fixes": len(report.get("fixes", [])),
        })
    if not rows:
        sys.exit(
            "No analysable traces were audited. Refusing to overwrite "
            f"{RESULTS} with an empty benchmark report."
        )

    md = ["# TraceRazor Benchmark Results", ""]
    md.append(
        "Measured by running `tracerazor audit` on every real public agent trace "
        "under `traces/external/` (tau-bench airline/retail; SWE-agent edit-format "
        "variants). These are real trajectories, not synthetic scenarios. "
        "Reproduce with `python -m benchmark.run_benchmarks`."
    )
    md.append("")
    md.append("| Source | Trace | TAS | Grade | Tokens | Waste | Est. savings | Fixes |")
    md.append("|---|---|---:|:-:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: (r["source"], r["trace"])):
        md.append(
            f"| {r['source']} | `{r['trace']}` | {r['tas']:.1f} | {r['grade']} | "
            f"{r['tokens']:,} | {r['waste_pct']:.0f}% | {r['savings']:,} | {r['n_fixes']} |"
        )
    md.append("")

    total_tokens = sum(r["tokens"] for r in rows)
    total_savings = sum(r["savings"] for r in rows)
    avg_tas = sum(r["tas"] for r in rows) / len(rows) if rows else 0.0
    overall_pct = (total_savings / total_tokens * 100.0) if total_tokens else 0.0
    md += [
        "## Summary",
        "",
        f"- Real traces benchmarked: **{len(rows)}**",
        f"- Average TAS: **{avg_tas:.1f}**",
        f"- Total tokens: **{total_tokens:,}**",
        f"- Total estimated savings: **{total_savings:,} tokens ({overall_pct:.0f}%)**",
        "",
        "Estimated savings are the sum of per-fix `estimated_token_savings`; they are "
        "projections, not a measured re-run. Token counts for external sources are "
        "approximated where the source did not record them, so read the relative "
        "ordering rather than absolute totals.",
        "",
    ]

    RESULTS.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {RESULTS} ({len(rows)} real traces, avg TAS {avg_tas:.1f})")


if __name__ == "__main__":
    main()
