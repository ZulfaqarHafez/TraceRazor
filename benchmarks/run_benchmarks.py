#!/usr/bin/env python3
"""
Run every trace under benchmarks/traces/ through `tracerazor audit` and
produce a markdown table of measured TAS scores, grades, tokens, waste,
and estimated savings. The output is written to benchmarks/RESULTS.md.

Usage:
    python benchmarks/run_benchmarks.py

Requires the `tracerazor` binary on PATH. Run `cargo build --release -p tracerazor`
first and add target/release to PATH if needed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRACES = HERE / "traces"
RESULTS = HERE / "RESULTS.md"


def find_binary() -> str:
    which = shutil.which("tracerazor")
    if which:
        return which
    # Fall back to cargo target dirs.
    repo = HERE.parent
    candidates = [
        repo / "target" / "release" / "tracerazor.exe",
        repo / "target" / "release" / "tracerazor",
        repo / "target" / "debug" / "tracerazor.exe",
        repo / "target" / "debug" / "tracerazor",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    sys.exit(
        "Could not find `tracerazor` binary. Run `cargo build --release -p tracerazor` first."
    )


def audit(binary: str, trace_path: Path) -> dict:
    result = subprocess.run(
        [binary, "audit", str(trace_path), "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        # exit 1 just means TAS < threshold, still valid JSON on stdout
        raise RuntimeError(
            f"tracerazor audit failed for {trace_path.name}:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def main() -> None:
    binary = find_binary()
    traces = sorted(TRACES.glob("*.json"))
    if not traces:
        sys.exit(f"No trace files found in {TRACES}")

    rows = []
    for t in traces:
        report = audit(binary, t)
        score = report["score"]
        savings = report.get("savings", {})
        total_tokens = report.get("total_tokens", 0)
        tokens_saved = savings.get("tokens_saved", 0)
        waste_pct = (tokens_saved / total_tokens * 100.0) if total_tokens else 0.0
        rows.append(
            {
                "trace": t.stem,
                "agent": report.get("agent_name", ""),
                "tas": score["score"],
                "grade": score.get("grade", "?"),
                "tokens": total_tokens,
                "savings": tokens_saved,
                "waste_pct": waste_pct,
                "n_fixes": len(report.get("fixes", [])),
            }
        )

    md = ["# TraceRazor Benchmark Results", ""]
    md.append(
        "Measured by running `tracerazor audit` on every trace under "
        "`benchmarks/traces/`. Each trace is a synthetic scenario that "
        "isolates a specific class of token waste. Reproduce with "
        "`python benchmarks/run_benchmarks.py`."
    )
    md.append("")
    md.append("| Trace | Agent | TAS | Grade | Tokens | Waste | Est. savings | Fixes |")
    md.append("|---|---|---:|:-:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| `{r['trace']}` | {r['agent']} | {r['tas']:.1f} | {r['grade']} | "
            f"{r['tokens']:,} | {r['waste_pct']:.0f}% | {r['savings']:,} | {r['n_fixes']} |"
        )
    md.append("")
    total_tokens = sum(r["tokens"] for r in rows)
    total_savings = sum(r["savings"] for r in rows)
    avg_tas = sum(r["tas"] for r in rows) / len(rows) if rows else 0.0
    overall_pct = (total_savings / total_tokens * 100.0) if total_tokens else 0.0
    md.append("## Summary")
    md.append("")
    md.append(f"- Traces benchmarked: **{len(rows)}**")
    md.append(f"- Average TAS: **{avg_tas:.1f}**")
    md.append(f"- Total tokens: **{total_tokens:,}**")
    md.append(f"- Total estimated savings: **{total_savings:,} tokens ({overall_pct:.0f}%)**")
    md.append("")
    md.append(
        "Estimated savings are the sum of per-fix `estimated_token_savings` "
        "from the report. To validate a specific patch set against a real "
        "re-run, use `tracerazor bench --before <old>.json --after <new>.json "
        "--fixes <fixes>.json`."
    )
    md.append("")

    RESULTS.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {RESULTS}")
    for line in md:
        print(line)


if __name__ == "__main__":
    main()