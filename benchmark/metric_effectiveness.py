"""Self-evaluation of the TAS sub-metrics on every real trace in the repo.

Audits the full real-trace corpus (public tau-bench / SWE-agent /
AgentInstruct exports plus the live Claude Code case-study traces)
hermetically and scores each sub-metric against pre-stated effectiveness
criteria. The output decides which metrics keep composite weight and which
are demoted to diagnostics — the decision rule is stated *before* the data
is looked at:

C1 — discriminative: corpus standard deviation >= 0.05. A metric that is
     (nearly) constant cannot move a composite; it only rescales it.
C2 — range-sane: corpus maximum >= 0.80. A normalised metric whose best
     observed value on ~60 real traces never approaches its top acts as a
     constant drag (miscalibrated normalisation), not a discriminator.
C3 — non-redundant: |Pearson r| < 0.85 against every *kept* metric. Of a
     collinear pair, the member with the larger corpus sd stays.

A metric failing any criterion is DEMOTED (composite weight 0) but stays
computed, reported and fix-driving — detection value is not the question;
composite value is. Nothing is deleted from the codebase.

Usage::

    python -m benchmark.metric_effectiveness --out docs/metric_effectiveness.md
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from benchmark.case_study import find_binary

REPO = Path(__file__).resolve().parent.parent

SD_MIN = 0.05      # C1
MAX_MIN = 0.80     # C2
COLLINEAR = 0.85   # C3

#: Trace sources: every real agent trace committed to the repository.
SOURCES = [
    ("tau-bench", "traces/external/tau_bench/*.json"),
    ("swe-agent", "traces/external/swe_agent/*.json"),
    ("agentinstruct", "traces/external/huggingface/agentinstruct/*.json"),
    ("live-coding", "benchmark/live/traces/*.before.json"),
    ("live-coding", "benchmark/live/traces/*.after.json"),
]


def audit(binary: str, trace: Path) -> dict | None:
    proc = subprocess.run(
        [binary, "audit", str(trace), "--format", "json", "--hermetic",
         "--store", "false", "--min-steps", "2"],
        capture_output=True, text=True,
    )
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        print(f"  skip {trace.name}: audit exit {proc.returncode}", file=sys.stderr)
        return None
    return json.loads(proc.stdout)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


def collect(binary: str) -> tuple[list[str], dict[str, list[float]], list[float]]:
    """Audit the corpus; return (metric codes, per-metric series, raw TAS series)."""
    series: dict[str, list[float]] = {}
    raw_tas: list[float] = []
    n = 0
    for source, glob in SOURCES:
        for trace in sorted(REPO.glob(glob)):
            report = audit(binary, trace)
            if report is None:
                continue
            mn = report["score"]["metric_normalised"]
            for code, val in mn.items():
                series.setdefault(code, []).append(float(val))
            raw_tas.append(float(report["score"]["raw_tas"]))
            n += 1
    print(f"audited {n} real traces", file=sys.stderr)
    if not series:
        sys.exit("error: no traces audited")
    return sorted(series.keys()), series, raw_tas


def evaluate(codes, series, raw_tas):
    stats = {}
    for c in codes:
        xs = series[c]
        n = len(xs)
        mean = sum(xs) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / n)
        stats[c] = {
            "mean": mean,
            "sd": sd,
            "min": min(xs),
            "max": max(xs),
            "ceiling": sum(x >= 0.999 for x in xs) / n,
            "r_tas": pearson(xs, raw_tas),
        }

    # C3 — greedy: order by sd descending; drop the weaker of a collinear pair.
    kept: list[str] = []
    fails: dict[str, list[str]] = {c: [] for c in codes}
    for c in codes:
        if stats[c]["sd"] < SD_MIN:
            fails[c].append(f"C1: sd {stats[c]['sd']:.3f} < {SD_MIN}")
        if stats[c]["max"] < MAX_MIN:
            fails[c].append(f"C2: max {stats[c]['max']:.2f} < {MAX_MIN}")
    for c in sorted(codes, key=lambda c: -stats[c]["sd"]):
        if fails[c]:
            continue
        clash = next(
            (k for k in kept if abs(pearson(series[c], series[k])) >= COLLINEAR),
            None,
        )
        if clash is not None:
            r = pearson(series[c], series[clash])
            fails[c].append(
                f"C3: |r|={abs(r):.2f} with {clash.upper()} "
                f"(sd {stats[clash]['sd']:.3f} >= {stats[c]['sd']:.3f})"
            )
            continue
        kept.append(c)
    return stats, sorted(kept), fails


def render(codes, series, stats, kept, fails, n_traces) -> str:
    lines = [
        "# Metric effectiveness — the self-evaluation behind the composite",
        "",
        f"Hermetic audits of **{n_traces} real traces** (tau-bench, SWE-agent,",
        "AgentInstruct ReAct, and the live Claude Code case-study corpus),",
        "regenerated by `python -m benchmark.metric_effectiveness`. Criteria",
        "are stated in the script docstring *before* the data is read:",
        f"C1 sd >= {SD_MIN}; C2 max >= {MAX_MIN}; C3 |r| < {COLLINEAR} vs kept metrics.",
        "",
        "| Metric | mean | sd | min | max | at ceiling | r(raw TAS) | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in codes:
        s = stats[c]
        verdict = "**KEEP (composite)**" if c in kept else (
            "demote to diagnostic — " + "; ".join(fails[c])
        )
        lines.append(
            f"| {c.upper()} | {s['mean']:.3f} | {s['sd']:.3f} | {s['min']:.2f} "
            f"| {s['max']:.2f} | {s['ceiling']:.0%} | {s['r_tas']:+.2f} | {verdict} |"
        )
    lines += [
        "",
        "## Pairwise |r| >= 0.6 (collinearity map)",
        "",
    ]
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            r = pearson(series[a], series[b])
            if abs(r) >= 0.6:
                lines.append(f"- {a.upper()} ~ {b.upper()}: r = {r:+.2f}")
    lines += [
        "",
        "## What demotion means",
        "",
        "Demoted metrics keep their detectors, per-step annotations, fix",
        "generation and report sections — they lose only their share of the",
        "composite TAS. The verbosity trio additionally keeps the AVS alert,",
        "which is its actual job. Composite weights for kept metrics are",
        "renormalised in `crates/tracerazor-core/src/scoring.rs`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    binary = find_binary()
    if binary is None:
        sys.exit("error: build the CLI first (cargo build --release -p tracerazor)")
    codes, series, raw_tas = collect(binary)
    stats, kept, fails = evaluate(codes, series, raw_tas)
    out = render(codes, series, stats, kept, fails, len(raw_tas))
    if args.out:
        args.out.write_text(out, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(out)
    print("kept:", ", ".join(c.upper() for c in kept), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
