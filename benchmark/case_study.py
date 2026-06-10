"""Measured case study harness (ship-plan 4.1).

Pipeline per task:  audit (before) -> apply fixes -> re-run the agent ->
``tracerazor bench`` before vs after.  This module implements the
*measurement* half: given pairs of before/after traces captured at constant
task outcome, it runs ``bench`` on every pair and aggregates **measured**
(not projected) token deltas with bootstrap confidence intervals and a
pass-rate-held check.

The live half (re-running the agent with the patched prompt) needs LLM
credentials and a tau-bench checkout; this harness consumes whatever pairs
that produces.  Pair layout::

    pairs_dir/
      <task>.before.json   # trace captured before fixes were applied
      <task>.after.json    # trace captured after fixes were applied
      <task>.fixes.json    # optional: audit fixes JSON for estimate accuracy

Usage::

    python -m benchmark.case_study --pairs-dir results/case_study \
        --out docs/case_study.md

The pass signal is each trace's ``task_value_score`` (tau-bench reward:
1.0 = solved).  A token saving that costs task success is a regression, not
a saving — any task whose pass flag flips from before to after is called
out and fails the constant-pass-rate requirement.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Bootstrap resamples for the confidence intervals (seeded, reproducible).
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
PASS_EPS = 1e-9


def find_binary() -> str | None:
    """Locate the tracerazor CLI like the test suite does."""
    env = os.environ.get("TRACERAZOR_BIN")
    if env and Path(env).is_file():
        return env
    for cand in ("release", "debug"):
        p = REPO / "target" / cand / "tracerazor"
        if p.is_file():
            return str(p)
    return None


def bootstrap_ci(
    values: list[float],
    n_resamples: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap ``1 - alpha`` CI of the mean.

    Returns ``(mean, lo, hi)``.  Deterministic for a given seed.  With a
    single observation the CI degenerates to the point estimate.
    """
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_resamples)
    )
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return mean, lo, hi


@dataclass
class TaskResult:
    """One task's measured before/after deltas."""

    task: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    pct_saved: float
    tas_before: float
    tas_after: float
    tas_delta: float
    pass_before: bool
    pass_after: bool
    estimated_tokens_saved: int | None

    @property
    def pass_held(self) -> bool:
        return self.pass_before == self.pass_after


def trace_passes(trace_path: Path) -> bool:
    """Read the task-outcome signal from a trace (tau-bench reward)."""
    d = json.loads(trace_path.read_text(encoding="utf-8"))
    tvs = d.get("task_value_score")
    if tvs is None:
        tvs = d.get("metadata", {}).get("reward", 1.0)
    return float(tvs) >= 1.0 - PASS_EPS


def run_bench(
    binary: str, before: Path, after: Path, fixes: Path | None
) -> dict:
    """Run ``tracerazor bench`` on one pair and return its JSON output."""
    cmd = [
        binary,
        "bench",
        "--before",
        str(before),
        "--after",
        str(after),
        "--format",
        "json",
    ]
    if fixes is not None:
        cmd += ["--fixes", str(fixes)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench failed for {before.name} vs {after.name} "
            f"(exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


def discover_pairs(pairs_dir: Path) -> list[tuple[str, Path, Path, Path | None]]:
    """Find ``<task>.before.json`` / ``<task>.after.json`` pairs."""
    pairs = []
    for before in sorted(pairs_dir.glob("*.before.json")):
        task = before.name[: -len(".before.json")]
        after = pairs_dir / f"{task}.after.json"
        if not after.is_file():
            print(f"skip {task}: no matching .after.json", file=sys.stderr)
            continue
        fixes = pairs_dir / f"{task}.fixes.json"
        pairs.append((task, before, after, fixes if fixes.is_file() else None))
    return pairs


def measure(
    binary: str, pairs: list[tuple[str, Path, Path, Path | None]]
) -> list[TaskResult]:
    results = []
    for task, before, after, fixes in pairs:
        bench = run_bench(binary, before, after, fixes)
        results.append(
            TaskResult(
                task=task,
                tokens_before=int(bench["before"]["tokens"]),
                tokens_after=int(bench["after"]["tokens"]),
                tokens_saved=int(bench["actual_tokens_saved"]),
                pct_saved=float(bench["pct_tokens_saved"]),
                tas_before=float(bench["before"]["tas"]),
                tas_after=float(bench["after"]["tas"]),
                tas_delta=float(bench["tas_delta"]),
                pass_before=trace_passes(before),
                pass_after=trace_passes(after),
                estimated_tokens_saved=bench.get("estimated_tokens_saved"),
            )
        )
    return results


def render_markdown(results: list[TaskResult], synthetic: bool = False) -> str:
    """Render the published case-study table from measured results."""
    if not results:
        raise ValueError("no task results to render")

    pct_mean, pct_lo, pct_hi = bootstrap_ci([r.pct_saved for r in results])
    tas_mean, tas_lo, tas_hi = bootstrap_ci([r.tas_delta for r in results])
    n = len(results)
    pass_before = sum(r.pass_before for r in results)
    pass_after = sum(r.pass_after for r in results)
    all_held = all(r.pass_held for r in results)

    lines = []
    if synthetic:
        lines += [
            "> **Synthetic plumbing check — NOT the case study.** These rows",
            "> come from constructed traces and validate the harness, not the",
            "> product. The published case study requires real agent re-runs.",
            "",
        ]
    lines += [
        "| Task | Tokens before | Tokens after | Saved | Saved % | TAS Δ | Pass held |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in results:
        held = "✅" if r.pass_held else "❌ FLIPPED"
        lines.append(
            f"| {r.task} | {r.tokens_before} | {r.tokens_after} "
            f"| {r.tokens_saved} | {r.pct_saved:.1f}% | {r.tas_delta:+.1f} | {held} |"
        )
    lines += [
        "",
        f"**Aggregate over {n} task(s):** mean token reduction "
        f"**{pct_mean:.1f}%** (95% bootstrap CI [{pct_lo:.1f}%, {pct_hi:.1f}%]); "
        f"mean TAS delta {tas_mean:+.1f} (95% CI [{tas_lo:+.1f}, {tas_hi:+.1f}]).",
        "",
        f"**Pass rate:** {pass_before}/{n} before → {pass_after}/{n} after — "
        + (
            "constant task outcome on every pair (the savings are at unchanged pass rate)."
            if all_held
            else "⚠️ **at least one task outcome flipped; the token delta on those "
            "tasks is not a saving.**"
        ),
    ]
    ests = [r for r in results if r.estimated_tokens_saved]
    if ests:
        acc = [
            r.tokens_saved / r.estimated_tokens_saved * 100.0
            for r in ests
            if r.estimated_tokens_saved
        ]
        acc_mean, acc_lo, acc_hi = bootstrap_ci(acc)
        lines += [
            "",
            f"**Estimate accuracy** (measured / audit-estimated savings, {len(ests)} "
            f"task(s) with fixes JSON): mean {acc_mean:.0f}% "
            f"(95% CI [{acc_lo:.0f}%, {acc_hi:.0f}%]).",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--pairs-dir",
        type=Path,
        required=True,
        help="directory of <task>.before.json / <task>.after.json pairs",
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="write the markdown table here"
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="label the output as a synthetic plumbing check",
    )
    args = ap.parse_args(argv)

    binary = find_binary()
    if binary is None:
        print(
            "error: tracerazor binary not found — build with "
            "`cargo build --release -p tracerazor` or set TRACERAZOR_BIN",
            file=sys.stderr,
        )
        return 2

    pairs = discover_pairs(args.pairs_dir)
    if not pairs:
        print(f"error: no before/after pairs in {args.pairs_dir}", file=sys.stderr)
        return 2

    results = measure(binary, pairs)
    table = render_markdown(results, synthetic=args.synthetic)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(table)

    if not all(r.pass_held for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
