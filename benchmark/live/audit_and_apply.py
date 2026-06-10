"""Stage 2 of the live case study: audit every *before* trace and apply fixes.

For each ``<pair>.before.json`` in the traces directory this script runs the
product's own pipeline, exactly as a user would:

1. ``tracerazor audit <pair>.before.json --format json --hermetic`` —
   the audit report, kept under ``results/`` for the record.
2. The report's fixes, **filtered to the subset ``apply`` actually
   applies** (safe prompt-level patches), are written to
   ``<pair>.fixes.json`` next to the pair so the measurement harness
   (``benchmark.case_study``) scores the audit's savings *estimate*
   against the *measured* delta for the fixes that really ran —
   estimating from unapplied fixes would inflate the denominator.
3. ``tracerazor apply <report> --to results/<pair>.prompt.txt`` — safe
   patches only (the product's default).  The resulting file is the ONLY
   delta between the before and after agent runs.

Fix patches are task-specific (e.g. ``goal_anchor`` quotes the task
objective), so prompts are built per pair, never pooled.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRACES_DIR = REPO / "benchmark" / "live" / "traces"
DEFAULT_RESULTS_DIR = REPO / "benchmark" / "live" / "results"

#: The fix types `tracerazor apply` applies without --all/--force (its
#: "safe" prompt-level set — keep in sync with the apply subcommand help).
APPLY_SAFE_TYPES = {
    "hedge_reduction",
    "verbosity_reduction",
    "caveman_prompt_insert",
    "reformulation_guard",
    "goal_anchor",
}


def find_binary() -> str:
    for cand in ("release", "debug"):
        p = REPO / "target" / cand / "tracerazor"
        if p.is_file():
            return str(p)
    sys.exit("error: build the CLI first: cargo build --release -p tracerazor")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces-dir", type=Path, default=DEFAULT_TRACES_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args(argv)

    binary = find_binary()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    befores = sorted(args.traces_dir.glob("*.before.json"))
    if not befores:
        print(f"error: no *.before.json in {args.traces_dir}", file=sys.stderr)
        return 2

    rows = []
    for before in befores:
        pair = before.name[: -len(".before.json")]
        report_path = args.results_dir / f"{pair}.report.json"
        prompt_path = args.results_dir / f"{pair}.prompt.txt"
        fixes_path = args.traces_dir / f"{pair}.fixes.json"

        audit = subprocess.run(
            [binary, "audit", str(before), "--format", "json",
             "--hermetic", "--store", "false"],
            capture_output=True, text=True,
        )
        if audit.returncode not in (0, 1):  # 1 = below threshold, still a report
            print(f"  {pair}: audit failed: {audit.stderr.strip()[:300]}",
                  file=sys.stderr)
            return 1
        report_path.write_text(audit.stdout, encoding="utf-8")
        report = json.loads(audit.stdout)
        applied_fixes = [
            f
            for f in report.get("fixes", [])
            if f.get("risk") == "safe" and f.get("fix_type") in APPLY_SAFE_TYPES
        ]
        if applied_fixes:
            fixes_path.write_text(
                json.dumps(applied_fixes, indent=2) + "\n", encoding="utf-8"
            )
        else:
            fixes_path.unlink(missing_ok=True)

        prompt_path.unlink(missing_ok=True)
        prompt_path.touch()
        applied = subprocess.run(
            [binary, "apply", str(report_path), "--to", str(prompt_path)],
            capture_output=True, text=True,
        )
        if applied.returncode != 0:
            print(f"  {pair}: apply failed: {applied.stderr.strip()[:300]}",
                  file=sys.stderr)
            return 1
        n_patches = applied.stdout.count("] ")  # "[k/n] fix_type" lines
        patch_chars = len(prompt_path.read_text(encoding="utf-8").strip())
        rows.append((pair, report["score"]["score"], n_patches, patch_chars))
        print(
            f"  {pair:24s} TAS={report['score']['score']:5.1f} "
            f"patches={n_patches} prompt_chars={patch_chars}"
        )

    empty = [p for p, _, n, c in rows if c == 0]
    print(f"done: {len(rows)} pair(s) audited; prompts in {args.results_dir}")
    if empty:
        print(
            "note: no safe patches for: "
            + ", ".join(empty)
            + " — run their after-condition without an appended prompt "
            "(measures the product honestly: nothing to apply, expect ~0 delta)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
