#!/usr/bin/env python3
"""Build a calibration manifest from common harness exports.

`calibrate.py` consumes a manifest of entries, each either a trace with a known
recoverable fraction or a measured before/after pair. Writing that by hand is
tedious, so this adapter converts the shapes a harness usually emits:

  CSV / JSONL, before/after pair mode:
      python -m calibration.adapt --csv runs.csv \
          --before-col before_path --after-col after_path --out manifest.json

  CSV / JSONL, labelled-trace mode:
      python -m calibration.adapt --jsonl runs.jsonl \
          --trace-col path --label-col recoverable_fraction --out manifest.json

  Filename-paired directory:
      python -m calibration.adapt --pair-dir runs/ \
          --before-suffix _before.json --after-suffix _after.json --out manifest.json

Column / key names are configurable, so this fits whatever your harness writes
without editing code. Trace files may be in any format the auditor understands
(TraceRazor raw, LangSmith, or OTEL JSON); the auditor figures out the rest.

Paths are written into the manifest as absolute paths, so the manifest can live
anywhere. The before/after recoverable fraction is computed later by
`calibrate.py` from the measured token totals of each run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List


def _resolve(base: Path, p: str) -> Path:
    """Resolve a possibly-relative path to an absolute path, so manifest entries
    are location-independent (calibrate.py resolves them against the manifest's
    own directory)."""
    q = Path(p)
    return (q if q.is_absolute() else base / q).resolve()


def _exists(path: Path, allow_missing: bool) -> bool:
    """True if the file is present. With allow_missing, a missing file returns
    False (caller skips the row); otherwise it is a hard error."""
    if path.is_file():
        return True
    if allow_missing:
        return False
    sys.exit(f"file not found: {path} (use --allow-missing to skip the row)")


def _rows_from_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rows_from_jsonl(path: Path) -> List[Dict[str, str]]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows


def _entries_from_rows(rows, base, args) -> List[dict]:
    pair_mode = bool(args.before_col and args.after_col)
    label_mode = bool(args.trace_col and args.label_col)
    if pair_mode and label_mode:
        sys.exit("choose one mode: --before-col/--after-col OR --trace-col/--label-col, not both")
    if not pair_mode and not label_mode:
        sys.exit("specify either --before-col and --after-col, or --trace-col and --label-col")

    entries: List[dict] = []
    if pair_mode:
        for i, r in enumerate(rows):
            if args.before_col not in r or args.after_col not in r:
                sys.exit(f"row {i}: missing '{args.before_col}' or '{args.after_col}'")
            before = _resolve(base, str(r[args.before_col]))
            after = _resolve(base, str(r[args.after_col]))
            if not _exists(before, args.allow_missing) or not _exists(after, args.allow_missing):
                continue  # --allow-missing: skip rows whose files are absent
            entries.append({"before": str(before), "after": str(after)})
    else:
        for i, r in enumerate(rows):
            if args.trace_col not in r or args.label_col not in r:
                sys.exit(f"row {i}: missing '{args.trace_col}' or '{args.label_col}'")
            trace = _resolve(base, str(r[args.trace_col]))
            if not _exists(trace, args.allow_missing):
                continue
            try:
                frac = float(r[args.label_col])
            except (TypeError, ValueError):
                sys.exit(f"row {i}: label '{r[args.label_col]}' is not a number")
            if not math.isfinite(frac) or not 0.0 <= frac <= 1.0:
                sys.exit(f"row {i}: recoverable_fraction {frac} out of [0,1]")
            entries.append({"trace": str(trace), "recoverable_fraction": frac})

    if not entries:
        sys.exit("no entries produced (all rows skipped or empty input)")
    return entries


def _entries_from_pair_dir(args) -> List[dict]:
    d = Path(args.pair_dir)
    if not d.is_dir():
        sys.exit(f"not a directory: {d}")
    if not args.before_suffix or not args.after_suffix:
        sys.exit("--before-suffix and --after-suffix must be non-empty")
    if args.before_suffix == args.after_suffix:
        sys.exit("--before-suffix and --after-suffix must differ")
    # Exclude after-files from the before set in case the suffixes overlap
    # (e.g. before-suffix '.json' would otherwise also match '*_after.json').
    befores = [b for b in sorted(d.glob(f"*{args.before_suffix}"))
               if not b.name.endswith(args.after_suffix)]
    entries: List[dict] = []
    for b in befores:
        stem = b.name[: -len(args.before_suffix)]
        a = d / f"{stem}{args.after_suffix}"
        if not a.is_file():
            if args.allow_missing:
                continue
            sys.exit(f"no after file for '{b.name}': expected {a.name}")
        entries.append({"before": str(b.resolve()), "after": str(a.resolve())})
    if not entries:
        sys.exit(f"no '*{args.before_suffix}' files paired in {d}")
    return entries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a calibration manifest from harness exports.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="CSV file of runs")
    src.add_argument("--jsonl", type=Path, help="JSONL file of runs")
    src.add_argument("--pair-dir", type=Path, help="directory of before/after files")

    ap.add_argument("--out", type=Path, default=Path("manifest.json"))
    ap.add_argument("--name", default="imported-dataset", help="dataset name")
    ap.add_argument("--base-dir", type=Path, default=None,
                    help="resolve relative paths against this dir (default: source file's dir)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="skip rows whose trace files are missing instead of erroring")

    # CSV/JSONL field mapping (choose pair mode or label mode).
    ap.add_argument("--before-col")
    ap.add_argument("--after-col")
    ap.add_argument("--trace-col")
    ap.add_argument("--label-col")

    # pair-dir naming.
    ap.add_argument("--before-suffix", default="_before.json")
    ap.add_argument("--after-suffix", default="_after.json")
    args = ap.parse_args(argv)

    if args.pair_dir:
        entries = _entries_from_pair_dir(args)
    else:
        src_path = args.csv or args.jsonl
        rows = _rows_from_csv(src_path) if args.csv else _rows_from_jsonl(src_path)
        base = args.base_dir or src_path.parent
        entries = _entries_from_rows(rows, base, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"name": args.name, "entries": entries}, indent=2) + "\n")
    pairs = sum(1 for e in entries if "before" in e)
    labelled = len(entries) - pairs
    print(f"Wrote {len(entries)} entries ({pairs} before/after, {labelled} labelled) to {args.out}")
    print(f"Next: python -m calibration.calibrate --dataset {args.out} "
          f"--out config/tas_weights.json --report config/calibration_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
