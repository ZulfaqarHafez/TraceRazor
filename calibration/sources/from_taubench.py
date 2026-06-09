#!/usr/bin/env python3
"""Emit a calibration JSONL from tau-bench historical trajectories.

tau-bench (github.com/sierra-research/tau-bench) commits real agent runs under
`historical_trajectories/<model>-<domain>.json`, each a list of episodes with
`task_id`, `reward` (1.0 = success), and `traj` (OpenAI messages). The same
tasks are solved by multiple models and trials, so successful runs of the same
task at differing token cost form genuine before/after pairs at equal quality.

This is reachable from a locked-down environment (GitHub is allowlisted even
when Hugging Face is not):

    git clone --depth 1 https://github.com/sierra-research/tau-bench
    python -m calibration.sources.from_taubench \
        --dir tau-bench/historical_trajectories --out taubench.jsonl
    python -m calibration.sources.from_messages --jsonl taubench.jsonl \
        --out-dir converted --manifest manifest.json
    python -m calibration.calibrate --dataset manifest.json \
        --out config/tas_weights.json --report config/calibration_report.md

Use `--within-model` to pair only same-model/different-trial runs (the purest
within-agent recoverable-waste signal); omit it to also pair across models.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="tau-bench trajectories -> calibration JSONL.")
    ap.add_argument("--dir", required=True, type=Path, help="historical_trajectories directory")
    ap.add_argument("--out", type=Path, default=Path("taubench.jsonl"))
    ap.add_argument("--min-reward", type=float, default=1.0, help="keep episodes with reward >= this")
    ap.add_argument("--within-model", action="store_true",
                    help="scope pairing to same model (instance id includes the model)")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(args.dir / "*.json")))
    if not files:
        sys.exit(f"no *.json under {args.dir}")
    n = 0
    with args.out.open("w", encoding="utf-8") as out:
        for f in files:
            base = os.path.basename(f).replace(".json", "")
            parts = base.split("-")
            model, domain = "-".join(parts[:-1]), parts[-1]
            for ep in json.load(open(f)):
                if float(ep.get("reward", 0)) < args.min_reward:
                    continue
                if "traj" not in ep:
                    continue
                task = f"{domain}-{ep.get('task_id')}"
                instance = f"{task}-{model}" if args.within_model else task
                out.write(json.dumps({
                    "instance_id": instance,
                    "model": f"{model}#t{ep.get('trial', 0)}",
                    "resolved": True,
                    # Task group (no model): same task never split across CV folds.
                    "group": task,
                    "messages": ep["traj"],
                }) + "\n")
                n += 1
    print(f"Wrote {n} successful episodes from {len(files)} files to {args.out}")
    print("Next: python -m calibration.sources.from_messages --jsonl "
          f"{args.out} --out-dir converted --manifest manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
