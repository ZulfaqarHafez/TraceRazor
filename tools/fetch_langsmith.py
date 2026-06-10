#!/usr/bin/env python3
"""Fetch LangSmith runs and write TraceRazor-ready trace files.

Usage:
    pip install langsmith
    export LANGSMITH_API_KEY=...
    python tools/fetch_langsmith.py --project my-project --limit 20 --out traces/langsmith/

Each root run (and its full child tree) is written as one JSON file that
`tracerazor audit -F langsmith` ingests directly. Flat `list_runs()` arrays
also work: the adapter rebuilds the tree from `parent_run_id`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True, help="LangSmith project name")
    ap.add_argument("--limit", type=int, default=20, help="max root runs")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    args = ap.parse_args()

    try:
        from langsmith import Client
    except ImportError:
        raise SystemExit("pip install langsmith (and set LANGSMITH_API_KEY)")

    client = Client()
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for run in client.list_runs(
        project_name=args.project, is_root=True, limit=args.limit
    ):
        tree = client.read_run(run.id, load_child_runs=True)
        payload = json.loads(json.dumps(tree.dict(), default=str))
        out = args.out / f"{run.id}.json"
        out.write_text(json.dumps(payload, indent=2))
        n += 1
        print(f"  {out}")
    print(f"Fetched {n} run trees. Audit the fleet with:")
    print(f"  tracerazor audit {args.out} -F langsmith")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
