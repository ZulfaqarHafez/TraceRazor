"""Deterministic repair command for TRICE bundled live tasks.

This script is intentionally tiny and dependency-free. It is used by
``examples/trice_adapter_profile_bundled_tasks.json`` to exercise the command
adapter/profile/receipt path in real fresh workspaces.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    task_id = os.environ.get("TRICE_TASK_ID", "")
    changed = repair(task_id, Path.cwd())
    receipt_path = Path(os.environ.get("TRICE_AGENT_RECEIPT", ".trice/agent_receipt.json"))
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": "trice-agent-receipt/v1",
                "model": "deterministic-bundled-repair",
                "token_accounting": {
                    "input_tokens": int(os.environ.get("TRICE_INPUT_TOKENS", "0") or 0),
                    "baseline_input_tokens": int(os.environ.get("TRICE_BASELINE_INPUT_TOKENS", "0") or 0),
                    "output_tokens": 0,
                },
                "trice_context": {
                    "condition": os.environ.get("TRICE_CONDITION"),
                    "context_mode": os.environ.get("TRICE_CONTEXT_MODE"),
                    "budget_ratio": os.environ.get("TRICE_BUDGET_RATIO"),
                    "realized_budget_ratio": os.environ.get("TRICE_REALIZED_BUDGET_RATIO"),
                },
                "changed_files": changed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def repair(task_id: str, root: Path) -> list[str]:
    if task_id == "csv-filter":
        code = (
            "import csv\n\n\n"
            "def filter_rows(path, min_score):\n"
            "    out = []\n"
            "    with open(path, newline=\"\", encoding=\"utf-8\") as fh:\n"
            "        for row in csv.DictReader(fh):\n"
            "            score = int(row[\"score\"])\n"
            "            if score >= min_score:\n"
            "                out.append({\"name\": row[\"name\"], \"score\": score})\n"
            "    return out\n"
        )
        (root / "filt.py").write_text(code, encoding="utf-8")
        return ["filt.py"]
    if task_id == "dedupe-helpers":
        (root / "textutil.py").write_text(
            "def normalize_name(name):\n"
            "    return \" \".join(str(name).split()).strip().lower()\n",
            encoding="utf-8",
        )
        (root / "utils_a.py").write_text(
            "from textutil import normalize_name\n\n\n"
            "def label_for(name):\n"
            "    return f\"user:{normalize_name(name)}\"\n",
            encoding="utf-8",
        )
        (root / "utils_b.py").write_text(
            "from textutil import normalize_name\n\n\n"
            "def greeting(name):\n"
            "    return f\"hello {normalize_name(name)}\"\n",
            encoding="utf-8",
        )
        return ["textutil.py", "utils_a.py", "utils_b.py"]
    if task_id == "fix-imports":
        replace(root / "mypkg" / "api.py", "from loaders import read_rows", "from .loaders import read_rows")
        (root / "mypkg" / "__init__.py").write_text(
            "from .api import run_pipeline\n\n__all__ = [\"run_pipeline\"]\n",
            encoding="utf-8",
        )
        return ["mypkg/__init__.py", "mypkg/api.py"]
    if task_id == "fix-offby-one":
        replace(root / "chunker.py", "size - 1", "size")
        return ["chunker.py"]
    if task_id == "implement-median":
        replace(
            root / "stats.py",
            "    raise NotImplementedError\n",
            (
                "    if not xs:\n"
                "        raise ValueError(\"median of empty sequence\")\n"
                "    values = sorted(xs)\n"
                "    mid = len(values) // 2\n"
                "    if len(values) % 2:\n"
                "        return values[mid]\n"
                "    return (values[mid - 1] + values[mid]) / 2\n"
            ),
        )
        return ["stats.py"]
    if task_id == "rename-api":
        replace(root / "core.py", "def fetch_data", "def load_records")
        replace(root / "report.py", "fetch_data", "load_records")
        replace(root / "cli.py", "fetch_data", "load_records")
        return ["cli.py", "core.py", "report.py"]
    raise SystemExit(f"unknown bundled task id: {task_id}")


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"old text not found in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
