"""Rebuild live-case-study traces from their persisted session transcripts.

Agent runs are expensive; converter improvements are cheap.  This utility
re-applies ``benchmark.convert_claude_code`` to the transcripts that
``run_live`` already captured — re-checking each preserved sandbox with
pytest for the task outcome — so accounting fixes never require re-running
the agent.

Looks for sandboxes under ``/tmp/lcs/<condition>/<pair>`` and their
transcripts under ``~/.claude/projects/-tmp-lcs-<condition>-<pair>/``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from benchmark.convert_claude_code import convert

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_TASKS_DIR = REPO / "benchmark" / "live" / "tasks"
DEFAULT_OUT_DIR = REPO / "benchmark" / "live" / "traces"
SANDBOX_ROOT = Path("/tmp/lcs")


def munge(path: Path) -> str:
    """Claude Code's project-directory name for a cwd."""
    return re.sub(r"[/.]", "-", str(path))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = ap.parse_args(argv)

    if not SANDBOX_ROOT.is_dir():
        print(f"error: no sandboxes under {SANDBOX_ROOT}", file=sys.stderr)
        return 2

    n = 0
    for cond_dir in sorted(SANDBOX_ROOT.iterdir()):
        condition = cond_dir.name
        for sandbox in sorted(cond_dir.iterdir()):
            pair = sandbox.name
            base = pair.split(".", 1)[0]
            prompt_file = args.tasks_dir / base / "prompt.md"
            if not prompt_file.is_file():
                print(f"  skip {pair}: no task prompt for '{base}'", file=sys.stderr)
                continue
            transcripts = sorted(
                Path.home().glob(f".claude/projects/{munge(sandbox)}/*.jsonl"),
                key=lambda p: p.stat().st_mtime,
            )
            if not transcripts:
                print(f"  skip {pair}/{condition}: no transcript", file=sys.stderr)
                continue
            check = subprocess.run(
                ["python3", "-m", "pytest", "-q", "--tb=no"],
                cwd=sandbox, capture_output=True, timeout=120,
            )
            passed = check.returncode == 0
            trace = convert(
                transcripts[-1],
                task=prompt_file.read_text(encoding="utf-8").strip(),
                task_value=1.0 if passed else 0.0,
                agent_name=f"claude-code ({args.model.split('-2')[0]})",
            )
            trace["trace_id"] = f"{pair}.{condition}"
            out = args.out_dir / f"{pair}.{condition}.json"
            out.write_text(
                json.dumps(trace, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"  {pair}.{condition}: {len(trace['steps'])} steps, "
                f"{trace['total_tokens']} tokens, pass={'Y' if passed else 'N'}"
            )
            n += 1
    print(f"reconverted {n} trace(s) into {args.out_dir}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
