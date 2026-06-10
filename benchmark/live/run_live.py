"""Live case-study runner: real agent runs, real traces, real pass checks.

Drives Claude Code headlessly over the task suite in ``benchmark/live/tasks``
(each task = ``prompt.md`` + a ``seed/`` project with a failing pytest
suite), converts every session transcript into a TraceRazor trace, and
writes ``<task>.<condition>.json`` pairs that
``python -m benchmark.case_study`` consumes directly.

Conditions:

- ``before`` — stock agent, just the task prompt.
- ``after``  — identical, plus ``--append-system-prompt`` loaded from the
  file produced by ``tracerazor apply`` on the *before* audits.  The only
  delta between conditions is the product's own emitted fixes.

Task outcome is objective: ``python3 -m pytest -q`` in the sandbox after
the agent finishes.  Green → ``task_value_score = 1.0``, red → ``0.0``.
The measurement harness enforces that pass flags hold from before to
after, so a token saving that costs correctness is called out, not
celebrated.

Usage::

    python3 -m benchmark.live.run_live --condition before
    python3 -m benchmark.live.run_live --condition after \
        --append-system-prompt-file results/optimized_prompt.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from benchmark.convert_claude_code import convert

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_TASKS_DIR = REPO / "benchmark" / "live" / "tasks"
DEFAULT_OUT_DIR = REPO / "benchmark" / "live" / "traces"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def run_one(
    task_dir: Path,
    condition: str,
    out_dir: Path,
    model: str,
    max_turns: int,
    timeout_s: int,
    extra_system_prompt: str | None,
    name_suffix: str = "",
) -> dict:
    task = task_dir.name + name_suffix
    prompt = (task_dir / "prompt.md").read_text(encoding="utf-8").strip()
    sandbox = Path("/tmp/lcs") / condition / task
    if sandbox.exists():
        shutil.rmtree(sandbox)
    shutil.copytree(task_dir / "seed", sandbox)

    # Tight permission envelope instead of skipping permissions: the agent
    # may edit files in its sandbox cwd and run pytest — nothing else (no
    # arbitrary shell, no network, no VCS). Denials outside the envelope are
    # identical across conditions, so they don't bias the comparison.
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "json",
        "--max-turns",
        str(max_turns),
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        "Bash(python3 -m pytest:*)",
        "Read",
        "Edit",
        "Write",
        "Glob",
        "Grep",
        "TodoWrite",
    ]
    if extra_system_prompt:
        cmd += ["--append-system-prompt", extra_system_prompt]

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=sandbox, capture_output=True, text=True, timeout=timeout_s
    )
    wall_s = time.time() - t0
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(
            f"{task}/{condition}: claude exited {proc.returncode}: "
            f"{proc.stderr.strip()[:500]}"
        )
    result = json.loads(proc.stdout)
    session_id = result["session_id"]

    check = subprocess.run(
        ["python3", "-m", "pytest", "-q", "--tb=no"],
        cwd=sandbox,
        capture_output=True,
        text=True,
        timeout=120,
    )
    passed = check.returncode == 0

    matches = list(Path.home().glob(f".claude/projects/*/{session_id}.jsonl"))
    if not matches:
        raise FileNotFoundError(f"{task}/{condition}: no transcript for {session_id}")
    trace = convert(
        matches[0],
        task=prompt,
        task_value=1.0 if passed else 0.0,
        agent_name=f"claude-code ({model.split('-2')[0]})",
    )
    trace["trace_id"] = f"{task}.{condition}"
    out_path = out_dir / f"{task}.{condition}.json"
    out_path.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    row = {
        "task": task,
        "condition": condition,
        "passed": passed,
        "steps": len(trace["steps"]),
        "trace_tokens": trace["total_tokens"],
        "num_turns": result.get("num_turns"),
        "cost_usd": result.get("total_cost_usd"),
        "wall_s": round(wall_s, 1),
        "session_id": session_id,
        "agent_exit": proc.returncode,
    }
    print(
        f"  {task:18s} pass={'Y' if passed else 'N'} "
        f"steps={row['steps']:3d} tokens={row['trace_tokens']:6d} "
        f"turns={row['num_turns']} cost=${row['cost_usd']:.3f} "
        f"wall={row['wall_s']}s",
        flush=True,
    )
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--condition", choices=("before", "after"), required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=600, help="per-task seconds")
    ap.add_argument(
        "--append-system-prompt-file",
        type=Path,
        default=None,
        help="extra system prompt (the applied TraceRazor fixes) — after runs",
    )
    ap.add_argument(
        "--task",
        action="append",
        default=None,
        help="run only this task (repeatable); default: all tasks",
    )
    ap.add_argument(
        "--name-suffix",
        default="",
        help="suffix for pair names, e.g. '.r2' for a second replicate",
    )
    args = ap.parse_args(argv)

    extra = None
    if args.append_system_prompt_file:
        extra = args.append_system_prompt_file.read_text(encoding="utf-8").strip()
        if not extra:
            print("error: --append-system-prompt-file is empty", file=sys.stderr)
            return 2

    task_dirs = sorted(
        d
        for d in args.tasks_dir.iterdir()
        if d.is_dir() and (d / "prompt.md").is_file()
    )
    if args.task:
        wanted = set(args.task)
        task_dirs = [d for d in task_dirs if d.name in wanted]
        missing = wanted - {d.name for d in task_dirs}
        if missing:
            print(f"error: unknown task(s): {sorted(missing)}", file=sys.stderr)
            return 2
    if not task_dirs:
        print("error: no tasks found", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"live run: {len(task_dirs)} task(s), condition={args.condition}, "
        f"model={args.model}",
        flush=True,
    )
    rows, failures = [], []
    for d in task_dirs:
        try:
            rows.append(
                run_one(
                    d,
                    args.condition,
                    args.out_dir,
                    args.model,
                    args.max_turns,
                    args.timeout,
                    extra,
                    args.name_suffix,
                )
            )
        except Exception as e:  # keep going; report at the end
            failures.append(f"{d.name}: {e}")
            print(f"  {d.name:18s} ERROR: {e}", flush=True)

    # Hyphenated so it can never match the harness's *.before.json /
    # *.after.json pair globs.  Merged, not overwritten: per-pair
    # invocations of the same condition accumulate into one log, with the
    # newest run per task winning.
    log_path = args.out_dir / f"runs-{args.condition}{args.name_suffix}.json"
    merged: dict[str, dict] = {}
    if log_path.is_file():
        for r in json.loads(log_path.read_text(encoding="utf-8")).get("runs", []):
            merged[r["task"]] = r
    for r in rows:
        merged[r["task"]] = r
    all_rows = sorted(merged.values(), key=lambda r: r["task"])
    log_path.write_text(
        json.dumps(
            {
                "condition": args.condition,
                "model": args.model,
                "max_turns": args.max_turns,
                "system_prompt_appended": bool(extra),
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_cost_usd": round(
                    sum(r["cost_usd"] or 0 for r in all_rows), 4
                ),
                "runs": all_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    passed = sum(r["passed"] for r in rows)
    print(
        f"done: {len(rows)}/{len(task_dirs)} runs, {passed}/{len(rows)} passed, "
        f"total cost ${sum(r['cost_usd'] or 0 for r in rows):.3f} — log: {log_path}"
    )
    if failures:
        print("failures:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
