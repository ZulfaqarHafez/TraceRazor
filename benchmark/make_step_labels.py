"""Generate a labelled step corpus for auditor detection P/R evaluation.

Runs the MockAgent with the *baseline* config (all waste injection on) over
a variety of tasks, then labels each step based on the runner's own injection
logic -- an independent ground truth that did NOT come from the auditor heuristics.

Independence guarantee: the ground-truth labels are derived from the data-
generation code (``teacher/runner.py`` ``run_task()``), which uses fixed string
templates to inject waste.  The auditor's detection heuristics (SRR Jaccard,
LDI identical-call, SHL regex, RDA step-count) are independent algorithms that
were NOT consulted to produce these labels.

Output:
    benchmark/labels/labelled_steps.jsonl  -- one step record per line
    benchmark/labels/labelled_traces.jsonl -- one trace record per line

Usage:
    python -m benchmark.make_step_labels [--n-tasks N] [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teacher.runner import FILLER, HEDGE, Task, run_task
from teacher.schemas import AgentConfig

# Tasks vary by number of required tools: 1-4 tools in round-robin
_TOOL_SETS = [
    ["search_db"],
    ["search_db", "send_email"],
    ["lookup", "validate", "write_record"],
    ["lookup", "validate", "call_api", "commit"],
]


def _label_step(step: dict, tool_sig_count: dict[tuple, int]) -> dict:
    """Assign a ground-truth label to one step based on runner injection logic."""
    stype = step.get("type", "")
    content = step.get("content", "")

    if stype == "reasoning":
        if content.startswith("Certainly!"):
            label, removable, reason = (
                "preamble", True,
                "HEDGE preamble injected by runner; removable via NO_HEDGING",
            )
        elif "Let me re-read the request again:" in content:
            label, removable, reason = (
                "reformulation", True,
                "Reformulation step injected by runner; removable via NO_REFORMULATION",
            )
        elif "Let me also consider several unlikely edge cases." in content:
            label, removable, reason = (
                "overdepth", True,
                "Over-depth speculation injected by runner; removable via STEP_BUDGET",
            )
        else:
            label, removable, reason = (
                "final_answer", False,
                "Final answer reasoning; not removable",
            )
    elif stype == "tool_call":
        sig = (step.get("tool_name"), json.dumps(step.get("tool_params", {}), sort_keys=True))
        if tool_sig_count.get(sig, 0) > 0:
            label, removable, reason = (
                "loop_repeat", True,
                "Identical tool call repeated by runner; removable via loop_breaker",
            )
        else:
            label, removable, reason = (
                "required_tool", False,
                "Required tool call; not removable",
            )
        tool_sig_count[sig] = tool_sig_count.get(sig, 0) + 1
    else:
        label, removable, reason = "unknown", False, "Unknown step type"

    return {
        "trace_id": step.get("_trace_id", ""),
        "step_id": step.get("id"),
        "step_type": stype,
        "content_preview": content[:150],
        "tokens": step.get("tokens", 0),
        "label": label,
        "removable": removable,
        "reason": reason,
    }


def label_trace(trace: dict) -> list[dict]:
    """Label all steps in a trace; inject trace_id into each step for convenience."""
    steps = trace.get("steps", [])
    tool_sig_count: dict[tuple, int] = {}
    labelled = []
    for s in steps:
        s = dict(s)
        s["_trace_id"] = trace.get("trace_id", "?")
        labelled.append(_label_step(s, tool_sig_count))
    return labelled


def generate(n_tasks: int = 200) -> tuple[list[dict], list[dict]]:
    """Return (all_steps, all_traces) labelled from MockAgent baseline runs."""
    cfg = AgentConfig()  # baseline: all waste injection active
    all_steps: list[dict] = []
    all_traces: list[dict] = []

    for i in range(n_tasks):
        tools = _TOOL_SETS[i % len(_TOOL_SETS)]
        task = Task(
            task_id=f"task_{i:04d}",
            goal=f"complete_objective_{i:04d}",
            required_tools=tools,
        )
        out = run_task(cfg, task)
        trace = out["trace"]
        labelled = label_trace(trace)
        all_steps.extend(labelled)
        all_traces.append({
            "trace_id": trace["trace_id"],
            "n_steps": len(trace["steps"]),
            "total_tokens": out["tokens"],
            "success": out["success"],
            "n_removable": sum(1 for s in labelled if s["removable"]),
            "n_required": sum(1 for s in labelled if not s["removable"]),
            "steps": labelled,
        })

    return all_steps, all_traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labelled step corpus")
    parser.add_argument("--n-tasks", type=int, default=200,
                        help="Number of tasks to run (default: 200)")
    parser.add_argument("--output-dir", default="benchmark/labels",
                        help="Output directory (default: benchmark/labels)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_tasks} labelled traces (MockAgent, baseline config)...")
    all_steps, all_traces = generate(args.n_tasks)

    steps_path = out_dir / "labelled_steps.jsonl"
    traces_path = out_dir / "labelled_traces.jsonl"

    with open(steps_path, "w") as fh:
        for s in all_steps:
            fh.write(json.dumps(s) + "\n")
    with open(traces_path, "w") as fh:
        for t in all_traces:
            fh.write(json.dumps(t) + "\n")

    n_removable = sum(1 for s in all_steps if s["removable"])
    n_total = len(all_steps)
    label_counts: dict[str, int] = {}
    for s in all_steps:
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1

    print(f"  {n_total} steps total: {n_removable} removable, {n_total - n_removable} not")
    print(f"  Label breakdown: {json.dumps(label_counts, indent=None)}")
    print(f"  Written to {steps_path}")
    print(f"  Written to {traces_path}")


if __name__ == "__main__":
    main()
