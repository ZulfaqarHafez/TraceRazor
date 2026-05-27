"""Convert a SWE-agent .traj file into TraceRazor's raw trace format.

SWE-agent trajectories have:
    {
      "environment": "swe_main",
      "trajectory": [
        {"thought": "...", "action": "...", "observation": "...",
         "response": "...", "execution_time": 0.24, "state": {...}}
      ],
      "history": [...],
      "info": {"submission": "...", "exit_status": "...", "model_stats": {...}}
    }

Mapping:
  trajectory[i].thought    -> reasoning step (paired with the action it precedes)
  trajectory[i].action     -> tool_call step (action verb = tool_name)
  trajectory[i].observation -> attached as `output` on the tool_call step

Token estimation: SWE-agent stores token totals in info.model_stats; we
distribute them roughly by character length per turn.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def parse_action(action: str) -> tuple[str, str]:
    """Return (tool_name, args_str) extracted from a SWE-agent action string.

    Actions look like:
        "create reproduce.py"
        "edit 1:10\nprint('hi')\nend_of_edit"
        "find_file Schema"
        "submit"
    The first whitespace-delimited token is the tool name; the rest is args.
    """
    if not action:
        return "noop", ""
    parts = action.split(None, 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def convert_trajectory(item: dict, agent_name: str, trace_id: str) -> dict:
    traj = item.get("trajectory") or []
    info = item.get("info") or {}
    exit_status = info.get("exit_status", "")
    # exit_status starts with "submitted" when the agent committed a patch.
    passed_flag = isinstance(exit_status, str) and exit_status.startswith("submitted")
    task_value = 1.0 if passed_flag else 0.0

    steps: list[dict] = []
    step_id = 0

    for turn in traj:
        thought = (turn.get("thought") or "").strip()
        action = (turn.get("action") or "").strip()
        observation = (turn.get("observation") or "")

        if thought:
            step_id += 1
            steps.append({
                "id": step_id,
                "type": "reasoning",
                "content": thought,
                "tokens": approx_tokens(thought),
            })

        if action:
            tool_name, args_str = parse_action(action)
            step_id += 1
            tool_success = True
            tool_error = None
            obs_lower = observation.lower()
            if "error" in obs_lower or "traceback" in obs_lower or "not found" in obs_lower:
                tool_success = False
                tool_error = observation[:500]
            steps.append({
                "id": step_id,
                "type": "tool_call",
                "content": f"Action: {tool_name} {args_str}".strip()[:400],
                "tokens": approx_tokens(action) + approx_tokens(observation),
                "tool_name": tool_name,
                "tool_params": {"args": args_str[:300]} if args_str else None,
                "tool_success": tool_success,
                "tool_error": tool_error,
                "output": observation[:2000],
            })

    return {
        "trace_id": trace_id,
        "agent_name": agent_name,
        "framework": "raw",
        "task_value_score": task_value,
        "steps": steps,
        "total_tokens": sum(s["tokens"] for s in steps),
        "metadata": {
            "source": "swe-agent/trajectories/demonstrations",
            "exit_status": exit_status,
            "passed": passed_flag,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="SWE-agent .traj file")
    ap.add_argument("--agent-name", default="swe-agent")
    ap.add_argument("--trace-id", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    trace_id = args.trace_id or Path(args.input).stem
    trace = convert_trajectory(data, agent_name=args.agent_name, trace_id=trace_id)
    Path(args.output).write_text(json.dumps(trace, indent=2))
    print(f"Wrote {len(trace['steps'])} steps to {args.output}")
    print(f"  passed={trace['task_value_score'] >= 1.0} total_tokens={trace['total_tokens']}")


if __name__ == "__main__":
    main()
