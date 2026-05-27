"""Convert tau-bench historical_trajectories JSON into TraceRazor's raw trace format.

Mapping:
  assistant turn with tool_calls         -> step_type = "tool_call"  (one step per call)
  assistant turn with plain content      -> step_type = "reasoning"
  tool turn                              -> attached as `output` on the previous tool_call
  user/system turns                      -> rolled into `input_context` of the next assistant turn

Token estimation: tau-bench trajectories don't carry per-turn token counts, so
we approximate via len(content) / 4 (rough OpenAI tokens-per-char). Good enough
for relative metrics (SRR/LDI/CCE/VDI) since we only care about ratios.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def convert_trajectory(item: dict, agent_name: str = "tau-bench-airline-gpt-4o") -> dict:
    traj = item["traj"]
    task_id = item["task_id"]
    reward = float(item.get("reward", 0.0))
    trial = item.get("trial", 0)

    steps: list[dict] = []
    pending_input_context: list[str] = []
    prev_tool_step: dict | None = None
    step_id = 0

    for turn in traj:
        role = turn.get("role")
        content = turn.get("content") or ""
        tool_calls = turn.get("tool_calls") or []

        if role in ("user", "system"):
            pending_input_context.append(f"{role}: {content}")
            continue

        if role == "tool":
            # Attach the tool result to the previous tool_call step.
            if prev_tool_step is not None:
                prev_tool_step["output"] = content[:2000]  # cap for cleanliness
                # Treat any tool result containing "error" as a failure.
                if any(k in content.lower() for k in ("error", '"error":', "exception")):
                    prev_tool_step["tool_success"] = False
                    prev_tool_step["tool_error"] = content[:500]
                else:
                    prev_tool_step["tool_success"] = True
                prev_tool_step["tokens"] += approx_tokens(content)
            continue

        if role != "assistant":
            continue

        if tool_calls:
            for tc in tool_calls:
                step_id += 1
                fn = tc.get("function", {})
                name = fn.get("name", "unknown")
                args_raw = fn.get("arguments", "")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = {"raw": str(args_raw)[:200]}
                step = {
                    "id": step_id,
                    "type": "tool_call",
                    "content": f"Calling {name} with {json.dumps(args)[:300]}",
                    "tokens": approx_tokens(args_raw if isinstance(args_raw, str) else json.dumps(args)),
                    "tool_name": name,
                    "tool_params": args,
                }
                if pending_input_context:
                    step["input_context"] = "\n".join(pending_input_context)[:2000]
                    pending_input_context = []
                steps.append(step)
                prev_tool_step = step
        else:
            # Pure assistant reasoning / final answer.
            step_id += 1
            step = {
                "id": step_id,
                "type": "reasoning",
                "content": content,
                "tokens": approx_tokens(content),
            }
            if pending_input_context:
                step["input_context"] = "\n".join(pending_input_context)[:2000]
                pending_input_context = []
            steps.append(step)
            prev_tool_step = None

    return {
        "trace_id": f"tau-bench-task-{task_id}-trial-{trial}",
        "agent_name": agent_name,
        "framework": "raw",
        "task_value_score": reward,  # 1.0 = passed, 0.0 = failed → drives TVI multiplier
        "steps": steps,
        "total_tokens": sum(s["tokens"] for s in steps),
        "metadata": {
            "source": "tau-bench/historical_trajectories",
            "model": agent_name,
            "reward": reward,
            "passed": reward >= 1.0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="tau-bench trajectory JSON file")
    ap.add_argument("--task-id", type=int, default=0, help="task_id to extract")
    ap.add_argument("--trial", type=int, default=0, help="trial index for the task")
    ap.add_argument("--agent-name", default="tau-bench-airline-gpt-4o")
    ap.add_argument("--output", required=True, help="output TraceRazor trace JSON path")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    candidates = [x for x in data if x["task_id"] == args.task_id and x.get("trial", 0) == args.trial]
    if not candidates:
        print(f"task_id={args.task_id} trial={args.trial} not found", file=sys.stderr)
        sys.exit(2)

    trace = convert_trajectory(candidates[0], agent_name=args.agent_name)
    Path(args.output).write_text(json.dumps(trace, indent=2))
    print(f"Wrote {len(trace['steps'])} steps to {args.output}")
    print(f"  task_value_score={trace['task_value_score']} total_tokens={trace['total_tokens']}")


if __name__ == "__main__":
    main()
