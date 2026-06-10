#!/usr/bin/env python3
"""Convert Hugging Face ``zai-org/AgentInstruct`` ReAct trajectories into
TraceRazor's raw trace format.

AgentInstruct rows look like::

    {"id": "os_3", "conversations": [
        {"from": "human", "loss": None, "value": "<system + task / OS output>"},
        {"from": "gpt",   "loss": ...,  "value": "Think: ...\\n\\nAct: bash\\n```bash\\n...```"},
        ...]}

Mapping into TraceRazor steps:
  gpt turn with ``Act: bash`` / ``Action: Operation``  -> step_type "tool_call"
      (tool_name "bash" or "sql"; the reasoning text becomes ``content``)
  gpt turn with ``Act: answer(...)`` / ``Action: Answer`` / ``Act: finish``  -> "reasoning"
  human turn after a tool_call  -> attached as that step's ``output``
  leading/other human turns      -> rolled into the next step's ``input_context``

Token estimation mirrors ``tools/convert_tau_bench.py`` (len/4) since the source
carries no per-turn token counts; only ratios matter for SRR/LDI/CCE/VDI/OBS.

Usage:
    # From the vendored real Hugging Face sample (offline, hermetic):
    python -m tools.convert_agentinstruct --bundled \\
        --out-dir traces/external/huggingface/agentinstruct

    # From a JSONL snapshot of dataset rows (one row per line):
    python -m tools.convert_agentinstruct --jsonl rows.jsonl --out-dir out/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Output substrings that signal a genuine shell/SQL *failure* (as opposed to the
# word "error" merely appearing in legitimate command output, e.g. a log file).
_FAILURE_SIGNALS = (
    "no such file or directory",
    "command not found",
    "permission denied",
    "syntax error",
    "cannot access",
    "cannot open",
    "cannot remove",
    "not a directory",
    "operation not permitted",
    "error 1064",  # MySQL syntax error code
    "you have an error in your sql syntax",
)

_CODE_FENCE = re.compile(r"```(?:bash|sql|sh)?\s*\n?(.*?)```", re.DOTALL)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _looks_failed(output: str) -> bool:
    low = output.lower()
    return any(sig in low for sig in _FAILURE_SIGNALS)


def _extract_code(value: str) -> Optional[str]:
    m = _CODE_FENCE.search(value)
    if m:
        return m.group(1).strip()
    return None


def _real_task_turns(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only the turns of the *real* task, excluding few-shot scaffolding.

    AgentInstruct rows embed the dataset's fixed one-shot demonstration (and the
    db split's "Ok." acknowledgement) before the real trajectory, and mark it
    via the ``loss`` flag: scaffolding gpt turns carry ``loss: false``, the real
    agent turns carry ``loss: true``. Auditing scaffolding as agent behaviour
    pseudo-replicates the same canned steps into every trace and misanchors the
    goal metrics, so it is excluded. The human turn immediately preceding the
    first real gpt turn (which states the real problem) is kept as context.

    When loss flags are absent, fall back to the textual boundary marker; when
    neither exists, the row has no scaffolding and is returned unchanged.
    """
    losses = [t.get("loss") for t in conversations if t.get("from") == "gpt"]
    if True in losses and False in losses:
        for i, t in enumerate(conversations):
            if t.get("from") == "gpt" and t.get("loss") is True:
                if i > 0 and conversations[i - 1].get("from") == "human":
                    return conversations[i - 1:]
                return conversations[i:]
    marker = "i will start a new problem"
    for i in range(len(conversations) - 1, -1, -1):
        t = conversations[i]
        if t.get("from") == "human" and marker in (t.get("value") or "").lower():
            return conversations[i:]
    return conversations


def _classify_gpt_turn(value: str) -> Dict[str, Any]:
    """Classify one assistant turn into an action.

    Returns a dict: {kind: "bash"|"sql"|"answer"|"finish"|"reasoning",
                     code: Optional[str]}.
    """
    low = value.lower()
    code = _extract_code(value)

    # SQL agent (db split): "Action: Operation" / "Action: Answer".
    if "action: operation" in low:
        return {"kind": "sql", "code": code}
    if "action: answer" in low:
        return {"kind": "answer", "code": None}

    # Bash agent (os split): "Act: bash" / "Act: answer(...)" / "Act: finish".
    if "act: bash" in low or (code is not None and "act:" in low):
        return {"kind": "bash", "code": code}
    if "act: answer" in low:
        return {"kind": "answer", "code": None}
    if "act: finish" in low:
        return {"kind": "finish", "code": None}

    # No recognised action (e.g. the "Ok." acknowledgement) -> plain reasoning.
    return {"kind": "reasoning", "code": None}


def convert_conversations(
    conversations: List[Dict[str, Any]],
    row_id: str,
    domain: str = "agentinstruct",
    instruction: Optional[str] = None,
    task_value_score: float = 1.0,
) -> Dict[str, Any]:
    """Convert one AgentInstruct conversation into a TraceRazor trace dict."""
    steps: List[Dict[str, Any]] = []
    pending_input: List[str] = []
    prev_tool_step: Optional[Dict[str, Any]] = None
    step_id = 0

    conversations = _real_task_turns(conversations)

    for turn in conversations:
        who = turn.get("from")
        value = turn.get("value") or ""

        if who == "human":
            # Environment / user turn.
            if prev_tool_step is not None and value.startswith("The output of the OS:"):
                # Tool/observation result for the previous tool call.
                prev_tool_step["output"] = value[:2000]
                prev_tool_step["tokens"] += approx_tokens(value)
                prev_tool_step["tool_success"] = not _looks_failed(value)
                if _looks_failed(value):
                    prev_tool_step["tool_error"] = value[:300]
                prev_tool_step = None
            elif prev_tool_step is not None:
                # Generic environment reply (e.g. raw MySQL output "[(0,)]").
                prev_tool_step["output"] = value[:2000]
                prev_tool_step["tokens"] += approx_tokens(value)
                prev_tool_step["tool_success"] = not _looks_failed(value)
                if _looks_failed(value):
                    prev_tool_step["tool_error"] = value[:300]
                prev_tool_step = None
            else:
                pending_input.append(value)
            continue

        if who != "gpt":
            continue

        action = _classify_gpt_turn(value)
        step_id += 1
        kind = action["kind"]

        if kind in ("bash", "sql"):
            tool_name = "bash" if kind == "bash" else "sql"
            param_key = "command" if kind == "bash" else "query"
            code = action["code"] or ""
            step = {
                "id": step_id,
                "type": "tool_call",
                "content": value,
                "tokens": approx_tokens(value),
                "tool_name": tool_name,
                "tool_params": {param_key: code},
            }
            if pending_input:
                step["input_context"] = "\n".join(pending_input)[:2000]
                pending_input = []
            steps.append(step)
            prev_tool_step = step
        else:
            # reasoning / answer / finish.
            step = {
                "id": step_id,
                "type": "reasoning",
                "content": value,
                "tokens": approx_tokens(value),
            }
            if pending_input:
                step["input_context"] = "\n".join(pending_input)[:2000]
                pending_input = []
            steps.append(step)
            prev_tool_step = None

    metadata: Dict[str, Any] = {
        "source": "huggingface:zai-org/AgentInstruct",
        "dataset": "zai-org/AgentInstruct",
        "row_id": row_id,
        "domain": domain,
    }
    if instruction:
        # Anchors goal-oriented metrics (GAR, TPE) on the real task objective.
        metadata["task"] = instruction

    return {
        "trace_id": f"agentinstruct-{row_id}",
        "agent_name": f"agentinstruct-{domain}",
        "framework": "raw",
        "task_value_score": task_value_score,
        "steps": steps,
        "total_tokens": sum(s["tokens"] for s in steps),
        "metadata": metadata,
    }


def convert_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a full dataset row (with id/domain/instruction/conversations)."""
    return convert_conversations(
        row["conversations"],
        row_id=row.get("id", "unknown"),
        domain=row.get("domain", "agentinstruct"),
        instruction=row.get("instruction"),
    )


def _load_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.bundled:
        from benchmark.data._agentinstruct_hf_sample import ROWS
        return list(ROWS)
    if args.jsonl:
        rows = []
        for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    raise SystemExit("specify --bundled or --jsonl <file>")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AgentInstruct ReAct -> TraceRazor traces.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bundled", action="store_true",
                     help="use the vendored real Hugging Face sample")
    src.add_argument("--jsonl", type=Path, help="JSONL of dataset rows (one per line)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="directory to write per-trace JSON files")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args)
    n = 0
    for row in rows:
        trace = convert_row(row)
        out = args.out_dir / f"{trace['trace_id']}.json"
        out.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        n += 1
        print(f"  {out.name}: {len(trace['steps'])} steps, {trace['total_tokens']} tokens")
    print(f"Wrote {n} traces to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
