#!/usr/bin/env python3
"""Convert OpenAI/Anthropic chat-completions logs into TraceRazor traces.

Usage:
    python tools/convert_openai.py conversation.json -o trace.json
    tracerazor audit trace.json

Accepts the artifact most developers actually have: a `{"messages": [...]}`
request body (or a bare `[{"role": ...}]` list) in either the OpenAI shape
(string `content`, `tool_calls`, `role: "tool"`) or the Anthropic shape
(`content` as a list of text / tool_use / tool_result blocks).

Mapping:
  assistant text          -> reasoning step
  assistant tool_calls /
  tool_use blocks         -> one tool_call step each (params from arguments)
  tool / tool_result      -> output + success on the matching tool_call step
                             (matched by tool_call_id / tool_use_id;
                             `is_error: true` or error-looking text => failure)
  user / system text      -> input_context of the NEXT step; the first user
                             message also becomes metadata.task so the
                             goal-oriented metrics (GAR, TPE) anchor on it

Token counts: per-message `usage` is used when present; otherwise the
fallback is len(text) // chars-per-token (default 4) — a rough estimate,
clearly inferior to real usage numbers. Pass exact counts if you have them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ERROR_RE = re.compile(r"\b(error|exception|traceback|failed|failure)\b", re.I)


def text_of(content) -> str:
    """Flatten OpenAI string content or Anthropic block-list content to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # Anthropic blocks
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def estimate_tokens(text: str, chars_per_token: int) -> int:
    return max(1, len(text) // chars_per_token) if text else 1


def convert(messages: list, agent_name: str, trace_id: str, framework: str,
            chars_per_token: int) -> dict:
    steps: list[dict] = []
    pending_context: list[str] = []  # user/system text awaiting the next step
    by_call_id: dict[str, dict] = {}  # tool_call_id -> step (for results)
    task: str | None = None

    def push(step: dict) -> None:
        if pending_context:
            step["input_context"] = "\n".join(pending_context)
            pending_context.clear()
        step["id"] = len(steps) + 1
        steps.append(step)

    def usage_tokens(msg: dict) -> int | None:
        u = msg.get("usage") or {}
        for key in ("total_tokens", "output_tokens", "completion_tokens"):
            if isinstance(u.get(key), int):
                return u[key]
        return None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role in ("user", "system", "developer"):
            text = text_of(content)
            if role == "user" and task is None and text:
                task = text
            if text:
                pending_context.append(text)
            # Anthropic puts tool results in user messages as blocks.
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        attach_result(by_call_id, block.get("tool_use_id"),
                                      text_of(block.get("content")),
                                      bool(block.get("is_error")))
            continue

        if role == "tool":  # OpenAI tool-result message
            attach_result(by_call_id, msg.get("tool_call_id"),
                          text_of(content), False)
            continue

        if role != "assistant":
            continue

        text = text_of(content)
        if text:
            push({
                "type": "reasoning",
                "content": text,
                "tokens": usage_tokens(msg) or estimate_tokens(text, chars_per_token),
            })

        # OpenAI tool calls
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            try:
                params = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                params = {"raw": fn.get("arguments")}
            step = {
                "type": "tool_call",
                "content": f"{fn.get('name', 'tool')}({fn.get('arguments', '')})",
                "tokens": estimate_tokens(fn.get("arguments") or "", chars_per_token),
                "tool_name": fn.get("name", "tool"),
                "tool_params": params,
                "tool_success": True,
            }
            push(step)
            if call.get("id"):
                by_call_id[call["id"]] = step

        # Anthropic tool_use blocks
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    step = {
                        "type": "tool_call",
                        "content": f"{block.get('name', 'tool')}({json.dumps(block.get('input', {}))})",
                        "tokens": estimate_tokens(json.dumps(block.get("input", {})),
                                                  chars_per_token),
                        "tool_name": block.get("name", "tool"),
                        "tool_params": block.get("input", {}),
                        "tool_success": True,
                    }
                    push(step)
                    if block.get("id"):
                        by_call_id[block["id"]] = step

    trace = {
        "trace_id": trace_id,
        "agent_name": agent_name,
        "framework": framework,
        "steps": steps,
    }
    if task:
        trace["metadata"] = {"task": task}
    return trace


def attach_result(by_call_id: dict, call_id, result_text: str, is_error: bool) -> None:
    step = by_call_id.get(call_id)
    if step is None:
        return
    step["output"] = result_text
    failed = is_error or bool(ERROR_RE.search(result_text or ""))
    step["tool_success"] = not failed
    if failed:
        step["tool_error"] = (result_text or "tool error")[:500]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="chat log JSON file")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output trace path")
    ap.add_argument("--agent-name", default="converted-agent")
    ap.add_argument("--trace-id", default=None)
    ap.add_argument("--framework", default=None,
                    help="default: detected (anthropic if tool_use blocks, else openai)")
    ap.add_argument("--chars-per-token", type=int, default=4,
                    help="token estimate divisor when usage is absent (default 4)")
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    messages = data.get("messages", data) if isinstance(data, dict) else data
    if not isinstance(messages, list):
        print("error: expected a messages array or {\"messages\": [...]}", file=sys.stderr)
        return 2

    blob = json.dumps(messages)
    framework = args.framework or ("anthropic" if '"tool_use"' in blob else "openai")
    trace_id = args.trace_id or f"converted-{int(time.time())}"

    trace = convert(messages, args.agent_name, trace_id, framework,
                    args.chars_per_token)
    if len(trace["steps"]) < 5:
        print(f"warning: only {len(trace['steps'])} steps converted — TraceRazor "
              "needs >= 5 steps to compute its metrics", file=sys.stderr)

    args.out.write_text(json.dumps(trace, indent=2) + "\n")
    print(f"wrote {args.out} ({len(trace['steps'])} steps); "
          f"audit with: tracerazor audit {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
