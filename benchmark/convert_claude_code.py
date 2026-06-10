"""Convert a Claude Code session transcript (JSONL) into a TraceRazor trace.

Claude Code (the CLI agent) writes one JSONL transcript per session under
``~/.claude/projects/<munged-cwd>/<session-id>.jsonl``.  Those transcripts
are *real* agent traces: every API call carries exact token usage, every
tool call carries its name, parameters and result.  This converter maps
them onto TraceRazor's trace schema so real Claude Code sessions can be
audited like any other agent run::

    python -m benchmark.convert_claude_code session.jsonl \
        --task "Fix the failing test" --task-value 1.0 --out trace.json

Mapping
-------
- Each *API message* (assistant lines grouped by ``message.id`` — the CLI
  writes one line per content block, repeating the same usage object on
  each) becomes one or more steps:

  - ``thinking`` / ``text`` blocks  -> one ``reasoning`` step per message
    (concatenated; they are a single model turn's reasoning).
  - each ``tool_use`` block         -> one ``tool_call`` step, joined with
    its ``tool_result`` (success flag, output text) from the following
    user line.

- **Token accounting is marginal**: each message's
  ``input_tokens + cache_creation_input_tokens + output_tokens`` — the new
  tokens this turn added — split equally across the steps it produced.
  Cache *reads* (the re-fed conversation prefix, billed at a discount) are
  excluded by default so a step's cost reflects what that step itself
  introduced; pass ``--include-cache-read`` for gross accounting.  The
  convention is recorded in the trace metadata either way.

- **The first turn's ``cache_creation`` is excluded** under marginal
  accounting: it is the one-time encoding of the harness prefix (system
  prompt + environment + task), and whether a given run pays it as a
  cache *write* (cold) or a cache *read* (warm) is decided by cache state
  at launch — observed swings of ±22k tokens on otherwise-identical runs.
  It is constant across compared conditions by construction, so excluding
  it removes infrastructure noise without touching agent behavior.  The
  first turn still counts its ``input_tokens + output_tokens``.

- ``input_context`` is the *new* context injected since the previous
  model turn (tool results + user text), attached to the **first** step
  of each message only — the context enters the model once per API call,
  so repeating it on every parallel tool step would manufacture
  context-bloat findings that don't exist.

- Sidechain entries (subagent traffic, ``isSidechain: true``) get
  ``agent_id = "subagent"``; the main thread carries no agent_id.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Truncation limits keep converted traces reviewable and diff-able while
#: leaving plenty of signal for the semantic metrics (SRR/CCE work on
#: openings and shingles, not on full file dumps).
MAX_CONTENT_CHARS = 2_000
MAX_OUTPUT_CHARS = 2_000
MAX_CONTEXT_CHARS = 4_000


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[+{len(text) - limit} chars]"


def _block_text(content) -> str:
    """Flatten a message ``content`` field (str or block list) to text."""
    if isinstance(content, str):
        return content
    parts = []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
    return "\n".join(p for p in parts if p)


def load_entries(path: Path) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") in ("assistant", "user"):
                entries.append(d)
    return entries


def _usage_tokens(
    usage: dict, include_cache_read: bool, first_turn: bool = False
) -> int:
    total = int(usage.get("input_tokens", 0) or 0) + int(
        usage.get("output_tokens", 0) or 0
    )
    # First-turn cache_creation is the harness-prefix encoding — an
    # infrastructure cost paid (or not) depending on cache warmth, so it is
    # only counted under gross accounting (see module docstring).
    if not first_turn or include_cache_read:
        total += int(usage.get("cache_creation_input_tokens", 0) or 0)
    if include_cache_read:
        total += int(usage.get("cache_read_input_tokens", 0) or 0)
    return total


def convert(
    transcript: Path,
    task: str | None = None,
    task_value: float | None = None,
    agent_name: str = "claude-code",
    include_cache_read: bool = False,
) -> dict:
    """Convert one transcript file into a TraceRazor trace dict."""
    entries = load_entries(transcript)

    # Tool results live in user entries; index them by tool_use_id.
    tool_results: dict[str, dict] = {}
    for e in entries:
        if e.get("type") != "user":
            continue
        content = e.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                tid = b.get("tool_use_id")
                if tid:
                    tool_results[tid] = b

    # Group assistant lines by API message id (one line per content block,
    # usage repeated on each line — count it once).
    messages: list[dict] = []  # {id, model, blocks, usage, sidechain, pos}
    by_id: dict[str, dict] = {}
    first_user_prompt: str | None = None
    pending_context: list[str] = []  # new context since last model turn

    for e in entries:
        msg = e.get("message", {})
        if e.get("type") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                if first_user_prompt is None and not e.get("isSidechain"):
                    first_user_prompt = content
                pending_context.append(content)
            elif isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_result":
                        pending_context.append(_block_text(b.get("content")))
                    elif b.get("type") == "text":
                        if first_user_prompt is None and not e.get("isSidechain"):
                            first_user_prompt = b.get("text", "")
                        pending_context.append(b.get("text", ""))
            continue

        mid = msg.get("id")
        if mid is None:
            continue
        rec = by_id.get(mid)
        if rec is None:
            rec = {
                "id": mid,
                "model": msg.get("model", ""),
                "blocks": [],
                "usage": msg.get("usage", {}) or {},
                "sidechain": bool(e.get("isSidechain")),
                "context": _clip(
                    "\n".join(c for c in pending_context if c), MAX_CONTEXT_CHARS
                ),
            }
            pending_context = []
            by_id[mid] = rec
            messages.append(rec)
        rec["blocks"].extend(
            b for b in msg.get("content", []) if isinstance(b, dict)
        )

    steps: list[dict] = []
    total_tokens = 0
    first_main_seen = False
    for rec in messages:
        first_turn = not rec["sidechain"] and not first_main_seen
        if first_turn:
            first_main_seen = True
        msg_tokens = _usage_tokens(rec["usage"], include_cache_read, first_turn)
        total_tokens += msg_tokens

        reasoning_parts = []
        tool_blocks = []
        for b in rec["blocks"]:
            btype = b.get("type")
            if btype == "thinking":
                reasoning_parts.append(b.get("thinking", ""))
            elif btype == "text":
                reasoning_parts.append(b.get("text", ""))
            elif btype == "tool_use":
                tool_blocks.append(b)

        reasoning_text = "\n".join(p for p in reasoning_parts if p.strip())
        n_steps = (1 if reasoning_text else 0) + len(tool_blocks)
        if n_steps == 0:
            continue
        share, rem = divmod(msg_tokens, n_steps)

        def push(step: dict, idx: int) -> None:
            step["id"] = len(steps) + 1
            step["tokens"] = share + (rem if idx == 0 else 0)
            # New context enters the model once per API call — attach it to
            # the first step only (see module docstring).
            step["input_context"] = (rec["context"] or None) if idx == 0 else None
            if rec["sidechain"]:
                step["agent_id"] = "subagent"
            steps.append(step)

        idx = 0
        if reasoning_text:
            push(
                {
                    "type": "reasoning",
                    "content": _clip(reasoning_text, MAX_CONTENT_CHARS),
                },
                idx,
            )
            idx += 1
        for b in tool_blocks:
            name = b.get("name", "unknown")
            params = b.get("input", {})
            res = tool_results.get(b.get("id", ""))
            step = {
                "type": "tool_call",
                "content": _clip(
                    f"{name}({json.dumps(params, ensure_ascii=False)})",
                    MAX_CONTENT_CHARS,
                ),
                "tool_name": name,
                "tool_params": params,
            }
            if res is not None:
                is_err = bool(res.get("is_error"))
                step["tool_success"] = not is_err
                out = _clip(_block_text(res.get("content")), MAX_OUTPUT_CHARS)
                if is_err:
                    step["tool_error"] = out or "tool error"
                else:
                    step["output"] = out or None
            push(step, idx)
            idx += 1

    # Drop None-valued optional fields for a clean trace file.
    for s in steps:
        for k in [k for k, v in s.items() if v is None]:
            del s[k]

    model = next((m["model"] for m in messages if m["model"]), "")
    trace: dict = {
        "trace_id": transcript.stem,
        "agent_name": agent_name,
        "framework": "claude-code",
        "total_tokens": total_tokens,
        "steps": steps,
        "metadata": {
            "source": "claude-code-transcript",
            "model": model,
            "token_accounting": (
                "gross (incl. cache reads and first-turn prefix encoding)"
                if include_cache_read
                else "marginal (input + cache_creation + output; cache reads "
                "and first-turn prefix encoding excluded)"
            ),
        },
    }
    goal = task or first_user_prompt
    if goal:
        trace["metadata"]["task"] = _clip(goal, MAX_CONTEXT_CHARS)
    if task_value is not None:
        trace["task_value_score"] = task_value
    return trace


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("transcript", type=Path, help="Claude Code session .jsonl")
    ap.add_argument("--task", help="task/goal text (defaults to first user prompt)")
    ap.add_argument(
        "--task-value",
        type=float,
        default=None,
        help="task outcome score 0.0–1.0 (e.g. 1.0 if the task's checks passed)",
    )
    ap.add_argument("--agent-name", default="claude-code")
    ap.add_argument(
        "--include-cache-read",
        action="store_true",
        help="count cache-read tokens in per-step totals (gross accounting)",
    )
    ap.add_argument("--out", type=Path, help="output trace path (default stdout)")
    args = ap.parse_args(argv)

    trace = convert(
        args.transcript,
        task=args.task,
        task_value=args.task_value,
        agent_name=args.agent_name,
        include_cache_read=args.include_cache_read,
    )
    text = json.dumps(trace, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out} ({len(trace['steps'])} steps, "
              f"{trace['total_tokens']} tokens)", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
