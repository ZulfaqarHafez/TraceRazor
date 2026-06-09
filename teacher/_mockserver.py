"""A tiny OpenAI-compatible chat-completions server (stdlib only).

Purpose: verify the *real* online verifier (`teacher.online`) end-to-end without
a paid API key. It speaks the exact `/v1/chat/completions` wire format -- tool
calls, tool results, and a `usage` block -- so `teacher.online.LLMClient` makes
genuine HTTP requests and parses genuine responses. Point the same client at
`https://api.openai.com` (or any OpenAI-compatible gateway) with a key and the
code path is identical.

The "model" is a deterministic scripted agent whose behaviour *responds to the
system prompt and runtime-policy markers the verifier sends*, so installing
interventions (NO_HEDGING, EFFICIENCY_RULES, STEP_BUDGET, loop guard) really
does shrink the token usage it reports -- which is what makes the before/after
verification meaningful rather than staged.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HEDGE = ("Certainly! I would be very happy to help you with this request. Let me "
          "carefully think through it step by step, because I want to be thorough. ")
_FILLER = ("Basically, to be honest, at the end of the day, essentially what this "
           "comes down to is the following point, more or less, generally speaking. ")


def _toklen(s: str) -> int:
    return max(1, len(s) // 4)


def _flags(system_text: str) -> dict:
    t = system_text
    return {
        "no_hedging": "NO_HEDGING" in t,
        "efficiency": "EFFICIENCY_RULES" in t,
        "step_budget": "STEP_BUDGET" in t or "[runtime] step_cap" in t,
        "loop_guard": "loop_breaker" in t or "[runtime] loop_guard" in t,
    }


def _reasoning(flags: dict) -> str:
    """Assistant 'thinking' that rides alongside a tool call; flags inflate it."""
    s = ""
    if not flags["no_hedging"]:
        s += _HEDGE
    if not flags["efficiency"]:
        s += _FILLER
    if not flags["step_budget"]:
        s += "Let me also consider several unlikely edge cases before proceeding. "
    return s + "Proceeding with the next required step."


def _script(flags: dict, required: list[str], order_id: str) -> list[dict]:
    """Deterministic per-run plan; each entry is one assistant turn.

    Non-final turns are tool-call turns carrying reasoning in their content
    (matching how real models interleave reasoning with tool calls); the final
    turn is content-only, which ends the agent loop.
    """
    reasoning = _reasoning(flags)
    turns: list[dict] = []
    for i, tool in enumerate(required):
        turns.append({"tool": tool, "args": {"order_id": order_id}, "content": reasoning})
        if i == 0 and not flags["loop_guard"]:
            turns.append({"tool": tool, "args": {"order_id": order_id}, "content": reasoning})
    final = "Final answer: the task has been completed successfully."
    if not flags["efficiency"]:
        final = _FILLER + final
    turns.append({"content": final})
    return turns


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):   # silence
        pass

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        messages = body.get("messages", [])

        system_text = " ".join(m.get("content") or "" for m in messages
                               if m.get("role") == "system")
        user_text = " ".join(m.get("content") or "" for m in messages
                             if m.get("role") == "user")
        # task hint embedded by the verifier: "... TOOLS: a,b,c | ID: ORD-1"
        required, order_id = [], "ORD-0"
        for chunk in user_text.split("|"):
            if "TOOLS:" in chunk:
                required = [x.strip() for x in chunk.split("TOOLS:")[1].split(",") if x.strip()]
            if "ID:" in chunk:
                order_id = chunk.split("ID:")[1].strip().split()[0]

        flags = _flags(system_text)
        turns = _script(flags, required, order_id)
        idx = sum(1 for m in messages if m.get("role") == "assistant")
        turn = turns[idx] if idx < len(turns) else turns[-1]

        prompt_tokens = sum(_toklen(m.get("content") or "") +
                            _toklen(json.dumps(m.get("tool_calls", ""))) for m in messages)

        if "tool" in turn:
            args = json.dumps(turn["args"])
            content = turn.get("content") or ""
            message = {
                "role": "assistant", "content": content or None,
                "tool_calls": [{
                    "id": f"call_{idx}", "type": "function",
                    "function": {"name": turn["tool"], "arguments": args}}],
            }
            completion_tokens = _toklen(turn["tool"] + args + content)
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": turn["content"]}
            completion_tokens = _toklen(turn["content"])
            finish = "stop"

        resp = {
            "id": f"chatcmpl-{idx}", "object": "chat.completion",
            "model": body.get("model", "mock-gpt"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
        }
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve_in_thread() -> tuple[str, "ThreadingHTTPServer"]:
    """Start the mock server on an ephemeral port; return (base_url, server)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    host, port = server.server_address
    return f"http://{host}:{port}/v1", server
