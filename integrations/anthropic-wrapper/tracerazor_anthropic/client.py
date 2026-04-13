"""
Drop-in Anthropic client that auto-records every messages.create() call
as a TraceRazor reasoning step.

The wrapper uses composition and proxies every attribute of the underlying
anthropic.Anthropic client through __getattr__. Only messages.create() is
intercepted. Streaming is not wrapped yet — use the SDK directly for that.
"""
from __future__ import annotations

from typing import Any, Optional

import anthropic
from tracerazor_sdk import Tracer


def _messages_to_context(system: Any, messages: Any) -> str:
    parts = []
    if system:
        parts.append(str(system))
    if messages:
        for m in messages:
            if isinstance(m, dict):
                content = m.get("content", "")
            else:
                content = getattr(m, "content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    else:
                        parts.append(str(block))
            else:
                parts.append(str(content))
    return "\n".join(parts)[:2000]


def _response_text(response: Any) -> str:
    blocks = getattr(response, "content", None) or []
    out = []
    for b in blocks:
        text = getattr(b, "text", None)
        if text:
            out.append(text)
    return "\n".join(out)


class _Messages:
    def __init__(self, inner: Any, tracer: Tracer) -> None:
        self._inner = inner
        self._tracer = tracer

    def create(self, **kwargs: Any) -> Any:
        response = self._inner.create(**kwargs)
        self._record(kwargs, response)
        return response

    def _record(self, kwargs: dict, response: Any) -> None:
        try:
            content = _response_text(response)
            usage = getattr(response, "usage", None)
            if usage is not None:
                tokens = int(
                    getattr(usage, "input_tokens", 0)
                    + getattr(usage, "output_tokens", 0)
                )
            else:
                tokens = max(len(content) // 4, 1)
            self._tracer.reasoning(
                content=content or "(empty)",
                tokens=tokens,
                input_context=_messages_to_context(
                    kwargs.get("system"), kwargs.get("messages")
                ),
                output=content or None,
            )
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class Anthropic:
    """
    Drop-in replacement for `anthropic.Anthropic`. Accepts all the same
    constructor kwargs plus:

        agent_name:  Name used in the TraceRazor report (default "claude-agent").
        tracer:      An existing Tracer to append to. If omitted, a new one
                     is created.
        server:      Optional URL of a running tracerazor-server.

    Call `.audit()` when your workflow is done to analyse the captured trace.
    """

    def __init__(
        self,
        *,
        agent_name: str = "claude-agent",
        tracer: Optional[Tracer] = None,
        server: Optional[str] = None,
        **anthropic_kwargs: Any,
    ) -> None:
        self._inner = anthropic.Anthropic(**anthropic_kwargs)
        self._tracer = tracer or Tracer(
            agent_name=agent_name, framework="anthropic", server=server
        )
        self.messages = _Messages(self._inner.messages, self._tracer)

    @property
    def tracer(self) -> Tracer:
        return self._tracer

    def audit(self):
        """Submit the captured trace for analysis and return the report."""
        return self._tracer.analyse()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)