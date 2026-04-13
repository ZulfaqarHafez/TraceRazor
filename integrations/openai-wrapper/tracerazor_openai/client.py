"""
Drop-in OpenAI client that auto-records every chat.completions.create()
call as a TraceRazor reasoning step.

The wrapper uses composition (not inheritance) to proxy every attribute of
the underlying openai.OpenAI client through __getattr__, so anything you
could do with a plain `openai.OpenAI()` you can still do here. Only the
`chat.completions.create()` path is intercepted.
"""
from __future__ import annotations

from typing import Any, Optional

import openai
from tracerazor_sdk import Tracer


def _messages_to_context(messages: Any) -> str:
    if not messages:
        return ""
    parts = []
    for m in messages:
        if isinstance(m, dict):
            parts.append(str(m.get("content", "")))
        else:
            parts.append(str(getattr(m, "content", "")))
    return "\n".join(parts)[:2000]


class _ChatCompletions:
    def __init__(self, inner: Any, tracer: Tracer) -> None:
        self._inner = inner
        self._tracer = tracer

    def create(self, **kwargs: Any) -> Any:
        response = self._inner.create(**kwargs)
        self._record(kwargs, response)
        return response

    def _record(self, kwargs: dict, response: Any) -> None:
        try:
            content = ""
            if getattr(response, "choices", None):
                msg = response.choices[0].message
                content = getattr(msg, "content", None) or ""
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "total_tokens", None) if usage else None
            if tokens is None:
                tokens = max(len(content) // 4, 1)
            self._tracer.reasoning(
                content=content or "(empty)",
                tokens=int(tokens),
                input_context=_messages_to_context(kwargs.get("messages")),
                output=content or None,
            )
        except Exception:
            # Never let instrumentation break the user's program.
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Chat:
    def __init__(self, inner: Any, tracer: Tracer) -> None:
        self._inner = inner
        self.completions = _ChatCompletions(inner.completions, tracer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class OpenAI:
    """
    Drop-in replacement for `openai.OpenAI`. Accepts all the same constructor
    kwargs plus:

        agent_name:  Name used in the TraceRazor report (default "openai-agent").
        tracer:      An existing Tracer to append to. If omitted, a new one
                     is created.
        server:      Optional URL of a running tracerazor-server. If omitted,
                     the local `tracerazor` binary is used.

    Call `.audit()` when your workflow is done to analyse the captured trace.
    """

    def __init__(
        self,
        *,
        agent_name: str = "openai-agent",
        tracer: Optional[Tracer] = None,
        server: Optional[str] = None,
        **openai_kwargs: Any,
    ) -> None:
        self._inner = openai.OpenAI(**openai_kwargs)
        self._tracer = tracer or Tracer(
            agent_name=agent_name, framework="openai", server=server
        )
        self.chat = _Chat(self._inner.chat, self._tracer)

    @property
    def tracer(self) -> Tracer:
        return self._tracer

    def audit(self):
        """Submit the captured trace for analysis and return the report."""
        return self._tracer.analyse()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)