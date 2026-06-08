"""LangGraph adapter.

Two ways to feed the Teacher from a LangGraph / LangChain agent:

1. **RunRecorder** -- a dependency-free recorder. Call ``llm`` / ``tool`` /
   ``final`` as the graph executes (e.g. from a tiny callback), then ``end()``
   to snapshot a trace. Works with no langchain installed, so it is fully
   testable offline.

2. **from_tracerazor_callback** -- if you already use the official
   ``tracerazor`` LangGraph integration (``TraceRazorCallback`` wraps a
   ``TraceBuilder``), this pulls the built trace straight out of it.

Both produce the same auditor-schema trace dicts that ``Diagnoser`` / ``Teacher``
consume; from there, ``Teacher.coach(traces)`` turns real runs into ranked,
human-approvable efficiency recommendations.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional


def _toklen(*parts: str) -> int:
    """Cheap len/4 token estimate (matches the repo's offline convention)."""
    return max(1, sum(len(p) for p in parts if p) // 4)


class RunRecorder:
    """Accumulates steps for a single agent run into an auditor-schema trace."""

    def __init__(self, agent_name: str, framework: str = "langgraph",
                 task_value_score: float = 1.0):
        self.agent_name = agent_name
        self.framework = framework
        self.task_value_score = task_value_score
        self._steps: list[dict] = []
        self._sid = 0
        self._ctx = ""

    def _add(self, step_type: str, content: str, tokens: int, **extra) -> None:
        self._sid += 1
        self._steps.append(
            {"id": self._sid, "type": step_type, "content": content,
             "tokens": int(tokens), **extra})

    # -- event hooks -------------------------------------------------------- #
    def context(self, text: str) -> "RunRecorder":
        self._ctx = text or self._ctx
        return self

    def llm(self, output: str, *, prompt: str = "", tokens: Optional[int] = None,
            input_context: Optional[str] = None) -> "RunRecorder":
        ctx = input_context if input_context is not None else self._ctx
        self._add("reasoning", output, tokens if tokens is not None else
                  _toklen(prompt, output), input_context=ctx or None)
        return self

    def tool(self, name: str, params: dict, *, output: str = "", success: bool = True,
             error: Optional[str] = None, tokens: Optional[int] = None) -> "RunRecorder":
        self._add("tool_call", f"Calling {name}",
                  tokens if tokens is not None else _toklen(str(params), output),
                  tool_name=name, tool_params=params or {}, tool_success=success,
                  tool_error=error, output=output or None)
        return self

    def final(self, answer: str, *, tokens: Optional[int] = None) -> "RunRecorder":
        self._add("reasoning", f"Final answer: {answer}",
                  tokens if tokens is not None else _toklen(answer))
        return self

    # -- snapshot ----------------------------------------------------------- #
    def end(self, trace_id: Optional[str] = None) -> dict:
        return {
            "trace_id": trace_id or str(uuid.uuid4()),
            "agent_name": self.agent_name,
            "framework": self.framework,
            "task_value_score": self.task_value_score,
            "steps": list(self._steps),
        }


class LangGraphAdapter:
    """Buffers LangGraph runs as traces for the Teacher (FrameworkAdapter)."""

    def __init__(self, agent_name: str = "langgraph-agent",
                 framework: str = "langgraph"):
        self.agent_name = agent_name
        self.framework = framework
        self._traces: list[dict] = []

    # -- recording ---------------------------------------------------------- #
    def new_run(self, task_value_score: float = 1.0) -> RunRecorder:
        """Start recording a fresh run; call ``add_run`` with its ``.end()``."""
        return RunRecorder(self.agent_name, self.framework, task_value_score)

    def add_run(self, recorder_or_trace) -> "LangGraphAdapter":
        """Add a finished ``RunRecorder`` (auto-ended) or a raw trace dict."""
        if isinstance(recorder_or_trace, RunRecorder):
            self._traces.append(recorder_or_trace.end())
        elif isinstance(recorder_or_trace, dict):
            self._traces.append(recorder_or_trace)
        else:
            raise TypeError("expected RunRecorder or trace dict")
        return self

    def from_tracerazor_callback(self, callback: Any) -> "LangGraphAdapter":
        """Pull the built trace out of a ``tracerazor`` ``TraceRazorCallback``."""
        builder = getattr(callback, "_builder", None)
        if builder is None or not hasattr(builder, "build"):
            raise TypeError("not a TraceRazorCallback (no _builder.build())")
        self._traces.append(builder.build())
        return self

    # -- FrameworkAdapter --------------------------------------------------- #
    def collect_traces(self) -> list[dict]:
        return list(self._traces)

    def reset(self) -> None:
        self._traces.clear()
