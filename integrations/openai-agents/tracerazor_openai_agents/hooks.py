"""
TraceRazor hooks adapter for the OpenAI Agents SDK.

Implements the ``RunHooks`` interface — all methods are async as required by
the SDK. Duck-typed to avoid a hard import at module load so the package
installs and imports without ``openai-agents`` present.

The SDK hook protocol (openai-agents >= 0.0.3):

    class RunHooks(Generic[TContext]):
        async def on_agent_start(self, context, agent): ...
        async def on_agent_end(self, context, agent, output): ...
        async def on_tool_start(self, context, agent, tool): ...
        async def on_tool_end(self, context, agent, tool, result): ...
        async def on_handoff(self, context, from_agent, to_agent): ...

Usage::

    from tracerazor_openai_agents import TraceRazorHooks
    from agents import Agent, Runner

    hooks = TraceRazorHooks(agent_name="support-agent", threshold=70)
    result = await Runner.run(
        agent,
        "I need a refund for order ORD-9182",
        hooks=hooks,
    )
    report = hooks.analyse()
    print(report.markdown())
    hooks.assert_passes()
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from .client import TraceRazorClient, TraceRazorReport


class TraceRazorHooks:
    """
    OpenAI Agents SDK ``RunHooks`` adapter for TraceRazor.

    Inherits from ``RunHooks`` only when the ``agents`` package is installed;
    otherwise works as a plain duck-typed object with the same async method
    signatures.

    Args:
        agent_name:       Name shown in reports (default: "openai-agent").
        framework:        Framework label (default: "openai-agents").
        threshold:        Minimum TAS for ``assert_passes()`` (default: 70).
        task_value_score: Answer quality 0–1 (default: 1.0).
        tracerazor_bin:   Path to binary; auto-detected if None.
    """

    def __init__(
        self,
        agent_name: str = "openai-agent",
        framework: str = "openai-agents",
        threshold: float = 70.0,
        task_value_score: float = 1.0,
        tracerazor_bin: Optional[str] = None,
    ):
        self._agent_name = agent_name
        self._framework = framework
        self._threshold = threshold
        self._task_value_score = task_value_score
        self._client = TraceRazorClient(bin_path=tracerazor_bin)
        self._report: Optional[TraceRazorReport] = None

        self._trace_id: str = str(uuid.uuid4())
        self._steps: List[Dict] = []
        self._step_counter: int = 1
        self._current_agent: Optional[str] = None

        # Track open tool call waiting for on_tool_end.
        self._pending_tool: Optional[Dict] = None
        # Track per-agent step timing.
        self._agent_start_time: float = 0.0

    # ── RunHooks protocol (all async) ─────────────────────────────────────────

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        """Called at the start of each agent run (including handoffs)."""
        self._current_agent = self._agent_name_from(agent)
        self._agent_start_time = time.time()

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        """Called when the agent produces its final output."""
        agent_id = self._agent_name_from(agent)
        output_str = str(output) if output is not None else ""
        elapsed = time.time() - self._agent_start_time
        tokens = max(int(len(output_str) / 4), 10)
        # Attribute ~1 token per 0.001 second of wall time (rough heuristic).
        tokens = max(tokens, int(elapsed * 50))
        self._commit({
            "step_type": "reasoning",
            "content": output_str[:300] or f"Agent {agent_id} completed",
            "tokens": tokens,
            "output": output_str[:500] if output_str else None,
            "agent_id": agent_id,
        })

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        """Called before each tool execution."""
        tool_name = self._tool_name_from(tool)
        agent_id = self._agent_name_from(agent)
        self._pending_tool = {
            "step_type": "tool_call",
            "content": f"Calling {tool_name}",
            "tool_name": tool_name,
            "tool_params": {},
            "agent_id": agent_id,
            "start_time": time.time(),
        }

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: str) -> None:
        """Called after a successful tool execution."""
        tool_name = self._tool_name_from(tool)
        if self._pending_tool and self._pending_tool.get("tool_name") == tool_name:
            p = self._pending_tool
        else:
            p = {
                "step_type": "tool_call",
                "content": f"Calling {tool_name}",
                "tool_name": tool_name,
                "tool_params": {},
                "agent_id": self._current_agent,
                "start_time": time.time(),
            }
        result_str = str(result) if result is not None else ""
        p["tool_success"] = True
        p["output"] = result_str[:500] if result_str else None
        p["tokens"] = max(int(len(result_str) / 4), 10)
        self._commit(p)
        self._pending_tool = None

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        """Called when one agent hands off to another."""
        from_id = self._agent_name_from(from_agent)
        to_id = self._agent_name_from(to_agent)
        # Record as a lightweight reasoning step so handoffs are visible in the trace.
        self._commit({
            "step_type": "reasoning",
            "content": f"Handoff from {from_id} to {to_id}",
            "tokens": 50,
            "agent_id": from_id,
        })
        self._current_agent = to_id

    # ── Analysis API ──────────────────────────────────────────────────────────

    def analyse(self) -> TraceRazorReport:
        """
        Finalise the trace and submit it to TraceRazor for analysis.

        Call this after ``Runner.run()`` completes.
        """
        if self._pending_tool is not None:
            p = self._pending_tool
            p.setdefault("tool_success", True)
            p.setdefault("tokens", 50)
            self._commit(p)
            self._pending_tool = None

        trace = {
            "trace_id": self._trace_id,
            "agent_name": self._agent_name,
            "framework": self._framework,
            "task_value_score": self._task_value_score,
            "steps": self._steps,
        }
        self._report = self._client.analyse(trace=trace, threshold=self._threshold)
        return self._report

    def assert_passes(self) -> None:
        """Analyse (if needed) and raise ``AssertionError`` if TAS < threshold."""
        if self._report is None:
            import asyncio
            # Synchronous wrapper for test environments.
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # In async context, user should have called await hooks.analyse() directly.
                    pass
            except RuntimeError:
                pass
            self.analyse()
        assert self._report is not None
        if not self._report.passes:
            raise AssertionError(
                f"TraceRazor: TAS {self._report.tas_score:.1f} is below "
                f"threshold {self._threshold}.\n\n{self._report.summary()}"
            )

    def set_task_value_score(self, score: float) -> None:
        """Update answer quality (0–1) before calling ``analyse()``."""
        self._task_value_score = max(0.0, min(1.0, score))

    @property
    def report(self) -> Optional[TraceRazorReport]:
        return self._report

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _commit(self, step: Dict) -> None:
        out: Dict = {
            "id": self._step_counter,
            "step_type": step["step_type"],
            "content": step.get("content", ""),
            "tokens": max(step.get("tokens", 50), 1),
        }
        for key in ("tool_name", "tool_params", "tool_success", "tool_error",
                    "input_context", "output", "agent_id"):
            if step.get(key) is not None:
                out[key] = step[key]
        self._steps.append(out)
        self._step_counter += 1

    @staticmethod
    def _agent_name_from(agent: Any) -> str:
        """Extract a string identifier from an agent object."""
        return (
            getattr(agent, "name", None)
            or getattr(agent, "role", None)
            or str(type(agent).__name__)
        )

    @staticmethod
    def _tool_name_from(tool: Any) -> str:
        """Extract a tool name from a tool object or string."""
        if isinstance(tool, str):
            return tool
        return (
            getattr(tool, "name", None)
            or getattr(tool, "__name__", None)
            or str(type(tool).__name__)
        )


# ── Try to inherit from RunHooks if openai-agents is installed ────────────────
# This gives proper type compatibility with Runner.run(hooks=...) while
# still working as a duck-typed object when openai-agents isn't present.

try:
    from agents import RunHooks  # type: ignore[import-untyped]

    class TraceRazorHooks(TraceRazorHooks, RunHooks):  # type: ignore[no-redef]
        """TraceRazorHooks with openai-agents RunHooks as base class."""

except ImportError:
    pass  # TraceRazorHooks already defined above as a standalone class
