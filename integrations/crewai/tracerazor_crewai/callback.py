"""
TraceRazor CrewAI callback.

Hooks into CrewAI's BaseCallback to capture every LLM call and tool
execution, then serialises them into a TraceRazor trace for analysis.

CrewAI callback events used:
  on_task_start / on_task_end  — one reasoning step per task attempt
  on_tool_use_start / on_tool_use_end / on_tool_error — tool steps

Usage::

    from tracerazor_crewai import TraceRazorCallback
    from crewai import Crew

    callback = TraceRazorCallback(agent_name="support-crew", threshold=70)
    crew = Crew(agents=[...], tasks=[...], callbacks=[callback])
    crew.kickoff()

    report = callback.analyse()
    print(report.markdown())

    # CI/CD gate:
    callback.assert_passes()
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from .client import TraceRazorClient, TraceRazorReport


class TraceRazorCallback:
    """
    CrewAI callback adapter for TraceRazor.

    Implements the CrewAI ``BaseCallback`` interface (duck-typed — no hard
    dependency on crewai at import time so the package installs without it).

    Args:
        agent_name:       Name shown in reports (default: "crewai-crew").
        framework:        Framework label (default: "crewai").
        threshold:        Minimum TAS score for ``assert_passes()`` (default: 70).
        task_value_score: Answer quality 0–1 (default: 1.0). Update after
                          validation with ``set_task_value_score()``.
        tracerazor_bin:   Path to the binary. Auto-detected if None.
    """

    def __init__(
        self,
        agent_name: str = "crewai-crew",
        framework: str = "crewai",
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

        # Pending tool step while waiting for on_tool_use_end.
        self._pending_tool: Optional[Dict] = None
        # Pending task step while waiting for on_task_end.
        self._pending_task: Optional[Dict] = None

    # ── CrewAI callback hooks ─────────────────────────────────────────────────

    def on_task_start(self, task: Any, agent: Any = None, **kwargs: Any) -> None:
        """Called when a CrewAI task begins execution."""
        agent_id = getattr(agent, "role", None) or getattr(agent, "name", None)
        if agent_id:
            self._current_agent = str(agent_id)

        task_desc = getattr(task, "description", None) or str(task)
        self._pending_task = {
            "id": self._step_counter,
            "step_type": "reasoning",
            "content": task_desc[:300],
            "input_context": task_desc,
            "agent_id": self._current_agent,
            "start_time": time.time(),
            "tokens": 0,
        }

    def on_task_end(self, task: Any, output: Any = None, **kwargs: Any) -> None:
        """Called when a CrewAI task finishes."""
        if self._pending_task is None:
            return
        p = self._pending_task
        output_str = str(output) if output is not None else ""
        p["tokens"] = p.get("tokens") or self._estimate_tokens(p.get("input_context", ""), output_str)
        p["output"] = output_str[:500] if output_str else None
        self._commit_step(p)
        self._pending_task = None

    def on_agent_action(self, agent: Any = None, action: Any = None, **kwargs: Any) -> None:
        """Called when an agent decides to take an action (LLM reasoning step)."""
        agent_id = getattr(agent, "role", None) or getattr(agent, "name", None)
        if agent_id:
            self._current_agent = str(agent_id)

        thought = getattr(action, "log", None) or getattr(action, "thought", None) or str(action or "")
        tool_name = getattr(action, "tool", None)

        if tool_name:
            # Agent chose a tool — start a tool step.
            tool_input = getattr(action, "tool_input", {})
            if isinstance(tool_input, str):
                tool_input = {"input": tool_input}
            self._pending_tool = {
                "id": self._step_counter,
                "step_type": "tool_call",
                "content": f"Calling {tool_name}",
                "tool_name": tool_name,
                "tool_params": tool_input,
                "agent_id": self._current_agent,
                "start_time": time.time(),
            }
        elif thought:
            # Pure reasoning step (no tool chosen yet).
            tokens = self._estimate_tokens(thought, "")
            self._commit_step({
                "id": self._step_counter,
                "step_type": "reasoning",
                "content": thought[:300],
                "tokens": tokens,
                "agent_id": self._current_agent,
            })

    def on_tool_use_start(
        self, tool_name: str, tool_input: Any = None, agent: Any = None, **kwargs: Any
    ) -> None:
        """Called when a tool starts executing."""
        agent_id = getattr(agent, "role", None) or getattr(agent, "name", None)
        if agent_id:
            self._current_agent = str(agent_id)

        params: Dict = {}
        if isinstance(tool_input, dict):
            params = tool_input
        elif tool_input is not None:
            params = {"input": str(tool_input)}

        self._pending_tool = {
            "id": self._step_counter,
            "step_type": "tool_call",
            "content": f"Calling {tool_name}",
            "tool_name": tool_name,
            "tool_params": params,
            "agent_id": self._current_agent,
            "start_time": time.time(),
        }

    def on_tool_use_end(
        self, tool_name: str, output: Any = None, agent: Any = None, **kwargs: Any
    ) -> None:
        """Called when a tool finishes successfully."""
        if self._pending_tool and self._pending_tool.get("tool_name") == tool_name:
            p = self._pending_tool
        else:
            # No matching pending tool — create a minimal completed step.
            p = {
                "id": self._step_counter,
                "step_type": "tool_call",
                "content": f"Calling {tool_name}",
                "tool_name": tool_name,
                "tool_params": {},
                "agent_id": self._current_agent,
                "start_time": time.time(),
            }
        output_str = str(output) if output is not None else ""
        p["tool_success"] = True
        p["output"] = output_str[:500] if output_str else None
        p["tokens"] = self._estimate_tokens(str(p.get("tool_params", "")), output_str)
        self._commit_step(p)
        self._pending_tool = None

    def on_tool_error(
        self, tool_name: str, error: Any = None, agent: Any = None, **kwargs: Any
    ) -> None:
        """Called when a tool raises an error."""
        if self._pending_tool and self._pending_tool.get("tool_name") == tool_name:
            p = self._pending_tool
        else:
            p = {
                "id": self._step_counter,
                "step_type": "tool_call",
                "content": f"Calling {tool_name} (failed)",
                "tool_name": tool_name,
                "tool_params": {},
                "agent_id": self._current_agent,
                "start_time": time.time(),
            }
        p["tool_success"] = False
        p["tool_error"] = str(error) if error is not None else "unknown error"
        p["tokens"] = p.get("tokens") or 50
        self._commit_step(p)
        self._pending_tool = None

    # ── Analysis API ──────────────────────────────────────────────────────────

    def analyse(self) -> TraceRazorReport:
        """
        Finalise the trace and submit it to TraceRazor for analysis.

        Call this after ``crew.kickoff()`` completes.
        Returns a :class:`TraceRazorReport` with TAS score and full report.
        """
        # Flush any pending steps.
        if self._pending_task is not None:
            p = self._pending_task
            p.setdefault("tokens", 100)
            self._commit_step(p)
            self._pending_task = None
        if self._pending_tool is not None:
            p = self._pending_tool
            p["tool_success"] = True
            p.setdefault("tokens", 50)
            self._commit_step(p)
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
        """The most recent report, or None if not yet analysed."""
        return self._report

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _commit_step(self, step: Dict) -> None:
        out: Dict = {
            "id": step["id"],
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
    def _estimate_tokens(input_text: str, output_text: str) -> int:
        return max(int((len(input_text) + len(output_text)) / 4), 10)
