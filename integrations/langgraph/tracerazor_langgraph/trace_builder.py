"""
Builds a TraceRazor-format trace dict from LangGraph callback events.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional


class _PendingStep:
    """Mutable step being built from start/end event pairs."""

    def __init__(self, step_type: str, step_id: int, start_time: float):
        self.step_id = step_id
        self.step_type = step_type
        self.start_time = start_time
        self.content: str = ""
        self.tokens: int = 0
        self.tool_name: Optional[str] = None
        self.tool_params: Optional[Dict] = None
        self.tool_success: Optional[bool] = None
        self.tool_error: Optional[str] = None
        self.input_context: Optional[str] = None
        self.output: Optional[str] = None
        self.agent_id: Optional[str] = None


class TraceBuilder:
    """
    Stateful builder that converts LangGraph callback events into a
    TraceRazor raw JSON trace.
    """

    def __init__(
        self,
        agent_name: str,
        framework: str,
        task_value_score: float = 1.0,
    ):
        self.agent_name = agent_name
        self.framework = framework
        self.task_value_score = task_value_score
        self._trace_id: str = str(uuid.uuid4())
        self._steps: List[Dict] = []
        self._step_counter: int = 1
        self._pending: Optional[_PendingStep] = None
        self._current_agent: Optional[str] = None

    # ── Step lifecycle ───────────────────────────────────────────────────────

    def start_reasoning_step(self, input_context: str = "") -> None:
        self._pending = _PendingStep("reasoning", self._step_counter, time.time())
        self._pending.input_context = input_context or None
        self._pending.agent_id = self._current_agent

    def end_reasoning_step(self, output: str = "", tokens: int = 0) -> None:
        if self._pending is None or self._pending.step_type != "reasoning":
            return
        p = self._pending
        p.output = output or None
        p.tokens = tokens or self._estimate_tokens(p.input_context or "", output)
        p.content = self._extract_content(p.input_context or "", output)
        self._commit(p)
        self._pending = None

    def start_tool_step(self, tool_name: str, tool_params: Dict) -> None:
        self._pending = _PendingStep("tool_call", self._step_counter, time.time())
        self._pending.tool_name = tool_name
        self._pending.tool_params = tool_params
        self._pending.content = f"Calling {tool_name}"
        self._pending.agent_id = self._current_agent

    def end_tool_step(
        self, output: str, success: bool, error: Optional[str] = None
    ) -> None:
        if self._pending is None or self._pending.step_type != "tool_call":
            return
        p = self._pending
        p.tool_success = success
        p.tool_error = error
        p.output = output or None
        p.tokens = self._estimate_tokens(str(p.tool_params or ""), output)
        self._commit(p)
        self._pending = None

    def abort_current_step(self, error: str) -> None:
        """Called when an LLM or tool raises an error mid-step."""
        if self._pending is None:
            return
        p = self._pending
        if p.step_type == "tool_call":
            p.tool_success = False
            p.tool_error = error
            p.tokens = p.tokens or 50
        else:
            p.content = f"[ERROR] {error}"
            p.tokens = p.tokens or 50
        self._commit(p)
        self._pending = None

    def note_agent_transition(self, agent_id: str) -> None:
        """Record a LangGraph node transition (agent handoff)."""
        self._current_agent = agent_id

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self) -> Dict:
        """Return the complete trace as a dict ready for JSON serialisation."""
        # Commit any pending step.
        if self._pending is not None:
            p = self._pending
            p.tokens = p.tokens or 100
            p.content = p.content or "incomplete step"
            self._commit(p)
            self._pending = None

        return {
            "trace_id": self._trace_id,
            "agent_name": self.agent_name,
            "framework": self.framework,
            "task_value_score": self.task_value_score,
            "steps": self._steps,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _commit(self, p: _PendingStep) -> None:
        step: Dict[str, Any] = {
            "id": p.step_id,
            "type": p.step_type,
            "content": p.content or "",
            "tokens": max(p.tokens, 1),
        }
        if p.tool_name is not None:
            step["tool_name"] = p.tool_name
        if p.tool_params is not None:
            step["tool_params"] = p.tool_params
        if p.tool_success is not None:
            step["tool_success"] = p.tool_success
        if p.tool_error is not None:
            step["tool_error"] = p.tool_error
        if p.input_context:
            step["input_context"] = p.input_context
        if p.output:
            step["output"] = p.output
        if p.agent_id:
            step["agent_id"] = p.agent_id
        self._steps.append(step)
        self._step_counter += 1

    @staticmethod
    def _estimate_tokens(input_text: str, output_text: str) -> int:
        """
        Rough token estimate: ~4 characters per token.
        Actual counts come from LLM usage metadata when available.
        """
        total_chars = len(input_text) + len(output_text)
        return max(int(total_chars / 4), 10)

    @staticmethod
    def _extract_content(input_context: str, output: str) -> str:
        """Build a concise content string from input/output."""
        if output:
            return output[:200]
        if input_context:
            # Use last 200 chars of input as the content summary.
            return input_context[-200:]
        return ""
