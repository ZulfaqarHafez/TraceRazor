"""
TraceRazor LangGraph callback handler.

Captures LangGraph/LangChain execution events and serialises them into
TraceRazor's raw JSON trace format for analysis.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .client import TraceRazorClient, TraceRazorReport
from .trace_builder import TraceBuilder


class TraceRazorCallback(BaseCallbackHandler):
    """
    LangGraph/LangChain callback that builds a TraceRazor trace from
    execution events and submits it for analysis when the run completes.

    Args:
        agent_name: Name of the agent (appears in the report header).
        framework:  Framework name (default: "langgraph").
        threshold:  Minimum TAS score for assertion checks (default: 70).
        semantic:   If True, enable OpenAI embedding-based analysis (Phase 2).
                    Requires OPENAI_API_KEY in environment.
        cost_per_million: Token cost in USD for savings estimates.
        tracerazor_bin: Path to the tracerazor CLI binary. Auto-detected if None.
        task_value_score: Quality of the final answer (0.0–1.0). Set this
                         after the run if you have ground-truth validation.
    """

    def __init__(
        self,
        agent_name: str = "langgraph-agent",
        framework: str = "langgraph",
        threshold: float = 70.0,
        semantic: bool = False,
        cost_per_million: float = 3.0,
        tracerazor_bin: Optional[str] = None,
        task_value_score: float = 1.0,
    ):
        super().__init__()
        self._builder = TraceBuilder(
            agent_name=agent_name,
            framework=framework,
            task_value_score=task_value_score,
        )
        self._threshold = threshold
        self._semantic = semantic
        self._cost_per_million = cost_per_million
        self._client = TraceRazorClient(bin_path=tracerazor_bin)
        self._report: Optional[TraceRazorReport] = None

    # ── LLM events ──────────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Called when an LLM starts generating."""
        context = prompts[0] if prompts else ""
        self._builder.start_reasoning_step(input_context=context)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when an LLM finishes generating."""
        output_text = ""
        total_tokens = 0

        if response.generations:
            first = response.generations[0]
            if first:
                output_text = getattr(first[0], "text", "") or ""

        # Extract token usage from LLMResult metadata.
        if response.llm_output:
            usage = response.llm_output.get("usage", response.llm_output.get("token_usage", {}))
            total_tokens = (
                usage.get("total_tokens")
                or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            )

        self._builder.end_reasoning_step(output=output_text, tokens=total_tokens)

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        self._builder.abort_current_step(error=str(error))

    # ── Tool events ──────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts executing."""
        tool_name = serialized.get("name", "unknown_tool")
        try:
            params = json.loads(input_str) if input_str.strip().startswith("{") else {"input": input_str}
        except json.JSONDecodeError:
            params = {"input": input_str}
        self._builder.start_tool_step(tool_name=tool_name, tool_params=params)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool finishes successfully."""
        self._builder.end_tool_step(output=output, success=True)

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when a tool raises an error."""
        self._builder.end_tool_step(
            output="",
            success=False,
            error=str(error),
        )

    # ── Chain events (LangGraph node transitions) ────────────────────────────

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """LangGraph node start — we use this for agent handoffs."""
        node_name = serialized.get("name", "")
        # Top-level chain start = beginning of the run; skip.
        if node_name and node_name not in ("RunnableSequence", "AgentExecutor", "CompiledGraph"):
            self._builder.note_agent_transition(node_name)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    # ── Analysis ─────────────────────────────────────────────────────────────

    def analyse(self) -> TraceRazorReport:
        """
        Finalise the trace and submit it to the TraceRazor CLI for analysis.
        Returns a TraceRazorReport with the TAS score and full report.

        Call this after your LangGraph invocation completes.
        """
        trace_dict = self._builder.build()
        self._report = self._client.analyse(
            trace=trace_dict,
            semantic=self._semantic,
            threshold=self._threshold,
            cost_per_million=self._cost_per_million,
        )
        return self._report

    def assert_passes(self) -> None:
        """
        Analyse the trace (if not already done) and raise AssertionError
        if TAS is below threshold. Useful for CI/CD and test assertions.
        """
        if self._report is None:
            self.analyse()
        assert self._report is not None
        if not self._report.passes:
            raise AssertionError(
                f"TraceRazor: TAS {self._report.tas_score:.1f} is below "
                f"threshold {self._threshold}.\n\n{self._report.summary()}"
            )

    @property
    def report(self) -> Optional[TraceRazorReport]:
        """The most recent analysis report, or None if not yet analysed."""
        return self._report

    def set_task_value_score(self, score: float) -> None:
        """
        Update the task value score (0.0–1.0) based on ground-truth validation.
        Must be called before analyse() for it to take effect.
        """
        self._builder.task_value_score = max(0.0, min(1.0, score))
