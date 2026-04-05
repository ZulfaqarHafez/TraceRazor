"""
Tracer — the main entry point for manual instrumentation.

Use as a context manager or call step methods directly.

Example:
    from tracerazor_sdk import Tracer

    with Tracer(agent_name="my-agent") as t:
        # After each LLM call:
        t.reasoning("model output text", tokens=820, input_context="full prompt")

        # After each tool call:
        t.tool("search_web", params={"q": "..."}, output="results", success=True, tokens=200)

    report = t.analyse()
    print(report.summary())
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .client import TraceRazorClient, TraceRazorReport
from .trace import Trace, TraceStep


class Tracer:
    """
    Manual instrumentation wrapper. Records reasoning steps and tool calls,
    then submits the trace for analysis.

    Args:
        agent_name:        Name of the agent (appears in all reports).
        framework:         Framework identifier, e.g. "openai", "anthropic",
                           "crewai", "autogen", "custom".
        threshold:         Minimum TAS score for assert_passes() (default 70).
        task_value_score:  Quality of the final answer (0.0–1.0). Update via
                           set_task_value() after ground-truth validation.
        bin_path:          Path to tracerazor binary (CLI mode). Auto-detected.
        server:            URL of tracerazor-server (HTTP mode). When set,
                           bin_path is ignored.
    """

    def __init__(
        self,
        agent_name: str,
        framework: str = "custom",
        threshold: float = 70.0,
        task_value_score: float = 1.0,
        bin_path: Optional[str] = None,
        server: Optional[str] = None,
    ):
        self._trace = Trace(
            agent_name=agent_name,
            framework=framework,
            task_value_score=task_value_score,
        )
        self._client = TraceRazorClient(
            bin_path=bin_path,
            server=server,
            threshold=threshold,
        )
        self._report: Optional[TraceRazorReport] = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Don't auto-analyse on exception — the trace may be incomplete.
        pass

    # ── Step recording ────────────────────────────────────────────────────────

    def reasoning(
        self,
        content: str,
        tokens: int,
        input_context: Optional[str] = None,
        output: Optional[str] = None,
    ) -> None:
        """
        Record one LLM reasoning step.

        Args:
            content:       The model's output text (or a summary of it).
            tokens:        Total token count for this LLM call.
            input_context: The full prompt sent to the LLM (optional but
                           improves CCE and SRR accuracy).
            output:        The model's raw output (optional).
        """
        step = TraceStep(
            id=len(self._trace.steps) + 1,
            type="reasoning",
            content=content[:500],
            tokens=max(tokens, 1),
            input_context=input_context,
            output=output,
        )
        self._trace.add_step(step)

    def tool(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        tokens: int = 0,
        input_context: Optional[str] = None,
    ) -> None:
        """
        Record one tool call step.

        Args:
            name:     Tool name (e.g. "search_web", "get_order_details").
            params:   Dict of parameters passed to the tool.
            output:   String output returned by the tool.
            success:  False if the tool raised an error or returned a failure.
                      Setting this accurately enables TCA misfire detection.
            error:    Error message if success=False.
            tokens:   Token count. If unknown, omit and a rough estimate is used.
            input_context: The LLM input that triggered this tool call.
        """
        estimated = tokens or self._estimate_tool_tokens(params, output)
        step = TraceStep(
            id=len(self._trace.steps) + 1,
            type="tool_call",
            content=f"Calling {name}",
            tokens=max(estimated, 1),
            tool_name=name,
            tool_params=params,
            tool_success=success,
            tool_error=error if not success else None,
            output=output,
            input_context=input_context,
        )
        self._trace.add_step(step)

    def set_task_value(self, score: float) -> None:
        """
        Set the task value score (0.0–1.0) based on outcome quality.
        Call this after you have validated the agent's answer.
        1.0 = correct answer, 0.0 = wrong answer.
        """
        self._trace.task_value_score = max(0.0, min(1.0, score))

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyse(self) -> TraceRazorReport:
        """
        Submit the collected trace for analysis and return the report.
        Call this after your agent finishes.
        """
        self._report = self._client.analyse(self._trace.to_dict())
        return self._report

    def assert_passes(self) -> None:
        """Analyse (if not done) and raise AssertionError if TAS < threshold."""
        if self._report is None:
            self.analyse()
        assert self._report is not None
        self._report.assert_passes()

    @property
    def report(self) -> Optional[TraceRazorReport]:
        """The most recent report, or None if analyse() hasn't been called."""
        return self._report

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_tool_tokens(
        params: Optional[Dict], output: Optional[str]
    ) -> int:
        param_chars = len(str(params)) if params else 0
        output_chars = len(output) if output else 0
        return max(int((param_chars + output_chars) / 4), 10)
