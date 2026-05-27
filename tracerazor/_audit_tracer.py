"""
Tracer: manual instrumentation for recording agent traces.

Use Tracer as a context manager around your agent loop. Call t.reasoning()
after each LLM turn and t.tool() after each tool call. Then call t.analyse()
to get a TraceRazorReport with efficiency scores and fix recommendations.

Example:
    from tracerazor import Tracer

    with Tracer(agent_name="support-agent", framework="openai") as t:
        response = llm.invoke(prompt)
        t.reasoning(response.text, tokens=response.usage.total_tokens)

        result = lookup_order(order_id="ORD-123")
        t.tool("lookup_order", params={"order_id": "ORD-123"},
               output=str(result), success=True, tokens=80)

    report = t.analyse()
    print(report.summary())
    report.assert_passes()   # raises AssertionError in CI if TAS < threshold
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ._audit_client import TraceRazorClient, TraceRazorReport
from ._audit_trace import TraceStep, _Trace


class Tracer:
    """
    Records reasoning steps and tool calls, then submits for analysis.

    Args:
        agent_name:        Name of the agent. Appears in all reports.
        framework:         Framework identifier, e.g. "openai", "anthropic",
                           "langgraph", "crewai", or "custom".
        threshold:         Minimum TAS score (0-100) for assert_passes().
                           Default is 70.
        task_value_score:  Quality of the final answer on a 0.0-1.0 scale.
                           Update with set_task_value() after validation.
                           1.0 means the agent produced a correct answer.
        bin_path:          Path to the tracerazor binary (CLI mode).
                           Auto-detected from PATH and TRACERAZOR_BIN if None.
        server:            URL of a running tracerazor-server (HTTP mode).
                           When set, bin_path is ignored.
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
        self._trace = _Trace(
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

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

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
            content:       The model's output text or a summary of it.
            tokens:        Total token count for this LLM call.
            input_context: Full prompt sent to the LLM. Optional, but
                           improves CCE and SRR accuracy.
            output:        Raw model output. Optional.
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
            name:          Tool name, e.g. "search_web" or "get_order".
            params:        Dict of parameters passed to the tool.
            output:        String output returned by the tool.
            success:       False if the tool raised an error or returned a
                           failure. Accurate values enable TCA misfire detection.
            error:         Error message when success=False.
            tokens:        Token count. Estimated from params/output if omitted.
            input_context: LLM input that triggered this tool call.
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
        Set the task value score after validating the agent's answer.

        Args:
            score: 0.0 to 1.0. Use 1.0 for a fully correct answer,
                   0.0 for a completely wrong one.
        """
        self._trace.task_value_score = max(0.0, min(1.0, score))

    def analyse(self) -> TraceRazorReport:
        """
        Submit the recorded trace for analysis and return the report.
        Call this after your agent finishes its work.
        """
        self._report = self._client.analyse(self._trace.to_dict())
        return self._report

    def assert_passes(self) -> None:
        """Analyse (if not already done) and raise AssertionError if TAS < threshold."""
        if self._report is None:
            self.analyse()
        assert self._report is not None
        self._report.assert_passes()

    @property
    def report(self) -> Optional[TraceRazorReport]:
        """The most recent report, or None if analyse() has not been called yet."""
        return self._report

    @staticmethod
    def _estimate_tool_tokens(
        params: Optional[Dict], output: Optional[str]
    ) -> int:
        param_chars = len(str(params)) if params else 0
        output_chars = len(output) if output else 0
        return max(int((param_chars + output_chars) / 4), 10)
