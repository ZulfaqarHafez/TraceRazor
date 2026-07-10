"""Compile rich runtime events to TraceRazor's stable native trace shape."""

from __future__ import annotations

from typing import Any, Iterable

from .models import (
    CaptureQuality,
    RunContext,
    RuntimeEvent,
    TaskOutcome,
    TokenProvenance,
    ToolStatus,
    run_capture_quality,
)


class NoAuditableEventsError(ValueError):
    """Raised when a run contains no event that can become a native step."""


def _native_type(event_type: str) -> str:
    if event_type in {"reasoning", "message"}:
        return "reasoning"
    if event_type == "tool_call":
        return "tool_call"
    if event_type == "handoff":
        return "handoff"
    return "unknown"


def compile_native_trace(
    events: Iterable[RuntimeEvent],
    *,
    context: RunContext | None = None,
    agent_name: str | None = None,
    framework: str | None = None,
    partial: bool = False,
) -> dict[str, Any]:
    """Compile events into the existing ``trace.schema.json`` representation.

    The returned trace is the in-memory analysis representation and may include
    raw content.  Persistence is a separate, policy-aware boundary handled by
    :mod:`tracerazor.runtime.persistence`.
    """

    values = list(events)
    if not values:
        raise NoAuditableEventsError("run contains no events")

    run_ids = {event.run_id for event in values}
    trace_ids = {event.trace_id for event in values}
    if len(run_ids) != 1:
        raise ValueError("all compiled events must belong to one run_id")
    if len(trace_ids) != 1:
        raise ValueError("all compiled events must belong to one trace_id")
    if context is not None:
        if context.run_id not in run_ids or context.trace_id not in trace_ids:
            raise ValueError("run context does not match compiled events")

    event_ids: set[str] = set()
    span_ids: set[str] = set()
    for event in values:
        if event.event_id in event_ids:
            raise ValueError(f"duplicate event_id in run: {event.event_id}")
        event_ids.add(event.event_id)
        if event.span_id in span_ids:
            raise ValueError(f"duplicate span_id in run: {event.span_id}")
        span_ids.add(event.span_id)

    ordered = sorted(values, key=lambda event: (event.sequence, event.timestamp, event.event_id))
    auditable = [event for event in ordered if event.auditable]
    if not auditable:
        raise NoAuditableEventsError("run contains no auditable step events")

    steps: list[dict[str, Any]] = []
    for index, event in enumerate(auditable, start=1):
        content = event.content
        if content is None:
            content = event.output if event.output is not None else event.event_type
        step: dict[str, Any] = {
            "id": index,
            "type": _native_type(event.event_type),
            "content": content,
            "tokens": event.tokens.total,
            "agent_id": event.agent_id,
        }
        if event.input_context is not None:
            step["input_context"] = event.input_context
        if event.output is not None:
            step["output"] = event.output
        if event.tool is not None:
            step["tool_name"] = event.tool.signature
            step["tool_params"] = {
                "arguments_sha256": event.tool.arguments_digest,
                "expected_failure": event.tool.expected_failure,
                "observation_size": event.tool.observation_size,
            }
            step["tool_success"] = event.tool.status is ToolStatus.SUCCESS
            if event.tool.status is ToolStatus.ERROR:
                step["tool_error"] = event.tool.error_type or "tool_error"
        steps.append(step)

    latest_task = next((event.task for event in reversed(ordered) if event.task is not None), None)
    quality = run_capture_quality(values, partial=partial)
    provenance = sorted(
        {event.tokens.provenance.value for event in auditable},
        key=lambda value: (value != TokenProvenance.PROVIDER_REPORTED.value, value),
    )
    result: dict[str, Any] = {
        "trace_id": next(iter(trace_ids)),
        "agent_name": agent_name or (context.agent_id if context else auditable[0].agent_id),
        "framework": framework or auditable[0].framework,
        "steps": steps,
        "total_tokens": sum(step["tokens"] for step in steps),
        "metadata": {
            "runtime_schema": "tracerazor-event/v1",
            "run_id": next(iter(run_ids)),
            "session_id": context.session_id if context else auditable[0].session_id,
            "capture_quality": quality.value,
            "degraded_ingest": quality is not CaptureQuality.COMPLETE,
            "token_provenance": provenance,
            "partial": partial,
        },
    }
    if latest_task is not None:
        result["metadata"]["task_outcome"] = latest_task.outcome.value
        result["metadata"]["verifier"] = latest_task.verifier
        if latest_task.outcome is TaskOutcome.PASSED:
            result["task_value_score"] = 1.0
        elif latest_task.outcome is TaskOutcome.FAILED:
            result["task_value_score"] = 0.0
    validate_native_trace_shape(result)
    return result


def validate_native_trace_shape(trace: dict[str, Any]) -> None:
    """Dependency-free validation of the compatibility-critical native shape."""

    for name in ("trace_id", "agent_name", "framework"):
        if not isinstance(trace.get(name), str):
            raise ValueError(f"native trace {name} must be a string")
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("native trace requires at least one step")
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"native step {index} must be an object")
        if step.get("id") != index:
            raise ValueError("native step IDs must be contiguous and one-based")
        if step.get("type") not in {"reasoning", "tool_call", "handoff", "unknown"}:
            raise ValueError(f"native step {index} has an unsupported type")
        if not isinstance(step.get("content"), str):
            raise ValueError(f"native step {index} content must be a string")
        tokens = step.get("tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError(f"native step {index} tokens must be a non-negative integer")


__all__ = ["NoAuditableEventsError", "compile_native_trace", "validate_native_trace_shape"]
