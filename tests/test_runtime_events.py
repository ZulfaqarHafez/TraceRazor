from __future__ import annotations

import json
from pathlib import Path

import pytest

from tracerazor.runtime import (
    CaptureQuality,
    RunContext,
    RuntimeEvent,
    TaskOutcome,
    TaskResult,
    TokenProvenance,
    TokenUsage,
    ToolCall,
    ToolStatus,
    arguments_digest,
    compile_native_trace,
    format_traceparent,
    parse_traceparent,
    validate_native_trace_shape,
)


def _context(agent_id: str = "parent") -> RunContext:
    return RunContext.create(agent_id=agent_id, session_id="session-1")


def test_token_provenance_controls_quality_and_enforcement():
    context = _context()
    estimated = RuntimeEvent.create(
        context,
        event_type="reasoning",
        host="codex",
        framework="openai-agents",
        tokens=TokenUsage(input=10, output=3, provenance=TokenProvenance.ESTIMATED),
        content="answer",
    )
    assert estimated.effective_capture.quality is CaptureQuality.DEGRADED
    assert estimated.effective_capture.issues == ("estimated_token_usage",)
    assert estimated.enforcement_eligible is False

    exact = RuntimeEvent.create(
        context,
        event_type="reasoning",
        host="codex",
        framework="openai-agents",
        tokens=TokenUsage(input=10, output=3, provenance=TokenProvenance.PROVIDER_REPORTED),
    )
    assert exact.effective_capture.quality is CaptureQuality.COMPLETE
    assert exact.enforcement_eligible is True


def test_missing_provenance_rejects_nonzero_counts():
    with pytest.raises(ValueError, match="missing token provenance"):
        TokenUsage(input=1, provenance=TokenProvenance.MISSING)
    with pytest.raises(ValueError, match="non-negative"):
        TokenUsage(input=-1, provenance=TokenProvenance.PROVIDER_REPORTED)


def test_tool_event_requires_digest_and_has_no_raw_arguments():
    args = {"token": "super-secret", "z": 2}
    tool = ToolCall.from_arguments(
        "lookup_order",
        args,
        status=ToolStatus.SUCCESS,
        duration_ms=12.5,
        observation_size=200,
    )
    assert tool.arguments_digest == arguments_digest({"z": 2, "token": "super-secret"})
    serialized = tool.to_dict()
    assert "super-secret" not in json.dumps(serialized)
    with pytest.raises(ValueError, match="SHA-256"):
        ToolCall(signature="x", arguments_digest="not-a-digest")


def test_w3c_traceparent_round_trip_and_child_context():
    parent = _context()
    value = parent.traceparent
    trace_id, parent_span, flags = parse_traceparent(value)
    assert value == format_traceparent(trace_id, parent_span, flags)
    assert trace_id == parent.trace_id
    assert parent_span == parent.span_id

    env = parent.spawn_env(child_agent_id="child", policy_path="tracerazor.toml")
    child = RunContext.from_env(env)
    assert child.run_id == parent.run_id
    assert child.trace_id == parent.trace_id
    assert child.session_id == parent.session_id
    assert child.parent_span_id == parent.span_id
    assert child.parent_agent_id == parent.agent_id
    assert child.agent_id == "child"
    assert child.span_id != parent.span_id
    assert env["TRACERAZOR_POLICY"] == "tracerazor.toml"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "00-00000000000000000000000000000000-1111111111111111-01",
        "00-11111111111111111111111111111111-0000000000000000-01",
        "01-11111111111111111111111111111111-1111111111111111-01",
    ],
)
def test_invalid_traceparent_is_rejected(bad):
    with pytest.raises(ValueError):
        parse_traceparent(bad)


def test_compile_parent_child_events_to_native_trace():
    parent = _context()
    child = RunContext.from_env(parent.spawn_env(child_agent_id="worker"))
    events = [
        RuntimeEvent.create(
            parent,
            event_type="reasoning",
            host="codex",
            framework="openai-agents",
            tokens=TokenUsage(input=100, output=20, provenance="provider_reported"),
            content="plan work",
            sequence=1,
        ),
        RuntimeEvent.create(
            child,
            event_type="tool_call",
            host="codex",
            framework="openai-agents",
            tokens=TokenUsage(input=40, output=10, provenance="provider_reported"),
            content="run search",
            output="found",
            tool=ToolCall.from_arguments("search", {"q": "x"}, status="success"),
            task=TaskResult(outcome=TaskOutcome.PASSED, verifier="pytest", evidence={"passed": True}),
            sequence=2,
        ),
    ]
    trace = compile_native_trace(events, agent_name="orchestrator", framework="openai-agents")
    validate_native_trace_shape(trace)
    assert trace["trace_id"] == parent.trace_id
    assert trace["agent_name"] == "orchestrator"
    assert trace["total_tokens"] == 170
    assert [step["agent_id"] for step in trace["steps"]] == ["parent", "worker"]
    assert trace["steps"][1]["type"] == "tool_call"
    assert trace["steps"][1]["tool_params"]["arguments_sha256"] == events[1].tool.arguments_digest
    assert trace["task_value_score"] == 1.0
    assert trace["metadata"]["degraded_ingest"] is False


def test_event_matches_shipped_json_schema_when_jsonschema_available():
    jsonschema = pytest.importorskip("jsonschema")
    context = _context()
    event = RuntimeEvent.create(
        context,
        event_type="reasoning",
        host="codex",
        framework="raw",
        tokens=TokenUsage(input=1, output=1, provenance="provider_reported"),
        content="hello",
    )
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "tracerazor_event.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft7Validator(schema).validate(event.to_dict())
