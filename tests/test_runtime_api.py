from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import hashlib
import json

from tracerazor.runtime import (
    AuditPolicy,
    TraceRazorProcessor,
    auto_instrument,
    configure,
    get_current_processor,
    register_instrumentation,
    unregister_instrumentation,
)


def test_configure_sets_process_default_without_optional_sdks(tmp_path):
    processor = configure(artifact_dir=tmp_path, agent_id="configured")
    assert isinstance(processor, TraceRazorProcessor)
    assert get_current_processor() is processor
    result = auto_instrument("sdk-that-does-not-exist")
    assert result.ok
    assert "sdk_that_does_not_exist" in result.unavailable


def test_custom_instrumentation_registry_is_lazy_and_isolated(tmp_path):
    processor = configure(artifact_dir=tmp_path)
    calls = []

    def installer(runtime):
        calls.append(runtime)
        return object()

    register_instrumentation("test-sdk", installer)
    try:
        result = auto_instrument("test-sdk", processor=processor)
        assert result.enabled == ("test_sdk",)
        assert calls == [processor]
        assert "handles" not in result.to_dict()
    finally:
        unregister_instrumentation("test-sdk")


def test_policy_loads_root_toml_and_nested_sections(tmp_path):
    path = tmp_path / "tracerazor.toml"
    path.write_text(
        """
schema_version = 1
mode = "enforce"
capture = "auto"
hermetic = true
privacy = "local-redacted"
persist_raw_content = false
artifact_dir = ".tracerazor/runs"
min_steps = 5

[quality]
verifier = "pytest"

[enforcement]
enabled = true
""".strip(),
        encoding="utf-8",
    )
    policy = AuditPolicy.load(path)
    assert policy.configured_for_enforcement
    assert policy.verifier == "pytest"
    assert policy.artifact_dir == ".tracerazor/runs"

    processor = configure(policy_path=path)
    assert processor.run_dir.parent == tmp_path / ".tracerazor" / "runs"


def test_off_policy_has_no_artifact_side_effect(tmp_path):
    processor = TraceRazorProcessor(
        policy=AuditPolicy(mode="off", artifact_dir=str(tmp_path)),
    )
    assert not processor.run_dir.exists()
    processor.finalize()
    assert not processor.run_dir.exists()


def test_openai_agents_processor_isolates_multiple_sdk_traces(tmp_path, monkeypatch):
    registered = []
    removed = []
    fake_tracing = SimpleNamespace(
        add_trace_processor=lambda processor: registered.append(processor),
        remove_trace_processor=lambda processor: removed.append(processor),
    )
    monkeypatch.setitem(sys.modules, "agents", SimpleNamespace(tracing=fake_tracing))

    def auditor(trace, *, hermetic, min_steps):
        assert hermetic is True
        trace_sha256 = hashlib.sha256(
            json.dumps(
                trace,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": "tracerazor-report/v1",
            "trace_id": trace["trace_id"],
            "agent_name": trace["agent_name"],
            "framework": trace["framework"],
            "total_steps": len(trace["steps"]),
            "total_tokens": trace["total_tokens"],
            "score": {
                "score": 80.0,
                "grade": "Good",
                "passes_threshold": True,
                "metric_normalised": {},
            },
            "diff": [],
            "savings": {"tokens_saved": 0, "reduction_pct": 0.0},
            "fixes": [],
            "summary": "ok",
            "manifest": {
                "trace_sha256": trace_sha256,
                "tool_version": "1.1.0",
                "created_at": "2026-07-10T00:00:00Z",
                "similarity_backend": "bow",
                "weights": {},
                "weights_sha256": "b" * 64,
                "threshold": 70.0,
                "min_steps": min_steps,
                "hermetic": True,
            },
        }

    runtime = TraceRazorProcessor(
        policy=AuditPolicy(artifact_dir=str(tmp_path), min_steps=2),
        auditor=auditor,
    )
    result = auto_instrument("openai_agents", processor=runtime)
    assert result.enabled == ("openai_agents",)
    adapter = result.handles["openai_agents"]
    assert registered == [adapter]

    def span(trace_id, span_id, parent_id, content):
        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_id": parent_id,
            "span_data": {
                "type": "generation",
                "content": content,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        }

    first = {"trace_id": "sdk-trace-one"}
    adapter.on_trace_start(first)
    adapter.on_span_end(span("sdk-trace-one", "sdk-span-1", "sdk-root-1", "one"))
    adapter.on_span_end(span("sdk-trace-one", "sdk-span-2", "sdk-span-1", "two"))
    adapter.on_trace_end(first)
    first_processor = adapter.processors[0]
    assert first_processor.finalized
    first_count = len(first_processor.events)

    # A late callback for a completed trace is ignored rather than recorded in
    # a finalized processor or raised into the SDK.
    adapter.on_span_end(span("sdk-trace-one", "sdk-span-late", "sdk-span-2", "late"))
    assert len(first_processor.events) == first_count

    second = {"trace_id": "sdk-trace-two"}
    adapter.on_trace_start(second)
    adapter.on_span_end(span("sdk-trace-two", "sdk-span-3", "sdk-root-2", "three"))
    adapter.on_span_end(span("sdk-trace-two", "sdk-span-4", "sdk-span-3", "four"))
    adapter.on_trace_end(second)

    assert len(adapter.processors) == 2
    assert all(processor.finalized for processor in adapter.processors)
    assert len({processor.run_dir for processor in adapter.processors}) == 2
    assert [event.span_id for event in adapter.processors[1].events] == [
        "sdk-span-3",
        "sdk-span-4",
    ]
    assert [event.parent_span_id for event in adapter.processors[1].events] == [
        "sdk-root-2",
        "sdk-span-3",
    ]
    assert all(
        event.metadata["sdk_trace_id"] == "sdk-trace-two"
        for event in adapter.processors[1].events
    )
    assert adapter.errors == ()
    adapter.shutdown()
    assert removed == [adapter]
