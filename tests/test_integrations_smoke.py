"""Hermetic regression tests for the framework integrations.

These guard against the class of bug where an integration callback calls
``TraceRazorClient.analyse(trace=..., threshold=...)`` even though ``analyse``
takes only ``trace`` (threshold moved to the client constructor). That bug
shipped in v1.1.0 and broke every integration at runtime with a ``TypeError``,
because nothing in CI imported the integrations or exercised ``analyse()``.

The tests are hermetic: they replace the real client with a fake whose
``analyse(self, trace)`` signature matches the real one exactly, so passing any
extra keyword (e.g. ``threshold=``) raises ``TypeError`` and fails the test —
without needing the Rust binary, a server, or the framework packages.
"""
from __future__ import annotations

import pytest

from tracerazor import TraceRazorClient, TraceRazorReport


def _make_report() -> TraceRazorReport:
    return TraceRazorReport(
        trace_id="t",
        agent_name="a",
        framework="f",
        total_steps=5,
        total_tokens=100,
        tas_score=88.0,
        grade="Good",
        passes=True,
        threshold=70.0,
    )


class _FakeClient:
    """Mirrors the real ``TraceRazorClient.analyse`` signature exactly."""

    def __init__(self) -> None:
        self.calls: list = []

    def analyse(self, trace):  # noqa: ANN001 - matches real signature
        self.calls.append(trace)
        return _make_report()


def _patched(cb) -> _FakeClient:
    fake = _FakeClient()
    cb._client = fake
    return fake


def test_crewai_callback_analyse_uses_correct_signature():
    from tracerazor.integrations.crewai import TraceRazorCallback

    cb = TraceRazorCallback(agent_name="x", tracerazor_bin="unused")
    fake = _patched(cb)
    report = cb.analyse()  # would raise TypeError under the v1.1.0 bug
    assert len(fake.calls) == 1
    assert "steps" in fake.calls[0]
    assert report.tas_score == 88.0


def test_openai_agents_hooks_analyse_uses_correct_signature():
    from tracerazor.integrations.openai_agents import TraceRazorHooks

    cb = TraceRazorHooks(agent_name="x", tracerazor_bin="unused")
    fake = _patched(cb)
    report = cb.analyse()
    assert len(fake.calls) == 1
    assert "steps" in fake.calls[0]
    assert report.tas_score == 88.0


def test_langgraph_callback_analyse_uses_correct_signature():
    pytest.importorskip("langchain_core")
    from tracerazor.integrations.langgraph import TraceRazorCallback

    cb = TraceRazorCallback(agent_name="x", tracerazor_bin="unused")
    fake = _patched(cb)
    report = cb.analyse()
    assert len(fake.calls) == 1
    assert report.tas_score == 88.0


def test_http_report_maps_full_server_response():
    """HTTP mode must map the full AuditResponse, not silently drop fields."""
    client = TraceRazorClient(server="http://localhost:9999")
    data = {
        "trace_id": "t",
        "agent_name": "a",
        "framework": "raw",
        "total_steps": 7,
        "total_tokens": 1500,
        "tas_score": 76.0,
        "grade": "Good",
        "tokens_saved": 300,
        "avs": 0.26,
        "fixes": [{"fix_type": "x"}],
        "anomalies": [],
        "report_markdown": "# REPORT",
    }
    report = client._parse_http_report(data)
    assert report.total_steps == 7
    assert report.total_tokens == 1500
    assert report.savings["tokens_saved"] == 300
    assert report.fixes == [{"fix_type": "x"}]
    assert report.metrics["avs"] == 0.26
    assert report.markdown() == "# REPORT"
