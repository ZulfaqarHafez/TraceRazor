from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from benchmark.hf_audit_stats import _audit as audit_hf_corpus
from benchmark.run_benchmarks import audit


def _completed(payload: dict) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["tracerazor", "audit"],
        returncode=0,
        stdout=json.dumps(payload),
        stderr="",
    )


def test_audit_ignores_structured_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: _completed(
            {
                "status": "skipped",
                "reason": "below_min_steps",
                "steps_found": 2,
                "min_steps": 5,
            }
        ),
    )

    assert audit("tracerazor", Path("short.json")) is None


def test_audit_rejects_unknown_success_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: _completed({"status": "partial"}),
    )

    with pytest.raises(RuntimeError, match="unexpected JSON contract"):
        audit("tracerazor", Path("partial.json"))


def test_audit_returns_scored_report(monkeypatch: pytest.MonkeyPatch) -> None:
    report = {"score": {"score": 83.0, "grade": "B"}, "total_tokens": 100}
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: _completed(report),
    )

    assert audit("tracerazor", Path("trace.json")) == report


def test_audit_rejects_non_object_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["tracerazor", "audit"],
            returncode=0,
            stdout="[]",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="unexpected JSON contract"):
        audit("tracerazor", Path("array.json"))


def test_hf_audit_ignores_structured_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: _completed(
            {
                "status": "skipped",
                "reason": "below_min_steps",
                "steps_found": 2,
                "min_steps": 5,
            }
        ),
    )

    assert audit_hf_corpus("tracerazor", Path("short.json")) is None
