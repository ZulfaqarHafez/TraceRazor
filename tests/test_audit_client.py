import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tracerazor import BelowMinStepsError, BinaryNotFoundError, TraceRazorClient, TraceRazorReport


def _cli_report(score: float = 80.0) -> dict:
    return {
        "trace_id": "trace-1",
        "agent_name": "agent",
        "framework": "custom",
        "total_steps": 5,
        "total_tokens": 123,
        "score": {"score": score, "grade": "Good"},
        "savings": {"tokens_saved": 20},
        "fixes": [{"fix_type": "dedupe"}],
        "anomalies": [],
    }


def test_client_init_is_lazy_when_binary_env_is_invalid(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACERAZOR_BIN", str(tmp_path / "missing-tracerazor"))

    client = TraceRazorClient()

    with pytest.raises(BinaryNotFoundError):
        client.analyse({"steps": []})


def test_cli_audit_uses_machine_flags_and_exit_code_pass_status(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 1, stdout=json.dumps(_cli_report(42.0)), stderr="")

    monkeypatch.setattr("tracerazor._audit_client.subprocess.run", fake_run)
    client = TraceRazorClient(
        bin_path=sys.executable,
        threshold=70,
        min_steps=2,
        weights=tmp_path / "weights.json",
        enhanced=True,
        store=False,
        timeout_s=9,
    )

    report = client.analyse({"steps": []})

    cmd, kwargs = calls[0]
    assert cmd[0] == sys.executable
    assert cmd[1:2] == ["audit"]
    assert "--format" in cmd and cmd[cmd.index("--format") + 1] == "json"
    assert "--threshold" in cmd and cmd[cmd.index("--threshold") + 1] == "70"
    assert "--hermetic" in cmd
    assert "--min-steps" in cmd and cmd[cmd.index("--min-steps") + 1] == "2"
    assert "--weights" in cmd and cmd[cmd.index("--weights") + 1] == str(tmp_path / "weights.json")
    assert "--enhanced" in cmd
    assert "--store" in cmd and cmd[cmd.index("--store") + 1] == "false"
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 9
    assert report.passes is False
    assert report.tas_score == 42.0
    assert report.fixes == [{"fix_type": "dedupe"}]


def test_cli_audit_can_disable_hermetic_per_call(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_cli_report()), stderr="")

    monkeypatch.setattr("tracerazor._audit_client.subprocess.run", fake_run)
    client = TraceRazorClient(bin_path=sys.executable, hermetic=True)

    client.analyse({"steps": []}, hermetic=False, store=None)

    cmd = calls[0]
    assert "--hermetic" not in cmd
    assert "--store" not in cmd


def test_cli_empty_stdout_raises_below_min_steps(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="Notice: trace has 1 step")

    monkeypatch.setattr("tracerazor._audit_client.subprocess.run", fake_run)
    client = TraceRazorClient(bin_path=sys.executable, min_steps=2)

    with pytest.raises(BelowMinStepsError, match="at least 2 steps"):
        client.analyse({"steps": []})


def test_http_audit_sends_bearer_and_maps_full_server_response(monkeypatch):
    post_calls = []

    class FakeRequests:
        @staticmethod
        def post(url, **kwargs):
            post_calls.append((url, kwargs))
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "trace_id": "t",
                    "agent_name": "a",
                    "framework": "raw",
                    "total_steps": 7,
                    "total_tokens": 1500,
                    "tas_score": 76.0,
                    "grade": "Good",
                    "tokens_saved": 300,
                    "avs": 0.26,
                    "manifest": {"hermetic": True},
                    "fixes": [{"fix_type": "x"}],
                    "anomalies": [],
                    "report_markdown": "# REPORT",
                },
            )

    monkeypatch.setitem(sys.modules, "requests", FakeRequests)
    client = TraceRazorClient(server="http://localhost:9999/", api_token="secret", timeout_s=3)

    report = client.analyse({"steps": [1]}, hermetic=True)

    url, kwargs = post_calls[0]
    assert url == "http://localhost:9999/api/audit"
    assert kwargs["headers"] == {"Authorization": "Bearer secret"}
    assert kwargs["json"] == {"trace": {"steps": [1]}, "hermetic": True}
    assert kwargs["timeout"] == 3
    assert report.total_steps == 7
    assert report.total_tokens == 1500
    assert report.savings["tokens_saved"] == 300
    assert report.fixes == [{"fix_type": "x"}]
    assert report.metrics["avs"] == 0.26
    assert report.metrics["manifest"] == {"hermetic": True}
    assert report.markdown() == "# REPORT"


def test_report_labels_projected_savings_as_estimated():
    report = TraceRazorReport(
        trace_id="t",
        agent_name="a",
        framework="custom",
        total_steps=5,
        total_tokens=100,
        tas_score=75,
        grade="Good",
        passes=True,
        threshold=70,
        savings={
            "tokens_saved": 20,
            "reduction_pct": 20,
            "monthly_runs": 50000,
            "monthly_runs_assumed": True,
            "monthly_savings_usd": 12.5,
        },
    )
    assert "Estimated 20 tokens" in report.summary()
    markdown = report.markdown()
    assert "ASSUMED 50,000/month" in markdown
    assert "$12.50/month (estimated)" in markdown
