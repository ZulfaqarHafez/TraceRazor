"""Contract and safety tests for the versioned TraceRazor MCP controls."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from tracerazor import mcp_server as mcp


ENVELOPE_KEYS = {
    "schema_version",
    "run_id",
    "ingest_quality",
    "estimate_status",
    "warnings",
    "data",
    "evidence_ref",
    "error",
    "ok",
}


def _run_dir(root: Path, run_id: str = "run-1") -> Path:
    path = root / ".tracerazor" / "runs" / run_id
    path.mkdir(parents=True)
    return path


def _report(
    *,
    run_id: str = "run-1",
    estimate_status: str = "provider_reported",
    degraded: bool = False,
) -> dict:
    return {
        "trace_id": run_id,
        "estimate_status": estimate_status,
        "score": {
            "metric_normalised": {"ldi": 0.25},
            "ldi": {"loops": [{"tool": "search"}]},
        },
        "fixes": [
            {
                "fix_type": "termination_guard",
                "target": "system_prompt",
                "patch": "Stop retrying unchanged calls.",
                "estimated_token_savings": 100,
                "risk": "dangerous",
            }
        ],
        "manifest": {
            "ingest_quality": {
                "degraded": degraded,
                "degraded_ingest": degraded,
                "token_coverage": 0.5 if degraded else 1.0,
                "content_coverage": 1.0,
                "warnings": ["degraded fixture"] if degraded else [],
            }
        },
    }


def _completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


def _assert_envelope(result: dict) -> None:
    assert ENVELOPE_KEYS <= set(result)
    assert result["schema_version"] == "tracerazor-mcp/v1"
    assert isinstance(result["warnings"], list)
    if result["ok"]:
        assert result["error"] is None
    else:
        assert set(result["error"]) == {"code", "message", "details", "retryable"}


def test_doctor_is_sdk_independent_and_reports_three_surfaces(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp, "find_binary", lambda: None)
    result = mcp.doctor(str(tmp_path))
    _assert_envelope(result)
    assert result["ok"] is True
    assert result["data"]["binary"]["status"] == "missing"
    assert result["data"]["policy"]["status"] == "missing"
    assert result["data"]["artifacts"]["status"] == "empty"


def test_audit_current_run_returns_stable_envelope(monkeypatch, tmp_path):
    run = _run_dir(tmp_path)
    (run / "trace.json").write_text("{}", encoding="utf-8")
    report = _report()
    seen = {}

    monkeypatch.setattr(mcp, "_resolve_binary", lambda: "tracerazor")

    def fake_run(args):
        seen["args"] = args
        return _completed(args, stdout=json.dumps(report))

    monkeypatch.setattr(mcp, "_run", fake_run)
    result = mcp.audit_current_run("run-1", str(tmp_path))
    _assert_envelope(result)
    assert result["run_id"] == "run-1"
    assert result["estimate_status"] == "provider_reported"
    assert seen["args"][1:3] == ["audit", str(run / "trace.json")]
    assert "--hermetic" in seen["args"]


def test_latest_findings_falls_back_to_report(monkeypatch, tmp_path):
    run = _run_dir(tmp_path)
    (run / "report.json").write_text(json.dumps(_report()), encoding="utf-8")
    result = mcp.latest_findings("run-1", str(tmp_path))
    _assert_envelope(result)
    assert result["data"]["count"] == 1
    assert result["data"]["findings"][0]["fix_type"] == "termination_guard"
    assert "returning report fixes" in result["warnings"][0]


def test_compare_runs_shells_json_compare_and_preserves_gate_result(monkeypatch, tmp_path):
    first = _run_dir(tmp_path, "before")
    second = _run_dir(tmp_path, "after")
    (first / "trace.json").write_text("{}", encoding="utf-8")
    (second / "trace.json").write_text("{}", encoding="utf-8")
    comparison = {
        "target": {"trace_id": "after"},
        "regression_detected": True,
    }
    seen = {}
    monkeypatch.setattr(mcp, "_resolve_binary", lambda: "tracerazor")

    def fake_run(args):
        seen["args"] = args
        return _completed(args, returncode=1, stdout=json.dumps(comparison))

    monkeypatch.setattr(mcp, "_run", fake_run)
    result = mcp.compare_runs("before", "after", str(tmp_path), 7.5)
    _assert_envelope(result)
    assert result["ok"] is True
    assert result["data"]["passed"] is False
    assert seen["args"][1:4] == ["compare", str(first / "trace.json"), str(second / "trace.json")]
    assert seen["args"][-4:] == ["--format", "json", "--regression-threshold", "7.5"]


def test_compare_rejects_traversal_before_subprocess(monkeypatch, tmp_path):
    called = False
    monkeypatch.setattr(mcp, "_resolve_binary", lambda: "tracerazor")

    def fake_run(args):
        nonlocal called
        called = True
        return _completed(args)

    monkeypatch.setattr(mcp, "_run", fake_run)
    result = mcp.compare_runs("../outside.json", "also-missing.json", str(tmp_path))
    _assert_envelope(result)
    assert result["ok"] is False
    assert result["error"]["code"] == "path_traversal"
    assert called is False


def test_explain_signal_attaches_run_specific_detail(tmp_path):
    run = _run_dir(tmp_path)
    (run / "report.json").write_text(json.dumps(_report()), encoding="utf-8")
    result = mcp.explain_signal("LDI", "run-1", str(tmp_path))
    _assert_envelope(result)
    assert result["data"]["code"] == "LDI"
    assert result["data"]["normalised_score"] == 0.25
    assert result["data"]["matching_fixes"][0]["fix_type"] == "termination_guard"


def test_preview_fix_is_always_dry_run(monkeypatch, tmp_path):
    run = _run_dir(tmp_path)
    (run / "report.json").write_text(json.dumps(_report()), encoding="utf-8")
    target = tmp_path / "AGENTS.md"
    target.write_text("original", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(mcp, "_resolve_binary", lambda: "tracerazor")

    def fake_run(args):
        seen["args"] = args
        return _completed(args, stdout="DRY RUN")

    monkeypatch.setattr(mcp, "_run", fake_run)
    result = mcp.preview_fix("run-1", "AGENTS.md", str(tmp_path))
    _assert_envelope(result)
    assert result["data"]["dry_run"] is True
    assert result["data"]["wrote"] is False
    assert "--dry-run" in seen["args"]
    assert "--force" not in seen["args"]
    assert target.read_text(encoding="utf-8") == "original"


def test_record_validation_is_atomic_and_limited_to_existing_run(tmp_path):
    run = _run_dir(tmp_path)
    result = mcp.record_validation(
        "run-1", {"status": "passed", "task_success": True}, str(tmp_path)
    )
    _assert_envelope(result)
    saved = json.loads((run / "validation.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == "tracerazor-validation/v1"
    assert saved["run_id"] == "run-1"
    assert saved["status"] == "passed"
    assert saved["task"]["outcome"] == "passed"
    assert saved["trust_level"] == "untrusted_mcp_record"
    assert not list(run.glob(".validation.*.tmp"))

    rejected = mcp.record_validation("../escape", {"status": "passed"}, str(tmp_path))
    _assert_envelope(rejected)
    assert rejected["ok"] is False
    assert rejected["error"]["code"] == "invalid_run_id"


def test_record_validation_rejects_symlink_destination(tmp_path):
    run = _run_dir(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("untouched", encoding="utf-8")
    try:
        os.symlink(outside, run / "validation.json")
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    result = mcp.record_validation("run-1", {"status": "passed"}, str(tmp_path))
    _assert_envelope(result)
    assert result["ok"] is False
    assert result["error"]["code"] == "unsafe_symlink"
    assert outside.read_text(encoding="utf-8") == "untouched"


def test_policy_refuses_enforcement_for_estimated_degraded_or_no_verifier(tmp_path):
    (tmp_path / "tracerazor.toml").write_text(
        'schema_version = 1\nmode = "enforce"\n\n[quality]\nverifier = ""\n\n[enforcement]\nenabled = true\n',
        encoding="utf-8",
    )
    run = _run_dir(tmp_path)
    (run / "report.json").write_text(
        json.dumps(_report(estimate_status="estimated", degraded=True)),
        encoding="utf-8",
    )
    result = mcp.check_policy("run-1", str(tmp_path))
    _assert_envelope(result)
    assert result["data"]["enforcement_requested"] is True
    assert result["data"]["enforce_eligible"] is False
    assert "quality.verifier is missing" in result["data"]["refusal_reasons"]
    assert "ingest quality is degraded" in result["data"]["refusal_reasons"]
    assert "usage contains estimated token counts" in result["data"]["refusal_reasons"]


def test_policy_allows_exact_clean_run_with_verifier(tmp_path):
    (tmp_path / "tracerazor.toml").write_text(
        'schema_version = 1\nmode = "enforce"\n\n[quality]\nverifier = "pytest -q"\n\n[enforcement]\nenabled = true\n',
        encoding="utf-8",
    )
    run = _run_dir(tmp_path)
    (run / "report.json").write_text(json.dumps(_report()), encoding="utf-8")
    (run / "validation.json").write_text(
        json.dumps(
            {
                "schema_version": "tracerazor-validation/v1",
                "run_id": "run-1",
                "trust_level": "trusted_executed_verifier",
                "task": {"outcome": "passed", "verifier": "pytest -q"},
            }
        ),
        encoding="utf-8",
    )
    result = mcp.check_policy("run-1", str(tmp_path))
    _assert_envelope(result)
    assert result["data"]["enforce_eligible"] is True
    assert result["data"]["refusal_reasons"] == []


def test_policy_requires_enabled_and_enforce_mode(tmp_path):
    (tmp_path / "tracerazor.toml").write_text(
        'schema_version = 1\nmode = "coach"\n\n[quality]\nverifier = "pytest -q"\n\n[enforcement]\nenabled = true\n',
        encoding="utf-8",
    )
    result = mcp.check_policy(cwd=str(tmp_path))
    _assert_envelope(result)
    assert result["data"]["enforcement_requested"] is False


def test_policy_rejects_failed_or_mismatched_validation(tmp_path):
    (tmp_path / "tracerazor.toml").write_text(
        'schema_version = 1\nmode = "enforce"\n\n[quality]\nverifier = "pytest -q"\n\n[enforcement]\nenabled = true\n',
        encoding="utf-8",
    )
    run = _run_dir(tmp_path)
    (run / "report.json").write_text(json.dumps(_report()), encoding="utf-8")
    (run / "validation.json").write_text(
        json.dumps(
            {
                "schema_version": "tracerazor-validation/v1",
                "run_id": "run-1",
                "task": {"outcome": "failed", "verifier": "different"},
            }
        ),
        encoding="utf-8",
    )
    result = mcp.check_policy("run-1", str(tmp_path))
    _assert_envelope(result)
    assert result["data"]["enforce_eligible"] is False
    assert "task verifier did not pass" in result["data"]["refusal_reasons"]
    assert (
        "recorded verifier does not match quality.verifier"
        in result["data"]["refusal_reasons"]
    )


def test_mcp_recorded_validation_never_authorizes_enforcement(tmp_path):
    (tmp_path / "tracerazor.toml").write_text(
        'schema_version = 1\nmode = "enforce"\n\n[quality]\nverifier = "pytest -q"\n\n[enforcement]\nenabled = true\n',
        encoding="utf-8",
    )
    run = _run_dir(tmp_path)
    (run / "report.json").write_text(json.dumps(_report()), encoding="utf-8")
    recorded = mcp.record_validation(
        "run-1", {"outcome": "passed", "verifier": "pytest -q"}, str(tmp_path)
    )
    _assert_envelope(recorded)
    result = mcp.check_policy("run-1", str(tmp_path))
    _assert_envelope(result)
    assert result["data"]["enforce_eligible"] is False
    assert (
        "trusted executed verifier receipt is missing"
        in result["data"]["refusal_reasons"]
    )


def test_record_validation_redacts_untrusted_keys_and_rejects_unknown_fields(tmp_path):
    run = _run_dir(tmp_path)
    secret = "secret-in-a-mapping-key"
    result = mcp.record_validation(
        "run-1",
        {
            "outcome": "passed",
            "verifier": "pytest -q",
            "evidence": {secret: "sensitive value"},
        },
        str(tmp_path),
    )
    _assert_envelope(result)
    persisted = (run / "validation.json").read_text(encoding="utf-8")
    assert secret not in persisted
    assert "redacted-key sha256=" in persisted

    rejected = mcp.record_validation(
        "run-1", {"outcome": "passed", "arbitrary": "value"}, str(tmp_path)
    )
    _assert_envelope(rejected)
    assert rejected["ok"] is False
    assert rejected["error"]["code"] == "invalid_validation"


def test_policy_uses_manifest_when_report_is_not_yet_available(tmp_path):
    (tmp_path / "tracerazor.toml").write_text(
        'schema_version = 1\nmode = "enforce"\n\n[quality]\nverifier = "pytest -q"\n',
        encoding="utf-8",
    )
    run = _run_dir(tmp_path)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "estimate_status": "estimated",
                "ingest_quality": {"degraded_ingest": False, "token_coverage": 1.0},
            }
        ),
        encoding="utf-8",
    )
    result = mcp.check_policy("run-1", str(tmp_path))
    _assert_envelope(result)
    assert result["data"]["enforce_eligible"] is False
    assert "usage contains estimated token counts" in result["data"]["refusal_reasons"]
    assert any("using manifest quality metadata" in item for item in result["warnings"])


def test_policy_refuses_runtime_manifest_with_partial_provider_coverage(tmp_path):
    (tmp_path / "tracerazor.toml").write_text(
        'schema_version = 1\nmode = "enforce"\n\n[quality]\nverifier = "pytest -q"\n\n[enforcement]\nenabled = true\n',
        encoding="utf-8",
    )
    run = _run_dir(tmp_path)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "estimate_status": "provider_reported",
                "ingest_quality": {
                    "status": "degraded",
                    "provider_token_coverage": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )
    result = mcp.check_policy("run-1", str(tmp_path))
    _assert_envelope(result)
    assert result["data"]["enforce_eligible"] is False
    assert "ingest quality is degraded" in result["data"]["refusal_reasons"]


def test_policy_uses_dependency_free_parser_on_python_310(monkeypatch, tmp_path):
    (tmp_path / "tracerazor.toml").write_text(
        'schema_version = 1\nmode = "enforce"\nprivacy = "local-redacted"\n\n[quality]\nverifier = "pytest -q"\n\n[enforcement]\nenabled = true\n',
        encoding="utf-8",
    )
    run = _run_dir(tmp_path)
    (run / "report.json").write_text(json.dumps(_report()), encoding="utf-8")
    monkeypatch.setattr(mcp, "tomllib", None)
    result = mcp.check_policy("run-1", str(tmp_path))
    _assert_envelope(result)
    assert result["data"]["policy"]["mode"] == "enforce"
    assert result["data"]["verifier_present"] is True


def test_verify_evidence_wraps_tamper_as_data_not_transport_error(monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mcp, "_resolve_binary", lambda: "tracerazor")
    monkeypatch.setattr(
        mcp,
        "_run",
        lambda args: _completed(
            args, returncode=1, stdout=json.dumps({"status": "tampered"})
        ),
    )
    result = mcp.verify_evidence("report.json", cwd=str(tmp_path))
    _assert_envelope(result)
    assert result["ok"] is True
    assert result["data"]["verified"] is False
    assert result["data"]["verdict"]["status"] == "tampered"


def test_redacted_runtime_evidence_is_not_mislabeled_tampered(monkeypatch, tmp_path):
    run = _run_dir(tmp_path)
    trace = {
        "trace_id": "run-1",
        "agent_name": "agent",
        "framework": "runtime",
        "steps": [{"id": 1, "type": "reasoning", "content": "[redacted]", "tokens": 1}],
        "metadata": {
            "tracerazor_redacted": True,
            "persisted_representation": "redacted_non_auditable",
        },
    }
    report = {
        "schema_version": "tracerazor-report/v1",
        "trace_id": "run-1",
        "persisted_representation": "redacted_auditor_report",
        "score": {"score": 75},
    }
    (run / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        mcp,
        "_resolve_binary",
        lambda: (_ for _ in ()).throw(AssertionError("redacted evidence must not invoke verify")),
    )

    current = mcp.audit_current_run("run-1", str(tmp_path))
    _assert_envelope(current)
    assert current["data"]["reused_in_memory_audit"] is True

    verified = mcp.verify_evidence(
        ".tracerazor/runs/run-1/report.json",
        ".tracerazor/runs/run-1/trace.json",
        str(tmp_path),
    )
    _assert_envelope(verified)
    assert verified["data"]["status"] == "non_replayable_redacted"
    assert verified["data"]["tampered"] is False

    legacy = mcp.verify_report(
        ".tracerazor/runs/run-1/report.json",
        ".tracerazor/runs/run-1/trace.json",
        str(tmp_path),
    )
    assert legacy["status"] == "non_replayable_redacted"
    assert legacy["tampered"] is False


def test_server_startup_binds_model_controlled_cwd_to_one_workspace(monkeypatch, tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    monkeypatch.delenv("TRACERAZOR_MCP_ROOT", raising=False)
    monkeypatch.setattr(mcp, "_SERVER_ROOT", None)
    monkeypatch.chdir(root)
    assert mcp._bind_server_root() == root.resolve()
    with pytest.raises(mcp.McpToolError, match="outside TRACERAZOR_MCP_ROOT"):
        mcp._workspace_root(str(outside))


def test_legacy_dict_tools_receive_additive_metadata(monkeypatch, tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mcp, "_resolve_binary", lambda: "tracerazor")
    monkeypatch.setattr(
        mcp,
        "_run",
        lambda args: _completed(args, stdout=json.dumps(_report())),
    )
    result = mcp.audit_trace("trace.json", cwd=str(tmp_path))
    assert result["trace_id"] == "run-1"
    assert result["passed"] is True
    assert result["_tracerazor"]["schema_version"] == "tracerazor-mcp/v1"
    assert result["_tracerazor"]["run_id"] == "run-1"
