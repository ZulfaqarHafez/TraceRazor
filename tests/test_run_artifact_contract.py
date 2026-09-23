"""One run-artifact contract across every writer.

``.tracerazor/runs/<run-id>/`` is written by the Rust lifecycle hook
(``tracerazor agent hook``), the Python runtime (``tracerazor.runtime``), and,
for validation.json, the MCP ``record_validation`` tool. Each artifact must
match its schema and carry the same field set whichever writer produced it,
and every run receipt must pass ``tracerazor agent verify-receipt``.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tracerazor._launcher import find_binary
from tracerazor.runtime import AuditPolicy, PrivacyMode, RunContext, TokenUsage
from tracerazor.runtime.processor import TraceRazorProcessor

REPO = Path(__file__).resolve().parent.parent
jsonschema = pytest.importorskip("jsonschema")


def _schema(name: str) -> dict:
    return json.loads((REPO / "schemas" / name).read_text(encoding="utf-8"))


def _validate(artifact: dict, schema_name: str) -> None:
    jsonschema.Draft202012Validator(_schema(schema_name)).validate(artifact)


def _load(run_dir: Path, name: str) -> dict:
    return json.loads((run_dir / name).read_text(encoding="utf-8"))


def _clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRACERAZOR_")}
    env.pop("TRACEPARENT", None)
    env.pop("traceparent", None)
    env.update(extra)
    return env


@pytest.fixture(scope="module")
def binary() -> str:
    found = find_binary()
    if found is None:
        pytest.skip("native TraceRazor binary not built")
    return found


def _claude_transcript(messages: int) -> str:
    lines = [{"type": "user", "session_id": "session-1",
              "message": {"role": "user", "content": "inspect carefully"}}]
    for index in range(messages):
        lines.append({
            "type": "assistant",
            "session_id": "session-1",
            "message": {
                "id": f"message-{index}",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": f"Step {index}: inspect carefully"}],
                "usage": {"input_tokens": 10, "cache_creation_input_tokens": 20,
                          "cache_read_input_tokens": 30, "output_tokens": 40},
            },
        })
    return "\n".join(json.dumps(line) for line in lines)


def _rust_run(binary: str, tmp_path: Path) -> Path:
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    transcript = home / "agent-session.jsonl"
    transcript.write_text(_claude_transcript(6), encoding="utf-8")
    result = subprocess.run(
        [binary, "agent", "hook", "--host", "claude", "--event", "session-end"],
        input=json.dumps({"session_id": "session-1", "transcript_path": str(transcript),
                          "cwd": str(project)}),
        capture_output=True, text=True, cwd=project,
        env=_clean_env(HOME=str(home), USERPROFILE=str(home),
                       TRACERAZOR_RUN_ID="run-rust-contract"),
    )
    assert result.returncode == 0, result.stderr
    return project / ".tracerazor" / "runs" / "run-rust-contract"


def _python_run(workspace: Path, **policy: object) -> TraceRazorProcessor:
    runs = workspace / ".tracerazor" / "runs"
    processor = TraceRazorProcessor(
        policy=AuditPolicy(artifact_dir=str(runs), min_steps=2, **policy),
        context=RunContext.create(agent_id="py-agent"),
    )
    for index in range(3):
        processor.record(
            "reasoning",
            content=f"python contract step {index}",
            tokens=TokenUsage(input=4, output=2, provenance="provider_reported"),
        )
    manifest = processor.finalize()
    assert manifest["audit"]["status"] == "completed", manifest
    return processor


def _verify_receipt(binary: str, run_dir: Path, *extra: str) -> tuple[int, dict]:
    result = subprocess.run(
        [binary, "agent", "verify-receipt", str(run_dir / "run-receipt.json"),
         "--format", "json", *extra],
        capture_output=True, text=True, env=_clean_env(),
    )
    return result.returncode, json.loads(result.stdout)


def _keys(value: dict, *nested: str) -> dict[str, set[str]]:
    shape = {"": set(value)}
    for name in nested:
        shape[name] = set(value[name])
    return shape


def test_rust_and_python_writers_share_one_artifact_contract(binary, tmp_path, monkeypatch):
    monkeypatch.delenv("TRACERAZOR_SIGNING_KEY", raising=False)
    rust_dir = _rust_run(binary, tmp_path)
    python_dir = _python_run(tmp_path).run_dir

    shapes = {}
    for label, run_dir in (("rust", rust_dir), ("python", python_dir)):
        manifest = _load(run_dir, "manifest.json")
        validation = _load(run_dir, "validation.json")
        receipt = _load(run_dir, "run-receipt.json")
        _validate(manifest, "tracerazor_run.schema.json")
        _validate(validation, "tracerazor_validation.schema.json")
        _validate(receipt, "tracerazor_run_receipt.schema.json")
        assert "run-receipt.json" in manifest["files"], label
        shapes[label] = (
            _keys(manifest, "policy", "audit", "ingest_quality"),
            _keys(validation, "enforcement"),
        )

        code, verdict = _verify_receipt(binary, run_dir)
        assert code == 0, (label, verdict)
        assert verdict["status"] == "unsigned", (label, verdict)
        assert verdict["hash_checks"] == {
            "persisted_trace": "verified",
            "report": "verified",
            "manifest_identity": "verified",
        }, (label, verdict)

    assert shapes["rust"] == shapes["python"]


def test_python_receipts_are_signed_by_the_shared_writer(binary, tmp_path, monkeypatch):
    keygen = subprocess.run([binary, "keygen"], capture_output=True, text=True, check=True)
    keys = dict(
        line.split("=", 1) for line in keygen.stdout.splitlines()
        if line.startswith("TRACERAZOR_")
    )
    monkeypatch.setenv("TRACERAZOR_SIGNING_KEY", keys["TRACERAZOR_SIGNING_KEY"])
    run_dir = _python_run(
        tmp_path, privacy=PrivacyMode.RAW, persist_raw_content=True
    ).run_dir

    receipt = _load(run_dir, "run-receipt.json")
    assert receipt["signed"] is True
    assert receipt["replayable"] is True
    code, verdict = _verify_receipt(
        binary, run_dir, "--verify-key", keys["TRACERAZOR_VERIFY_KEY"]
    )
    assert code == 0, verdict
    assert verdict["status"] == "valid"
    assert verdict["authenticated"] is True
    assert verdict["signer_pinned"] is True


def test_mcp_record_validation_keeps_the_contract_and_audit_facts(binary, tmp_path):
    from tracerazor import mcp_server

    processor = _python_run(tmp_path)
    run_dir = processor.run_dir
    result = mcp_server.record_validation(
        processor.context.run_id,
        {"outcome": "passed", "verifier": "pytest -q"},
        cwd=str(tmp_path),
    )
    assert result["ok"] is True, result
    validation = _load(run_dir, "validation.json")
    _validate(validation, "tracerazor_validation.schema.json")
    assert validation["trust_level"] == "untrusted_mcp_record"
    assert validation["status"] == "passed"
    # The runtime's audit facts survive the MCP record instead of being dropped.
    assert validation["audit_status"] == "completed"
    assert validation["enforcement_eligible"] is False
