"""Tests for the TraceRazor MCP server (tracerazor.mcp_server).

Server construction depends on the optional `mcp` SDK and is guarded with
importorskip. The always-run tests cover the SDK-agnostic surface: the
`--selftest` JSON contract (tolerating SDK present *or* absent) and the tool
functions' missing-binary teach path.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MODULE = "tracerazor.mcp_server"

EXPECTED_TOOLS = {
    "audit_trace",
    "audit_current_run",
    "check_policy",
    "compare_runs",
    "convert_transcript",
    "doctor",
    "explain_signal",
    "latest_findings",
    "list_claude_sessions",
    "preview_fix",
    "record_validation",
    "verify_evidence",
    "verify_report",
}


def _mod():
    return importlib.import_module(MODULE)


# ── always-run: --selftest contract ──────────────────────────────────────────


def test_selftest_subprocess_shape():
    """`python -m tracerazor.mcp_server --selftest` prints JSON and exits 0/1,
    for both SDK-present and SDK-absent installs."""
    proc = subprocess.run(
        [sys.executable, "-m", MODULE, "--selftest"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode in (0, 1), proc.stderr
    data = json.loads(proc.stdout)
    if proc.returncode == 0:
        # SDK present: a catalog of {name, description}.
        assert isinstance(data, list)
        names = {t["name"] for t in data}
        assert EXPECTED_TOOLS <= names
        assert all(t.get("description") for t in data)
    else:
        # SDK absent: error + copy-pasteable install hint.
        assert data["error"]
        assert data["install"] == 'pip install "tracerazor[mcp]"'


def test_tool_catalog_matches_expected_names():
    mod = _mod()
    names = {t["name"] for t in mod._tool_catalog()}
    assert names == EXPECTED_TOOLS
    assert all(t["description"] for t in mod._tool_catalog())


# ── always-run: missing-binary teach path ────────────────────────────────────


def test_resolve_binary_raises_teaching_error(monkeypatch, tmp_path):
    mod = _mod()
    monkeypatch.setenv("TRACERAZOR_BIN", str(tmp_path / "does-not-exist"))
    with pytest.raises(mod.BinaryNotFoundError) as ei:
        mod._resolve_binary()
    msg = str(ei.value)
    assert "TRACERAZOR_BIN" in msg
    assert "cargo build --release -p tracerazor" in msg


def test_audit_trace_missing_binary_teaches(monkeypatch, tmp_path):
    mod = _mod()
    monkeypatch.setenv("TRACERAZOR_BIN", str(tmp_path / "nope"))
    with pytest.raises(mod.BinaryNotFoundError) as ei:
        mod.audit_trace(str(tmp_path / "trace.json"))
    assert "tracerazor-trice" in str(ei.value)


def test_convert_transcript_missing_binary_teaches(monkeypatch, tmp_path):
    mod = _mod()
    monkeypatch.setenv("TRACERAZOR_BIN", str(tmp_path / "nope"))
    with pytest.raises(mod.BinaryNotFoundError):
        mod.convert_transcript(str(tmp_path / "t.jsonl"))


def test_verify_report_missing_binary_teaches(monkeypatch, tmp_path):
    mod = _mod()
    monkeypatch.setenv("TRACERAZOR_BIN", str(tmp_path / "nope"))
    with pytest.raises(mod.BinaryNotFoundError):
        mod.verify_report(str(tmp_path / "report.json"))


# ── always-run: list_claude_sessions needs no binary ─────────────────────────


def test_list_claude_sessions_empty(tmp_path):
    mod = _mod()
    assert mod.list_claude_sessions(str(tmp_path)) == []


def test_list_claude_sessions_reads_index(tmp_path):
    mod = _mod()
    idx_dir = tmp_path / ".tracerazor" / "claude-code"
    idx_dir.mkdir(parents=True)
    entries = [{"trace_id": "s1", "tas_score": 72.0}]
    (idx_dir / "index.json").write_text(json.dumps(entries), encoding="utf-8")
    assert mod.list_claude_sessions(str(tmp_path)) == entries


def test_list_claude_sessions_tolerates_garbage(tmp_path):
    mod = _mod()
    idx_dir = tmp_path / ".tracerazor" / "claude-code"
    idx_dir.mkdir(parents=True)
    (idx_dir / "index.json").write_text("{ not json", encoding="utf-8")
    assert mod.list_claude_sessions(str(tmp_path)) == []


# ── always-run: child processes must not inherit the MCP stdio pipe ─────────


def test_run_detaches_stdin(monkeypatch):
    """_run must pass stdin=DEVNULL: under an MCP stdio host the server's stdin
    is the live protocol pipe, and a child inheriting it (the pip console-script
    wrapper chain on Windows) blocks forever, deadlocking the tool call."""
    mod = _mod()
    seen = {}

    def fake_run(args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod._run(["whatever"])
    assert seen.get("stdin") == subprocess.DEVNULL


# ── SDK-dependent: server construction ───────────────────────────────────────


def test_build_server_registers_tools():
    pytest.importorskip("mcp")
    mod = _mod()
    server = mod._build_server()
    assert server is not None
    assert getattr(server, "name", "tracerazor") == "tracerazor"
