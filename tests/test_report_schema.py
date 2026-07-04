"""Validate the audit report JSON against schemas/report.schema.json.

Skips cleanly when either `jsonschema` or a TraceRazor binary is unavailable,
so the suite stays green on machines without a release build. Point at a
specific binary with TRACERAZOR_BIN; otherwise the usual release path is tried.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "schemas" / "report.schema.json"
TRACE = REPO / "traces" / "support-agent-run-2847.json"


def _find_binary():
    env = os.environ.get("TRACERAZOR_BIN")
    if env:
        p = Path(env)
        return p if p.exists() else None
    for name in ("tracerazor.exe", "tracerazor"):
        cand = REPO / "target" / "release" / name
        if cand.exists():
            return cand
    return None


def test_audit_report_matches_schema():
    jsonschema = pytest.importorskip("jsonschema")

    binary = _find_binary()
    if binary is None:
        pytest.skip("no TraceRazor binary (set TRACERAZOR_BIN or build --release)")
    if not TRACE.exists():
        pytest.skip(f"trace fixture missing: {TRACE}")

    proc = subprocess.run(
        [str(binary), "audit", str(TRACE), "--hermetic", "--format", "json"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    report = json.loads(proc.stdout)

    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    # Raises jsonschema.ValidationError on any contract violation.
    jsonschema.validate(instance=report, schema=schema)

    # Spot-check the stable contract surface the schema pins.
    assert report["schema_version"] == "tracerazor-report/v1"
    assert report["trace_id"] == "support-agent-run-2847"
    assert report["manifest"]["hermetic"] is True
    assert isinstance(report["fixes"], list)
