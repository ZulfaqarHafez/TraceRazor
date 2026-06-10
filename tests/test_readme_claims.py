"""README claims must survive contact with the shipped repository.

Ship-plan Phase 0 acceptance: every trace path the README references exists,
the quickstart audit command actually runs, the version string is single, and
the CLI help carries no internal ticket IDs.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = (REPO / "README.md").read_text(encoding="utf-8")


def _binary():
    env = os.environ.get("TRACERAZOR_BIN")
    if env and Path(env).is_file():
        return env
    for cand in ("release", "debug"):
        p = REPO / "target" / cand / "tracerazor"
        if p.is_file():
            return str(p)
    return None


def test_referenced_trace_paths_exist():
    paths = set(re.findall(r"traces/[\w./-]+\.json", README))
    assert paths, "README should reference shipped trace files"
    missing = [p for p in sorted(paths) if not (REPO / p).is_file()]
    assert not missing, f"README references non-existent files: {missing}"


def test_one_version_everywhere():
    import tracerazor

    pyproject = (REPO / "pyproject.toml").read_text()
    pv = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    assert tracerazor.__version__ == pv
    banner = re.search(r"TraceRazor v(\d+\.\d+\.\d+)", README)
    assert banner and banner.group(1) == pv, "README banner version drifted"
    cargo = (REPO / "Cargo.toml").read_text()
    cv = re.search(r'^version = "([^"]+)"', cargo, re.M).group(1)
    assert cv == pv, f"Cargo workspace {cv} != pyproject {pv}"


def test_quickstart_audit_command_runs():
    binary = _binary()
    if binary is None:
        pytest.skip("tracerazor binary not built")
    env = dict(os.environ, HOME="/tmp/tr-readme-test")
    out = subprocess.run(
        [binary, "audit", str(REPO / "traces/support-agent-run-2847.json"),
         "--hermetic"],
        capture_output=True, text=True, env=env,
    )
    assert out.returncode == 0, out.stderr
    assert "TRACERAZOR SCORE" in out.stdout


def test_cli_help_has_no_ticket_ids():
    binary = _binary()
    if binary is None:
        pytest.skip("tracerazor binary not built")
    out = subprocess.run([binary, "--help"], capture_output=True, text=True)
    assert not re.search(r"\(E-\d+\)", out.stdout), "internal ticket IDs in --help"


def test_find_binary_locates_repo_build(monkeypatch):
    # Phase 0.3 regression: a source checkout's cargo output is one level up
    # from the package, and must be found without TRACERAZOR_BIN.
    if not (REPO / "target").exists():
        pytest.skip("no cargo build output")
    monkeypatch.delenv("TRACERAZOR_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda *_: None)
    from tracerazor._audit_client import TraceRazorClient

    found = TraceRazorClient._find_binary()
    # Either the source build (one level up) or a wheel-bundled binary
    # (tracerazor/bin/) is a correct resolution in a checkout.
    assert str(REPO / "target") in found or str(REPO / "tracerazor" / "bin") in found
