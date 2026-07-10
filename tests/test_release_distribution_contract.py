from __future__ import annotations

import json
import sys

from tracerazor._trice.release_evidence import build_release_evidence_card
from tracerazor._trice.install import _install_score, _run_probe


def _check(card, name):
    return next(row for row in card["checks"] if row["name"] == name)


def test_release_evidence_requires_platform_wheel_and_excludes_sdist(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "tracerazor-1.1.0-py3-none-manylinux_2_35_x86_64.whl").write_bytes(
        b"platform wheel"
    )
    binary = tmp_path / "tracerazor"
    binary.write_bytes(b"native binary")

    card = build_release_evidence_card(dist_dir=dist, cli_binary_path=binary)
    assert _check(card, "wheel_present")["passed"] is True
    assert _check(card, "sdist_absent")["passed"] is True

    (dist / "tracerazor-1.1.0.tar.gz").write_bytes(b"source-only distribution")
    unsafe = build_release_evidence_card(dist_dir=dist, cli_binary_path=binary)
    assert _check(unsafe, "sdist_absent")["passed"] is False
    assert unsafe["release_evidence_level"] != "release_evidence_ready"


def test_installability_score_cannot_hide_one_failed_check():
    names = [
        "wheel_present",
        "venv_created",
        "wheel_installs",
        "version_matches",
        "schemas_importable",
        "trice_api_importable",
        "runtime_api_importable",
        "agent_assets_shipped",
        "mcp_catalog_importable",
        "trice_console_works",
        "mcp_selftest_works",
        "sample_audit_works",
        "rust_cli_bundled",
        "rust_cli_from_distribution",
        "agent_console_works",
    ]
    checks = [{"name": name, "passed": True} for name in names]
    assert _install_score(checks) == 100
    for failed_name in names:
        failed = [
            {"name": row["name"], "passed": row["name"] != failed_name}
            for row in checks
        ]
        assert _install_score(failed) < 100


def test_import_probe_parses_complete_stdout_before_excerpt_truncation(tmp_path):
    payload = {
        "mcp_tools": [f"tool-{index:03d}" for index in range(80)],
        "version": "1.1.0",
    }
    command = [sys.executable, "-c", f"print({json.dumps(payload)!r})"]

    receipt, probe = _run_probe(
        command,
        cwd=tmp_path,
        timeout_s=10.0,
        scrub_roots=[tmp_path],
    )

    assert receipt["ok"] is True
    assert len(receipt["stdout_excerpt"]) == 500
    assert probe == payload
    assert probe["version"] == "1.1.0"


def test_import_probe_scrubs_structured_paths_after_parsing(tmp_path):
    payload = {
        "nested": {"artifact": str(tmp_path / "private" / "wheel.whl")},
        "version": "1.1.0",
    }
    command = [sys.executable, "-c", f"print({json.dumps(payload)!r})"]

    receipt, probe = _run_probe(
        command,
        cwd=tmp_path,
        timeout_s=10.0,
        scrub_roots=[tmp_path],
    )

    assert probe["nested"]["artifact"].replace("\\", "/") == "<tmp>/private/wheel.whl"
    assert str(tmp_path).lower() not in json.dumps(receipt).lower()
    assert str(tmp_path).lower() not in json.dumps(probe).lower()
