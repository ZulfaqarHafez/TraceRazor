import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_tracerazor_trice_module_schema_and_patch_validation():
    schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "patch"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    payload = json.loads(schema.stdout)
    assert payload["title"] == "TRICE deterministic patch spec"

    verdict = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "validate-patch",
            "examples/trice_patch_fix_offbyone.json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(verdict.stdout)["ok"] is True

    suite_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "suite"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(suite_schema.stdout)["title"] == "TRICE live suite manifest"

    suite_verdict = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "validate-suite",
            "examples/trice_suite_fix_offbyone.json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(suite_verdict.stdout)["task_count"] == 1
