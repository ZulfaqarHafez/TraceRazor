import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_tracerazor_trice_module_schema_and_patch_validation(tmp_path):
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

    bundle_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "bundle"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(bundle_schema.stdout)["title"] == "TRICE evidence bundle manifest"

    adapter_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "adapter-profile"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(adapter_schema.stdout)["title"] == "TRICE adapter profile"

    adapter_verdict = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "validate-adapter",
            "examples/trice_adapter_profile_echo.json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(adapter_verdict.stdout)["ok"] is True

    receipt_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "receipt"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(receipt_schema.stdout)["title"] == "TRICE run receipt"

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

    repo = tmp_path / "repo"
    shutil.copytree(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one" / "seed", repo)
    repair = tmp_path / "repair_cli.py"
    repair.write_text(
        "from pathlib import Path\n"
        "p = Path('chunker.py')\n"
        "p.write_text(p.read_text(encoding='utf-8').replace('size - 1', 'size'), encoding='utf-8')\n",
        encoding="utf-8",
    )
    command_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "run",
            "--",
            "--repo",
            str(repo),
            "--task-id",
            "cli-command-fix",
            "--prompt",
            "Fix chunker.py without editing tests.",
            "--verify-cmd",
            "python -m pytest -q --tb=short",
            "--repair-cmd",
            f"{sys.executable} {repair}",
            "--repair-timeout-s",
            "30",
            "--out-dir",
            str(tmp_path / "cli-out"),
            "--rounds",
            "1",
            "--user-feedback",
            "real runs, not replay; target 60% savings",
            "--json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    result = json.loads(command_run.stdout)
    assert result["rounds"][0]["optimized"]["passed"] is True
    assert result["rounds"][0]["optimized"]["modified_files"] == ["chunker.py"]
    receipt_verdict = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "validate-receipt",
            result["rounds"][0]["optimized"]["receipt_path"],
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(receipt_verdict.stdout)["adapter_type"] == "command"
