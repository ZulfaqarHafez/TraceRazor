import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_tracerazor_trice_module_schema_and_patch_validation(tmp_path):
    doctor = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "doctor", "--format", "json", "--offline"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert doctor.returncode in (0, 1)
    doctor_payload = json.loads(doctor.stdout)
    assert doctor_payload["schema_version"] == "trice-doctor/v1"
    assert doctor_payload["offline"] is True
    assert "schemas" in doctor_payload["checks"]

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

    claim_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "claim-card"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(claim_schema.stdout)["title"] == "TRICE deterministic claim card"

    readiness_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "suite-readiness"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(readiness_schema.stdout)["title"] == "TRICE suite readiness preflight"

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

    remote_source = tmp_path / "remote-git-list.json"
    remote_suite = tmp_path / "remote-suite.json"
    remote_source.write_text(
        json.dumps(
            {
                "name": "cli-scaffold-suite",
                "adapter_profile": "profiles/codex-adapter.json",
                "tasks": [
                    {
                        "task_id": "cli-remote-task",
                        "url": "https://github.com/example/project.git",
                        "rev": "0123456789abcdef0123456789abcdef01234567",
                        "prompt": "Fix the failing parser test without editing tests.",
                        "verify_cmd": ["python", "-m", "pytest", "-q"],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    scaffold = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "suite",
            "scaffold",
            "--source",
            str(remote_source),
            "--out",
            str(remote_suite),
            "--json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    scaffold_payload = json.loads(scaffold.stdout)
    assert scaffold_payload["schema_version"] == "trice-suite/v1"
    assert remote_suite.is_file()
    assert scaffold_payload["tasks"][0]["git"]["url"] == "https://github.com/example/project.git"

    readiness_out = tmp_path / "readiness.json"
    readiness = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "suite",
            "readiness",
            str(remote_suite),
            "--out",
            str(readiness_out),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    readiness_payload = json.loads(readiness.stdout)
    assert readiness_payload["readiness"]["schema_version"] == "trice-suite-readiness/v1"
    assert readiness_payload["readiness"]["readiness_level"] == "smoke_ready"
    assert readiness_out.is_file()
    assert readiness_out.with_suffix(".md").is_file()
    assert readiness_out.with_suffix(".tex").is_file()
    assert readiness_out.with_suffix(".svg").is_file()

    readiness_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "suite",
            "verify-readiness",
            str(readiness_out),
            "--manifest",
            str(remote_suite),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    readiness_verify_payload = json.loads(readiness_verify.stdout)
    assert readiness_verify_payload["ok"] is True
    assert readiness_verify_payload["checked_inputs"] == ["suite_manifest"]

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

    claim_card = tmp_path / "claim-card.json"
    claim = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "claim",
            "--suite-result",
            "benchmark/trice/results/v2-broad-smoke/trice_suite_results.json",
            "--manifest",
            "benchmark/trice/results/v2-broad-smoke/trice_suite_evidence_manifest.json",
            "--out",
            str(claim_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    claim_payload = json.loads(claim.stdout)
    assert claim_payload["claim_card"]["schema_version"] == "trice-claim-card/v1"
    assert claim_payload["claim_card"]["claim_level"] == "smoke"
    assert claim_payload["claim_card"]["claim_allowed"] is False
    assert claim_card.is_file()
    assert claim_card.with_suffix(".md").is_file()
    assert claim_card.with_suffix(".tex").is_file()
    assert claim_card.with_suffix(".svg").is_file()

    verify_claim = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "verify-claim",
            str(claim_card),
            "--suite-result",
            "benchmark/trice/results/v2-broad-smoke/trice_suite_results.json",
            "--manifest",
            "benchmark/trice/results/v2-broad-smoke/trice_suite_evidence_manifest.json",
        ],
        capture_output=True,
        text=True,
    )
    assert verify_claim.returncode == 0, verify_claim.stderr
    verify_payload = json.loads(verify_claim.stdout)
    assert verify_payload["ok"] is True
    assert verify_payload["claim_level"] == "smoke"
    assert verify_payload["checked_inputs"] == ["suite_result", "suite_manifest"]
