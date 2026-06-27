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
    assert "openssf_scorecard" in doctor_payload["checks"]

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

    artifact_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "artifact-card"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(artifact_schema.stdout)["title"] == "TRICE artifact review card"

    protocol_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "protocol-lock"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(protocol_schema.stdout)["title"] == "TRICE protocol lock"

    design_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "design-card"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(design_schema.stdout)["title"] == "TRICE statistical design card"

    reproduction_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "reproduction-card"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(reproduction_schema.stdout)["title"] == "TRICE reproduction card"

    release_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "release-card"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(release_schema.stdout)["title"] == "TRICE release card"

    contract_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "contract-card"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(contract_schema.stdout)["title"] == "TRICE public contract card"

    release_evidence_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "release-evidence"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(release_evidence_schema.stdout)["title"] == "TRICE release evidence"

    integrity_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "integrity"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(integrity_schema.stdout)["title"] == "TRICE integrity card"

    crates_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "crates-card"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(crates_schema.stdout)["title"] == "TRICE crates publish card"

    install_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "install-card"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(install_schema.stdout)["title"] == "TRICE installability card"

    research_schema = subprocess.run(
        [sys.executable, "-m", "tracerazor.trice", "schema", "research-card"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(research_schema.stdout)["title"] == "TRICE research card"

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

    protocol_out = tmp_path / "protocol-lock.json"
    protocol = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "protocol",
            "--manifest",
            str(remote_suite),
            "--out",
            str(protocol_out),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    protocol_payload = json.loads(protocol.stdout)
    assert protocol_payload["protocol_lock"]["schema_version"] == "trice-protocol-lock/v1"
    assert protocol_payload["protocol_lock"]["protocol_level"] == "smoke_protocol_locked"
    assert protocol_out.is_file()
    assert protocol_out.with_suffix(".md").is_file()
    assert protocol_out.with_suffix(".tex").is_file()
    assert protocol_out.with_suffix(".svg").is_file()

    protocol_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "verify-protocol",
            str(protocol_out),
            "--manifest",
            str(remote_suite),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    protocol_verify_payload = json.loads(protocol_verify.stdout)
    assert protocol_verify_payload["ok"] is True
    assert protocol_verify_payload["checked_inputs"] == ["suite_manifest"]

    design_card = tmp_path / "design-card.json"
    design = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "design",
            "--protocol",
            str(protocol_out),
            "--suite-result",
            "benchmark/trice/results/v2-broad-smoke/trice_suite_results.json",
            "--out",
            str(design_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    design_payload = json.loads(design.stdout)
    assert design_payload["design_card"]["schema_version"] == "trice-design-card/v1"
    assert design_payload["design_card"]["design_level"] == "smoke_design_observed"
    assert design_payload["design_card"]["claim_design_ready"] is False
    assert design_card.is_file()
    assert design_card.with_suffix(".md").is_file()
    assert design_card.with_suffix(".tex").is_file()
    assert design_card.with_suffix(".svg").is_file()

    design_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "verify-design",
            str(design_card),
            "--protocol",
            str(protocol_out),
            "--suite-result",
            "benchmark/trice/results/v2-broad-smoke/trice_suite_results.json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    design_verify_payload = json.loads(design_verify.stdout)
    assert design_verify_payload["ok"] is True
    assert design_verify_payload["checked_inputs"] == ["protocol_lock", "suite_result"]

    reproduction_card = tmp_path / "reproduction-card.json"
    reproduction = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "reproduction",
            "--out",
            str(reproduction_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    reproduction_payload = json.loads(reproduction.stdout)
    assert reproduction_payload["reproduction_card"]["schema_version"] == "trice-reproduction-card/v1"
    assert reproduction_payload["reproduction_card"]["reproduction_level"] == "reviewer_replay_ready_smoke"
    assert reproduction_card.is_file()
    assert reproduction_card.with_suffix(".md").is_file()
    assert reproduction_card.with_suffix(".tex").is_file()
    assert reproduction_card.with_suffix(".svg").is_file()

    reproduction_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "verify-reproduction",
            str(reproduction_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    reproduction_verify_payload = json.loads(reproduction_verify.stdout)
    assert reproduction_verify_payload["ok"] is True
    assert "paper_manifest" in reproduction_verify_payload["checked_inputs"]
    assert "broad_evidence_bundle" in reproduction_verify_payload["checked_inputs"]

    contract_card = tmp_path / "contract-card.json"
    contract = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "contract",
            "--out",
            str(contract_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    contract_payload = json.loads(contract.stdout)
    assert contract_payload["contract_card"]["schema_version"] == "trice-contract-card/v1"
    assert contract_payload["contract_card"]["contract_level"] == "library_contract_locked"
    assert contract_payload["contract_card"]["contract_score"] == 100
    assert contract_card.is_file()
    assert contract_card.with_suffix(".md").is_file()
    assert contract_card.with_suffix(".tex").is_file()
    assert contract_card.with_suffix(".svg").is_file()

    contract_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "verify-contract",
            str(contract_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    contract_verify_payload = json.loads(contract_verify.stdout)
    assert contract_verify_payload["ok"] is True
    assert contract_verify_payload["contract_level"] == "library_contract_locked"
    assert "trice_contract_card.schema.json" in contract_verify_payload["checked_inputs"]

    research_card = tmp_path / "research-card.json"
    research = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "research",
            "--out",
            str(research_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    research_payload = json.loads(research.stdout)
    assert research_payload["research_card"]["schema_version"] == "trice-research-card/v1"
    assert research_payload["research_card"]["research_level"] == "research_basis_locked"
    assert research_payload["research_card"]["source_count"] >= 150
    assert research_card.with_suffix(".md").is_file()
    assert research_card.with_suffix(".tex").is_file()
    assert research_card.with_suffix(".svg").is_file()

    research_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "verify-research",
            str(research_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    research_verify_payload = json.loads(research_verify.stdout)
    assert research_verify_payload["ok"] is True
    assert research_verify_payload["checked_inputs"] == ["ledger"]

    crates_card = tmp_path / "crates-card.json"
    crates = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "crates",
            "--offline",
            "--out",
            str(crates_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    crates_payload = json.loads(crates.stdout)
    assert crates_payload["crates_card"]["schema_version"] == "trice-crates-card/v1"
    assert crates_payload["crates_card"]["crates_card_level"] == "publish_plan_locked"
    assert crates_payload["crates_card"]["local_publish_plan_locked"] is True
    assert crates_payload["crates_card"]["cargo_install_claim_allowed"] is False
    assert crates_card.with_suffix(".md").is_file()
    assert crates_card.with_suffix(".tex").is_file()
    assert crates_card.with_suffix(".svg").is_file()

    crates_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "verify-crates",
            str(crates_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    crates_verify_payload = json.loads(crates_verify.stdout)
    assert crates_verify_payload["ok"] is True
    assert crates_verify_payload["local_publish_plan_locked"] is True

    install_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "verify-install",
            "docs/trice_install_card.json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    install_verify_payload = json.loads(install_verify.stdout)
    assert install_verify_payload["ok"] is True
    assert install_verify_payload["install_level"] in {"python_trice_install_ready", "full_cli_install_ready"}

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "tracerazor-1.0.3-py3-none-any.whl").write_bytes(b"fake wheel\n")
    (dist / "tracerazor-1.0.3.tar.gz").write_bytes(b"fake sdist\n")
    cli_binary = tmp_path / ("tracerazor.exe" if sys.platform.startswith("win") else "tracerazor")
    cli_binary.write_bytes(b"fake cli\n")
    release_evidence_card = tmp_path / "release-evidence.json"
    release_evidence = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "release-evidence",
            "--out",
            str(release_evidence_card),
            "--dist-dir",
            str(dist),
            "--cli-binary",
            str(cli_binary),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    release_evidence_payload = json.loads(release_evidence.stdout)
    assert release_evidence_payload["release_evidence"]["schema_version"] == "trice-release-evidence/v1"
    assert release_evidence_payload["release_evidence"]["release_evidence_level"] == "release_evidence_ready"
    assert release_evidence_card.with_suffix(".md").is_file()
    assert release_evidence_card.with_suffix(".tex").is_file()
    assert release_evidence_card.with_suffix(".svg").is_file()
    assert release_evidence_card.with_name("release-evidence.checksums.txt").is_file()

    release_evidence_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "verify-release-evidence",
            str(release_evidence_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    release_evidence_verify_payload = json.loads(release_evidence_verify.stdout)
    assert release_evidence_verify_payload["ok"] is True
    assert "release-evidence.checksums.txt" in release_evidence_verify_payload["checked_sidecars"]

    release_card = tmp_path / "release-card.json"
    release = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "release",
            "--offline",
            "--out",
            str(release_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    release_payload = json.loads(release.stdout)
    assert release_payload["release_card"]["schema_version"] == "trice-release-card/v1"
    assert release_payload["release_card"]["release_level"] == "local_release_candidate"
    assert release_payload["release_card"]["public_release_ready"] is False
    assert release_card.is_file()
    assert release_card.with_suffix(".md").is_file()
    assert release_card.with_suffix(".tex").is_file()
    assert release_card.with_suffix(".svg").is_file()

    release_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "verify-release",
            str(release_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    release_verify_payload = json.loads(release_verify.stdout)
    assert release_verify_payload["ok"] is True
    assert "artifact_card" in release_verify_payload["checked_inputs"]
    assert "reproduction_card" in release_verify_payload["checked_inputs"]
    assert "contract_card" in release_verify_payload["checked_inputs"]

    integrity_card = tmp_path / "integrity-card.json"
    integrity = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "integrity",
            "--release",
            str(release_card),
            "--release-evidence",
            str(release_evidence_card),
            "--crates",
            str(crates_card),
            "--install",
            "docs/trice_install_card.json",
            "--research",
            str(research_card),
            "--out",
            str(integrity_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    integrity_payload = json.loads(integrity.stdout)
    assert integrity_payload["integrity_card"]["schema_version"] == "trice-integrity-card/v1"
    assert integrity_payload["integrity_card"]["integrity_level"] == "proof_graph_integrity_locked"
    assert integrity_card.with_suffix(".md").is_file()
    assert integrity_card.with_suffix(".tex").is_file()
    assert integrity_card.with_suffix(".svg").is_file()

    integrity_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "verify-integrity",
            str(integrity_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    integrity_verify_payload = json.loads(integrity_verify.stdout)
    assert integrity_verify_payload["ok"] is True
    assert "release_evidence" in integrity_verify_payload["checked_inputs"]
    assert "crates_card" in integrity_verify_payload["checked_inputs"]
    assert "install_card" in integrity_verify_payload["checked_inputs"]
    assert "research_card" in integrity_verify_payload["checked_inputs"]

    artifact_card = tmp_path / "artifact-card.json"
    artifact = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "artifact",
            "--out",
            str(artifact_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    artifact_payload = json.loads(artifact.stdout)
    assert artifact_payload["artifact_card"]["schema_version"] == "trice-artifact-card/v1"
    assert artifact_payload["artifact_card"]["artifact_level"] == "review_ready_smoke"
    assert artifact_payload["artifact_card"]["claim_allowed"] is False
    assert artifact_card.is_file()
    assert artifact_card.with_suffix(".md").is_file()
    assert artifact_card.with_suffix(".tex").is_file()
    assert artifact_card.with_suffix(".svg").is_file()

    artifact_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "verify-artifact",
            str(artifact_card),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    artifact_verify_payload = json.loads(artifact_verify.stdout)
    assert artifact_verify_payload["ok"] is True
    assert "readme" in artifact_verify_payload["checked_inputs"]
    assert "protocol_lock" in artifact_verify_payload["checked_inputs"]
    assert "design_card" in artifact_verify_payload["checked_inputs"]
    assert "reproduction_card" in artifact_verify_payload["checked_inputs"]
    assert "contract_card" in artifact_verify_payload["checked_inputs"]
    assert "trice_artifact_card.schema.json" in artifact_verify_payload["checked_schemas"]
    assert "trice_protocol_lock.schema.json" in artifact_verify_payload["checked_schemas"]
    assert "trice_design_card.schema.json" in artifact_verify_payload["checked_schemas"]
    assert "trice_reproduction_card.schema.json" in artifact_verify_payload["checked_schemas"]
    assert "trice_release_card.schema.json" in artifact_verify_payload["checked_schemas"]
    assert "trice_contract_card.schema.json" in artifact_verify_payload["checked_schemas"]
    assert "trice_release_evidence.schema.json" in artifact_verify_payload["checked_schemas"]

    paper_manifest_verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracerazor.trice",
            "verify",
            "paper/trice_v3_research_manifest.json",
            "--result",
            "benchmark/trice/results/v2-smoke/trice_v2_live_results.json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(paper_manifest_verify.stdout)["ok"] is True

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
