from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from benchmark.agent_native.evaluate import (
    EvaluationError,
    _expected_receipts,
    _expected_run_bindings,
    evaluate_records,
    load_jsonl,
    load_protocol,
    main,
    protocol_sha256,
    task_manifest_sha256,
    verify_evidence_index,
)


HERE = Path(__file__).resolve().parents[1]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _protocol() -> dict:
    return load_protocol(HERE / "protocol.json")


def _task_manifest(protocol: dict, *, synthetic: bool = True) -> dict:
    hosts = protocol["design"]["host_categories"]
    strata = protocol["design"]["workload_strata"]
    cells = [(host, stratum) for host in hosts for stratum in strata]
    rotations = [
        ["no_tracerazor", "coach", "verified_optimizer"],
        ["coach", "verified_optimizer", "no_tracerazor"],
        ["verified_optimizer", "no_tracerazor", "coach"],
    ]
    tasks = []
    block_number = 0
    for task_number in range(50):
        host, stratum = cells[task_number % len(cells)]
        randomization = []
        for repetition in range(1, 4):
            randomization.append(
                {
                    "repetition": repetition,
                    "block_id": f"block-{task_number:03d}-{repetition}",
                    "condition_order": rotations[block_number % len(rotations)],
                }
            )
            block_number += 1
        tasks.append(
            {
                "task_id": f"task-{task_number:03d}",
                "content_sha256": _sha(f"held-out-content-{task_number}"),
                "host_category": host,
                "workload_stratum": stratum,
                "verifier_id": "task-oracle/v1",
                "randomization": randomization,
            }
        )
    return {
        "schema_version": "tracerazor-agent-task-manifest/v1",
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_sha256(protocol),
        "synthetic": synthetic,
        "generated_at": "2026-07-10T12:00:00Z",
        "tasks": tasks,
    }


def _passing_records(protocol: dict, task_manifest: dict) -> list[dict]:
    study_id = protocol["study_id"]
    records: list[dict] = [
        {
            "record_type": "study_manifest",
            "study_id": study_id,
            "protocol_sha256": protocol_sha256(protocol),
            "task_manifest_sha256": task_manifest_sha256(task_manifest),
            "collection_started_at": "2026-07-11T00:00:00Z",
            "protocol_locked_before_collection": True,
            "task_manifest_locked_before_collection": True,
            "result_mode": "synthetic" if task_manifest["synthetic"] else "real",
        }
    ]
    token_counts = {"no_tracerazor": 1000, "coach": 930, "verified_optimizer": 850}
    base_time = datetime(2026, 7, 11, tzinfo=timezone.utc)
    run_number = 0
    for task_number, task in enumerate(task_manifest["tasks"]):
        for randomization in task["randomization"]:
            repetition = randomization["repetition"]
            for condition in token_counts:
                order_index = randomization["condition_order"].index(condition) + 1
                started_at = base_time + timedelta(seconds=(run_number * 10) + order_index)
                run_key = f"{task['task_id']}-{condition}-{repetition}"
                intervention = condition != "no_tracerazor"
                accepted = condition == "verified_optimizer" and task_number < 10
                records.append(
                    {
                        "record_type": "run",
                        "study_id": study_id,
                        "run_id": run_key,
                        "task_id": task["task_id"],
                        "task_input_sha256": task["content_sha256"],
                        "initial_state_sha256": _sha(
                            f"clean-initial-state-{task['task_id']}"
                        ),
                        "execution_environment_sha256": _sha(
                            f"execution-environment-{task['host_category']}"
                        ),
                        "disposable_workspace_id": f"workspace-{run_key}",
                        "host_category": task["host_category"],
                        "host_version": "host-v1",
                        "model_id": "test-model",
                        "model_version": "model-v1",
                        "agent_config_sha256": _sha(f"agent-config-{task['task_id']}-{repetition}"),
                        "workload_stratum": task["workload_stratum"],
                        "condition": condition,
                        "repetition": repetition,
                        "randomization_block": randomization["block_id"],
                        "order_index": order_index,
                        "started_at": started_at.isoformat().replace("+00:00", "Z"),
                        "held_out": True,
                        "token_usage": {
                            "provenance": "provider_reported",
                            "value": token_counts[condition],
                        },
                        "task_success": True,
                        "verifier_id": task["verifier_id"],
                        "run_receipt_sha256": _sha(f"run-receipt-{run_key}"),
                        "verifier_receipt_sha256": _sha(f"verifier-receipt-{run_key}"),
                        "recommendation_issued": intervention,
                        "recommendation_issuer_id": (
                            "tracerazor-coach/v1" if intervention else None
                        ),
                        "recommendation_id": f"recommendation-{run_key}" if intervention else None,
                        "recommendation_adjudicated": intervention,
                        "recommendation_adjudicator_id": (
                            "independent-reviewer/v1" if intervention else None
                        ),
                        "recommendation_actionable": intervention,
                        "adjudication_receipt_sha256": (
                            _sha(f"adjudication-{run_key}") if intervention else None
                        ),
                        "intervention_accepted": accepted,
                    }
                )
            run_number += 1

    for matrix_cell in protocol["release_matrix"]:
        records.append(
            {
                "record_type": "install_probe",
                "study_id": study_id,
                "matrix_cell": matrix_cell,
                "attempt_id": f"install-{matrix_cell}",
                "succeeded": True,
                "time_to_first_audit_seconds": 120.0,
            }
        )
    for probe_number in range(20):
        records.extend(
            [
                {
                    "record_type": "activation_probe",
                    "study_id": study_id,
                    "probe_id": f"activation-positive-{probe_number}",
                    "expected_activation": True,
                    "activated": probe_number < 18,
                },
                {
                    "record_type": "activation_probe",
                    "study_id": study_id,
                    "probe_id": f"activation-negative-{probe_number}",
                    "expected_activation": False,
                    "activated": probe_number == 0,
                },
            ]
        )
    for probe_number in range(100):
        records.append(
            {
                "record_type": "capture_probe",
                "study_id": study_id,
                "probe_id": f"capture-{probe_number}",
                "token_provenance": "provider_reported",
                "provider_total_tokens": 1000 + probe_number,
                "tracerazor_total_tokens": 1000 + probe_number,
                "parent_child_expected": True,
                "parent_child_linked": True,
            }
        )
    for probe_number in range(50):
        records.extend(
            [
                {
                    "record_type": "overhead_probe",
                    "study_id": study_id,
                    "probe_id": f"overhead-{probe_number}",
                    "baseline_wall_ms": 1000.0,
                    "instrumented_wall_ms": 1010.0,
                    "event_latencies_ms": [1.0] * 10,
                },
                {
                    "record_type": "safety_probe",
                    "study_id": study_id,
                    "probe_id": f"safety-{probe_number}",
                    "sandbox_escape": False,
                    "redacted_secret_leak": False,
                },
            ]
        )
    return records


def _run(records: list[dict], *, synthetic: bool = True) -> dict:
    protocol = _protocol()
    manifest = _task_manifest(protocol, synthetic=synthetic)
    # Callers commonly mutate a record set created with a different manifest;
    # bind the study record to this manifest when only mode changes.
    records[0]["task_manifest_sha256"] = task_manifest_sha256(manifest)
    records[0]["result_mode"] = "synthetic" if synthetic else "real"
    return evaluate_records(records, protocol, manifest)


def _fixture(*, synthetic: bool = True) -> tuple[dict, dict, list[dict]]:
    protocol = _protocol()
    manifest = _task_manifest(protocol, synthetic=synthetic)
    return protocol, manifest, _passing_records(protocol, manifest)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _run_record(records: list[dict], task: str, condition: str, repetition: int) -> dict:
    return next(
        record
        for record in records
        if record.get("record_type") == "run"
        and record["task_id"] == task
        and record["condition"] == condition
        and record["repetition"] == repetition
    )


def test_statistical_pass_is_release_incomplete_and_synthetic_never_passes() -> None:
    protocol, manifest, records = _fixture()

    report = evaluate_records(records, protocol, manifest)

    assert report["statistical_status"] == "pass"
    assert report["release_status"] == "incomplete"
    assert report["status"] == "release_incomplete"
    assert all(gate["status"] == "pass" for gate in report["gates"])
    assert report["efficacy"]["per_condition_median_paired_token_reduction"] == {
        "coach": pytest.approx(0.07),
        "verified_optimizer": pytest.approx(0.15),
    }
    assert len(report["efficacy"]["accepted_optimizer_tasks"]) == 10


def test_locked_protocol_cannot_be_weakened() -> None:
    protocol = _protocol()
    protocol["gates"]["coach_median_token_reduction_min"] = 0.0

    with pytest.raises(EvaluationError, match="locked preregistration"):
        evaluate_records([], protocol, {})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_task_id", "duplicate task_id"),
        ("duplicate_content", "duplicate task content"),
        ("unbalanced_cell", "host/stratum cell"),
        ("fixed_order", "not balanced"),
    ],
)
def test_task_manifest_rejects_identity_balance_and_order_bypasses(
    mutation: str, message: str
) -> None:
    protocol, manifest, records = _fixture()
    if mutation == "duplicate_task_id":
        manifest["tasks"][1]["task_id"] = manifest["tasks"][0]["task_id"]
    elif mutation == "duplicate_content":
        manifest["tasks"][1]["content_sha256"] = manifest["tasks"][0]["content_sha256"]
    elif mutation == "unbalanced_cell":
        target = (manifest["tasks"][0]["host_category"], manifest["tasks"][0]["workload_stratum"])
        for task in manifest["tasks"]:
            if (task["host_category"], task["workload_stratum"]) == target:
                task["host_category"] = "gemini_cli"
                task["workload_stratum"] = "support"
    else:
        for task in manifest["tasks"]:
            for randomization in task["randomization"]:
                randomization["condition_order"] = list(CONDITIONS_FOR_TEST)
    records[0]["task_manifest_sha256"] = task_manifest_sha256(manifest)

    report = evaluate_records(records, protocol, manifest)

    assert report["status"] == "incomplete"
    assert any(message in error for error in report["errors"])


CONDITIONS_FOR_TEST = ("no_tracerazor", "coach", "verified_optimizer")


def test_external_task_manifest_hash_is_binding() -> None:
    protocol, manifest, records = _fixture()
    records[0]["task_manifest_sha256"] = "0" * 64

    report = evaluate_records(records, protocol, manifest)

    assert report["status"] == "incomplete"
    assert any("external task manifest hash mismatch" in error for error in report["errors"])


def test_initial_state_cannot_drift_between_repetitions() -> None:
    protocol, manifest, records = _fixture()
    for condition in CONDITIONS_FOR_TEST:
        _run_record(records, "task-000", condition, 2)["initial_state_sha256"] = _sha(
            "different-but-pair-matched-state"
        )

    report = evaluate_records(records, protocol, manifest)

    assert report["status"] == "incomplete"
    assert any("drifts across runs" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("model", "differs on invariants"),
        ("workspace_contamination", "drifts across runs"),
        ("environment_drift", "drifts across runs"),
        ("workspace_reuse", "workspace identity"),
        ("precommitted_order", "order_index differs"),
        ("actual_order", "timestamps do not match"),
        ("receipt_reuse", "receipt digest"),
        ("partial_acceptance", "run-level"),
        ("adjudication_without_receipt", "adjudication needs"),
        ("adjudication_without_identity", "adjudicator_id"),
        ("issuer_is_adjudicator", "must differ"),
        ("duplicate_recommendation", "recommendation_id"),
    ],
)
def test_run_pair_receipt_acceptance_and_adjudication_bypasses_are_rejected(
    mutation: str, message: str
) -> None:
    protocol, manifest, records = _fixture()
    target = _run_record(records, "task-000", "coach", 1)
    if mutation == "model":
        target["model_version"] = "different-model-version"
    elif mutation == "workspace_contamination":
        target["initial_state_sha256"] = _sha("dirty-workspace")
    elif mutation == "environment_drift":
        target["execution_environment_sha256"] = _sha("different-environment")
    elif mutation == "workspace_reuse":
        other = _run_record(records, "task-000", "verified_optimizer", 1)
        target["disposable_workspace_id"] = other["disposable_workspace_id"]
    elif mutation == "precommitted_order":
        target["order_index"] = 3 if target["order_index"] != 3 else 2
    elif mutation == "actual_order":
        target["started_at"] = "2030-01-01T00:00:00Z"
    elif mutation == "receipt_reuse":
        other = _run_record(records, "task-000", "verified_optimizer", 1)
        target["run_receipt_sha256"] = other["run_receipt_sha256"]
    elif mutation == "partial_acceptance":
        _run_record(records, "task-000", "verified_optimizer", 1)[
            "intervention_accepted"
        ] = False
    elif mutation == "adjudication_without_receipt":
        target["adjudication_receipt_sha256"] = None
    elif mutation == "adjudication_without_identity":
        target["recommendation_adjudicator_id"] = None
    elif mutation == "issuer_is_adjudicator":
        target["recommendation_adjudicator_id"] = target["recommendation_issuer_id"]
    else:
        other = _run_record(records, "task-000", "verified_optimizer", 1)
        target["recommendation_id"] = other["recommendation_id"]

    report = evaluate_records(records, protocol, manifest)

    assert report["status"] == "incomplete"
    assert any(message in error for error in report["errors"])


def test_estimated_successful_pair_is_never_efficacy_evidence() -> None:
    protocol, manifest, records = _fixture()
    target = _run_record(records, "task-000", "coach", 1)
    target["token_usage"] = {"provenance": "estimated", "value": 1}

    report = evaluate_records(records, protocol, manifest)

    assert report["statistical_status"] == "incomplete"
    assert report["design"]["measured_successful_pairs"]["coach"] == 149
    assert report["efficacy"]["per_condition_median_paired_token_reduction"]["coach"] == pytest.approx(
        0.07
    )
    assert any("lack provider-reported usage" in error for error in report["errors"])


def test_failed_pair_with_tiny_token_count_cannot_improve_efficacy() -> None:
    protocol, manifest, records = _fixture()
    target = _run_record(records, "task-000", "coach", 1)
    target["task_success"] = False
    target["token_usage"] = {"provenance": "provider_reported", "value": 1}

    report = evaluate_records(records, protocol, manifest)

    assert report["design"]["successful_pairs"]["coach"] == 149
    assert report["design"]["measured_successful_pairs"]["coach"] == 149
    assert report["efficacy"]["per_condition_median_paired_token_reduction"]["coach"] == pytest.approx(
        0.07
    )


def test_per_condition_gate_prevents_optimizer_from_masking_bad_coach() -> None:
    protocol, manifest, records = _fixture()
    for record in records:
        if record.get("record_type") == "run" and record["condition"] == "coach":
            record["token_usage"]["value"] = 1000
        elif record.get("record_type") == "run" and record["condition"] == "verified_optimizer":
            record["token_usage"]["value"] = 700

    report = evaluate_records(records, protocol, manifest)
    by_id = {gate["id"]: gate for gate in report["gates"]}

    assert report["statistical_status"] == "fail"
    assert by_id["coach_median_token_reduction"]["status"] == "fail"
    assert by_id["coach_token_reduction_ci_lower"]["status"] == "fail"
    assert by_id["verified_optimizer_median_token_reduction"]["status"] == "pass"


def test_baseline_solvability_and_cluster_noninferiority_are_real_gates() -> None:
    protocol, manifest, records = _fixture()
    baseline = [
        record
        for record in records
        if record.get("record_type") == "run" and record["condition"] == "no_tracerazor"
    ]
    for record in baseline[:31]:
        record["task_success"] = False
    coach = [
        record
        for record in records
        if record.get("record_type") == "run" and record["condition"] == "coach"
    ]
    for record in coach[:45]:
        record["task_success"] = False

    report = evaluate_records(records, protocol, manifest)
    by_id = {gate["id"]: gate for gate in report["gates"]}

    assert by_id["baseline_solvability"]["status"] == "fail"
    assert by_id["coach_task_success_noninferiority"]["status"] == "fail"
    assert by_id["coach_task_success_noninferiority"]["observed"]["ci_95"][0] < -0.02


def test_acceptance_count_rate_and_adjudication_minimum_cannot_be_bypassed() -> None:
    protocol, manifest, records = _fixture()
    for repetition in (1, 2, 3):
        _run_record(records, "task-009", "verified_optimizer", repetition)[
            "intervention_accepted"
        ] = False
    issued = [
        record
        for record in records
        if record.get("record_type") == "run" and record["recommendation_issued"]
    ]
    for record in issued[29:]:
        record["recommendation_adjudicated"] = False
        record["recommendation_adjudicator_id"] = None
        record["recommendation_actionable"] = False
        record["adjudication_receipt_sha256"] = None

    report = evaluate_records(records, protocol, manifest)
    by_id = {gate["id"]: gate for gate in report["gates"]}

    assert report["statistical_status"] == "incomplete"
    assert by_id["accepted_optimizer_tasks"]["status"] == "fail"
    assert by_id["accepted_optimizer_task_rate"]["status"] == "fail"
    assert any("29 independently adjudicated" in error for error in report["errors"])


def test_signed_receipt_verification_requires_hash_signature_and_exact_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    manifest = _task_manifest(protocol, synthetic=False)
    report_file = tmp_path / "report.json"
    trace_file = tmp_path / "trace.json"
    report_file.write_text('{"signed":true}', encoding="utf-8")
    digest = hashlib.sha256(report_file.read_bytes()).hexdigest()
    run = next(
        record
        for record in _passing_records(protocol, manifest)
        if record.get("record_type") == "run"
    )
    run["run_receipt_sha256"] = digest
    bindings = _expected_run_bindings([run])
    trace_file.write_text(
        json.dumps({"metadata": {"evaluation_binding": bindings[digest]}}),
        encoding="utf-8",
    )
    index = {
        "schema_version": "tracerazor-agent-evidence-index/v1",
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_sha256(protocol),
        "task_manifest_sha256": task_manifest_sha256(manifest),
        "receipts": [
            {
                "kind": "run",
                "sha256": digest,
                "report_path": report_file.name,
                "trace_path": trace_file.name,
            }
        ],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setattr("benchmark.agent_native.evaluate.shutil.which", lambda _: "tracerazor")
    monkeypatch.setattr(
        "benchmark.agent_native.evaluate.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, json.dumps({"status": "verified", "signature": "ok", "trace_hash": "ok"}), ""
        ),
    )

    verified = verify_evidence_index(
        index_path,
        {"run": {digest}, "verifier": set(), "adjudication": set()},
        protocol,
        manifest,
        expected_run_bindings=bindings,
    )
    assert verified["status"] == "verified"

    monkeypatch.setattr(
        "benchmark.agent_native.evaluate.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, json.dumps({"status": "verified", "signature": "missing", "trace_hash": "ok"}), ""
        ),
    )
    unsigned = verify_evidence_index(
        index_path,
        {"run": {digest}, "verifier": set(), "adjudication": set()},
        protocol,
        manifest,
        expected_run_bindings=bindings,
    )
    assert unsigned["status"] == "failed"
    assert any("not Ed25519 authenticated" in error for error in unsigned["errors"])

    missing = verify_evidence_index(
        index_path,
        {"run": {digest}, "verifier": {_sha("missing")}, "adjudication": set()},
        protocol,
        manifest,
        expected_run_bindings=bindings,
    )
    assert missing["status"] == "failed"
    assert any("missing 1 verifier" in error for error in missing["errors"])

    trace_file.write_text(
        json.dumps(
            {
                "metadata": {
                    "evaluation_binding": {
                        **bindings[digest],
                        "initial_state_sha256": _sha("contaminated-state"),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    tampered_binding = verify_evidence_index(
        index_path,
        {"run": {digest}, "verifier": set(), "adjudication": set()},
        protocol,
        manifest,
        expected_run_bindings=bindings,
    )
    assert tampered_binding["status"] == "failed"
    assert any("evaluation binding" in error for error in tampered_binding["errors"])


def test_evidence_index_rejects_path_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = _protocol()
    manifest = _task_manifest(protocol, synthetic=False)
    outside = tmp_path.parent / "outside-report.json"
    outside.write_text("{}", encoding="utf-8")
    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    index = {
        "schema_version": "tracerazor-agent-evidence-index/v1",
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_sha256(protocol),
        "task_manifest_sha256": task_manifest_sha256(manifest),
        "receipts": [
            {
                "kind": "run",
                "sha256": digest,
                "report_path": "../outside-report.json",
                "trace_path": "trace.json",
            }
        ],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setattr("benchmark.agent_native.evaluate.shutil.which", lambda _: "tracerazor")

    result = verify_evidence_index(
        index_path,
        {"run": {digest}, "verifier": set(), "adjudication": set()},
        protocol,
        manifest,
    )

    assert result["status"] == "failed"
    assert any("escapes the evidence directory" in error for error in result["errors"])


def test_cli_requires_task_manifest_and_synthetic_pass_exits_two(tmp_path: Path) -> None:
    protocol, manifest, records = _fixture()
    results_path = tmp_path / "results.jsonl"
    manifest_path = tmp_path / "tasks.json"
    output_path = tmp_path / "report.json"
    _write_jsonl(results_path, records)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert len(load_jsonl(results_path)) == len(records)
    assert (
        main(
            [
                "--input",
                str(results_path),
                "--task-manifest",
                str(manifest_path),
                "--output",
                str(output_path),
            ]
        )
        == 2
    )
    rendered = json.loads(output_path.read_text(encoding="utf-8"))
    assert rendered["statistical_status"] == "pass"
    assert rendered["status"] == "release_incomplete"


def test_all_machine_contract_files_are_valid_json() -> None:
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    result_schema = json.loads((HERE / "result.schema.json").read_text(encoding="utf-8"))
    task_schema = json.loads((HERE / "task_manifest.schema.json").read_text(encoding="utf-8"))
    evidence_schema = json.loads((HERE / "evidence_index.schema.json").read_text(encoding="utf-8"))

    assert protocol_sha256(protocol) == "1c3cd823b132b731b77d02e9e433d4375b58f749b0fb67b9828714dbe2a788d0"
    assert result_schema["$schema"].endswith("2020-12/schema")
    assert task_schema["properties"]["tasks"]["minItems"] == 50
    assert evidence_schema["properties"]["receipts"]["minItems"] == 1
