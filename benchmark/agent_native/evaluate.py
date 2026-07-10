"""Locked evaluation for TraceRazor's agent-native product gates.

This module deliberately separates a statistical result from release evidence.
Estimated token counts never enter efficacy statistics, synthetic studies never
pass the release gate, and real studies need every expected signed receipt to be
verified by the native TraceRazor verifier before this CLI can exit zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROTOCOL_SCHEMA = "tracerazor-agent-eval-protocol/v1"
TASK_MANIFEST_SCHEMA = "tracerazor-agent-task-manifest/v1"
EVIDENCE_INDEX_SCHEMA = "tracerazor-agent-evidence-index/v1"
REPORT_SCHEMA = "tracerazor-agent-eval-report/v1"

# This digest locks every threshold and analysis rule in protocol.json. A
# change after collection starts requires a new study_id; every change requires
# an intentional code-lock update here.
LOCKED_PROTOCOL_SHA256 = "1c3cd823b132b731b77d02e9e433d4375b58f749b0fb67b9828714dbe2a788d0"

CONDITIONS = ("no_tracerazor", "coach", "verified_optimizer")
INTERVENTIONS = ("coach", "verified_optimizer")
KNOWN_RECORD_TYPES = {
    "study_manifest",
    "run",
    "install_probe",
    "activation_probe",
    "capture_probe",
    "overhead_probe",
    "safety_probe",
}
SHA_FIELDS = {
    "task_input_sha256",
    "initial_state_sha256",
    "execution_environment_sha256",
    "agent_config_sha256",
    "run_receipt_sha256",
    "verifier_receipt_sha256",
}
RUN_REQUIRED = {
    "record_type",
    "study_id",
    "run_id",
    "task_id",
    "task_input_sha256",
    "initial_state_sha256",
    "execution_environment_sha256",
    "disposable_workspace_id",
    "host_category",
    "host_version",
    "model_id",
    "model_version",
    "agent_config_sha256",
    "workload_stratum",
    "condition",
    "repetition",
    "randomization_block",
    "order_index",
    "started_at",
    "held_out",
    "token_usage",
    "task_success",
    "verifier_id",
    "run_receipt_sha256",
    "verifier_receipt_sha256",
    "recommendation_issued",
    "recommendation_issuer_id",
    "recommendation_id",
    "recommendation_adjudicated",
    "recommendation_adjudicator_id",
    "recommendation_actionable",
    "adjudication_receipt_sha256",
    "intervention_accepted",
}


class EvaluationError(ValueError):
    """Raised when an evaluation input cannot be safely interpreted."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(protocol))


def task_manifest_sha256(task_manifest: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(task_manifest))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationError(f"cannot read {label} {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{source}: invalid {label} JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be a JSON object")
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationError(f"cannot read {source}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise EvaluationError(f"{source}:{line_number}: every JSONL line must be an object")
        records.append(value)
    if not records:
        raise EvaluationError(f"{source}: no records")
    return records


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = _load_json(path, "protocol")
    _validate_protocol(protocol)
    return protocol


def load_task_manifest(path: str | Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _load_json(path, "task manifest")
    errors, _, _ = _validate_task_manifest(manifest, protocol)
    if errors:
        raise EvaluationError("invalid task manifest: " + "; ".join(errors))
    return manifest


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise EvaluationError(f"unsupported protocol schema: {protocol.get('schema_version')!r}")
    actual_digest = protocol_sha256(protocol)
    if actual_digest != LOCKED_PROTOCOL_SHA256:
        raise EvaluationError(
            "protocol differs from the locked preregistration; use a new study_id and update the "
            f"code lock intentionally (expected {LOCKED_PROTOCOL_SHA256}, got {actual_digest})"
        )
    if protocol.get("preregistration", {}).get("status") != "locked":
        raise EvaluationError("protocol must be locked before evaluation")
    design = protocol.get("design", {})
    if design.get("minimum_held_out_tasks", 0) < 50:
        raise EvaluationError("protocol must require at least 50 held-out tasks")
    if design.get("minimum_tasks_per_host_stratum", 0) < 5:
        raise EvaluationError("protocol cannot weaken balanced host/stratum cells")
    if design.get("repetitions_per_condition") != 3:
        raise EvaluationError("protocol must require exactly three repetitions")
    if tuple(item.get("id") for item in design.get("conditions", [])) != CONDITIONS:
        raise EvaluationError("protocol conditions differ from the locked design")
    if len(set(design.get("host_categories", []))) != 3:
        raise EvaluationError("protocol must contain exactly three hosts")
    if tuple(design.get("workload_strata", [])) != (
        "coding",
        "tool_heavy_research",
        "support",
    ):
        raise EvaluationError("protocol strata differ from the locked design")
    if set(design.get("pair_invariants", [])) != {
        "task_input_sha256",
        "initial_state_sha256",
        "execution_environment_sha256",
        "host_category",
        "host_version",
        "model_id",
        "model_version",
        "agent_config_sha256",
        "verifier_id",
        "randomization_block",
    }:
        raise EvaluationError("protocol pair invariants differ from the locked design")
    if design.get("efficacy_token_provenance") != "provider_reported":
        raise EvaluationError("efficacy token provenance must be provider_reported")


def _validate_task_manifest(
    manifest: Mapping[str, Any], protocol: Mapping[str, Any]
) -> tuple[list[str], dict[str, Mapping[str, Any]], dict[tuple[str, int], Mapping[str, Any]]]:
    errors: list[str] = []
    required = {
        "schema_version",
        "study_id",
        "protocol_sha256",
        "synthetic",
        "generated_at",
        "tasks",
    }
    if set(manifest) != required:
        errors.append(f"task manifest fields must be exactly {sorted(required)!r}")
    if manifest.get("schema_version") != TASK_MANIFEST_SCHEMA:
        errors.append("unsupported task manifest schema")
    if manifest.get("study_id") != protocol.get("study_id"):
        errors.append("task manifest study_id does not match protocol")
    if manifest.get("protocol_sha256") != protocol_sha256(protocol):
        errors.append("task manifest protocol_sha256 does not match locked protocol")
    if not _is_bool(manifest.get("synthetic")):
        errors.append("task manifest synthetic must be boolean")
    if _parse_datetime(manifest.get("generated_at")) is None:
        errors.append("task manifest generated_at must be an ISO-8601 timestamp")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        return errors + ["task manifest tasks must be an array"], {}, {}
    if len(tasks) < protocol["design"]["minimum_held_out_tasks"]:
        errors.append(
            f"task manifest has {len(tasks)} tasks; "
            f"{protocol['design']['minimum_held_out_tasks']} required"
        )

    task_index: dict[str, Mapping[str, Any]] = {}
    content_digests: set[str] = set()
    randomization_index: dict[tuple[str, int], Mapping[str, Any]] = {}
    block_ids: set[str] = set()
    cell_counts: Counter[tuple[str, str]] = Counter()
    position_counts: Counter[tuple[str, int]] = Counter()
    task_fields = {
        "task_id",
        "content_sha256",
        "host_category",
        "workload_stratum",
        "verifier_id",
        "randomization",
    }
    hosts = set(protocol["design"]["host_categories"])
    strata = set(protocol["design"]["workload_strata"])
    repetitions = set(range(1, protocol["design"]["repetitions_per_condition"] + 1))

    for task_number, task in enumerate(tasks, 1):
        label = f"task manifest task {task_number}"
        if not isinstance(task, dict):
            errors.append(f"{label} must be an object")
            continue
        if set(task) != task_fields:
            errors.append(f"{label} fields must be exactly {sorted(task_fields)!r}")
            continue
        task_id = task.get("task_id")
        digest = task.get("content_sha256")
        if not isinstance(task_id, str) or not task_id:
            errors.append(f"{label} task_id must be non-empty")
            continue
        if task_id in task_index:
            errors.append(f"duplicate task_id {task_id!r}")
        else:
            task_index[task_id] = task
        if not _is_sha256(digest):
            errors.append(f"{label} content_sha256 is invalid")
        elif digest in content_digests:
            errors.append(f"duplicate task content digest {digest!r}")
        else:
            content_digests.add(str(digest))
        host = task.get("host_category")
        stratum = task.get("workload_stratum")
        if host not in hosts:
            errors.append(f"{label} host_category is not preregistered")
        if stratum not in strata:
            errors.append(f"{label} workload_stratum is not preregistered")
        if host in hosts and stratum in strata:
            cell_counts[(str(host), str(stratum))] += 1
        if not isinstance(task.get("verifier_id"), str) or not task["verifier_id"]:
            errors.append(f"{label} verifier_id must be non-empty")
        randomization = task.get("randomization")
        if not isinstance(randomization, list) or len(randomization) != len(repetitions):
            errors.append(f"{label} needs exactly three randomization entries")
            continue
        seen_repetitions: set[int] = set()
        for entry in randomization:
            if not isinstance(entry, dict) or set(entry) != {
                "repetition",
                "block_id",
                "condition_order",
            }:
                errors.append(f"{label} has malformed randomization entry")
                continue
            repetition = entry.get("repetition")
            block_id = entry.get("block_id")
            order = entry.get("condition_order")
            if repetition not in repetitions or repetition in seen_repetitions:
                errors.append(f"{label} randomization repetitions must be unique 1..3")
                continue
            seen_repetitions.add(int(repetition))
            if not isinstance(block_id, str) or not block_id or block_id in block_ids:
                errors.append(f"{label} randomization block_id must be globally unique")
            else:
                block_ids.add(block_id)
            if not isinstance(order, list) or len(order) != 3 or set(order) != set(CONDITIONS):
                errors.append(f"{label} condition_order must be a permutation of all conditions")
                continue
            for position, condition in enumerate(order, 1):
                position_counts[(str(condition), position)] += 1
            randomization_index[(task_id, int(repetition))] = entry
        if seen_repetitions != repetitions:
            errors.append(f"{label} randomization does not cover repetitions 1..3")

    minimum_per_cell = protocol["design"]["minimum_tasks_per_host_stratum"]
    for host in sorted(hosts):
        for stratum in sorted(strata):
            count = cell_counts[(host, stratum)]
            if count < minimum_per_cell:
                errors.append(
                    f"host/stratum cell {host}/{stratum} has {count} tasks; {minimum_per_cell} required"
                )
    # Precommitted order must be position-balanced; this prevents a fixed-order
    # manifest from being labelled as randomized after outcomes are known.
    all_position_counts = [position_counts[(condition, position)] for condition in CONDITIONS for position in (1, 2, 3)]
    if all_position_counts and max(all_position_counts) - min(all_position_counts) > 1:
        errors.append("precommitted condition order is not balanced across order positions")
    return errors, task_index, randomization_index


def _required(record: Mapping[str, Any], fields: set[str], label: str, errors: list[str]) -> bool:
    missing = sorted(fields.difference(record))
    extra = sorted(set(record).difference(fields))
    if missing:
        errors.append(f"{label}: missing fields {missing!r}")
    if extra:
        errors.append(f"{label}: unexpected fields {extra!r}")
    return not missing and not extra


def _validate_records(
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    task_index: Mapping[str, Mapping[str, Any]],
    randomization_index: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[list[str], dict[str, list[Mapping[str, Any]]]]:
    errors: list[str] = []
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    study_id = protocol["study_id"]
    run_ids: set[str] = set()
    disposable_workspace_ids: set[str] = set()
    task_starting_states: dict[str, tuple[str, str]] = {}
    receipt_digests: set[str] = set()
    recommendation_ids: set[str] = set()
    probe_ids: dict[str, set[str]] = defaultdict(set)
    expected_mode = "synthetic" if task_manifest["synthetic"] else "real"

    manifest_fields = {
        "record_type",
        "study_id",
        "protocol_sha256",
        "task_manifest_sha256",
        "collection_started_at",
        "protocol_locked_before_collection",
        "task_manifest_locked_before_collection",
        "result_mode",
    }
    probe_fields = {
        "install_probe": {
            "record_type", "study_id", "matrix_cell", "attempt_id", "succeeded",
            "time_to_first_audit_seconds",
        },
        "activation_probe": {
            "record_type", "study_id", "probe_id", "expected_activation", "activated",
        },
        "capture_probe": {
            "record_type", "study_id", "probe_id", "token_provenance",
            "parent_child_expected", "parent_child_linked", "provider_total_tokens",
            "tracerazor_total_tokens",
        },
        "overhead_probe": {
            "record_type", "study_id", "probe_id", "baseline_wall_ms",
            "instrumented_wall_ms", "event_latencies_ms",
        },
        "safety_probe": {
            "record_type", "study_id", "probe_id", "sandbox_escape", "redacted_secret_leak",
        },
    }

    for number, record in enumerate(records, 1):
        label = f"record {number}"
        if not isinstance(record, dict):
            errors.append(f"{label}: must be an object")
            continue
        record_type = record.get("record_type")
        if record_type not in KNOWN_RECORD_TYPES:
            errors.append(f"{label}: unknown record_type {record_type!r}")
            continue
        grouped[str(record_type)].append(record)
        if record.get("study_id") != study_id:
            errors.append(f"{label}: study_id does not match protocol")

        if record_type == "study_manifest":
            if not _required(record, manifest_fields, label, errors):
                continue
            if record["protocol_sha256"] != protocol_sha256(protocol):
                errors.append(f"{label}: protocol_sha256 mismatch")
            if record["task_manifest_sha256"] != task_manifest_sha256(task_manifest):
                errors.append(f"{label}: external task manifest hash mismatch")
            if record["protocol_locked_before_collection"] is not True:
                errors.append(f"{label}: protocol was not locked before collection")
            if record["task_manifest_locked_before_collection"] is not True:
                errors.append(f"{label}: task manifest was not locked before collection")
            if record["result_mode"] != expected_mode:
                errors.append(f"{label}: result_mode does not match external task manifest")
            if _parse_datetime(record["collection_started_at"]) is None:
                errors.append(f"{label}: invalid collection_started_at")
            continue

        if record_type == "run":
            if not _required(record, RUN_REQUIRED, label, errors):
                continue
            string_fields = {
                "run_id", "task_id", "host_category", "host_version", "model_id",
                "model_version", "workload_stratum", "condition", "randomization_block",
                "verifier_id", "disposable_workspace_id",
            }
            if any(not isinstance(record[field], str) or not record[field] for field in string_fields):
                errors.append(f"{label}: run identity and pair-invariant fields must be non-empty strings")
            run_id = str(record["run_id"])
            if run_id in run_ids:
                errors.append(f"{label}: duplicate run_id {run_id!r}")
            run_ids.add(run_id)
            workspace_id = str(record["disposable_workspace_id"])
            if workspace_id in disposable_workspace_ids:
                errors.append(f"{label}: disposable workspace identity {workspace_id!r} is reused")
            disposable_workspace_ids.add(workspace_id)
            task_id = str(record["task_id"])
            starting_state = (
                str(record["initial_state_sha256"]),
                str(record["execution_environment_sha256"]),
            )
            previous_state = task_starting_states.setdefault(task_id, starting_state)
            if previous_state != starting_state:
                errors.append(
                    f"{label}: task initial state or execution environment drifts across runs"
                )
            for field in SHA_FIELDS:
                digest = record[field]
                if not _is_sha256(digest):
                    errors.append(f"{label}: {field} is not a SHA-256 digest")
                elif digest in receipt_digests and field.endswith("receipt_sha256"):
                    errors.append(f"{label}: receipt digest {digest!r} is reused")
                elif field.endswith("receipt_sha256"):
                    receipt_digests.add(str(digest))
            if record["held_out"] is not True:
                errors.append(f"{label}: task is not held out")
            if not isinstance(record["repetition"], int) or _is_bool(record["repetition"]):
                errors.append(f"{label}: repetition must be an integer")
            if not isinstance(record["order_index"], int) or _is_bool(record["order_index"]):
                errors.append(f"{label}: order_index must be an integer")
            if _parse_datetime(record["started_at"]) is None:
                errors.append(f"{label}: started_at must be an ISO-8601 timestamp")
            for field in (
                "task_success", "recommendation_issued", "recommendation_adjudicated",
                "recommendation_actionable", "intervention_accepted",
            ):
                if not _is_bool(record[field]):
                    errors.append(f"{label}: {field} must be boolean")
            usage = record["token_usage"]
            if not isinstance(usage, dict) or set(usage).difference({"provenance", "value"}):
                errors.append(f"{label}: malformed token_usage")
            else:
                provenance = usage.get("provenance")
                value = usage.get("value")
                if provenance not in {"provider_reported", "estimated", "missing"}:
                    errors.append(f"{label}: token provenance is absent or invalid")
                elif provenance in {"provider_reported", "estimated"} and (
                    not isinstance(value, int) or _is_bool(value) or value < 0
                ):
                    errors.append(f"{label}: {provenance} usage needs a non-negative integer")
                elif provenance == "missing" and "value" in usage:
                    errors.append(f"{label}: missing usage cannot carry a value")

            recommendation_id = record["recommendation_id"]
            issuer_id = record["recommendation_issuer_id"]
            adjudicator_id = record["recommendation_adjudicator_id"]
            adjudication_digest = record["adjudication_receipt_sha256"]
            if record["recommendation_issued"]:
                if not isinstance(issuer_id, str) or not issuer_id:
                    errors.append(f"{label}: issued recommendation needs recommendation_issuer_id")
                if not isinstance(recommendation_id, str) or not recommendation_id:
                    errors.append(f"{label}: issued recommendation needs recommendation_id")
                elif recommendation_id in recommendation_ids:
                    errors.append(f"{label}: recommendation_id {recommendation_id!r} is reused")
                else:
                    recommendation_ids.add(recommendation_id)
            else:
                if recommendation_id is not None:
                    errors.append(f"{label}: unissued recommendation must have null recommendation_id")
                if issuer_id is not None:
                    errors.append(f"{label}: unissued recommendation must have null issuer_id")
            if record["recommendation_adjudicated"]:
                if not record["recommendation_issued"]:
                    errors.append(f"{label}: unissued recommendation cannot be adjudicated")
                if not isinstance(adjudicator_id, str) or not adjudicator_id:
                    errors.append(f"{label}: adjudication needs an independent adjudicator_id")
                elif adjudicator_id == issuer_id:
                    errors.append(f"{label}: adjudicator_id must differ from recommendation issuer")
                if not _is_sha256(adjudication_digest):
                    errors.append(f"{label}: adjudication needs a receipt digest")
                elif adjudication_digest in receipt_digests:
                    errors.append(f"{label}: adjudication receipt digest is reused")
                else:
                    receipt_digests.add(str(adjudication_digest))
            else:
                if adjudication_digest is not None:
                    errors.append(f"{label}: unadjudicated recommendation must have null receipt")
                if adjudicator_id is not None:
                    errors.append(f"{label}: unadjudicated recommendation must have null adjudicator_id")
            if record["recommendation_actionable"] and not record["recommendation_adjudicated"]:
                errors.append(f"{label}: actionable label requires independent adjudication")
            if record["intervention_accepted"] and (
                record["condition"] != "verified_optimizer" or not record["task_success"]
            ):
                errors.append(f"{label}: only a successful verified-optimizer run may be accepted")

            task = task_index.get(str(record["task_id"]))
            if task is None:
                errors.append(f"{label}: task_id is absent from external task manifest")
                continue
            if record["task_input_sha256"] != task["content_sha256"]:
                errors.append(f"{label}: task input does not match held-out content digest")
            if record["host_category"] != task["host_category"]:
                errors.append(f"{label}: host differs from task-manifest assignment")
            if record["workload_stratum"] != task["workload_stratum"]:
                errors.append(f"{label}: stratum differs from task-manifest assignment")
            if record["verifier_id"] != task["verifier_id"]:
                errors.append(f"{label}: verifier differs from task-manifest assignment")
            randomization = randomization_index.get((str(record["task_id"]), record["repetition"]))
            if randomization is None:
                errors.append(f"{label}: no preregistered randomization block")
            else:
                if record["randomization_block"] != randomization["block_id"]:
                    errors.append(f"{label}: randomization block differs from task manifest")
                order = randomization["condition_order"]
                if record["condition"] not in order:
                    errors.append(f"{label}: condition is not preregistered")
                elif record["order_index"] != order.index(record["condition"]) + 1:
                    errors.append(f"{label}: order_index differs from preregistered randomized order")
            continue

        fields = probe_fields[str(record_type)]
        # Capture probes may omit the two token fields only for estimated/missing values.
        if record_type == "capture_probe":
            required = fields - {"provider_total_tokens", "tracerazor_total_tokens"}
            missing = sorted(required.difference(record))
            extra = sorted(set(record).difference(fields))
            if missing or extra:
                errors.append(f"{label}: malformed capture probe fields")
                continue
        elif not _required(record, fields, label, errors):
            continue
        id_field = "attempt_id" if record_type == "install_probe" else "probe_id"
        identifier = record[id_field]
        if not isinstance(identifier, str) or not identifier:
            errors.append(f"{label}: {id_field} must be non-empty")
        elif identifier in probe_ids[str(record_type)]:
            errors.append(f"{label}: duplicate {record_type} id {identifier!r}")
        else:
            probe_ids[str(record_type)].add(identifier)

        if record_type == "install_probe":
            if not isinstance(record["matrix_cell"], str) or not _is_bool(record["succeeded"]):
                errors.append(f"{label}: malformed install probe")
            if not _is_number(record["time_to_first_audit_seconds"]) or record[
                "time_to_first_audit_seconds"
            ] < 0:
                errors.append(f"{label}: invalid time_to_first_audit_seconds")
        elif record_type == "activation_probe":
            if not _is_bool(record["expected_activation"]) or not _is_bool(record["activated"]):
                errors.append(f"{label}: activation labels must be boolean")
        elif record_type == "capture_probe":
            provenance = record["token_provenance"]
            if provenance not in {"provider_reported", "estimated", "missing"}:
                errors.append(f"{label}: invalid token_provenance")
            if provenance == "provider_reported" and any(
                not isinstance(record.get(field), int) or _is_bool(record.get(field)) or record[field] < 0
                for field in ("provider_total_tokens", "tracerazor_total_tokens")
            ):
                errors.append(f"{label}: provider capture requires both exact token counts")
            if not _is_bool(record["parent_child_expected"]) or not _is_bool(
                record["parent_child_linked"]
            ):
                errors.append(f"{label}: linkage labels must be boolean")
            if record["parent_child_linked"] and not record["parent_child_expected"]:
                errors.append(f"{label}: unexpected child cannot count as linked")
        elif record_type == "overhead_probe":
            if not _is_number(record["baseline_wall_ms"]) or record["baseline_wall_ms"] <= 0:
                errors.append(f"{label}: baseline wall time must be positive")
            if not _is_number(record["instrumented_wall_ms"]) or record["instrumented_wall_ms"] < 0:
                errors.append(f"{label}: instrumented wall time must be non-negative")
            latencies = record["event_latencies_ms"]
            if not isinstance(latencies, list) or not latencies or any(
                not _is_number(value) or value < 0 for value in latencies
            ):
                errors.append(f"{label}: event latencies must be non-negative numbers")
        elif record_type == "safety_probe":
            if not _is_bool(record["sandbox_escape"]) or not _is_bool(record["redacted_secret_leak"]):
                errors.append(f"{label}: safety labels must be boolean")

    if len(grouped["study_manifest"]) != 1:
        errors.append("exactly one study_manifest record is required")
    return errors, grouped


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise EvaluationError("cannot calculate quantile without values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _cluster_bootstrap_ci(
    values_by_task: Mapping[str, Sequence[float]],
    *,
    samples: int,
    seed: int,
    confidence: float,
    statistic: Callable[[Sequence[float]], float],
) -> tuple[float, float]:
    task_ids = sorted(values_by_task)
    if not task_ids:
        raise EvaluationError("cannot bootstrap without task clusters")
    rng = random.Random(seed)
    results: list[float] = []
    for _ in range(samples):
        sample: list[float] = []
        for _task in task_ids:
            chosen = task_ids[rng.randrange(len(task_ids))]
            sample.extend(values_by_task[chosen])
        results.append(float(statistic(sample)))
    alpha = (1.0 - confidence) / 2.0
    return _quantile(results, alpha), _quantile(results, 1.0 - alpha)


def _nearest_rank_p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def _gate(
    gate_id: str, observed: Any, threshold: Any, passed: bool | None, detail: str
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "status": "incomplete" if passed is None else ("pass" if passed else "fail"),
        "observed": observed,
        "threshold": threshold,
        "detail": detail,
    }


def _expected_receipts(records: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = {"run": set(), "verifier": set(), "adjudication": set()}
    for record in records:
        if record.get("record_type") != "run":
            continue
        if _is_sha256(record.get("run_receipt_sha256")):
            expected["run"].add(str(record["run_receipt_sha256"]))
        if _is_sha256(record.get("verifier_receipt_sha256")):
            expected["verifier"].add(str(record["verifier_receipt_sha256"]))
        if record.get("recommendation_adjudicated") and _is_sha256(
            record.get("adjudication_receipt_sha256")
        ):
            expected["adjudication"].add(str(record["adjudication_receipt_sha256"]))
    return expected


EVALUATION_BINDING_FIELDS = (
    "study_id",
    "run_id",
    "task_id",
    "task_input_sha256",
    "initial_state_sha256",
    "execution_environment_sha256",
    "disposable_workspace_id",
    "host_category",
    "host_version",
    "model_id",
    "model_version",
    "agent_config_sha256",
    "condition",
    "repetition",
    "randomization_block",
    "verifier_id",
    "recommendation_id",
    "recommendation_issuer_id",
)


def _expected_run_bindings(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map each signed run-report digest to its trace-bound study identity."""

    bindings: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("record_type") != "run" or not _is_sha256(
            record.get("run_receipt_sha256")
        ):
            continue
        bindings[str(record["run_receipt_sha256"])] = {
            field: record.get(field) for field in EVALUATION_BINDING_FIELDS
        }
    return bindings


def verify_evidence_index(
    index_path: str | Path,
    expected_receipts: Mapping[str, set[str]],
    protocol: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    *,
    verifier_binary: str = "tracerazor",
    expected_run_bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Hash and cryptographically verify every indexed receipt.

    Each receipt report is hashed against its digest in the results, then passed
    to ``tracerazor verify``. Only Ed25519-authenticated verdicts with matching
    trace hashes count as verified.
    """

    source = Path(index_path).resolve()
    index = _load_json(source, "evidence index")
    errors: list[str] = []
    if set(index) != {
        "schema_version",
        "study_id",
        "protocol_sha256",
        "task_manifest_sha256",
        "receipts",
    }:
        errors.append("evidence index fields differ from the locked schema")
    if index.get("schema_version") != EVIDENCE_INDEX_SCHEMA:
        errors.append("unsupported evidence index schema")
    if index.get("study_id") != protocol["study_id"]:
        errors.append("evidence index study_id mismatch")
    if index.get("protocol_sha256") != protocol_sha256(protocol):
        errors.append("evidence index protocol digest mismatch")
    if index.get("task_manifest_sha256") != task_manifest_sha256(task_manifest):
        errors.append("evidence index task-manifest digest mismatch")
    receipts = index.get("receipts")
    if not isinstance(receipts, list):
        errors.append("evidence index receipts must be an array")
        receipts = []
    binary = shutil.which(verifier_binary)
    if binary is None:
        errors.append(f"native verifier not found: {verifier_binary}")

    indexed: dict[str, set[str]] = {"run": set(), "verifier": set(), "adjudication": set()}
    receipt_rows: list[tuple[str, str, Path, Path]] = []
    base = source.parent
    for number, receipt in enumerate(receipts, 1):
        label = f"evidence receipt {number}"
        if not isinstance(receipt, dict) or set(receipt) != {
            "kind",
            "sha256",
            "report_path",
            "trace_path",
        }:
            errors.append(f"{label}: malformed fields")
            continue
        kind = receipt.get("kind")
        digest = receipt.get("sha256")
        if kind not in indexed or not _is_sha256(digest):
            errors.append(f"{label}: invalid kind or digest")
            continue
        if digest in indexed[str(kind)]:
            errors.append(f"{label}: duplicate indexed receipt")
            continue
        indexed[str(kind)].add(str(digest))
        paths: list[Path] = []
        for field in ("report_path", "trace_path"):
            raw_path = receipt[field]
            if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
                errors.append(f"{label}: {field} must be a relative path")
                paths = []
                break
            unresolved = base / raw_path
            if unresolved.is_symlink():
                errors.append(f"{label}: symlinked receipt paths are forbidden")
                paths = []
                break
            resolved = unresolved.resolve()
            try:
                resolved.relative_to(base)
            except ValueError:
                errors.append(f"{label}: {field} escapes the evidence directory")
                paths = []
                break
            if not resolved.is_file():
                errors.append(f"{label}: {field} does not exist")
                paths = []
                break
            if resolved.stat().st_size > 50 * 1024 * 1024:
                errors.append(f"{label}: {field} exceeds the 50 MiB verifier limit")
                paths = []
                break
            paths.append(resolved)
        if len(paths) == 2:
            if _sha256_bytes(paths[0].read_bytes()) != digest:
                errors.append(f"{label}: report hash does not match result receipt digest")
            else:
                if kind == "run" and expected_run_bindings is not None:
                    expected_binding = expected_run_bindings.get(str(digest))
                    try:
                        trace_value = json.loads(paths[1].read_text(encoding="utf-8"))
                    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                        errors.append(f"{label}: cannot read trace evaluation binding: {exc}")
                        continue
                    actual_binding = (
                        trace_value.get("metadata", {}).get("evaluation_binding")
                        if isinstance(trace_value, dict)
                        and isinstance(trace_value.get("metadata"), dict)
                        else None
                    )
                    if expected_binding is None or actual_binding != expected_binding:
                        errors.append(
                            f"{label}: signed trace evaluation binding does not match run record"
                        )
                        continue
                receipt_rows.append((str(kind), str(digest), paths[0], paths[1]))

    for kind in sorted(indexed):
        missing = expected_receipts[kind] - indexed[kind]
        extra = indexed[kind] - expected_receipts[kind]
        if missing:
            errors.append(f"evidence index is missing {len(missing)} {kind} receipts")
        if extra:
            errors.append(f"evidence index has {len(extra)} unexpected {kind} receipts")

    verified_counts = {"run": 0, "verifier": 0, "adjudication": 0}
    if not errors and binary is not None:
        for kind, digest, report_path, trace_path in receipt_rows:
            try:
                completed = subprocess.run(
                    [binary, "verify", str(report_path), str(trace_path), "--format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"{kind} receipt {digest}: verifier failed to run: {exc}")
                continue
            try:
                verdict = json.loads(completed.stdout)
            except json.JSONDecodeError:
                verdict = {}
            if completed.returncode != 0 or not isinstance(verdict, dict):
                errors.append(f"{kind} receipt {digest}: native verifier rejected receipt")
                continue
            if not (
                verdict.get("status") == "verified"
                and verdict.get("signature") == "ok"
                and verdict.get("trace_hash") == "ok"
            ):
                errors.append(f"{kind} receipt {digest}: receipt is not Ed25519 authenticated")
                continue
            verified_counts[kind] += 1

    verified = not errors and all(
        verified_counts[kind] == len(expected_receipts[kind]) for kind in verified_counts
    )
    return {
        "status": "verified" if verified else "failed",
        "expected": {kind: len(values) for kind, values in expected_receipts.items()},
        "indexed": {kind: len(values) for kind, values in indexed.items()},
        "verified": verified_counts,
        "index_sha256": _sha256_bytes(source.read_bytes()),
        "errors": errors,
    }


def _incomplete_report(
    protocol: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    errors: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        "study_id": protocol.get("study_id"),
        "protocol_sha256": protocol_sha256(protocol),
        "task_manifest_sha256": task_manifest_sha256(task_manifest),
        "synthetic": bool(task_manifest.get("synthetic", True)),
        "statistical_status": "incomplete",
        "release_status": "incomplete",
        "status": "incomplete",
        "record_count": len(records),
        "design": {},
        "efficacy": {},
        "operations": {},
        "gates": [],
        "evidence_authentication": {"status": "not_evaluated", "errors": []},
        "errors": list(errors),
        "warnings": ["No efficacy conclusion is permitted from an incomplete study."],
    }


def _evaluate_records(
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    *,
    evidence_authentication: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate, aggregate, and gate one locked study."""

    _validate_protocol(protocol)
    manifest_errors, task_index, randomization_index = _validate_task_manifest(
        task_manifest, protocol
    )
    if manifest_errors:
        return _incomplete_report(protocol, task_manifest, records, manifest_errors)
    validation_errors, grouped = _validate_records(
        records, protocol, task_manifest, task_index, randomization_index
    )
    if validation_errors:
        return _incomplete_report(protocol, task_manifest, records, validation_errors)

    runs = grouped["run"]
    run_index: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    design_errors: list[str] = []
    provenance_counts: Counter[str] = Counter()
    repetitions = range(1, protocol["design"]["repetitions_per_condition"] + 1)
    for run in runs:
        key = (str(run["task_id"]), str(run["condition"]), int(run["repetition"]))
        if run["condition"] not in CONDITIONS:
            design_errors.append(f"run {run['run_id']!r} has an unregistered condition")
        elif int(run["repetition"]) not in repetitions:
            design_errors.append(f"run {run['run_id']!r} has a repetition outside 1..3")
        elif key in run_index:
            design_errors.append(f"duplicate task/condition/repetition cell {key!r}")
        else:
            run_index[key] = run
        provenance_counts[str(run["token_usage"]["provenance"])] += 1

    task_ids = sorted(task_index)
    expected_cells = {
        (task_id, condition, repetition)
        for task_id in task_ids
        for condition in CONDITIONS
        for repetition in repetitions
    }
    missing = expected_cells - set(run_index)
    extra = set(run_index) - expected_cells
    if missing:
        design_errors.append(f"{len(missing)} task/condition/repetition cells are missing")
    if extra:
        design_errors.append(f"{len(extra)} runs are not in the external task manifest")

    pair_invariants = protocol["design"]["pair_invariants"]
    if not missing and not extra:
        for task_id in task_ids:
            accepted_flags: list[bool] = []
            for repetition in repetitions:
                block_runs = [run_index[(task_id, condition, repetition)] for condition in CONDITIONS]
                baseline = block_runs[0]
                for run in block_runs[1:]:
                    mismatched = [field for field in pair_invariants if run[field] != baseline[field]]
                    if mismatched:
                        design_errors.append(
                            f"pair {task_id!r}/r{repetition} differs on invariants {mismatched!r}"
                        )
                ordered = sorted(block_runs, key=lambda run: _parse_datetime(run["started_at"]))
                if [run["order_index"] for run in ordered] != [1, 2, 3]:
                    design_errors.append(
                        f"pair {task_id!r}/r{repetition} execution timestamps do not match randomized order"
                    )
                accepted_flags.append(bool(run_index[(task_id, "verified_optimizer", repetition)]["intervention_accepted"]))
            if any(accepted_flags) and not all(accepted_flags):
                design_errors.append(
                    f"optimizer acceptance for task {task_id!r} is run-level; all three repetitions must agree"
                )

    if design_errors:
        report = _incomplete_report(protocol, task_manifest, records, design_errors)
        report["design"] = {
            "held_out_tasks": len(task_ids),
            "runs": len(runs),
            "token_provenance_counts": dict(sorted(provenance_counts.items())),
        }
        return report

    bootstrap = protocol["bootstrap"]
    baseline_runs = [run for run in runs if run["condition"] == "no_tracerazor"]
    baseline_solvability = sum(bool(run["task_success"]) for run in baseline_runs) / len(baseline_runs)
    reductions: dict[str, list[float]] = {condition: [] for condition in INTERVENTIONS}
    reductions_by_task: dict[str, dict[str, list[float]]] = {
        condition: defaultdict(list) for condition in INTERVENTIONS
    }
    quality_differences: dict[str, dict[str, list[float]]] = {
        condition: defaultdict(list) for condition in INTERVENTIONS
    }
    successful_pairs: Counter[str] = Counter()
    measured_pairs: Counter[str] = Counter()
    missing_measured_pairs: Counter[str] = Counter()

    for task_id in task_ids:
        for repetition in repetitions:
            baseline = run_index[(task_id, "no_tracerazor", repetition)]
            for condition in INTERVENTIONS:
                intervention = run_index[(task_id, condition, repetition)]
                quality_differences[condition][task_id].append(
                    float(int(intervention["task_success"]) - int(baseline["task_success"]))
                )
                if not baseline["task_success"] or not intervention["task_success"]:
                    continue
                successful_pairs[condition] += 1
                baseline_usage = baseline["token_usage"]
                intervention_usage = intervention["token_usage"]
                if (
                    baseline_usage["provenance"] != "provider_reported"
                    or intervention_usage["provenance"] != "provider_reported"
                    or baseline_usage.get("value", 0) <= 0
                ):
                    missing_measured_pairs[condition] += 1
                    continue
                reduction = (baseline_usage["value"] - intervention_usage["value"]) / baseline_usage[
                    "value"
                ]
                measured_pairs[condition] += 1
                reductions[condition].append(float(reduction))
                reductions_by_task[condition][task_id].append(float(reduction))

    medians: dict[str, float | None] = {}
    reduction_cis: dict[str, list[float | None]] = {}
    quality_cis: dict[str, list[float]] = {}
    quality_deltas: dict[str, float] = {}
    incomplete_reasons: list[str] = []
    for condition in INTERVENTIONS:
        medians[condition] = (
            float(statistics.median(reductions[condition])) if reductions[condition] else None
        )
        if missing_measured_pairs[condition]:
            incomplete_reasons.append(
                f"{missing_measured_pairs[condition]} successful {condition} pairs lack provider-reported usage"
            )
        if reductions_by_task[condition]:
            lower, upper = _cluster_bootstrap_ci(
                reductions_by_task[condition],
                samples=int(bootstrap["samples"]),
                seed=int(bootstrap["seed"]),
                confidence=float(bootstrap["confidence"]),
                statistic=lambda values: float(statistics.median(values)),
            )
            reduction_cis[condition] = [lower, upper]
        else:
            reduction_cis[condition] = [None, None]
        quality_deltas[condition] = float(
            statistics.mean(
                value for task_values in quality_differences[condition].values() for value in task_values
            )
        )
        quality_lower, quality_upper = _cluster_bootstrap_ci(
            quality_differences[condition],
            samples=int(bootstrap["samples"]),
            seed=int(bootstrap["seed"]),
            confidence=float(bootstrap["confidence"]),
            statistic=lambda values: float(statistics.mean(values)),
        )
        quality_cis[condition] = [quality_lower, quality_upper]

    accepted_tasks: list[str] = []
    accepted_reductions: list[float] = []
    for task_id in task_ids:
        optimizer_runs = [
            run_index[(task_id, "verified_optimizer", repetition)] for repetition in repetitions
        ]
        if all(run["intervention_accepted"] for run in optimizer_runs):
            if not all(run["task_success"] and _is_sha256(run["verifier_receipt_sha256"]) for run in optimizer_runs):
                incomplete_reasons.append(
                    f"accepted task {task_id!r} lacks three successful verifier receipts"
                )
                continue
            task_reductions = reductions_by_task["verified_optimizer"].get(task_id, [])
            if len(task_reductions) != 3:
                incomplete_reasons.append(
                    f"accepted task {task_id!r} lacks three matched provider-reported baseline comparisons"
                )
                continue
            accepted_tasks.append(task_id)
            accepted_reductions.extend(task_reductions)
    accepted_rate = len(accepted_tasks) / len(task_ids)
    accepted_median = (
        float(statistics.median(accepted_reductions)) if accepted_reductions else None
    )

    adjudicated = [
        run for run in runs if run["recommendation_issued"] and run["recommendation_adjudicated"]
    ]
    if len(adjudicated) < protocol["minimum_probe_counts"]["recommendations_adjudicated"]:
        incomplete_reasons.append(
            f"only {len(adjudicated)} independently adjudicated recommendations; "
            f"{protocol['minimum_probe_counts']['recommendations_adjudicated']} required"
        )
    recommendation_precision = (
        sum(bool(run["recommendation_actionable"]) for run in adjudicated) / len(adjudicated)
        if adjudicated
        else None
    )

    install = grouped["install_probe"]
    expected_matrix = set(protocol["release_matrix"])
    observed_matrix = {str(probe["matrix_cell"]) for probe in install}
    if expected_matrix != observed_matrix:
        incomplete_reasons.append(
            f"install matrix coverage differs: missing={sorted(expected_matrix - observed_matrix)!r}, "
            f"unknown={sorted(observed_matrix - expected_matrix)!r}"
        )
    install_success = sum(bool(probe["succeeded"]) for probe in install) / len(install) if install else None
    first_audit_median = (
        float(statistics.median(probe["time_to_first_audit_seconds"] for probe in install))
        if install
        else None
    )

    minimums = protocol["minimum_probe_counts"]
    activation = grouped["activation_probe"]
    activation_positive = [probe for probe in activation if probe["expected_activation"]]
    activation_negative = [probe for probe in activation if not probe["expected_activation"]]
    if len(activation_positive) < minimums["activation_expected"]:
        incomplete_reasons.append("insufficient expected-activation probes")
    if len(activation_negative) < minimums["activation_unrelated"]:
        incomplete_reasons.append("insufficient unrelated activation probes")
    true_positive = sum(bool(probe["activated"]) for probe in activation_positive)
    false_positive = sum(bool(probe["activated"]) for probe in activation_negative)
    activation_precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    )
    activation_recall = true_positive / len(activation_positive) if activation_positive else None
    unrelated_rate = false_positive / len(activation_negative) if activation_negative else None

    capture = grouped["capture_probe"]
    provider_capture = [probe for probe in capture if probe["token_provenance"] == "provider_reported"]
    if len(provider_capture) < minimums["provider_usage"]:
        incomplete_reasons.append("insufficient provider-usage probes")
    provider_agreement = (
        sum(
            probe["provider_total_tokens"] == probe["tracerazor_total_tokens"]
            for probe in provider_capture
        )
        / len(provider_capture)
        if provider_capture
        else None
    )
    linkage = [probe for probe in capture if probe["parent_child_expected"]]
    if len(linkage) < minimums["parent_child_linkage"]:
        incomplete_reasons.append("insufficient parent/child linkage probes")
    linkage_rate = (
        sum(bool(probe["parent_child_linked"]) for probe in linkage) / len(linkage)
        if linkage
        else None
    )

    overhead = grouped["overhead_probe"]
    if len(overhead) < minimums["wall_clock_overhead"]:
        incomplete_reasons.append("insufficient wall-clock overhead probes")
    overhead_ratios = [
        (probe["instrumented_wall_ms"] - probe["baseline_wall_ms"]) / probe["baseline_wall_ms"]
        for probe in overhead
    ]
    wall_median = float(statistics.median(overhead_ratios)) if overhead_ratios else None
    event_latencies = [float(value) for probe in overhead for value in probe["event_latencies_ms"]]
    if len(event_latencies) < minimums["event_latency_samples"]:
        incomplete_reasons.append("insufficient event-latency samples")
    event_p95 = _nearest_rank_p95(event_latencies) if event_latencies else None

    safety = grouped["safety_probe"]
    if len(safety) < minimums["safety"]:
        incomplete_reasons.append("insufficient safety probes")
    sandbox_escapes = sum(bool(probe["sandbox_escape"]) for probe in safety)
    secret_leaks = sum(bool(probe["redacted_secret_leak"]) for probe in safety)

    gates_config = protocol["gates"]
    gates: list[dict[str, Any]] = []

    def minimum(gate_id: str, observed: float | int | None, threshold: float | int, detail: str) -> None:
        gates.append(_gate(gate_id, observed, {"min_inclusive": threshold}, None if observed is None else observed >= threshold, detail))

    def maximum(gate_id: str, observed: float | int | None, threshold: float | int, detail: str) -> None:
        gates.append(_gate(gate_id, observed, {"max_inclusive": threshold}, None if observed is None else observed <= threshold, detail))

    def maximum_exclusive(gate_id: str, observed: float | None, threshold: float, detail: str) -> None:
        gates.append(_gate(gate_id, observed, {"max_exclusive": threshold}, None if observed is None else observed < threshold, detail))

    minimum("baseline_solvability", baseline_solvability, protocol["design"]["minimum_baseline_success_rate"], "Provider-independent baseline task success rate.")
    for condition in INTERVENTIONS:
        minimum(
            f"{condition}_matched_success_pairs",
            measured_pairs[condition],
            protocol["design"]["minimum_matched_success_pairs_per_condition"],
            "Successful matched pairs with provider-reported usage.",
        )
        median_gate = f"{condition.replace('verified_', '')}_median_token_reduction_min"
        ci_gate = f"{condition.replace('verified_', '')}_token_reduction_ci_lower_min_exclusive"
        minimum(
            f"{condition}_median_token_reduction",
            medians[condition],
            gates_config[median_gate],
            "Median reduction over matched successful provider-reported pairs only.",
        )
        ci_lower = reduction_cis[condition][0]
        threshold = gates_config[ci_gate]
        gates.append(
            _gate(
                f"{condition}_token_reduction_ci_lower",
                ci_lower,
                {"min_exclusive": threshold},
                None if ci_lower is None else ci_lower > threshold,
                "Task-cluster bootstrap confidence-interval lower bound.",
            )
        )
        quality_lower = quality_cis[condition][0]
        margin = gates_config["task_success_noninferiority_margin"]
        gates.append(
            _gate(
                f"{condition}_task_success_noninferiority",
                {"point_delta": quality_deltas[condition], "ci_95": quality_cis[condition]},
                {"ci_lower_min_inclusive": -margin},
                quality_lower >= -margin,
                "Paired task-cluster bootstrap CI for intervention minus baseline success.",
            )
        )

    minimum(
        "accepted_optimizer_median_token_reduction",
        accepted_median,
        gates_config["accepted_optimizer_median_token_reduction_min"],
        "Median matched reduction for task-level accepted optimizer interventions.",
    )
    minimum("accepted_optimizer_tasks", len(accepted_tasks), gates_config["accepted_optimizer_tasks_min"], "Tasks with all three optimizer repetitions accepted and successful.")
    minimum("accepted_optimizer_task_rate", accepted_rate, gates_config["accepted_optimizer_task_rate_min"], "Accepted optimizer tasks divided by all held-out tasks.")
    minimum("install_success", install_success, gates_config["install_success_rate_min"], "Clean-machine success rate.")
    maximum_exclusive("time_to_first_audit", first_audit_median, gates_config["median_time_to_first_audit_seconds_max_exclusive"], "Median seconds to first automatic audit.")
    minimum("activation_precision", activation_precision, gates_config["activation_precision_min"], "Skill activation precision.")
    minimum("activation_recall", activation_recall, gates_config["activation_recall_min"], "Skill activation recall.")
    maximum("unrelated_activation", unrelated_rate, gates_config["unrelated_activation_rate_max"], "Unrelated-task activation rate.")
    minimum("provider_usage_agreement", provider_agreement, gates_config["provider_usage_exact_agreement_min"], "Exact provider token agreement rate.")
    minimum("parent_child_linkage", linkage_rate, gates_config["parent_child_linkage_min"], "Expected parent/child linkage rate.")
    maximum_exclusive("wall_clock_overhead", wall_median, gates_config["wall_clock_overhead_median_max_exclusive"], "Median wall-clock overhead ratio.")
    maximum_exclusive("event_latency", event_p95, gates_config["event_latency_p95_ms_max_exclusive"], "Nearest-rank event latency p95.")
    minimum("recommendation_precision", recommendation_precision, gates_config["recommendation_precision_min"], "Actionable independently adjudicated recommendations divided by adjudicated recommendations.")
    maximum("sandbox_escape", sandbox_escapes, gates_config["sandbox_escape_max"], "Automatic edit sandbox escapes.")
    maximum("redacted_secret_leak", secret_leaks, gates_config["redacted_secret_leak_max"], "Secrets in redacted exports.")

    if incomplete_reasons or any(gate["status"] == "incomplete" for gate in gates):
        statistical_status = "incomplete"
    elif any(gate["status"] == "fail" for gate in gates):
        statistical_status = "fail"
    else:
        statistical_status = "pass"

    expected_receipts = _expected_receipts(records)
    if evidence_authentication is None:
        evidence = {
            "status": "missing",
            "expected": {kind: len(values) for kind, values in expected_receipts.items()},
            "verified": {kind: 0 for kind in expected_receipts},
            "errors": ["no signed receipt evidence index was verified"],
        }
    else:
        evidence = dict(evidence_authentication)

    synthetic = bool(task_manifest["synthetic"])
    if statistical_status == "fail":
        release_status = "fail"
        status = "fail"
    elif statistical_status == "incomplete":
        release_status = "incomplete"
        status = "incomplete"
    elif synthetic:
        release_status = "incomplete"
        status = "release_incomplete"
    elif evidence.get("status") != "verified":
        release_status = "incomplete"
        status = "release_incomplete"
    else:
        release_status = "pass"
        status = "pass"

    warnings = [
        f"Excluded {provenance_counts.get('estimated', 0)} estimated and "
        f"{provenance_counts.get('missing', 0)} missing-token runs from efficacy."
    ]
    if synthetic:
        warnings.append("Synthetic studies are evaluator tests, never product or release evidence.")
    if statistical_status == "pass" and release_status != "pass":
        warnings.append(
            "Statistical gates passed, but release evidence is incomplete until every signed receipt is authenticated."
        )

    return {
        "schema_version": REPORT_SCHEMA,
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_sha256(protocol),
        "task_manifest_sha256": task_manifest_sha256(task_manifest),
        "synthetic": synthetic,
        "statistical_status": statistical_status,
        "release_status": release_status,
        "status": status,
        "record_count": len(records),
        "design": {
            "held_out_tasks": len(task_ids),
            "runs": len(runs),
            "host_stratum_cells": dict(
                sorted(
                    (
                        f"{task['host_category']}/{task['workload_stratum']}",
                        sum(
                            1
                            for candidate in task_index.values()
                            if candidate["host_category"] == task["host_category"]
                            and candidate["workload_stratum"] == task["workload_stratum"]
                        ),
                    )
                    for task in task_index.values()
                )
            ),
            "token_provenance_counts": dict(sorted(provenance_counts.items())),
            "baseline_solvability": baseline_solvability,
            "successful_pairs": dict(successful_pairs),
            "measured_successful_pairs": dict(measured_pairs),
        },
        "efficacy": {
            "per_condition_median_paired_token_reduction": medians,
            "per_condition_token_reduction_ci_95": reduction_cis,
            "per_condition_task_success_delta": quality_deltas,
            "per_condition_task_success_delta_ci_95": quality_cis,
            "accepted_optimizer_tasks": accepted_tasks,
            "accepted_optimizer_task_rate": accepted_rate,
            "accepted_optimizer_median_paired_token_reduction": accepted_median,
            "recommendations_adjudicated": len(adjudicated),
            "recommendation_precision": recommendation_precision,
            "bootstrap_seed": bootstrap["seed"],
            "bootstrap_samples": bootstrap["samples"],
        },
        "operations": {
            "install_probe_count": len(install),
            "install_success_rate": install_success,
            "median_time_to_first_audit_seconds": first_audit_median,
            "activation_precision": activation_precision,
            "activation_recall": activation_recall,
            "unrelated_activation_rate": unrelated_rate,
            "provider_usage_probe_count": len(provider_capture),
            "provider_usage_exact_agreement": provider_agreement,
            "parent_child_probe_count": len(linkage),
            "parent_child_linkage": linkage_rate,
            "wall_clock_probe_count": len(overhead),
            "median_wall_clock_overhead": wall_median,
            "event_latency_sample_count": len(event_latencies),
            "event_latency_p95_ms": event_p95,
            "safety_probe_count": len(safety),
            "sandbox_escapes": sandbox_escapes,
            "redacted_secret_leaks": secret_leaks,
        },
        "gates": gates,
        "evidence_authentication": evidence,
        "errors": incomplete_reasons,
        "warnings": warnings,
    }


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate records without granting a release-authentication shortcut.

    Signed evidence can affect release status only through the CLI path, which
    calls :func:`verify_evidence_index` itself. A caller-provided mapping is not
    accepted as proof of cryptographic verification.
    """

    return _evaluate_records(records, protocol, task_manifest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the locked TraceRazor agent-native efficacy study."
    )
    parser.add_argument("--input", type=Path, help="JSONL study results")
    parser.add_argument("--task-manifest", type=Path, help="locked external held-out task manifest")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("protocol.json"),
        help="locked protocol JSON",
    )
    parser.add_argument("--evidence-index", type=Path, help="index of signed receipt report/trace pairs")
    parser.add_argument("--output", type=Path, help="write evaluation report here")
    parser.add_argument("--print-protocol-sha256", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        if args.print_protocol_sha256:
            print(protocol_sha256(protocol))
            return 0
        if args.input is None or args.task_manifest is None:
            parser.error("--input and --task-manifest are required")
        task_manifest = load_task_manifest(args.task_manifest, protocol)
        records = load_jsonl(args.input)
        evidence = None
        if args.evidence_index is not None:
            evidence = verify_evidence_index(
                args.evidence_index,
                _expected_receipts(records),
                protocol,
                task_manifest,
                expected_run_bindings=_expected_run_bindings(records),
            )
        report = _evaluate_records(
            records,
            protocol,
            task_manifest,
            evidence_authentication=evidence,
        )
        report["input_sha256"] = _sha256_bytes(args.input.read_bytes())
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except EvaluationError as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    if report["status"] == "pass":
        return 0
    if report["status"] == "fail":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
