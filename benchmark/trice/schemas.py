"""Schema helpers for TRICE public contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapters import CommandRepairAdapter, JsonPatchAdapter
from .receipt import validate_run_receipt, validate_run_receipt_file
from .suite import validate_suite_manifest_file as _validate_suite_manifest_file

REPO = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO / "schemas"
PATCH_SPEC_SCHEMA = "trice_patch_spec.schema.json"
EVIDENCE_MANIFEST_SCHEMA = "trice_evidence_manifest.schema.json"
SUITE_MANIFEST_SCHEMA = "trice_suite_manifest.schema.json"
BUNDLE_MANIFEST_SCHEMA = "trice_bundle_manifest.schema.json"
ADAPTER_PROFILE_SCHEMA = "trice_adapter_profile.schema.json"
RUN_RECEIPT_SCHEMA = "trice_run_receipt.schema.json"


def schema_path(name: str) -> Path:
    """Return the local path to a shipped TRICE schema."""

    aliases = {
        "patch": PATCH_SPEC_SCHEMA,
        "patch_spec": PATCH_SPEC_SCHEMA,
        "trice_patch_spec": PATCH_SPEC_SCHEMA,
        PATCH_SPEC_SCHEMA: PATCH_SPEC_SCHEMA,
        "manifest": EVIDENCE_MANIFEST_SCHEMA,
        "evidence": EVIDENCE_MANIFEST_SCHEMA,
        "evidence_manifest": EVIDENCE_MANIFEST_SCHEMA,
        "trice_evidence_manifest": EVIDENCE_MANIFEST_SCHEMA,
        EVIDENCE_MANIFEST_SCHEMA: EVIDENCE_MANIFEST_SCHEMA,
        "suite": SUITE_MANIFEST_SCHEMA,
        "suite_manifest": SUITE_MANIFEST_SCHEMA,
        "trice_suite_manifest": SUITE_MANIFEST_SCHEMA,
        SUITE_MANIFEST_SCHEMA: SUITE_MANIFEST_SCHEMA,
        "bundle": BUNDLE_MANIFEST_SCHEMA,
        "bundle_manifest": BUNDLE_MANIFEST_SCHEMA,
        "trice_bundle_manifest": BUNDLE_MANIFEST_SCHEMA,
        BUNDLE_MANIFEST_SCHEMA: BUNDLE_MANIFEST_SCHEMA,
        "adapter": ADAPTER_PROFILE_SCHEMA,
        "adapter_profile": ADAPTER_PROFILE_SCHEMA,
        "trice_adapter_profile": ADAPTER_PROFILE_SCHEMA,
        ADAPTER_PROFILE_SCHEMA: ADAPTER_PROFILE_SCHEMA,
        "receipt": RUN_RECEIPT_SCHEMA,
        "run_receipt": RUN_RECEIPT_SCHEMA,
        "trice_run_receipt": RUN_RECEIPT_SCHEMA,
        RUN_RECEIPT_SCHEMA: RUN_RECEIPT_SCHEMA,
    }
    filename = aliases.get(name)
    if not filename:
        raise ValueError(f"unknown TRICE schema: {name}")
    path = SCHEMA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"TRICE schema not found: {path}")
    return path


def load_schema(name: str) -> dict[str, Any]:
    """Load a shipped TRICE schema as a dictionary."""

    return json.loads(schema_path(name).read_text(encoding="utf-8"))


def validate_patch_spec(data: dict[str, Any]) -> dict[str, Any]:
    """Dependency-free validation for the deterministic patch contract.

    This mirrors the runtime constraints of ``JsonPatchAdapter``. Full JSON
    Schema validation can be done with any Draft 2020-12 validator using the
    shipped schema file.
    """

    adapter = JsonPatchAdapter.from_dict(data)
    if not adapter.edits:
        raise ValueError("patch spec requires at least one edit")
    return {
        "ok": True,
        "name": adapter.name,
        "edit_count": len(adapter.edits),
        "allow_test_edits": adapter.allow_test_edits,
    }


def validate_patch_spec_file(path: str | Path) -> dict[str, Any]:
    return validate_patch_spec(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_adapter_profile(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema_version") != "trice-adapter-profile/v1":
        raise ValueError("adapter profile schema_version must be 'trice-adapter-profile/v1'")
    if data.get("type") != "command":
        raise ValueError("only command adapter profiles are supported in v1")
    adapter = CommandRepairAdapter.from_dict(data)
    return {
        "ok": True,
        "name": adapter.name,
        "type": "command",
        "command": list(adapter.command),
        "timeout_s": adapter.timeout_s,
        "allow_test_edits": adapter.allow_test_edits,
        "agent_receipt_path": adapter.agent_receipt_path,
    }


def validate_adapter_profile_file(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    verdict = validate_adapter_profile(data)
    adapter = CommandRepairAdapter.from_file(path)
    verdict["command"] = list(adapter.command)
    verdict["agent_receipt_path"] = adapter.agent_receipt_path
    return verdict


def validate_suite_manifest_file(path: str | Path) -> dict[str, Any]:
    return _validate_suite_manifest_file(path)
