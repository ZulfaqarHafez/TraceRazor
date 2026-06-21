"""TRICE run receipt validation.

Run receipts are the smallest auditable unit in a live rollout. They describe
the adapter envelope and changed workspace, while the evidence manifest provides
tamper detection around the receipt file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA_VERSION = "trice-run-receipt/v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_KNOWN_ADAPTER_TYPES = {"json_patch", "command", "command_profile", "managed_python"}


def validate_run_receipt(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError(f"run receipt schema_version must be {RECEIPT_SCHEMA_VERSION!r}")
    adapter_type = _required_str(data, "adapter_type")
    if adapter_type not in _KNOWN_ADAPTER_TYPES:
        raise ValueError(f"unsupported receipt adapter_type: {adapter_type}")
    _required_str(data, "adapter_name")
    _required_str(data, "task_id")
    _require_hex64(data, "prompt_sha256")
    changed_files = data.get("changed_files")
    if not isinstance(changed_files, list) or not all(isinstance(item, str) and item for item in changed_files):
        raise ValueError("run receipt changed_files must be a list of strings")
    changed_file_count = data.get("changed_file_count")
    if changed_file_count != len(changed_files):
        raise ValueError("run receipt changed_file_count must match changed_files")

    if adapter_type in {"command", "command_profile"}:
        command = data.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            raise ValueError("command run receipt requires command argv")
        _require_hex64(data, "command_sha256")
        _require_hex64(data, "workspace_before_sha256")
        _require_hex64(data, "workspace_after_sha256")
        if not isinstance(data.get("exit_code"), int):
            raise ValueError("command run receipt requires integer exit_code")
        if not isinstance(data.get("timeout_s"), int) or data["timeout_s"] < 1:
            raise ValueError("command run receipt requires positive timeout_s")
        if not isinstance(data.get("allow_test_edits"), bool):
            raise ValueError("command run receipt requires boolean allow_test_edits")
        forbidden = data.get("forbidden_changed_files", [])
        if not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden):
            raise ValueError("command run receipt forbidden_changed_files must be a list")

    for field in ("stdout_sha256", "stderr_sha256"):
        if data.get(field) is not None:
            _require_hex64(data, field)

    input_tokens = _agent_reported_input_tokens(data.get("agent_reported"))
    trice_context = _validate_trice_context(data.get("trice_context"))
    return {
        "ok": True,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "adapter_type": adapter_type,
        "adapter_name": data["adapter_name"],
        "changed_file_count": len(changed_files),
        "agent_reported_input_tokens": input_tokens,
        "trice_context_mode": trice_context.get("context_mode"),
        "trice_input_tokens": trice_context.get("input_tokens"),
        "trice_baseline_input_tokens": trice_context.get("baseline_input_tokens"),
    }


def validate_run_receipt_file(path: str | Path) -> dict[str, Any]:
    return validate_run_receipt(json.loads(Path(path).read_text(encoding="utf-8")))


def _required_str(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"run receipt requires non-empty {field}")
    return value


def _require_hex64(data: dict[str, Any], field: str) -> None:
    value = data.get(field)
    if not isinstance(value, str) or not _HEX64.match(value):
        raise ValueError(f"run receipt {field} must be a 64-character lowercase hex SHA-256")


def _agent_reported_input_tokens(agent_reported: Any) -> int | None:
    if agent_reported is None:
        return None
    if not isinstance(agent_reported, dict):
        raise ValueError("run receipt agent_reported must be an object or null")
    token_accounting = agent_reported.get("token_accounting")
    if token_accounting is not None:
        if not isinstance(token_accounting, dict):
            raise ValueError("run receipt agent_reported.token_accounting must be an object")
        input_tokens = token_accounting.get("input_tokens")
        if input_tokens is not None:
            if not isinstance(input_tokens, int) or input_tokens < 0:
                raise ValueError("run receipt input_tokens must be a non-negative integer")
            return input_tokens
    input_tokens = agent_reported.get("input_tokens")
    if input_tokens is not None:
        if not isinstance(input_tokens, int) or input_tokens < 0:
            raise ValueError("run receipt input_tokens must be a non-negative integer")
        return input_tokens
    return None


def _validate_trice_context(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("run receipt trice_context must be an object when present")
    if value.get("schema_version") not in (None, "trice-context-envelope/v1"):
        raise ValueError("run receipt trice_context has unsupported schema_version")
    mode = value.get("context_mode")
    if mode is not None and mode not in {"full_context", "trice_policy"}:
        raise ValueError("run receipt trice_context.context_mode is unsupported")
    condition = value.get("condition")
    if condition is not None and (not isinstance(condition, str) or not condition):
        raise ValueError("run receipt trice_context.condition must be a non-empty string")
    for field in (
        "input_tokens",
        "baseline_input_tokens",
        "policy_tokens",
        "budget_tokens",
    ):
        item = value.get(field)
        if item is not None and (not isinstance(item, int) or item < 0):
            raise ValueError(f"run receipt trice_context.{field} must be a non-negative integer")
    for field in ("budget_ratio", "realized_budget_ratio", "projected_input_savings_pct"):
        item = value.get(field)
        if item is not None and not isinstance(item, (int, float)):
            raise ValueError(f"run receipt trice_context.{field} must be numeric")
    for field in ("policy_sha256", "compressed_context_sha256"):
        item = value.get(field)
        if item is not None and (not isinstance(item, str) or not _HEX64.match(item)):
            raise ValueError(f"run receipt trice_context.{field} must be a SHA-256 hex digest")
    action_counts = value.get("policy_action_counts")
    if action_counts is not None:
        if not isinstance(action_counts, dict) or not all(isinstance(k, str) and isinstance(v, int) and v >= 0 for k, v in action_counts.items()):
            raise ValueError("run receipt trice_context.policy_action_counts must map strings to non-negative integers")
    return value
