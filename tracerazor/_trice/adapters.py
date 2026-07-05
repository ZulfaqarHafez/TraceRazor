"""Adapter contracts for deterministic TRICE live rollouts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


DEFAULT_FORBIDDEN_PREFIXES = ("tests/", "test/")
SECRET_VALUE_RE = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|passwd|authorization|bearer)\s*[:=]\s*([^\s\"']+)"
)


class RepairAdapter(Protocol):
    """Deterministic edit adapter used by ``run_live_learning_loop``."""

    name: str

    def apply_fix(self, task: Any, workspace: Path) -> list[str]:
        """Apply a deterministic intervention and return modified relative paths."""


@dataclass(frozen=True)
class PatchEdit:
    op: str
    path: str
    old: str | None = None
    new: str | None = None
    content: str | None = None


@dataclass
class JsonPatchAdapter:
    """Apply a declarative JSON patch spec in a fresh workspace.

    Supported edit operations:
    - ``replace``: replace ``old`` with ``new`` in ``path``.
    - ``write``: write ``content`` to ``path``.

    The adapter is intentionally small. It is for deterministic evaluation and
    evidence generation, not a general patch language.
    """

    edits: list[PatchEdit]
    name: str = "json-patch-adapter"
    allow_test_edits: bool = False
    forbidden_prefixes: tuple[str, ...] = DEFAULT_FORBIDDEN_PREFIXES
    applied_empty_ok: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "JsonPatchAdapter":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JsonPatchAdapter":
        edits = [
            PatchEdit(
                op=str(e.get("op") or e.get("type") or ""),
                path=str(e["path"]),
                old=e.get("old"),
                new=e.get("new"),
                content=e.get("content"),
            )
            for e in data.get("edits", [])
        ]
        return cls(
            edits=edits,
            name=str(data.get("name") or "json-patch-adapter"),
            allow_test_edits=bool(data.get("allow_test_edits", False)),
            forbidden_prefixes=_effective_forbidden_prefixes(data.get("forbidden_prefixes")),
            applied_empty_ok=bool(data.get("applied_empty_ok", False)),
            metadata=dict(data.get("metadata") or {}),
        )

    def apply_fix(self, task: Any, workspace: Path) -> list[str]:
        if not self.edits:
            raise ValueError("patch spec has no edits")
        changed: list[str] = []
        for edit in self.edits:
            rel = _clean_rel_path(edit.path)
            if not self.allow_test_edits and _is_forbidden(rel, self.forbidden_prefixes):
                raise ValueError(f"refusing to edit forbidden path: {rel}")
            target = _resolve_in_workspace(workspace, rel)
            if edit.op == "replace":
                if edit.old is None or edit.new is None:
                    raise ValueError(f"replace edit for {rel} needs old and new")
                text = target.read_text(encoding="utf-8")
                if edit.old not in text:
                    if self.applied_empty_ok:
                        continue
                    raise ValueError(f"old text not found in {rel}")
                target.write_text(text.replace(edit.old, edit.new), encoding="utf-8")
                changed.append(rel)
            elif edit.op == "write":
                if edit.content is None:
                    raise ValueError(f"write edit for {rel} needs content")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(edit.content, encoding="utf-8")
                changed.append(rel)
            else:
                raise ValueError(f"unsupported patch op: {edit.op!r}")
        return sorted(set(changed))


@dataclass
class CommandRepairAdapter:
    """Run a deterministic repair command in a fresh workspace.

    This adapter is the bridge from TRICE's deterministic evidence gate to a
    user's own CLI agent, scripted repair, or harness wrapper. The command is
    executed with ``cwd`` set to the copied workspace, then TRICE fingerprints
    the workspace before and after to record changed files. Test edits are
    refused by default so measured savings stay tied to source repair rather
    than benchmark mutation.
    """

    command: tuple[str, ...]
    name: str = "command-repair-adapter"
    timeout_s: int = 600
    allow_test_edits: bool = False
    forbidden_prefixes: tuple[str, ...] = DEFAULT_FORBIDDEN_PREFIXES
    expected_exit_codes: tuple[int, ...] = (0,)
    applied_empty_ok: bool = False
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_receipt_path: str = ".trice/agent_receipt.json"
    last_receipt: dict[str, Any] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_file(cls, path: str | Path) -> "CommandRepairAdapter":
        p = Path(path).resolve()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(_expand_profile_placeholders(data, p.parent))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommandRepairAdapter":
        return cls(
            command=_parse_argv(data.get("command") or data.get("repair_cmd"), "command"),
            name=str(data.get("name") or "command-repair-adapter"),
            timeout_s=int(data.get("timeout_s") or data.get("repair_timeout_s") or 600),
            allow_test_edits=bool(data.get("allow_test_edits", False)),
            forbidden_prefixes=_effective_forbidden_prefixes(data.get("forbidden_prefixes")),
            expected_exit_codes=tuple(int(c) for c in data.get("expected_exit_codes", [0])),
            applied_empty_ok=bool(data.get("applied_empty_ok", False)),
            env={str(k): str(v) for k, v in dict(data.get("env") or {}).items()},
            metadata=dict(data.get("metadata") or {}),
            agent_receipt_path=str(data.get("agent_receipt_path") or ".trice/agent_receipt.json"),
        )

    def apply_fix(self, task: Any, workspace: Path) -> list[str]:
        if not self.command:
            raise ValueError("repair command is empty")
        root = workspace.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"workspace not found: {root}")
        before = _snapshot_workspace(root)
        agent_receipt_abs = _resolve_in_workspace(root, _clean_rel_path(self.agent_receipt_path))
        proc_env = os.environ.copy()
        proc_env.update(self.env)
        trice_context = _task_trice_context(task)
        reserved_env = {
            "TRICE_TASK_ID": str(getattr(task, "task_id", "")),
            "TRICE_PROMPT": str(getattr(task, "prompt", "")),
            "TRICE_WORKSPACE": str(root),
            "TRICE_AGENT_RECEIPT": str(agent_receipt_abs),
            **_trice_context_env(trice_context),
        }
        proc_env.update(reserved_env)
        proc = subprocess.run(
            list(self.command),
            cwd=root,
            env=proc_env,
            capture_output=True,
            text=True,
            timeout=max(1, int(self.timeout_s)),
        )
        redacted_stdout = _redact_secrets(proc.stdout, proc_env)
        redacted_stderr = _redact_secrets(proc.stderr, proc_env)
        if proc.returncode not in self.expected_exit_codes:
            raise RuntimeError(
                "repair command failed with exit code "
                f"{proc.returncode}: {_excerpt((redacted_stdout + chr(10) + redacted_stderr).strip())}"
            )
        after = _snapshot_workspace(root)
        changed = _changed_paths(before, after)
        forbidden = [rel for rel in changed if _is_forbidden(rel, self.forbidden_prefixes)]
        agent_reported = _read_agent_receipt(agent_receipt_abs)
        self.last_receipt = {
            "schema_version": "trice-run-receipt/v1",
            "adapter_type": "command",
            "adapter_name": self.name,
            "task_id": str(getattr(task, "task_id", "")),
            "prompt_sha256": _sha256_text(str(getattr(task, "prompt", ""))),
            "command": list(self.command),
            "command_sha256": _sha256_json(list(self.command)),
            "timeout_s": self.timeout_s,
            "expected_exit_codes": list(self.expected_exit_codes),
            "exit_code": proc.returncode,
            "workspace_before_sha256": _snapshot_digest(before),
            "workspace_after_sha256": _snapshot_digest(after),
            "changed_files": changed,
            "changed_file_count": len(changed),
            "forbidden_changed_files": forbidden,
            "allow_test_edits": self.allow_test_edits,
            "forbidden_prefixes": list(self.forbidden_prefixes),
            "env_keys": sorted(self.env),
            "reserved_env_keys": sorted(reserved_env),
            "agent_receipt_path": self.agent_receipt_path,
            "agent_reported": agent_reported,
            "trice_context": trice_context,
            "stdout_sha256": _sha256_text(redacted_stdout),
            "stderr_sha256": _sha256_text(redacted_stderr),
            "stdout_excerpt": _excerpt(redacted_stdout, 600),
            "stderr_excerpt": _excerpt(redacted_stderr, 600),
            "output_redacted": redacted_stdout != proc.stdout or redacted_stderr != proc.stderr,
            "metadata": self.metadata,
        }
        if forbidden and not self.allow_test_edits:
            raise ValueError(f"refusing to edit forbidden path(s): {', '.join(forbidden)}")
        if not changed and not self.applied_empty_ok:
            raise ValueError("repair command completed but changed no files")
        return changed


def _clean_rel_path(path: str) -> str:
    rel = Path(path.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"patch path must stay inside workspace: {path}")
    return rel.as_posix()


def _resolve_in_workspace(workspace: Path, rel: str) -> Path:
    root = workspace.resolve()
    target = (root / rel).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"patch path escapes workspace: {rel}")
    return target


def _is_forbidden(rel: str, prefixes: tuple[str, ...]) -> bool:
    rel_l = rel.lower().replace("\\", "/")
    return any(rel_l == p.rstrip("/") or rel_l.startswith(p.lower()) for p in prefixes)


def _effective_forbidden_prefixes(value: Any) -> tuple[str, ...]:
    prefixes = list(DEFAULT_FORBIDDEN_PREFIXES)
    if isinstance(value, (list, tuple)):
        prefixes.extend(str(item) for item in value if str(item).strip())
    out: list[str] = []
    for prefix in prefixes:
        rel = _clean_rel_path(str(prefix).strip())
        if not rel.endswith("/"):
            rel = f"{rel}/"
        if rel not in out:
            out.append(rel)
    return tuple(out)


def _redact_secrets(text: str, env: dict[str, str]) -> str:
    redacted = SECRET_VALUE_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    sensitive_values = {
        value
        for key, value in env.items()
        if len(value) >= 8
        and any(marker in key.lower() for marker in ("key", "secret", "token", "password", "passwd"))
    }
    for value in sorted(sensitive_values, key=len, reverse=True):
        redacted = redacted.replace(value, "[REDACTED]")
    return redacted


_SNAPSHOT_IGNORES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".trice",
}


def _parse_argv(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return tuple(shlex.split(value, posix=False))
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return tuple(value)
    if isinstance(value, tuple) and value and all(isinstance(item, str) and item for item in value):
        return tuple(value)
    raise ValueError(f"{field_name} must be a non-empty string or list of strings")


def _expand_profile_placeholders(data: dict[str, Any], profile_dir: Path) -> dict[str, Any]:
    expanded = dict(data)
    root = Path(__file__).resolve().parents[2]
    replacements = {
        "{{profile_dir}}": profile_dir.as_posix(),
        "{{repo_root}}": root.as_posix(),
    }

    def expand_value(value: Any) -> Any:
        if isinstance(value, str):
            out = value
            for key, replacement in replacements.items():
                out = out.replace(key, replacement)
            return out
        if isinstance(value, list):
            return [expand_value(item) for item in value]
        if isinstance(value, dict):
            return {str(k): expand_value(v) for k, v in value.items()}
        return value

    return {str(k): expand_value(v) for k, v in expanded.items()}


def _snapshot_workspace(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(part in _SNAPSHOT_IGNORES for part in rel_parts):
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        snapshot[rel] = _file_digest(path)
    return snapshot


def _changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    changed = [rel for rel in sorted(set(before) | set(after)) if before.get(rel) != after.get(rel)]
    return changed


def _file_digest(path: Path) -> str:
    if path.is_symlink():
        return "symlink:" + os.readlink(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_digest(snapshot: dict[str, str]) -> str:
    return _sha256_json(snapshot)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_agent_receipt(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"agent receipt must be a JSON object: {path}")
    return data


def _task_trice_context(task: Any) -> dict[str, Any]:
    value = getattr(task, "trice_context", None)
    return dict(value) if isinstance(value, dict) else {}


def _trice_context_env(context: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}
    scalar_fields = {
        "condition": "TRICE_CONDITION",
        "context_mode": "TRICE_CONTEXT_MODE",
        "input_tokens": "TRICE_INPUT_TOKENS",
        "baseline_input_tokens": "TRICE_BASELINE_INPUT_TOKENS",
        "policy_tokens": "TRICE_POLICY_TOKENS",
        "budget_tokens": "TRICE_BUDGET_TOKENS",
        "budget_ratio": "TRICE_BUDGET_RATIO",
        "realized_budget_ratio": "TRICE_REALIZED_BUDGET_RATIO",
        "projected_input_savings_pct": "TRICE_PROJECTED_INPUT_SAVINGS_PCT",
        "policy_sha256": "TRICE_POLICY_SHA256",
        "compressed_context_sha256": "TRICE_COMPRESSED_CONTEXT_SHA256",
        "policy_path": "TRICE_POLICY_PATH",
        "compressed_context_path": "TRICE_CONTEXT_PATH",
        "trace_path": "TRICE_TRACE_PATH",
    }
    for field, env_key in scalar_fields.items():
        value = context.get(field)
        if value is not None:
            env[env_key] = str(value)
    if "TRICE_TRACE_PATH" not in env and context.get("decision_trace_path") is not None:
        env["TRICE_TRACE_PATH"] = str(context["decision_trace_path"])
    verify_cmd = context.get("verify_cmd")
    if isinstance(verify_cmd, (list, tuple)) and all(isinstance(item, str) for item in verify_cmd):
        env["TRICE_VERIFY_CMD_JSON"] = json.dumps(list(verify_cmd), sort_keys=True, separators=(",", ":"))
    action_counts = context.get("policy_action_counts")
    if isinstance(action_counts, dict):
        env["TRICE_POLICY_ACTION_COUNTS_JSON"] = json.dumps(action_counts, sort_keys=True, separators=(",", ":"))
    return env


def _excerpt(text: str, limit: int = 1200) -> str:
    normalized = text.replace("\r\n", "\n").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."
