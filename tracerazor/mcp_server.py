"""Local-first MCP control surface for TraceRazor.

The legacy ``audit_trace``, ``convert_transcript``, ``list_claude_sessions``,
and ``verify_report`` tools retain their 1.x return shapes.  Dict-shaped legacy
results receive additive ``_tracerazor`` metadata.  New tools always return the
versioned :data:`MCP_SCHEMA_VERSION` envelope.

The MCP SDK remains optional.  Tool functions, ``doctor()``, and catalog
inspection do not import it; only server construction does.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback below
    tomllib = None  # type: ignore[assignment]

from tracerazor._launcher import find_binary, recovery_message
from tracerazor.errors import BinaryNotFoundError


MCP_SCHEMA_VERSION = "tracerazor-mcp/v1"
VALIDATION_SCHEMA_VERSION = "tracerazor-validation/v1"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SERVER_ROOT: Path | None = None


class McpToolError(Exception):
    """An expected, machine-readable MCP tool failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = retryable

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }


def _resolve_binary() -> str:
    """Return the auditor binary path or raise the existing teaching error."""
    env = os.environ.get("TRACERAZOR_BIN")
    if env and not os.path.isfile(env):
        raise BinaryNotFoundError(recovery_message())
    binary = find_binary()
    if binary is None:
        raise BinaryNotFoundError(recovery_message())
    return binary


def _run(args: list[str]) -> subprocess.CompletedProcess:
    """Run a child without allowing it to consume the MCP protocol pipe."""
    return subprocess.run(
        args,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _json_safe(value: Any) -> Any:
    """Convert TOML/path/datetime values to an MCP JSON-serializable shape."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


# -- result envelopes ---------------------------------------------------------


def _quality_from(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    direct = payload.get("ingest_quality")
    if isinstance(direct, dict):
        return direct
    manifest = payload.get("manifest")
    if isinstance(manifest, dict) and isinstance(manifest.get("ingest_quality"), dict):
        return manifest["ingest_quality"]
    nested = payload.get("report") or payload.get("data")
    if nested is not payload:
        return _quality_from(nested)
    return None


def _normalise_estimate_status(value: Any) -> str | None:
    if isinstance(value, bool):
        return "estimated" if value else "provider_reported"
    if not isinstance(value, str):
        return None
    value = value.strip().lower().replace("-", "_")
    aliases = {
        "exact": "provider_reported",
        "reported": "provider_reported",
        "provider": "provider_reported",
        "provider_reported": "provider_reported",
        "estimated": "estimated",
        "estimate": "estimated",
        "inferred": "estimated",
        "mixed": "mixed",
        "partial": "mixed",
        "missing": "missing",
        "unknown": "unknown",
    }
    return aliases.get(value)


def _estimate_status_from(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "unknown"
    candidates: list[Any] = [
        payload.get("estimate_status"),
        payload.get("token_provenance"),
        payload.get("usage_provenance"),
        payload.get("tokens_estimated"),
    ]
    for key in ("manifest", "usage", "capture_quality"):
        child = payload.get(key)
        if isinstance(child, dict):
            candidates.extend(
                [
                    child.get("estimate_status"),
                    child.get("token_provenance"),
                    child.get("usage_provenance"),
                    child.get("tokens_estimated"),
                ]
            )
    seen: set[str] = set()
    for candidate in candidates:
        if isinstance(candidate, list):
            for item in candidate:
                status = _normalise_estimate_status(item)
                if status:
                    seen.add(status)
        else:
            status = _normalise_estimate_status(candidate)
            if status:
                seen.add(status)
    if "estimated" in seen and "provider_reported" in seen:
        return "mixed"
    if "mixed" in seen:
        return "mixed"
    for status in ("estimated", "missing", "provider_reported", "unknown"):
        if status in seen:
            return status
    return "unknown"


def _warnings_from(payload: Any) -> list[str]:
    quality = _quality_from(payload)
    if not quality:
        return []
    warnings = quality.get("warnings")
    if not isinstance(warnings, list):
        return []
    return [str(item) for item in warnings if str(item).strip()]


def _run_id_from(payload: Any, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    if not isinstance(payload, dict):
        return None
    for key in ("run_id", "trace_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("manifest", "target", "report"):
        child = payload.get(key)
        if isinstance(child, dict):
            found = _run_id_from(child)
            if found:
                return found
    return None


def _metadata(
    payload: Any = None,
    *,
    run_id: str | None = None,
    warnings: list[str] | None = None,
    evidence_ref: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_warnings = _warnings_from(payload)
    if warnings:
        merged_warnings.extend(str(item) for item in warnings if str(item).strip())
    # Preserve order while removing duplicates.
    merged_warnings = list(dict.fromkeys(merged_warnings))
    return {
        "schema_version": MCP_SCHEMA_VERSION,
        "run_id": _run_id_from(payload, run_id),
        "ingest_quality": _quality_from(payload),
        "estimate_status": _estimate_status_from(payload),
        "warnings": merged_warnings,
        "evidence_ref": evidence_ref,
        "error": error,
    }


def _envelope(
    data: Any = None,
    *,
    run_id: str | None = None,
    warnings: list[str] | None = None,
    evidence_ref: str | None = None,
    error: dict[str, Any] | McpToolError | None = None,
    metadata_source: Any = None,
) -> dict[str, Any]:
    if isinstance(error, McpToolError):
        error = error.as_dict()
    source = data if metadata_source is None else metadata_source
    result = _metadata(
        source,
        run_id=run_id,
        warnings=warnings,
        evidence_ref=evidence_ref,
        error=error,
    )
    result["ok"] = error is None
    result["data"] = data
    return result


def _error_envelope(error: Exception, *, run_id: str | None = None) -> dict[str, Any]:
    if isinstance(error, McpToolError):
        tool_error = error
    elif isinstance(error, BinaryNotFoundError):
        tool_error = McpToolError("binary_not_found", str(error))
    else:
        tool_error = McpToolError("internal_error", str(error))
    return _envelope(None, run_id=run_id, error=tool_error)


def _legacy_result(
    result: dict[str, Any],
    *,
    run_id: str | None = None,
    warnings: list[str] | None = None,
    evidence_ref: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result["_tracerazor"] = _metadata(
        result,
        run_id=run_id,
        warnings=warnings,
        evidence_ref=evidence_ref,
        error=error,
    )
    return result


# -- filesystem boundary ------------------------------------------------------


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _reject_link_components(path: Path) -> None:
    """Reject symlinks/junctions in every existing component of ``path``."""
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if _is_link(current):
            raise McpToolError(
                "unsafe_symlink",
                "symlinks and junctions are not accepted by MCP path inputs",
                details={"path": str(current)},
            )
        if not current.exists():
            break


def _workspace_root(cwd: str = ".") -> Path:
    raw = Path(cwd)
    if ".." in raw.parts:
        raise McpToolError("path_traversal", "cwd may not contain '..'")
    candidate = Path(os.path.abspath(raw))
    _reject_link_components(candidate)
    if not candidate.is_dir():
        raise McpToolError(
            "workspace_not_found", "workspace root does not exist", details={"cwd": str(cwd)}
        )
    resolved = candidate.resolve(strict=True)

    configured = os.environ.get("TRACERAZOR_MCP_ROOT")
    if not configured and _SERVER_ROOT is not None:
        configured = os.fspath(_SERVER_ROOT)
    if configured:
        configured_raw = Path(configured)
        if ".." in configured_raw.parts:
            raise McpToolError("path_traversal", "TRACERAZOR_MCP_ROOT may not contain '..'")
        boundary = Path(os.path.abspath(configured_raw))
        _reject_link_components(boundary)
        if not boundary.is_dir():
            raise McpToolError("workspace_not_found", "TRACERAZOR_MCP_ROOT does not exist")
        boundary = boundary.resolve(strict=True)
        if not _contained(resolved, boundary):
            raise McpToolError(
                "path_outside_workspace",
                "cwd is outside TRACERAZOR_MCP_ROOT",
                details={"cwd": str(resolved), "root": str(boundary)},
            )
    return resolved


def _contained(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            [os.path.normcase(str(path)), os.path.normcase(str(root))]
        ) == os.path.normcase(str(root))
    except ValueError:
        return False


def _safe_path(
    value: str | os.PathLike[str],
    root: Path,
    *,
    must_exist: bool = True,
    kind: str | None = None,
) -> Path:
    raw_text = os.fspath(value)
    if not raw_text or "\x00" in raw_text:
        raise McpToolError("invalid_path", "path is empty or contains a NUL byte")
    raw = Path(raw_text)
    if ".." in raw.parts:
        raise McpToolError(
            "path_traversal", "path may not contain '..'", details={"path": raw_text}
        )
    candidate = raw if raw.is_absolute() else root / raw
    candidate = Path(os.path.abspath(candidate))
    _reject_link_components(candidate)
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise McpToolError(
            "path_not_found", "path does not exist", details={"path": raw_text}
        ) from exc
    if not _contained(resolved, root):
        raise McpToolError(
            "path_outside_workspace",
            "path is outside the MCP workspace boundary",
            details={"path": str(resolved), "root": str(root)},
        )
    if must_exist and kind == "file" and not resolved.is_file():
        raise McpToolError("not_a_file", "expected a file", details={"path": str(resolved)})
    if must_exist and kind == "dir" and not resolved.is_dir():
        raise McpToolError("not_a_directory", "expected a directory", details={"path": str(resolved)})
    if not must_exist and kind == "file" and resolved.exists() and not resolved.is_file():
        raise McpToolError("not_a_file", "expected a file", details={"path": str(resolved)})
    return resolved


def _evidence_ref(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise McpToolError(
            "invalid_json", "artifact is not valid JSON", details={"path": str(path)}
        ) from exc
    except OSError as exc:
        raise McpToolError(
            "artifact_read_failed", str(exc), details={"path": str(path)}
        ) from exc


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise McpToolError(
            "invalid_run_id",
            "run_id must contain only letters, digits, '.', '_' or '-'",
            details={"run_id": run_id},
        )
    return run_id


def _runs_root(root: Path, *, must_exist: bool = True) -> Path:
    return _safe_path(".tracerazor/runs", root, must_exist=must_exist, kind="dir" if must_exist else None)


def _run_directories(root: Path) -> list[Path]:
    try:
        runs = _runs_root(root)
    except McpToolError as exc:
        if exc.code == "path_not_found":
            return []
        raise
    directories: list[Path] = []
    for child in runs.iterdir():
        if _is_link(child):
            continue
        if child.is_dir():
            directories.append(child.resolve(strict=True))
    directories.sort(key=lambda item: item.stat().st_mtime_ns, reverse=True)
    return directories


def _resolve_run_dir(run_id: str | None, root: Path) -> tuple[Path, str]:
    selected = run_id or os.environ.get("TRACERAZOR_RUN_ID")
    if selected:
        selected = _validate_run_id(selected)
        path = _safe_path(
            f".tracerazor/runs/{selected}", root, must_exist=True, kind="dir"
        )
        return path, selected
    directories = _run_directories(root)
    if not directories:
        raise McpToolError("run_not_found", "no run artifacts were found")
    return directories[0], directories[0].name


def _claude_sessions(root: Path) -> list[dict[str, Any]]:
    try:
        path = _safe_path(
            ".tracerazor/claude-code/index.json", root, must_exist=True, kind="file"
        )
    except McpToolError as exc:
        if exc.code == "path_not_found":
            return []
        raise
    try:
        data = _read_json(path)
    except McpToolError:
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _claude_entry(root: Path, run_id: str | None) -> dict[str, Any] | None:
    entries = _claude_sessions(root)
    if run_id:
        _validate_run_id(run_id)
        return next(
            (
                entry
                for entry in entries
                if entry.get("trace_id") == run_id or entry.get("run_id") == run_id
            ),
            None,
        )
    return entries[0] if entries else None


def _path_from_claude_entry(
    entry: dict[str, Any], field: str, root: Path
) -> Path | None:
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        return None
    try:
        return _safe_path(value, root, must_exist=True, kind="file")
    except McpToolError:
        return None


def _run_artifact(
    run_id: str | None,
    filename: str,
    root: Path,
    *,
    claude_field: str | None = None,
) -> tuple[Path, str, list[str]]:
    warnings: list[str] = []
    try:
        run_dir, selected = _resolve_run_dir(run_id, root)
        path = _safe_path(run_dir / filename, root, must_exist=True, kind="file")
        return path, selected, warnings
    except McpToolError as run_error:
        if run_error.code not in {"path_not_found", "run_not_found"}:
            raise
        entry = _claude_entry(root, run_id)
        if entry and claude_field:
            path = _path_from_claude_entry(entry, claude_field, root)
            if path:
                selected = str(entry.get("run_id") or entry.get("trace_id") or run_id or "")
                warnings.append("using Claude Code session index fallback")
                return path, selected or path.stem, warnings
        raise McpToolError(
            "artifact_not_found",
            f"{filename} was not found for the requested run",
            details={"run_id": run_id},
        ) from run_error


def _trace_reference(value: str, root: Path) -> tuple[Path, str | None, list[str]]:
    # A matching run id is preferred; otherwise treat the value as an explicit
    # path inside the workspace.
    if RUN_ID_RE.fullmatch(value) and value not in {".", ".."}:
        try:
            path, selected, warnings = _run_artifact(
                value, "trace.json", root, claude_field="trace"
            )
            return path, selected, warnings
        except McpToolError as exc:
            if exc.code not in {"artifact_not_found", "path_not_found"}:
                raise
    path = _safe_path(value, root, must_exist=True)
    if path.is_dir():
        path = _safe_path(path / "trace.json", root, must_exist=True, kind="file")
    elif not path.is_file():
        raise McpToolError("not_a_file", "trace reference is not a file")
    return path, None, []


# -- legacy 1.x tools ---------------------------------------------------------


def audit_trace(
    path: str,
    hermetic: bool = True,
    min_steps: int | None = None,
    threshold: int | None = None,
    cwd: str = ".",
) -> dict[str, Any]:
    """Audit a trace and retain the historical top-level report shape."""
    binary = _resolve_binary()
    try:
        root = _workspace_root(cwd)
        trace_path = _safe_path(path, root, must_exist=True, kind="file")
    except McpToolError as exc:
        result = {"error": "audit failed", "exit_code": 2, "stderr": exc.message}
        return _legacy_result(result, error=exc.as_dict())
    args = [binary, "audit", str(trace_path), "--format", "json"]
    if hermetic:
        args.append("--hermetic")
    if min_steps is not None:
        args += ["--min-steps", str(min_steps)]
    if threshold is not None:
        args += ["--threshold", str(threshold)]
    res = _run(args)
    evidence = _evidence_ref(trace_path, root)
    if res.returncode == 2:
        error = McpToolError(
            "audit_failed",
            (res.stderr or "audit failed").strip(),
            details={"exit_code": 2},
        )
        result = {"error": "audit failed", "exit_code": 2, "stderr": error.message}
        return _legacy_result(result, evidence_ref=evidence, error=error.as_dict())
    report = _loads(res.stdout)
    if not isinstance(report, dict):
        result = {
            "passed": res.returncode == 0,
            "audited": False,
            "message": (res.stderr or "").strip()
            or "no report produced (trace below --min-steps?)",
            "exit_code": res.returncode,
        }
        return _legacy_result(result, evidence_ref=evidence)
    report["passed"] = res.returncode == 0
    return _legacy_result(report, evidence_ref=evidence)


def convert_transcript(path: str, format: str = "auto", cwd: str = ".") -> dict[str, Any]:
    """Normalize an export while retaining the historical trace shape."""
    binary = _resolve_binary()
    try:
        root = _workspace_root(cwd)
        source = _safe_path(path, root, must_exist=True, kind="file")
    except McpToolError as exc:
        result = {"error": "convert failed", "exit_code": 2, "stderr": exc.message}
        return _legacy_result(result, error=exc.as_dict())
    if format == "claude-code" or str(source).endswith(".jsonl"):
        args = [binary, "claude", "convert", str(source)]
    else:
        args = [binary, "import", str(source), "--from", format]
    res = _run(args)
    evidence = _evidence_ref(source, root)
    if res.returncode != 0:
        error = McpToolError(
            "convert_failed",
            (res.stderr or "convert failed").strip(),
            details={"exit_code": res.returncode},
        )
        result = {
            "error": "convert failed",
            "exit_code": res.returncode,
            "stderr": error.message,
        }
        return _legacy_result(result, evidence_ref=evidence, error=error.as_dict())
    trace = _loads(res.stdout)
    if not isinstance(trace, dict):
        error = McpToolError("invalid_cli_output", "converter produced no JSON object")
        result = {
            "error": "converter produced no JSON on stdout",
            "exit_code": res.returncode,
            "stderr": (res.stderr or "").strip(),
        }
        return _legacy_result(result, evidence_ref=evidence, error=error.as_dict())
    return _legacy_result(trace, evidence_ref=evidence)


def list_claude_sessions(cwd: str = ".") -> list[dict[str, Any]]:
    """Return the legacy list shape for the Claude Code session index."""
    try:
        return _claude_sessions(_workspace_root(cwd))
    except McpToolError:
        return []


def _verify_result(binary: str, report: Path, trace: Path | None) -> tuple[dict[str, Any], int]:
    base = [binary, "verify", str(report)]
    if trace is not None:
        base.append(str(trace))
    probe = _run(base + ["--format", "json"])
    if probe.returncode in (0, 1):
        parsed = _loads(probe.stdout)
        if isinstance(parsed, dict):
            return parsed, probe.returncode
    res = _run(base)
    status = {0: "verified", 1: "tampered"}.get(res.returncode, "error")
    return (
        {
            "status": status,
            "exit_code": res.returncode,
            "stdout": (res.stdout or "").strip(),
            "stderr": (res.stderr or "").strip(),
        },
        res.returncode,
    )


def _is_redacted_non_replayable(report: Path, trace: Path | None = None) -> bool:
    try:
        report_value = _read_json(report)
    except McpToolError:
        report_value = None
    if isinstance(report_value, dict) and report_value.get("persisted_representation") == "redacted_auditor_report":
        return True
    if trace is not None:
        try:
            trace_value = _read_json(trace)
        except McpToolError:
            trace_value = None
        metadata = trace_value.get("metadata") if isinstance(trace_value, dict) else None
        if isinstance(metadata, dict) and metadata.get("tracerazor_redacted") is True:
            return True
    return False


def _non_replayable_verdict() -> dict[str, Any]:
    return {
        "status": "non_replayable_redacted",
        "verified": False,
        "tampered": False,
        "exit_code": 0,
        "reason": (
            "the default local-redacted artifact was audited from raw content in memory, "
            "but the persisted representation intentionally cannot be re-scored"
        ),
    }


def verify_report(
    report_path: str, trace_path: str | None = None, cwd: str = "."
) -> dict[str, Any]:
    """Re-verify a report while retaining the historical verdict shape."""
    try:
        root = _workspace_root(cwd)
        report = _safe_path(report_path, root, must_exist=True, kind="file")
        trace = (
            _safe_path(trace_path, root, must_exist=True, kind="file")
            if trace_path is not None
            else None
        )
    except McpToolError as exc:
        # Preserve the 1.x teaching contract: a configured-but-missing binary
        # is surfaced before a legacy path verdict. Existing redacted evidence
        # bypasses binary resolution only after its marker has been read.
        _resolve_binary()
        result = {"status": "error", "exit_code": 2, "stdout": "", "stderr": exc.message}
        return _legacy_result(result, error=exc.as_dict())
    if _is_redacted_non_replayable(report, trace):
        return _legacy_result(
            _non_replayable_verdict(), evidence_ref=_evidence_ref(report, root)
        )
    binary = _resolve_binary()
    result, exit_code = _verify_result(binary, report, trace)
    error = None
    if exit_code not in (0, 1):
        error = McpToolError(
            "verify_failed",
            str(result.get("stderr") or "verification failed"),
            details={"exit_code": exit_code},
        ).as_dict()
    return _legacy_result(
        result,
        run_id=_run_id_from(result),
        evidence_ref=_evidence_ref(report, root),
        error=error,
    )


# -- versioned control tools --------------------------------------------------


def doctor(cwd: str = ".") -> dict[str, Any]:
    """Inspect binary, project policy, and local artifacts without the MCP SDK."""
    try:
        root = _workspace_root(cwd)
    except Exception as exc:
        return _error_envelope(exc)

    warnings: list[str] = []
    configured_binary = os.environ.get("TRACERAZOR_BIN")
    if configured_binary and not os.path.isfile(configured_binary):
        binary_data: dict[str, Any] = {
            "status": "misconfigured",
            "path": configured_binary,
            "version": None,
        }
        warnings.append("TRACERAZOR_BIN points to a missing file")
    else:
        binary = find_binary()
        if binary:
            version_result = _run([binary, "--version"])
            binary_data = {
                "status": "ready" if version_result.returncode == 0 else "error",
                "path": binary,
                "version": (version_result.stdout or version_result.stderr or "").strip() or None,
            }
            if version_result.returncode != 0:
                warnings.append("the TraceRazor binary did not pass --version")
        else:
            binary_data = {"status": "missing", "path": None, "version": None}
            warnings.append("the TraceRazor auditor binary was not found")

    policy_path = root / "tracerazor.toml"
    try:
        policy_path = _safe_path(policy_path, root, must_exist=False, kind="file")
        if policy_path.is_file():
            try:
                if tomllib is not None:
                    with policy_path.open("rb") as handle:
                        policy = tomllib.load(handle)
                else:
                    from tracerazor.runtime import AuditPolicy

                    policy = AuditPolicy.load(policy_path).to_dict()
                policy_data = {
                    "status": "ready",
                    "path": _evidence_ref(policy_path, root),
                    "mode": policy.get("mode", "coach") if isinstance(policy, dict) else "coach",
                }
            except (OSError, ValueError) as exc:
                policy_data = {
                    "status": "invalid",
                    "path": _evidence_ref(policy_path, root),
                    "error": str(exc),
                }
                warnings.append("tracerazor.toml is invalid")
        else:
            policy_data = {"status": "missing", "path": "tracerazor.toml", "mode": "coach"}
            warnings.append("tracerazor.toml was not found; coach defaults apply")
    except McpToolError as exc:
        return _error_envelope(exc)

    try:
        runs = _run_directories(root)
        claude = _claude_sessions(root)
        artifact_data = {
            "status": "ready" if runs or claude else "empty",
            "runs_path": ".tracerazor/runs",
            "run_count": len(runs),
            "latest_run_id": runs[0].name if runs else None,
            "claude_session_count": len(claude),
        }
    except McpToolError as exc:
        return _error_envelope(exc)

    data = {
        "workspace": str(root),
        "binary": binary_data,
        "policy": policy_data,
        "artifacts": artifact_data,
    }
    return _envelope(data, run_id=artifact_data["latest_run_id"], warnings=warnings)


def audit_current_run(
    run_id: str | None = None,
    cwd: str = ".",
    min_steps: int | None = None,
) -> dict[str, Any]:
    """Audit the selected run's trace, falling back to the Claude index."""
    try:
        root = _workspace_root(cwd)
        trace, selected, warnings = _run_artifact(
            run_id, "trace.json", root, claude_field="trace"
        )
        trace_value = _read_json(trace)
        metadata = trace_value.get("metadata") if isinstance(trace_value, dict) else None
        if isinstance(metadata, dict) and metadata.get("tracerazor_redacted") is True:
            report_path = _safe_path(
                trace.parent / "report.json", root, must_exist=False, kind="file"
            )
            if not report_path.is_file():
                raise McpToolError(
                    "non_replayable_redacted",
                    "stored trace is redacted and no in-memory auditor report was persisted",
                )
            report = _read_json(report_path)
            if not isinstance(report, dict):
                raise McpToolError("invalid_report", "report.json must be an object")
            warnings.append(
                "stored trace is redacted; returning the report produced from raw content in memory"
            )
            report["reused_in_memory_audit"] = True
            return _envelope(
                report,
                run_id=selected,
                warnings=warnings,
                evidence_ref=_evidence_ref(report_path, root),
                metadata_source=report,
            )
        binary = _resolve_binary()
        args = [binary, "audit", str(trace), "--format", "json", "--hermetic"]
        if min_steps is not None:
            args += ["--min-steps", str(min_steps)]
        res = _run(args)
        if res.returncode == 2:
            raise McpToolError(
                "audit_failed",
                (res.stderr or "audit failed").strip(),
                details={"exit_code": 2},
            )
        report = _loads(res.stdout)
        if not isinstance(report, dict):
            raise McpToolError(
                "no_report",
                (res.stderr or "no report produced (trace below --min-steps?)").strip(),
                details={"exit_code": res.returncode},
            )
        report["passed"] = res.returncode == 0
        return _envelope(
            report,
            run_id=selected,
            warnings=warnings,
            evidence_ref=_evidence_ref(trace, root),
        )
    except Exception as exc:
        return _error_envelope(exc, run_id=run_id)


def latest_findings(run_id: str | None = None, cwd: str = ".") -> dict[str, Any]:
    """Return stored findings, or derive them additively from a stored report."""
    try:
        root = _workspace_root(cwd)
        warnings: list[str] = []
        source: Path
        selected: str
        try:
            source, selected, warnings = _run_artifact(run_id, "findings.json", root)
            artifact = _read_json(source)
            findings = artifact.get("findings") if isinstance(artifact, dict) else artifact
            metadata_source = artifact
        except McpToolError as exc:
            if exc.code not in {"artifact_not_found", "path_not_found"}:
                raise
            try:
                source, selected, warnings = _run_artifact(
                    run_id, "report.json", root, claude_field="report"
                )
                artifact = _read_json(source)
                findings = artifact.get("fixes", []) if isinstance(artifact, dict) else []
                metadata_source = artifact
                warnings.append("findings.json missing; returning report fixes")
            except McpToolError as report_exc:
                if report_exc.code not in {"artifact_not_found", "path_not_found"}:
                    raise
                source, selected, warnings = _run_artifact(
                    run_id, "fixes.json", root, claude_field="fixes"
                )
                artifact = _read_json(source)
                if isinstance(artifact, list):
                    findings = artifact
                elif isinstance(artifact, dict):
                    findings = artifact.get("fixes", [])
                else:
                    raise McpToolError(
                        "invalid_findings", "fixes artifact must be an object or array"
                    )
                metadata_source = artifact
                warnings.append("returning Claude Code fixes fallback")
        if not isinstance(findings, list):
            findings = [findings] if findings is not None else []
        data = {
            "source": _evidence_ref(source, root),
            "count": len(findings),
            "findings": findings,
        }
        return _envelope(
            data,
            run_id=selected,
            warnings=warnings,
            evidence_ref=_evidence_ref(source, root),
            metadata_source=metadata_source,
        )
    except Exception as exc:
        return _error_envelope(exc, run_id=run_id)


def compare_runs(
    baseline: str,
    target: str,
    cwd: str = ".",
    regression_threshold: float = 10.0,
) -> dict[str, Any]:
    """Compare two run ids or trace paths using the native JSON command."""
    try:
        if not math.isfinite(regression_threshold) or regression_threshold < 0:
            raise McpToolError(
                "invalid_threshold", "regression_threshold must be a finite non-negative number"
            )
        root = _workspace_root(cwd)
        baseline_path, baseline_id, baseline_warnings = _trace_reference(baseline, root)
        target_path, target_id, target_warnings = _trace_reference(target, root)
        binary = _resolve_binary()
        res = _run(
            [
                binary,
                "compare",
                str(baseline_path),
                str(target_path),
                "--format",
                "json",
                "--regression-threshold",
                str(regression_threshold),
            ]
        )
        parsed = _loads(res.stdout)
        if res.returncode not in (0, 1) or not isinstance(parsed, dict):
            raise McpToolError(
                "compare_failed",
                (res.stderr or "compare produced no JSON").strip(),
                details={"exit_code": res.returncode},
            )
        warnings = baseline_warnings + target_warnings
        if res.returncode == 1:
            warnings.append("comparison regression gate failed")
        data = {
            "passed": res.returncode == 0,
            "exit_code": res.returncode,
            "baseline_ref": _evidence_ref(baseline_path, root),
            "target_ref": _evidence_ref(target_path, root),
            "comparison": parsed,
        }
        return _envelope(
            data,
            run_id=target_id or _run_id_from(parsed),
            warnings=warnings,
            evidence_ref=_evidence_ref(target_path, root),
            metadata_source=parsed,
        )
    except Exception as exc:
        return _error_envelope(exc)


SIGNALS: dict[str, dict[str, Any]] = {
    "srr": {"name": "Step Redundancy Rate", "diagnoses": "near-duplicate reasoning steps", "direction": "higher is cleaner", "fixes": ["reformulation_guard"]},
    "ldi": {"name": "Loop Detection Index", "diagnoses": "repeated or parametric tool-call loops", "direction": "higher is cleaner", "fixes": ["termination_guard"]},
    "tca": {"name": "Tool Call Accuracy", "diagnoses": "failed calls followed by retries", "direction": "higher is cleaner", "fixes": ["tool_schema"]},
    "tur": {"name": "Token Utilisation Ratio", "diagnoses": "tokens already attributed to low-value steps", "direction": "higher is cleaner", "fixes": []},
    "cce": {"name": "Context Efficiency", "diagnoses": "duplicated context carried across steps", "direction": "higher is cleaner", "fixes": ["context_compression"]},
    "rda": {"name": "Reasoning Depth Appropriateness", "diagnoses": "reasoning deeper than the task requires", "direction": "higher is cleaner", "fixes": ["verbosity_reduction"]},
    "isr": {"name": "Information Sufficiency Rate", "diagnoses": "steps that add little new information", "direction": "higher is cleaner", "fixes": ["verbosity_reduction"]},
    "dbo": {"name": "Decision Branch Optimality", "diagnoses": "sub-optimal tool sequences and branch thrashing", "direction": "higher is cleaner", "fixes": ["termination_guard"]},
    "vdi": {"name": "Verbosity Density", "diagnoses": "verbose prose relative to useful content", "direction": "higher is cleaner", "fixes": ["verbosity_reduction"]},
    "shl": {"name": "Sycophancy and Hedging Level", "diagnoses": "hedging and sycophantic phrasing", "direction": "higher is cleaner", "fixes": ["hedge_reduction"]},
    "ccr": {"name": "Compressibility", "diagnoses": "content that can be compressed without losing task value", "direction": "higher is cleaner", "fixes": ["context_compression"]},
    "gar": {"name": "Goal Advancement Rate", "diagnoses": "steps that do not advance the declared task", "direction": "higher is cleaner", "fixes": ["goal_anchor"]},
    "csd": {"name": "Context Semantic Drift", "diagnoses": "drift away from the task context", "direction": "higher is cleaner", "fixes": ["goal_anchor"]},
    "obs": {"name": "Observation Token Share", "diagnoses": "tool-output accumulation relative to recoverable reasoning", "direction": "higher is cleaner", "fixes": ["context_compression"]},
}


def explain_signal(
    signal: str, run_id: str | None = None, cwd: str = "."
) -> dict[str, Any]:
    """Explain a named signal and, when available, attach run-specific facts."""
    code = signal.strip().lower()
    if code not in SIGNALS:
        return _error_envelope(
            McpToolError(
                "unknown_signal",
                f"unknown signal: {signal}",
                details={"available": sorted(SIGNALS)},
            ),
            run_id=run_id,
        )
    try:
        root = _workspace_root(cwd)
        report: dict[str, Any] | None = None
        selected = run_id
        evidence: str | None = None
        warnings: list[str] = []
        try:
            report_path, selected, fallback_warnings = _run_artifact(
                run_id, "report.json", root, claude_field="report"
            )
            artifact = _read_json(report_path)
            if isinstance(artifact, dict):
                report = artifact
                evidence = _evidence_ref(report_path, root)
                warnings.extend(fallback_warnings)
        except McpToolError:
            if run_id:
                raise
            warnings.append("no report found; returning the static signal definition")

        definition = dict(SIGNALS[code])
        definition["code"] = code.upper()
        definition["tas_note"] = "TAS is ordinal; compare the same workload over time."
        if report:
            score = report.get("score") if isinstance(report.get("score"), dict) else {}
            normalised = score.get("metric_normalised") if isinstance(score, dict) else {}
            definition["normalised_score"] = (
                normalised.get(code) if isinstance(normalised, dict) else None
            )
            definition["detail"] = score.get(code) if isinstance(score, dict) else None
            fixes = report.get("fixes", [])
            definition["matching_fixes"] = [
                item
                for item in fixes
                if isinstance(item, dict)
                and item.get("fix_type") in definition.get("fixes", [])
            ]
        return _envelope(
            definition,
            run_id=selected,
            warnings=warnings,
            evidence_ref=evidence,
            metadata_source=report,
        )
    except Exception as exc:
        return _error_envelope(exc, run_id=run_id)


def preview_fix(
    run_id: str,
    target_path: str,
    cwd: str = ".",
    include_needs_review: bool = False,
) -> dict[str, Any]:
    """Preview CLI patch application.  This tool always passes ``--dry-run``."""
    try:
        root = _workspace_root(cwd)
        try:
            source, selected, warnings = _run_artifact(run_id, "fixes.json", root)
        except McpToolError as exc:
            if exc.code not in {"artifact_not_found", "path_not_found"}:
                raise
            source, selected, warnings = _run_artifact(
                run_id, "report.json", root, claude_field="report"
            )
            warnings.append("fixes.json missing; previewing fixes from report.json")
        target = _safe_path(target_path, root, must_exist=False, kind="file")
        binary = _resolve_binary()
        args = [binary, "apply", str(source), "--to", str(target), "--dry-run"]
        if include_needs_review:
            args.append("--all")
            warnings.append("needs-review fixes included; dangerous fixes remain excluded")
        res = _run(args)
        if res.returncode != 0:
            raise McpToolError(
                "preview_failed",
                (res.stderr or "fix preview failed").strip(),
                details={"exit_code": res.returncode},
            )
        data = {
            "dry_run": True,
            "wrote": False,
            "source": _evidence_ref(source, root),
            "target": _evidence_ref(target, root),
            "stdout": (res.stdout or "").strip(),
            "stderr": (res.stderr or "").strip(),
            "exit_code": res.returncode,
        }
        return _envelope(
            data,
            run_id=selected,
            warnings=warnings,
            evidence_ref=_evidence_ref(source, root),
        )
    except Exception as exc:
        return _error_envelope(exc, run_id=run_id)


def record_validation(
    run_id: str, validation: dict[str, Any], cwd: str = "."
) -> dict[str, Any]:
    """Atomically write ``validation.json`` inside an existing run directory."""
    try:
        if not isinstance(validation, dict):
            raise McpToolError("invalid_validation", "validation must be a JSON object")
        allowed_top = {
            "status",
            "outcome",
            "passed",
            "task_success",
            "verifier",
            "score",
            "evidence",
            "metadata",
            "task",
        }
        unknown = sorted(set(validation) - allowed_top)
        if unknown:
            raise McpToolError(
                "invalid_validation",
                "validation contains unsupported fields",
                details={"fields": unknown},
            )
        root = _workspace_root(cwd)
        run_dir, selected = _resolve_run_dir(run_id, root)
        destination = _safe_path(
            run_dir / "validation.json", root, must_exist=False, kind="file"
        )
        task_input = validation.get("task")
        if task_input is None:
            task_input = {}
        if not isinstance(task_input, dict):
            raise McpToolError("invalid_validation", "task must be a JSON object")
        allowed_task = {"outcome", "passed", "verifier", "score", "evidence"}
        unknown_task = sorted(set(task_input) - allowed_task)
        if unknown_task:
            raise McpToolError(
                "invalid_validation",
                "task validation contains unsupported fields",
                details={"fields": unknown_task},
            )

        outcome = task_input.get("outcome", validation.get("outcome", validation.get("status")))
        passed = task_input.get(
            "passed", validation.get("passed", validation.get("task_success"))
        )
        if outcome is None and isinstance(passed, bool):
            outcome = "passed" if passed else "failed"
        if outcome is None:
            outcome = "unknown"
        if not isinstance(outcome, str) or outcome.lower() not in {
            "passed",
            "failed",
            "unknown",
            "cancelled",
        }:
            raise McpToolError(
                "invalid_validation",
                "outcome must be passed, failed, unknown, or cancelled",
            )
        outcome = outcome.lower()
        if passed is not None and not isinstance(passed, bool):
            raise McpToolError("invalid_validation", "passed must be a boolean")
        if isinstance(passed, bool) and passed != (outcome == "passed"):
            raise McpToolError(
                "invalid_validation", "passed conflicts with the declared outcome"
            )
        verifier = task_input.get("verifier", validation.get("verifier"))
        if verifier is not None and (not isinstance(verifier, str) or not verifier.strip()):
            raise McpToolError("invalid_validation", "verifier must be a non-empty string")
        score = task_input.get("score", validation.get("score"))
        if score is not None and (
            isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
        ):
            raise McpToolError("invalid_validation", "score must be a finite number")

        try:
            from tracerazor.runtime import AuditPolicy
            from tracerazor.runtime.persistence import artifact_for_persistence

            policy_path = root / "tracerazor.toml"
            persistence_policy = (
                AuditPolicy.load(policy_path) if policy_path.is_file() else AuditPolicy()
            )
            evidence = artifact_for_persistence(
                task_input.get("evidence", validation.get("evidence", {})),
                persistence_policy,
            )
            metadata = artifact_for_persistence(
                validation.get("metadata", {}), persistence_policy
            )
        except (OSError, ValueError) as exc:
            raise McpToolError("invalid_policy", str(exc)) from exc

        # Keep only the versioned validation contract.  Free-form evidence and
        # metadata are privacy-filtered before the atomic write.
        payload: dict[str, Any] = {
            "task": {
                "outcome": outcome,
                "verifier": verifier,
                "score": score,
                "evidence": evidence,
            },
            "metadata": metadata,
            "trust_level": "untrusted_mcp_record",
        }
        payload["status"] = outcome
        payload["schema_version"] = VALIDATION_SCHEMA_VERSION
        payload["run_id"] = selected
        payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
        encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if len(encoded.encode("utf-8")) > 1_048_576:
            raise McpToolError("validation_too_large", "validation exceeds the 1 MiB limit")

        descriptor = -1
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=".validation.", suffix=".tmp", dir=str(run_dir)
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                descriptor = -1
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path = _safe_path(temporary, root, must_exist=True, kind="file")
            # Recheck immediately before replacement to prevent a pre-existing
            # validation.json symlink from redirecting the explicit write.
            _reject_link_components(destination)
            os.replace(temporary_path, destination)
            temporary = None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

        report: Any = None
        report_path = run_dir / "report.json"
        if report_path.is_file() and not _is_link(report_path):
            try:
                report = _read_json(report_path)
            except McpToolError:
                report = None
        data = {"recorded": True, "validation": payload}
        return _envelope(
            data,
            run_id=selected,
            evidence_ref=_evidence_ref(destination, root),
            metadata_source=report,
        )
    except Exception as exc:
        return _error_envelope(exc, run_id=run_id)


def _quality_is_degraded(quality: dict[str, Any] | None) -> bool:
    if not quality:
        return False
    if quality.get("degraded_ingest") is True or quality.get("degraded") is True:
        return True
    if str(quality.get("status", "")).lower() in {"degraded", "partial"}:
        return True
    for key in ("token_coverage", "content_coverage", "provider_token_coverage"):
        value = quality.get(key)
        if isinstance(value, (int, float)) and value < 1.0:
            return True
    return False


def check_policy(run_id: str | None = None, cwd: str = ".") -> dict[str, Any]:
    """Read ``tracerazor.toml`` and calculate safe enforcement eligibility."""
    try:
        root = _workspace_root(cwd)
        policy_path = _safe_path("tracerazor.toml", root, must_exist=False, kind="file")
        warnings: list[str] = []
        if policy_path.is_file():
            try:
                if tomllib is not None:
                    with policy_path.open("rb") as handle:
                        policy = tomllib.load(handle)
                else:
                    # Reuse the dependency-free, fail-closed Python 3.10
                    # policy parser from the public runtime package.
                    from tracerazor.runtime import AuditPolicy

                    policy = AuditPolicy.load(policy_path).to_dict()
            except (OSError, ValueError) as exc:
                raise McpToolError("invalid_policy", str(exc)) from exc
            policy_source = "project"
        else:
            policy = {
                "schema_version": 1,
                "mode": "coach",
                "capture": "auto",
                "hermetic": True,
                "privacy": "local-redacted",
                "persist_raw_content": False,
                "quality": {"verifier": ""},
                "enforcement": {"enabled": False},
            }
            policy_source = "defaults"
            warnings.append("tracerazor.toml missing; non-enforcing coach defaults apply")

        if not isinstance(policy, dict):
            raise McpToolError("invalid_policy", "tracerazor.toml must contain a table")
        quality_policy = policy.get("quality") if isinstance(policy.get("quality"), dict) else {}
        enforcement_policy = (
            policy.get("enforcement") if isinstance(policy.get("enforcement"), dict) else {}
        )
        mode = str(policy.get("mode", "coach")).lower()
        verifier = quality_policy.get("verifier")
        verifier_present = isinstance(verifier, str) and bool(verifier.strip())
        enforcement_requested = mode == "enforce" and enforcement_policy.get("enabled") is True

        report: dict[str, Any] | None = None
        selected = run_id
        evidence: str | None = _evidence_ref(policy_path, root) if policy_path.is_file() else None
        try:
            report_path, selected, fallback_warnings = _run_artifact(
                run_id, "report.json", root, claude_field="report"
            )
            artifact = _read_json(report_path)
            if isinstance(artifact, dict):
                report = artifact
                warnings.extend(fallback_warnings)
        except McpToolError as report_error:
            if report_error.code not in {"artifact_not_found", "path_not_found"}:
                raise
            try:
                manifest_path, selected, manifest_warnings = _run_artifact(
                    run_id, "manifest.json", root
                )
                artifact = _read_json(manifest_path)
                if not isinstance(artifact, dict):
                    raise McpToolError("invalid_manifest", "manifest.json must be an object")
                report = artifact
                warnings.extend(manifest_warnings)
                warnings.append("report.json missing; using manifest quality metadata")
            except McpToolError as manifest_error:
                if manifest_error.code not in {"artifact_not_found", "path_not_found"}:
                    raise
                if run_id:
                    raise report_error
                warnings.append("no report or manifest available for enforcement-quality checks")

        quality = _quality_from(report)
        estimate_status = _estimate_status_from(report)
        reasons: list[str] = []
        if not verifier_present:
            reasons.append("quality.verifier is missing")
        if _quality_is_degraded(quality):
            reasons.append("ingest quality is degraded")
        if estimate_status in {"estimated", "mixed"}:
            reasons.append("usage contains estimated token counts")
        elif estimate_status == "missing":
            reasons.append("usage token counts are missing")
        elif estimate_status == "unknown":
            reasons.append("usage provenance is unknown")

        validation_artifact: dict[str, Any] | None = None
        if selected:
            run_dir, _ = _resolve_run_dir(selected, root)
            validation_path = run_dir / "validation.json"
            if validation_path.is_file() and not _is_link(validation_path):
                loaded_validation = _read_json(validation_path)
                if not isinstance(loaded_validation, dict):
                    reasons.append("task validation evidence is invalid")
                else:
                    validation_artifact = loaded_validation
            else:
                reasons.append("task validation evidence is missing")
        else:
            reasons.append("task validation evidence is missing")

        if validation_artifact is not None:
            if validation_artifact.get("trust_level") != "trusted_executed_verifier":
                reasons.append("trusted executed verifier receipt is missing")
            task_validation = validation_artifact.get("task")
            if not isinstance(task_validation, dict):
                reasons.append("task validation result is missing")
            else:
                recorded_outcome = str(task_validation.get("outcome", "unknown")).lower()
                if recorded_outcome != "passed":
                    reasons.append("task verifier did not pass")
                recorded_verifier = task_validation.get("verifier")
                if not isinstance(recorded_verifier, str) or recorded_verifier.strip() != str(
                    verifier or ""
                ).strip():
                    reasons.append("recorded verifier does not match quality.verifier")
            if validation_artifact.get("enforcement_eligible") is False:
                runtime_reasons = validation_artifact.get("ineligible_reasons")
                if isinstance(runtime_reasons, list):
                    reasons.extend(
                        f"runtime validation: {item}"
                        for item in runtime_reasons
                        if isinstance(item, str) and item
                    )
                else:
                    reasons.append("runtime validation marked enforcement ineligible")
        if isinstance(report, dict) and report.get("enforcement_eligible") is False:
            report_reasons = report.get("enforcement_ineligible_reasons")
            if isinstance(report_reasons, list):
                reasons.extend(
                    f"run manifest: {item}"
                    for item in report_reasons
                    if isinstance(item, str) and item
                )
            else:
                reasons.append("run manifest marked enforcement ineligible")
        reasons = list(dict.fromkeys(reasons))
        enforce_eligible = not reasons
        if enforcement_requested and not enforce_eligible:
            warnings.append("enforcement refused: " + "; ".join(reasons))

        data = {
            "source": policy_source,
            "policy_path": _evidence_ref(policy_path, root) if policy_path.is_file() else None,
            "policy": _json_safe(policy),
            "mode": mode,
            "enforcement_requested": enforcement_requested,
            "enforce_eligible": enforce_eligible,
            "refusal_reasons": reasons,
            "verifier_present": verifier_present,
            "validation_recorded": validation_artifact is not None,
        }
        return _envelope(
            data,
            run_id=selected,
            warnings=warnings,
            evidence_ref=evidence,
            metadata_source=report,
        )
    except Exception as exc:
        return _error_envelope(exc, run_id=run_id)


def verify_evidence(
    report_path: str, trace_path: str | None = None, cwd: str = "."
) -> dict[str, Any]:
    """Versioned-envelope counterpart to the legacy ``verify_report`` tool."""
    try:
        root = _workspace_root(cwd)
        report = _safe_path(report_path, root, must_exist=True, kind="file")
        trace = (
            _safe_path(trace_path, root, must_exist=True, kind="file")
            if trace_path is not None
            else None
        )
        if _is_redacted_non_replayable(report, trace):
            verdict = _non_replayable_verdict()
            return _envelope(
                {
                    "verified": False,
                    "tampered": False,
                    "status": "non_replayable_redacted",
                    "exit_code": 0,
                    "verdict": verdict,
                },
                warnings=[
                    "local-redacted evidence is intentionally non-replayable; use its run receipt or retain raw content by explicit policy"
                ],
                evidence_ref=_evidence_ref(report, root),
                metadata_source=verdict,
            )
        binary = _resolve_binary()
        verdict, exit_code = _verify_result(binary, report, trace)
        if exit_code not in (0, 1):
            raise McpToolError(
                "verify_failed",
                str(verdict.get("stderr") or "verification failed"),
                details={"exit_code": exit_code},
            )
        warnings = ["evidence is tampered or mismatched"] if exit_code == 1 else []
        data = {"verified": exit_code == 0, "exit_code": exit_code, "verdict": verdict}
        return _envelope(
            data,
            run_id=_run_id_from(verdict),
            warnings=warnings,
            evidence_ref=_evidence_ref(report, root),
            metadata_source=verdict,
        )
    except Exception as exc:
        return _error_envelope(exc)


# Ordered source of truth for registration and --selftest.
TOOL_SPECS = [
    ("audit_trace", audit_trace, "Audit a trace hermetically and return the 1.x report shape with additive TraceRazor metadata."),
    ("convert_transcript", convert_transcript, "Normalize an external trace export into TraceRazor trace JSON."),
    ("list_claude_sessions", list_claude_sessions, "List audited Claude Code sessions using the legacy list shape."),
    ("verify_report", verify_report, "Re-verify a report or evidence bundle using the legacy verdict shape."),
    ("doctor", doctor, "Inspect binary, policy, and artifact readiness without requiring the MCP SDK."),
    ("audit_current_run", audit_current_run, "Audit the selected local run, with Claude session fallback."),
    ("latest_findings", latest_findings, "Return findings for the latest or selected local run."),
    ("compare_runs", compare_runs, "Compare two run ids or trace paths using native JSON output."),
    ("explain_signal", explain_signal, "Explain a TraceRazor signal and attach run-specific evidence when available."),
    ("preview_fix", preview_fix, "Preview safe fix application with apply --dry-run; never writes the target."),
    ("record_validation", record_validation, "Record advisory, untrusted validation metadata under an existing run; this never authorizes enforcement."),
    ("check_policy", check_policy, "Evaluate project policy and enforcement eligibility for a run."),
    ("verify_evidence", verify_evidence, "Verify evidence using the stable tracerazor-mcp/v1 envelope."),
]


def _tool_catalog() -> list[dict[str, str]]:
    return [{"name": name, "description": desc} for name, _fn, desc in TOOL_SPECS]


def _bind_server_root() -> Path:
    global _SERVER_ROOT
    configured = os.environ.get("TRACERAZOR_MCP_ROOT") or os.path.abspath(os.getcwd())
    candidate = Path(configured)
    if ".." in candidate.parts:
        raise McpToolError("path_traversal", "MCP server root may not contain '..'")
    candidate = Path(os.path.abspath(candidate))
    _reject_link_components(candidate)
    if not candidate.is_dir():
        raise McpToolError("workspace_not_found", "MCP server root does not exist")
    _SERVER_ROOT = candidate.resolve(strict=True)
    return _SERVER_ROOT


def _build_server():
    """Construct FastMCP lazily so SDK-free inspection keeps working."""
    # Bind the process to the host-selected startup directory before exposing
    # any model-controlled ``cwd`` argument. A tool may select a subdirectory,
    # but it cannot retarget this server at an unrelated checkout.
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("tracerazor")
    for name, fn, desc in TOOL_SPECS:
        server.tool(name=name, description=desc)(fn)
    return server


def _selftest() -> int:
    try:
        _build_server()
    except ImportError:
        print(
            json.dumps(
                {
                    "error": "the MCP SDK is not installed",
                    "install": 'pip install "tracerazor[mcp]"',
                }
            )
        )
        return 1
    print(json.dumps(_tool_catalog(), indent=2))
    return 0


def main() -> int:
    if "--selftest" in sys.argv[1:]:
        return _selftest()
    try:
        _bind_server_root()
        server = _build_server()
    except (ImportError, McpToolError) as exc:
        sys.stderr.write(
            "tracerazor-mcp: could not start the local server: "
            f"{exc}. Install the SDK with: pip install \"tracerazor[mcp]\"\n"
        )
        return 1
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
