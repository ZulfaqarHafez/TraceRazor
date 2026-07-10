"""Atomic, privacy-aware local event spooling."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import time
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping

from .models import PrivacyMode, RuntimeEvent, content_digest, utc_now
from .policy import AuditPolicy


_ARTIFACT_NAMES = {
    "manifest.json",
    "events.jsonl",
    "trace.json",
    "report.json",
    "findings.json",
    "validation.json",
}

_TERMINAL_RUN_STATUSES = {"completed", "partial", "cancelled", "error"}
_RECEIPT_NAME_RE = re.compile(r"^[0-9a-f]{16}\.json$")
_METRIC_CODES = {
    "srr", "ldi", "tca", "rda", "isr", "tur", "cce", "dbo",
    "vdi", "shl", "ccr", "csd", "gar", "obs", "avs",
}
_KNOWN_FIX_TYPES = {
    "tool_schema",
    "prompt_insert",
    "termination_guard",
    "context_compression",
    "verbosity_reduction",
    "hedge_reduction",
    "caveman_prompt_insert",
    "reformulation_guard",
    "goal_anchor",
    "dedupe",
}

# Keys in runtime-owned artifact envelopes are public structure and need to
# remain readable.  All other mapping keys are untrusted user content (tool
# arguments, verifier evidence, metadata, excerpts, and so on), so redaction
# must cover keys as well as values.  Secrets are sometimes supplied as map
# keys by hostile or simply unusual tool output.
_SAFE_STRUCTURE_KEYS = {
    "schema_version",
    "run_id",
    "recorded_at",
    "findings",
    "finding_id",
    "detector_id",
    "signal",
    "severity",
    "summary",
    "recommendation",
    "status",
    "task",
    "outcome",
    "verifier",
    "passed",
    "score",
    "evidence",
    "metadata",
    "enforcement_eligible",
    "ineligible_reasons",
    "error",
    "type",
    "detail",
}


def redacted_value(value: str) -> str:
    """Represent sensitive content without retaining any substring of it."""

    return f"[redacted sha256={content_digest(value)} chars={len(value)}]"


def _redacted_bytes(value: bytes) -> str:
    return f"[redacted-bytes sha256={sha256(value).hexdigest()} bytes={len(value)}]"


def _redacted_key(value: Any) -> str:
    text = str(value)
    return f"[redacted-key sha256={content_digest(text)} chars={len(text)}]"


def _redact_nested(value: Any, *, preserve_structure: bool = False) -> Any:
    if isinstance(value, str):
        return redacted_value(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _redacted_bytes(bytes(value))
    if isinstance(value, Mapping):
        return {
            (
                str(key)
                if preserve_structure and str(key) in _SAFE_STRUCTURE_KEYS
                else _redacted_key(key)
            ): _redact_nested(item, preserve_structure=preserve_structure)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_nested(item, preserve_structure=preserve_structure) for item in value]
    if isinstance(value, tuple):
        return [_redact_nested(item, preserve_structure=preserve_structure) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    # ``json.dumps(default=str)`` would otherwise persist arbitrary object
    # representations verbatim.  Hash the representation in memory and emit
    # only a type-neutral marker.  A hostile ``__str__`` implementation may
    # fail, so keep a fail-closed fallback that never serializes the object.
    try:
        rendered = str(value)
    except Exception:  # pragma: no cover - defensive custom-object boundary
        rendered = f"<{type(value).__module__}.{type(value).__qualname__}>"
    return redacted_value(rendered)


def event_for_persistence(event: RuntimeEvent, policy: AuditPolicy) -> dict[str, Any]:
    value = event.to_dict()
    value["capture"]["privacy"] = policy.privacy.value
    if policy.persist_raw_content:
        return value

    for field_name in ("content", "input_context", "output"):
        if value[field_name] is not None:
            value[field_name] = redacted_value(value[field_name])
    value["metadata"] = _redact_nested(value["metadata"])
    if value["task"] is not None:
        value["task"]["evidence"] = _redact_nested(value["task"]["evidence"])
        if value["task"].get("verifier") is not None:
            value["task"]["verifier"] = redacted_value(value["task"]["verifier"])
    if value["tool"] is not None and value["tool"].get("error_type") is not None:
        value["tool"]["error_type"] = redacted_value(value["tool"]["error_type"])
    return value


def artifact_for_persistence(value: Any, policy: AuditPolicy) -> Any:
    """Redact caller-provided findings/validation structures by policy."""

    if policy.persist_raw_content:
        return value
    return _redact_nested(value, preserve_structure=True)


def native_trace_for_persistence(trace: Mapping[str, Any], policy: AuditPolicy) -> dict[str, Any]:
    value = json.loads(json.dumps(trace, ensure_ascii=False, default=str))
    source_trace_sha256 = sha256(
        json.dumps(trace, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode(
            "utf-8"
        )
    ).hexdigest()
    metadata = value.setdefault("metadata", {})
    if policy.persist_raw_content:
        # Raw evidence must remain byte/schema faithful to the exact native
        # audit input so ``tracerazor verify report.json trace.json`` can replay
        # it.  Persistence annotations belong in the run manifest.
        return value
    for step in value.get("steps", []):
        for field_name in ("content", "input_context", "output", "tool_error"):
            raw = step.get(field_name)
            if isinstance(raw, str):
                step[field_name] = redacted_value(raw)
        if "tool_params" in step:
            step["tool_params"] = _redact_nested(step["tool_params"])
    if isinstance(metadata, Mapping):
        # Preserve structural identifiers while redacting any free-form task or
        # objective text that may have been copied into metadata.
        for key in ("task", "goal", "objective"):
            if isinstance(metadata.get(key), str):
                metadata[key] = redacted_value(metadata[key])
        issues = list(metadata.get("capture_issues") or [])
        if "redacted_persisted_trace" not in issues:
            issues.append("redacted_persisted_trace")
        metadata.update(
            {
                "tracerazor_redacted": True,
                "source_trace_sha256": source_trace_sha256,
                "persisted_representation": "redacted_non_auditable",
                "capture_quality": "degraded",
                "degraded_ingest": True,
                "capture_issues": issues,
            }
        )
    return value


def _safe_number(value: Any, default: int | float = 0) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return value


def _safe_code(value: Any, *, allowed: set[str] | None = None) -> str:
    text = str(value or "")
    if allowed is not None:
        return text if text in allowed else "redacted"
    return text if re.fullmatch(r"[a-zA-Z0-9_.:-]{1,80}", text) else "redacted"


def report_for_persistence(
    report: Mapping[str, Any],
    policy: AuditPolicy,
    *,
    source_trace_sha256: str,
) -> dict[str, Any]:
    """Return a useful report artifact without persisting audit excerpts.

    The native report is allowed to contain step descriptions, grounding
    literals, generated patches, and arbitrary extension keys.  The default
    runtime artifact is therefore an allow-listed projection.  Its scores and
    counts remain usable, while prose, excerpts, tool errors, and patches are
    represented only by fixed privacy markers.
    """

    raw = json.loads(json.dumps(report, ensure_ascii=False, default=str))
    if policy.persist_raw_content:
        # Do not mutate signed/verifiable native reports in raw mode.
        return raw

    score_raw = raw.get("score") if isinstance(raw.get("score"), Mapping) else {}
    score: dict[str, Any] = {}
    for key in ("score", "raw_tas", "task_value_score", "vae", "passes_threshold", "avs"):
        value = score_raw.get(key)
        if value is None or isinstance(value, (bool, int, float)):
            score[key] = value
    metric_normalised = score_raw.get("metric_normalised")
    score["metric_normalised"] = {
        key: value
        for key, value in (metric_normalised.items() if isinstance(metric_normalised, Mapping) else ())
        if key in _METRIC_CODES and isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for metric_code in _METRIC_CODES:
        metric = score_raw.get(metric_code)
        if isinstance(metric, Mapping):
            score[metric_code] = {
                key: value
                for key, value in metric.items()
                if key in {"score", "pass", "target"}
                and (value is None or isinstance(value, (bool, int, float)))
            }
    score["grade"] = _safe_code(
        score_raw.get("grade"), allowed={"Excellent", "Good", "Fair", "Poor"}
    )

    diff: list[dict[str, Any]] = []
    for line in raw.get("diff") or []:
        if not isinstance(line, Mapping):
            continue
        diff.append(
            {
                "action": _safe_code(line.get("action"), allowed={"keep", "delete", "trim"}),
                "step_id": int(_safe_number(line.get("step_id"), 0)),
                "step_type": _safe_code(line.get("step_type")),
                "description": "[redacted audit description]",
                "justification": "[redacted audit justification]",
                "tokens_actual": int(_safe_number(line.get("tokens_actual"), 0)),
                "tokens_suggested": (
                    None
                    if line.get("tokens_suggested") is None
                    else int(_safe_number(line.get("tokens_suggested"), 0))
                ),
            }
        )

    fixes: list[dict[str, Any]] = []
    for fix in raw.get("fixes") or []:
        if not isinstance(fix, Mapping):
            continue
        fixes.append(
            {
                "fix_type": (
                    str(fix.get("fix_type"))
                    if str(fix.get("fix_type")) in _KNOWN_FIX_TYPES
                    else "redacted"
                ),
                "target": "[redacted fix target]",
                "patch": "[redacted fix patch]",
                "estimated_token_savings": int(
                    _safe_number(fix.get("estimated_token_savings"), 0)
                ),
                "risk": _safe_code(
                    fix.get("risk"), allowed={"safe", "needs_review", "dangerous", "redacted"}
                ),
            }
        )

    manifest_raw = raw.get("manifest") if isinstance(raw.get("manifest"), Mapping) else {}
    ingest_raw = (
        manifest_raw.get("ingest_quality")
        if isinstance(manifest_raw.get("ingest_quality"), Mapping)
        else {}
    )
    manifest = {
        "trace_sha256": _safe_code(manifest_raw.get("trace_sha256")),
        "tool_version": _safe_code(manifest_raw.get("tool_version")),
        "created_at": _safe_code(manifest_raw.get("created_at")),
        "similarity_backend": _safe_code(manifest_raw.get("similarity_backend")),
        "weights": {
            key: value
            for key, value in (
                manifest_raw.get("weights", {}).items()
                if isinstance(manifest_raw.get("weights"), Mapping)
                else ()
            )
            if key in _METRIC_CODES and isinstance(value, (int, float)) and not isinstance(value, bool)
        },
        "weights_sha256": _safe_code(manifest_raw.get("weights_sha256")),
        "threshold": _safe_number(manifest_raw.get("threshold"), 0),
        "cost_per_million_tokens": _safe_number(
            manifest_raw.get("cost_per_million_tokens"), 0
        ),
        "min_steps": int(_safe_number(manifest_raw.get("min_steps"), 0)),
        "hermetic": manifest_raw.get("hermetic") is True,
        "baseline_tokens": manifest_raw.get("baseline_tokens")
        if isinstance(manifest_raw.get("baseline_tokens"), (int, type(None)))
        else None,
        "historical_median_steps": _safe_number(
            manifest_raw.get("historical_median_steps"), 0
        ),
        "n_historical_sequences": int(
            _safe_number(manifest_raw.get("n_historical_sequences"), 0)
        ),
        "ingest_quality": {
            key: value
            for key, value in ingest_raw.items()
            if key
            in {
                "content_coverage",
                "degraded",
                "degraded_ingest",
                "placeholder_content_pct",
                "step_count",
                "token_coverage",
                "zero_token_pct",
            }
            and (value is None or isinstance(value, (bool, int, float)))
        },
        "signature": _safe_code(manifest_raw.get("signature"))
        if manifest_raw.get("signature")
        else None,
        "signing_key_pub": _safe_code(manifest_raw.get("signing_key_pub"))
        if manifest_raw.get("signing_key_pub")
        else None,
    }

    result: dict[str, Any] = {
        "schema_version": "tracerazor-report/v1",
        "trace_id": _safe_code(raw.get("trace_id")),
        "agent_name": "[redacted agent name]",
        "framework": "[redacted framework]",
        "total_steps": int(_safe_number(raw.get("total_steps"), 0)),
        "total_tokens": int(_safe_number(raw.get("total_tokens"), 0)),
        "analysis_duration_ms": int(_safe_number(raw.get("analysis_duration_ms"), 0)),
        "score": score,
        "diff": diff,
        "savings": {
            key: value
            for key, value in (
                raw.get("savings", {}).items()
                if isinstance(raw.get("savings"), Mapping)
                else ()
            )
            if key
            in {
                "tokens_saved",
                "reduction_pct",
                "cost_saved_per_run_usd",
                "monthly_savings_usd",
                "latency_saved_seconds",
                "monthly_runs",
                "monthly_runs_assumed",
            }
            and (value is None or isinstance(value, (bool, int, float)))
        },
        "fixes": fixes,
        "summary": "[redacted audit summary]",
        "summary_oneliner": "[redacted audit summary]",
        "anomalies": [
            {
                key: value
                for key, value in item.items()
                if key in {"value", "z_score", "baseline_mean", "baseline_std"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
            for item in (raw.get("anomalies") or [])
            if isinstance(item, Mapping)
        ],
        "features": {},
        "manifest": manifest,
        "tracerazor_redacted": True,
        "source_trace_sha256": source_trace_sha256,
        "persisted_representation": "redacted_auditor_report",
        "persisted_evidence_quality": {
            "status": "degraded",
            "issues": [
                "redacted_persisted_trace",
                "audit_excerpts_removed",
                "fix_patches_removed",
            ],
        },
    }
    return result


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(info, "st_file_attributes", 0) & reparse)


def reject_link_components(path: str | os.PathLike[str]) -> None:
    """Reject existing symlink/junction components before a local write."""

    candidate = Path(path).absolute()
    components = [candidate]
    components.extend(candidate.parents)
    for component in reversed(components):
        if _is_link_or_reparse(component):
            raise ValueError(f"runtime artifact path contains a symlink or junction: {component}")


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> None:
    """Durably replace an artifact with exact bytes."""

    destination = Path(path)
    reject_link_components(destination)
    reject_link_components(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reject_link_components(destination.parent)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Durably replace a JSON artifact without exposing a partial file."""

    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n").encode(
        "utf-8"
    )
    atomic_write_bytes(path, payload)


@contextmanager
def _exclusive_lock(
    lock_path: Path,
    *,
    timeout: float = 10.0,
    stale_after: float = 30.0,
) -> Iterator[None]:
    """Portable process lock based on atomic file creation."""

    deadline = time.monotonic() + timeout
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                os.write(fd, f"{os.getpid()} {time.time()}\n".encode("ascii"))
            finally:
                os.close(fd)
            break
        except (FileExistsError, PermissionError):
            # Windows may surface O_EXCL contention as PermissionError while
            # another thread/process owns the lock file.  The owner may unlink
            # it between this exception and our retry, so existence is not a
            # reliable discriminator here; a real permission problem cleanly
            # becomes the timeout below.
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_after:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for event spool lock {lock_path}")
            time.sleep(0.005)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


class DiskSpoolReceiver:
    """A local-only, append-only receiver for runtime events.

    No socket is opened and no network service is contacted.  Multiple threads
    or processes may append to the same spool.  A crash-truncated final line is
    reported as partial and ignored during recovery; malformed interior lines
    remain hard errors.
    """

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        *,
        policy: AuditPolicy | None = None,
        lock_timeout: float = 10.0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "events.jsonl"
        self.lock_path = self.run_dir / ".events.lock"
        self.policy = policy or AuditPolicy()
        self.lock_timeout = lock_timeout
        self.partial_tail = False

    @contextmanager
    def locked(self, *, stale_after: float = 300.0) -> Iterator[None]:
        reject_link_components(self.run_dir)
        with _exclusive_lock(
            self.lock_path,
            timeout=self.lock_timeout,
            stale_after=stale_after,
        ):
            yield

    def _terminal_manifest_status(self) -> str | None:
        try:
            value = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        status = value.get("status")
        return str(status) if status in _TERMINAL_RUN_STATUSES else None

    def receive(self, event: RuntimeEvent | Mapping[str, Any]) -> None:
        if not isinstance(event, RuntimeEvent):
            event = RuntimeEvent.from_dict(event)
        payload = event_for_persistence(event, self.policy)
        line = (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
        with self.locked(stale_after=300.0):
            terminal = self._terminal_manifest_status()
            if terminal is not None:
                raise RuntimeError(f"cannot append to finalized TraceRazor run ({terminal})")
            reject_link_components(self.path)
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(self.path, flags, 0o600)
            try:
                offset = 0
                while offset < len(line):
                    written = os.write(fd, line[offset:])
                    if written <= 0:  # pragma: no cover - defensive OS boundary
                        raise OSError("event spool append made no progress")
                    offset += written
                os.fsync(fd)
            finally:
                os.close(fd)

    append = receive

    def records(
        self,
        *,
        allow_partial_tail: bool = True,
        assume_locked: bool = False,
    ) -> list[dict[str, Any]]:
        if not assume_locked:
            with self.locked(stale_after=300.0):
                return self.records(allow_partial_tail=allow_partial_tail, assume_locked=True)
        self.partial_tail = False
        reject_link_components(self.path)
        try:
            payload = self.path.read_bytes()
        except FileNotFoundError:
            return []
        lines = payload.splitlines(keepends=True)
        records: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            is_last = index == len(lines) - 1
            try:
                decoded = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if allow_partial_tail and is_last and not line.endswith((b"\n", b"\r")):
                    self.partial_tail = True
                    continue
                raise ValueError(f"malformed event spool record at line {index + 1}") from exc
            if not isinstance(decoded, dict):
                raise ValueError(f"event spool record at line {index + 1} is not an object")
            records.append(decoded)
        return records

    def events(self, *, allow_partial_tail: bool = True) -> list[RuntimeEvent]:
        return [RuntimeEvent.from_dict(record) for record in self.records(allow_partial_tail=allow_partial_tail)]

    def write_receipt(self, name: str, value: Mapping[str, Any], *, assume_locked: bool = False) -> Path:
        if not _RECEIPT_NAME_RE.fullmatch(name):
            raise ValueError("receipt name must be a W3C span ID plus .json")
        if not assume_locked:
            with self.locked(stale_after=300.0):
                return self.write_receipt(name, value, assume_locked=True)
        directory = self.run_dir / "receipts"
        reject_link_components(directory)
        path = directory / name
        if path.parent != directory or path.name != name:
            raise ValueError("receipt path escapes the run receipt directory")
        atomic_write_json(path, value)
        return path

    def receipts(self, *, assume_locked: bool = False) -> list[dict[str, Any]]:
        if not assume_locked:
            with self.locked(stale_after=300.0):
                return self.receipts(assume_locked=True)
        directory = self.run_dir / "receipts"
        reject_link_components(directory)
        if not directory.exists():
            return []
        result: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            if not _RECEIPT_NAME_RE.fullmatch(path.name) or _is_link_or_reparse(path):
                raise ValueError(f"unsafe run receipt path: {path}")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"run receipt is not an object: {path}")
            result.append(value)
        return result

    def write_artifact(self, name: str, value: Any, *, redact: bool = False) -> Path:
        if name not in _ARTIFACT_NAMES or name == "events.jsonl":
            raise ValueError(f"unsupported or unsafe runtime artifact name: {name!r}")
        if redact:
            value = artifact_for_persistence(value, self.policy)
        path = self.run_dir / name
        atomic_write_json(path, value)
        return path

    def write_artifact_bytes(self, name: str, value: bytes) -> Path:
        if name != "trace.json":
            raise ValueError("exact-byte runtime writes are restricted to trace.json")
        path = self.run_dir / name
        atomic_write_bytes(path, value)
        return path


def recover_partial_run(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Mark an interrupted ``running`` manifest partial and return it."""

    path = Path(run_dir) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot recover run manifest {path}: {exc}") from exc
    if manifest.get("status") == "running":
        manifest["status"] = "partial"
        manifest["capture_quality"] = "partial"
        manifest["degraded_ingest"] = True
        manifest["enforcement_eligible"] = False
        reasons = list(manifest.get("enforcement_ineligible_reasons") or [])
        if "interrupted_run" not in reasons:
            reasons.append("interrupted_run")
        manifest["enforcement_ineligible_reasons"] = reasons
        manifest["ended_at"] = utc_now()
        atomic_write_json(path, manifest)
    return manifest


__all__ = [
    "DiskSpoolReceiver",
    "artifact_for_persistence",
    "atomic_write_bytes",
    "atomic_write_json",
    "event_for_persistence",
    "native_trace_for_persistence",
    "reject_link_components",
    "report_for_persistence",
    "recover_partial_run",
    "redacted_value",
]
