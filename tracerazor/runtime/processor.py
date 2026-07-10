"""Run lifecycle and processor API for automatic local capture."""

from __future__ import annotations

import os
import json
import re
import subprocess
import tempfile
import threading
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from tracerazor._launcher import find_binary
from tracerazor.errors import AuditError, BelowMinStepsError, BinaryNotFoundError

from .compiler import NoAuditableEventsError, compile_native_trace
from .guardrails import GuardrailConfig, StreamingGuardrailDetector
from .models import (
    CaptureInfo,
    CaptureQuality,
    RunContext,
    RuntimeEvent,
    RUN_SCHEMA_VERSION,
    TaskResult,
    TokenProvenance,
    TokenUsage,
    ToolCall,
    new_span_id,
    run_capture_quality,
    utc_now,
)
from .persistence import (
    DiskSpoolReceiver,
    artifact_for_persistence,
    native_trace_for_persistence,
    reject_link_components,
    report_for_persistence,
)
from .policy import AuditPolicy


class RuntimeAuditor(Protocol):
    """Dependency-injection boundary for the native hermetic auditor."""

    def __call__(
        self,
        trace: Mapping[str, Any],
        *,
        hermetic: bool,
        min_steps: int,
    ) -> Mapping[str, Any]: ...


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _audit_with_installed_binary(
    trace: Mapping[str, Any],
    *,
    hermetic: bool,
    min_steps: int,
) -> Mapping[str, Any]:
    """Audit an in-memory trace with the resolved native CLI.

    No absolute TAS threshold is passed: runtime finalization diagnoses a run;
    it does not turn an ordinal score into a universal quality gate.
    """

    binary = find_binary()
    if binary is None:
        raise BinaryNotFoundError("installed TraceRazor native auditor was not found")
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as handle:
            handle.write(_canonical_json_bytes(trace))
            temp_name = handle.name
        command = [binary, "audit", temp_name, "--format", "json", "--min-steps", str(min_steps)]
        if hermetic:
            command.append("--hermetic")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            raise AuditError(f"native auditor exited with code {result.returncode}")
        if not result.stdout.strip():
            raise BelowMinStepsError("native auditor returned no report")
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise AuditError("native auditor returned invalid JSON") from exc
        if not isinstance(report, Mapping):
            raise AuditError("native auditor report is not an object")
        return report
    except subprocess.TimeoutExpired as exc:
        raise AuditError("native auditor timed out") from exc
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


class TraceRazorProcessor:
    """Collect, validate, spool, and finalize one agent run.

    A processor writes a ``running`` manifest immediately, so an abrupt process
    exit remains observable and can be marked partial on the next startup.
    Raw content stays only in this object's in-memory event list under the
    default ``local-redacted`` policy.
    """

    def __init__(
        self,
        *,
        context: RunContext | None = None,
        policy: AuditPolicy | None = None,
        artifact_dir: str | os.PathLike[str] | None = None,
        policy_path: str | os.PathLike[str] | None = None,
        agent_id: str = "agent",
        host: str = "unknown",
        host_version: str | None = None,
        framework: str = "unknown",
        framework_version: str | None = None,
        receiver: DiskSpoolReceiver | None = None,
        guardrails: StreamingGuardrailDetector | GuardrailConfig | None = None,
        auditor: RuntimeAuditor | None = None,
    ) -> None:
        self.policy = policy or AuditPolicy()
        self.context = context or RunContext.from_env(agent_id=agent_id)
        self.host = host
        self.host_version = host_version
        self.framework = framework
        self.framework_version = framework_version
        self.policy_path = Path(policy_path).resolve() if policy_path is not None else None
        requested_base = Path(artifact_dir if artifact_dir is not None else self.policy.artifact_dir)
        if ".." in requested_base.parts:
            raise ValueError("runtime artifact_dir must not contain parent-directory traversal")
        if self.policy_path is not None:
            policy_root = self.policy_path.parent
            if requested_base.is_absolute():
                try:
                    requested_base.relative_to(policy_root)
                except ValueError as exc:
                    raise ValueError("runtime artifact_dir escapes the policy root") from exc
                base = requested_base
            else:
                base = policy_root / requested_base
        else:
            base = requested_base if requested_base.is_absolute() else Path.cwd() / requested_base
        reject_link_components(base)
        base = base.resolve(strict=False)
        self.run_dir = base / self.context.run_id
        try:
            self.run_dir.relative_to(base)
        except ValueError as exc:  # pragma: no cover - run_id validates first
            raise ValueError("runtime run directory escapes artifact_dir") from exc
        self.receiver = receiver or DiskSpoolReceiver(self.run_dir, policy=self.policy)
        if self.receiver.run_dir.resolve() != self.run_dir.resolve():
            raise ValueError("receiver run directory does not match processor run directory")
        if isinstance(guardrails, StreamingGuardrailDetector):
            self.guardrails = guardrails
        else:
            self.guardrails = StreamingGuardrailDetector(guardrails)
        if self.guardrails.run_id not in {None, self.context.run_id}:
            raise ValueError("guardrail detector run_id does not match processor context")
        self._events: list[RuntimeEvent] = []
        self._analysis_events: list[RuntimeEvent] | None = None
        self._task: TaskResult | None = None
        self._sequence = 0
        self._lock = threading.RLock()
        self._started_at = utc_now()
        self._ended_at: str | None = None
        self._status = "running"
        self._finalized = False
        self._last_trace: dict[str, Any] | None = None
        self._last_receipt: dict[str, Any] | None = None
        self._child_receipts: list[dict[str, Any]] = []
        self._audit_status = "not_started"
        self._audit_issues: list[str] = []
        self._audit_error: BaseException | str | None = None
        self._source_trace_sha256: str | None = None
        self._raw_trace_bytes: bytes | None = None
        self._auditor = auditor or _audit_with_installed_binary
        if self.policy.captures and not self.is_child:
            self._initialize_manifest()

    @property
    def is_child(self) -> bool:
        return self.context.parent_agent_id is not None

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def status(self) -> str:
        return self._status

    @property
    def native_trace(self) -> dict[str, Any] | None:
        return self._last_trace

    @property
    def receipt(self) -> dict[str, Any] | None:
        return self._last_receipt

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeError("cannot add events to a finalized TraceRazor run")

    def _validate_event_context(self, event: RuntimeEvent) -> None:
        if event.run_id != self.context.run_id:
            raise ValueError("event run_id does not match processor context")
        if event.trace_id != self.context.trace_id:
            raise ValueError("event trace_id does not match processor context")
        if event.session_id != self.context.session_id:
            raise ValueError("event session_id does not match processor context")

    def emit(self, event: RuntimeEvent | Mapping[str, Any]) -> RuntimeEvent:
        """Validate and durably spool one event."""

        with self._lock:
            self._ensure_open()
            if not isinstance(event, RuntimeEvent):
                event = RuntimeEvent.from_dict(event)
            self._validate_event_context(event)
            if event.sequence == 0:
                event = replace(event, sequence=self._sequence + 1)
            self._sequence = max(self._sequence, event.sequence)
            if any(existing.event_id == event.event_id for existing in self._events):
                raise ValueError(f"duplicate event_id in run: {event.event_id}")
            if self.policy.captures:
                self.receiver.receive(event)
            self._events.append(event)
            self.guardrails.observe(event)
            if event.task is not None:
                self._task = event.task
            if self.policy.captures and not self.is_child:
                self._write_manifest()
            return event

    receive = emit
    process = emit

    def record(
        self,
        event_type: str,
        *,
        tokens: TokenUsage | None = None,
        tool: ToolCall | None = None,
        task: TaskResult | None = None,
        capture: CaptureInfo | None = None,
        content: str | None = None,
        input_context: str | None = None,
        output: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> RuntimeEvent:
        event = RuntimeEvent.create(
            self.context,
            event_type=event_type,
            host=self.host,
            host_version=self.host_version,
            framework=self.framework,
            framework_version=self.framework_version,
            tokens=tokens or TokenUsage(),
            tool=tool,
            task=task,
            capture=capture or CaptureInfo(privacy=self.policy.privacy),
            content=content,
            input_context=input_context,
            output=output,
            metadata=metadata or {},
            span_id=span_id or new_span_id(),
            parent_span_id=parent_span_id or self.context.span_id,
        )
        return self.emit(event)

    def spawn_env(
        self,
        *,
        child_agent_id: str | None = None,
        base: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        return self.context.spawn_env(
            child_agent_id=child_agent_id,
            policy_path=self.policy_path,
            base=base,
        )

    def _manifest_events(self) -> list[RuntimeEvent]:
        return self._analysis_events if self._analysis_events is not None else self._events

    def _token_coverage(self, events: Sequence[RuntimeEvent]) -> float:
        auditable = [event for event in events if event.auditable]
        if not auditable:
            return 0.0
        exact = sum(event.tokens.provenance is TokenProvenance.PROVIDER_REPORTED for event in auditable)
        return exact / len(auditable)

    def _manifest(self) -> dict[str, Any]:
        events = self._manifest_events()
        partial = self._status in {"partial", "cancelled", "error"}
        quality = run_capture_quality(events, partial=partial)
        eligible, reasons = self.policy.enforcement_eligibility(
            events,
            self._task,
            partial=partial,
        )
        issues = sorted(
            {
                issue
                for event in events
                for issue in event.effective_capture.issues
            }
        )
        issues = sorted(set(issues).union(self._audit_issues))
        if self._audit_status != "completed":
            eligible = False
            reasons = list(reasons) + ["audit_not_completed"]
        if self.receiver.partial_tail and "truncated_event_spool" not in issues:
            issues.append("truncated_event_spool")
            quality = CaptureQuality.PARTIAL
            eligible = False
            reasons = list(reasons) + ["truncated_event_spool"]
        files = ["manifest.json"]
        for name in ("events.jsonl", "trace.json", "findings.json", "validation.json", "report.json"):
            if (self.run_dir / name).exists():
                files.append(name)
        receipt_files = []
        receipt_dir = self.run_dir / "receipts"
        if receipt_dir.exists():
            receipt_files = [f"receipts/{path.name}" for path in sorted(receipt_dir.glob("*.json"))]
            files.extend(receipt_files)
        agent_ids = sorted({event.agent_id for event in events})
        child_agent_ids = sorted(
            {
                event.agent_id
                for event in events
                if event.parent_agent_id is not None
            }
        )
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": self._status,
            "run_id": self.context.run_id,
            "trace_id": self.context.trace_id,
            "session_id": self.context.session_id,
            "agent_id": self.context.agent_id,
            "parent_agent_id": self.context.parent_agent_id,
            "host": self.host,
            "host_version": self.host_version,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "agent_ids": agent_ids,
            "child_agent_ids": child_agent_ids,
            "child_receipt_count": len(self._child_receipts),
            "event_count": len(events),
            "step_count": sum(event.auditable for event in events),
            "total_tokens": sum(event.tokens.total for event in events if event.auditable),
            "capture_quality": quality.value,
            "degraded_ingest": quality is not CaptureQuality.COMPLETE,
            "ingest_quality": {
                "status": quality.value,
                "provider_token_coverage": self._token_coverage(events),
                "issues": issues,
            },
            "privacy": self.policy.privacy.value,
            "raw_content_persisted": self.policy.persist_raw_content,
            "policy": {
                "mode": self.policy.mode,
                "hermetic": True,
                "min_steps": self.policy.min_steps,
                "path": os.fspath(self.policy_path) if self.policy_path else None,
            },
            "enforcement_eligible": eligible,
            "enforcement_ineligible_reasons": sorted(set(reasons)),
            "audit": {
                "status": self._audit_status,
                "hermetic": True,
                "absolute_tas_gate_used": False,
                "source_trace_sha256": self._source_trace_sha256,
                "issues": sorted(set(self._audit_issues)),
                "report_available": (self.run_dir / "report.json").exists(),
            },
            "files": sorted(files),
        }

    def _write_manifest(self) -> None:
        self.receiver.write_artifact("manifest.json", self._manifest())

    def _initialize_manifest(self) -> None:
        """Claim root-artifact ownership for the sole parent process."""

        with self.receiver.locked(stale_after=300.0):
            path = self.run_dir / "manifest.json"
            if path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"existing run manifest is unreadable: {path}") from exc
                if existing.get("run_id") == self.context.run_id:
                    raise RuntimeError(
                        "a parent TraceRazor processor already owns this run_id; "
                        "spawned processors must carry parent_agent_id"
                    )
            self._write_manifest()

    def _child_receipt(self) -> dict[str, Any]:
        quality = run_capture_quality(self._events, partial=self._status != "completed")
        return {
            "schema_version": "tracerazor-run-receipt/v1",
            "status": self._status,
            "run_id": self.context.run_id,
            "trace_id": self.context.trace_id,
            "session_id": self.context.session_id,
            "agent_id": self.context.agent_id,
            "parent_agent_id": self.context.parent_agent_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "event_count": len(self._events),
            "step_count": sum(event.auditable for event in self._events),
            "event_ids": sorted(event.event_id for event in self._events),
            "capture_quality": quality.value,
            "degraded_ingest": quality is not CaptureQuality.COMPLETE,
            "raw_content_persisted": self.policy.persist_raw_content,
            "audit_status": "deferred_to_parent",
        }

    def _aggregate_events_locked(self) -> list[RuntimeEvent]:
        own = {event.event_id: event for event in self._events}
        aggregated: dict[str, RuntimeEvent] = {}
        for record in self.receiver.records(assume_locked=True):
            event = RuntimeEvent.from_dict(record)
            self._validate_event_context(event)
            if event.event_id in own:
                aggregated[event.event_id] = own[event.event_id]
                continue
            if not self.policy.persist_raw_content:
                issues = set(event.effective_capture.issues)
                issues.add("redacted_child_content")
                event = replace(
                    event,
                    capture=CaptureInfo(
                        quality=CaptureQuality.DEGRADED,
                        privacy=self.policy.privacy,
                        issues=tuple(sorted(issues)),
                    ),
                )
            aggregated[event.event_id] = event
        for event_id, event in own.items():
            aggregated.setdefault(event_id, event)
        values = sorted(
            aggregated.values(),
            key=lambda event: (event.sequence, event.timestamp, event.event_id),
        )
        if self._task is None:
            latest_task = next((event.task for event in reversed(values) if event.task is not None), None)
            if latest_task is not None:
                self._task = latest_task
        self._child_receipts = self.receiver.receipts(assume_locked=True)
        return values

    def _validate_child_receipts(self, events: Sequence[RuntimeEvent]) -> None:
        """Require one terminal receipt for every observed child execution."""

        observed: dict[tuple[str, str], dict[str, Any]] = {}
        malformed_observation = False
        for event in events:
            if event.parent_agent_id is None:
                continue
            if event.parent_span_id is None:
                malformed_observation = True
                continue
            key = (event.agent_id, event.parent_span_id)
            group = observed.setdefault(
                key,
                {
                    "event_ids": set(),
                    "parent_agent_ids": set(),
                },
            )
            group["event_ids"].add(event.event_id)
            group["parent_agent_ids"].add(event.parent_agent_id)

        valid_keys: set[tuple[str, str]] = set()
        invalid = malformed_observation
        for receipt in self._child_receipts:
            try:
                key = (str(receipt["agent_id"]), str(receipt["span_id"]))
                group = observed.get(key)
                receipt_event_id_list = list(receipt.get("event_ids") or [])
                receipt_event_ids = set(receipt_event_id_list)
                structurally_valid = (
                    receipt.get("schema_version") == "tracerazor-run-receipt/v1"
                    and receipt.get("status") == "completed"
                    and receipt.get("run_id") == self.context.run_id
                    and receipt.get("trace_id") == self.context.trace_id
                    and receipt.get("session_id") == self.context.session_id
                    and group is not None
                    and receipt.get("parent_agent_id") in group["parent_agent_ids"]
                    and receipt_event_ids == group["event_ids"]
                    and len(receipt_event_id_list) == len(receipt_event_ids)
                    and receipt.get("event_count") == len(group["event_ids"])
                )
            except (KeyError, TypeError, ValueError):
                structurally_valid = False
                key = ("", "")
            if structurally_valid and key not in valid_keys:
                valid_keys.add(key)
            else:
                invalid = True
        if invalid or valid_keys != set(observed):
            self._status = "partial"
            if "incomplete_child_run" not in self._audit_issues:
                self._audit_issues.append("incomplete_child_run")

    def _run_audit(self, trace: Mapping[str, Any]) -> Mapping[str, Any]:
        report = self._auditor(
            trace,
            hermetic=True,
            min_steps=self.policy.min_steps,
        )
        if not isinstance(report, Mapping):
            raw = getattr(report, "raw", None)
            if not isinstance(raw, Mapping):
                raise AuditError("runtime auditor must return a report mapping")
            report = raw
        report_manifest = report.get("manifest")
        if not isinstance(report_manifest, Mapping) or report_manifest.get("hermetic") is not True:
            raise AuditError("runtime auditor did not prove hermetic execution")
        expected_trace_sha256 = _canonical_sha256(trace)
        if report_manifest.get("trace_sha256") != expected_trace_sha256:
            raise AuditError("runtime auditor report is not bound to the audited trace bytes")
        return report

    def _persist_parent_artifacts(
        self,
        *,
        trace: Mapping[str, Any] | None,
        report: Mapping[str, Any] | None,
        findings: Sequence[Mapping[str, Any]] | None,
        error: BaseException | str | None,
    ) -> None:
        if trace is not None:
            if self.policy.persist_raw_content:
                if self._raw_trace_bytes is None:
                    raise RuntimeError("raw trace audit bytes were not retained for persistence")
                self.receiver.write_artifact_bytes("trace.json", self._raw_trace_bytes)
            else:
                self.receiver.write_artifact(
                    "trace.json",
                    native_trace_for_persistence(trace, self.policy),
                )
        if trace is not None and report is not None and self._source_trace_sha256 is not None:
            self.receiver.write_artifact(
                "report.json",
                report_for_persistence(
                    report,
                    self.policy,
                    source_trace_sha256=self._source_trace_sha256,
                ),
            )

        persisted_findings = artifact_for_persistence(list(findings or ()), self.policy)
        if report is not None:
            for rank, fix in enumerate(report.get("fixes") or [], start=1):
                if not isinstance(fix, Mapping):
                    continue
                raw_fix_type = str(fix.get("fix_type") or "unknown")
                known_fix_types = {
                    "tool_schema",
                    "prompt_insert",
                    "termination_guard",
                    "context_compression",
                    "verbosity_reduction",
                    "hedge_reduction",
                    "caveman_prompt_insert",
                    "reformulation_guard",
                    "goal_anchor",
                    "dedupe",  # compatibility with early Python reports
                }
                fix_type = raw_fix_type if raw_fix_type in known_fix_types else "redacted"
                raw_risk = str(fix.get("risk") or "needs_review")
                risk = (
                    raw_risk
                    if raw_risk in {"safe", "needs_review", "dangerous"}
                    else "needs_review"
                )
                estimated = fix.get("estimated_token_savings", 0)
                if isinstance(estimated, bool) or not isinstance(estimated, (int, float)):
                    estimated = 0
                finding_seed = (
                    f"{self.context.run_id}:{rank}:{raw_fix_type}:"
                    f"{fix.get('target', '')}"
                ).encode("utf-8")
                persisted_findings.append(
                    {
                        "schema_version": "tracerazor-audit-finding/v1",
                        "finding_id": f"tra_{sha256(finding_seed).hexdigest()[:24]}",
                        "source": "native_audit",
                        "rank": rank,
                        "signal_id": f"audit.fix.{fix_type}",
                        "fix_type": fix_type,
                        "risk": risk,
                        "estimated_token_savings": max(0, int(estimated)),
                        "estimate_status": "estimated",
                        "evidence_ref": f"report.json#/fixes/{rank - 1}",
                        "patch_in_findings": False,
                    }
                )
        for finding in self.guardrails.to_dicts():
            if not self.policy.persist_raw_content:
                finding = dict(finding)
                finding["evidence"] = artifact_for_persistence(finding.get("evidence", {}), self.policy)
            persisted_findings.append(finding)
        self.receiver.write_artifact(
            "findings.json",
            {
                "schema_version": "tracerazor-findings/v1",
                "run_id": self.context.run_id,
                "findings": persisted_findings,
            },
        )

        events = self._manifest_events()
        partial = self._status != "completed"
        eligible, reasons = self.policy.enforcement_eligibility(events, self._task, partial=partial)
        if self._audit_status != "completed":
            eligible = False
            reasons = list(reasons) + ["audit_not_completed"]
        persisted_task: dict[str, Any] | None = None
        if self._task is not None:
            persisted_task = self._task.to_dict()
            if not self.policy.persist_raw_content:
                persisted_task["verifier"] = (
                    artifact_for_persistence(persisted_task["verifier"], self.policy)
                    if persisted_task.get("verifier") is not None
                    else None
                )
                persisted_task["evidence"] = artifact_for_persistence(
                    persisted_task["evidence"], self.policy
                )
        validation: dict[str, Any] = {
            "schema_version": "tracerazor-validation/v1",
            "run_id": self.context.run_id,
            "task": persisted_task,
            "audit_status": self._audit_status,
            "audit_issues": sorted(set(self._audit_issues)),
            "enforcement_eligible": eligible,
            "ineligible_reasons": sorted(set(reasons)),
        }
        effective_error = error if error is not None else self._audit_error
        if effective_error is not None:
            message = str(effective_error)
            validation["error"] = {
                "type": (
                    type(effective_error).__name__
                    if isinstance(effective_error, BaseException)
                    else "RuntimeError"
                ),
                "detail": (
                    message
                    if self.policy.persist_raw_content
                    else artifact_for_persistence(message, self.policy)
                ),
            }
        self.receiver.write_artifact("validation.json", validation)

    def finalize(
        self,
        *,
        status: str = "completed",
        task: TaskResult | None = None,
        findings: Sequence[Mapping[str, Any]] | None = None,
        error: BaseException | str | None = None,
    ) -> dict[str, Any]:
        """Audit raw memory, then atomically persist policy-safe artifacts."""

        if status not in {"completed", "partial", "cancelled", "error"}:
            raise ValueError("status must be completed, partial, cancelled, or error")
        with self._lock:
            if self._finalized:
                return self._last_receipt if self.is_child and self._last_receipt else self._manifest()
            if task is not None:
                self._task = task
            self._status = status
            self._ended_at = utc_now()
            if not self.policy.captures:
                self._finalized = True
                return self._manifest()

            if self.is_child:
                receipt = self._child_receipt()
                self.receiver.write_receipt(f"{self.context.span_id}.json", receipt)
                self._last_receipt = receipt
                self._finalized = True
                return receipt

            report: Mapping[str, Any] | None = None
            trace: dict[str, Any] | None = None
            with self.receiver.locked(stale_after=300.0):
                try:
                    self._analysis_events = self._aggregate_events_locked()
                    self._validate_child_receipts(self._analysis_events)
                except Exception as exc:
                    self._analysis_events = list(self._events)
                    self._status = "partial"
                    self._audit_status = "failed"
                    self._audit_issues.append("event_aggregation_failed")
                    self._audit_error = exc
                try:
                    trace = compile_native_trace(
                        self._manifest_events(),
                        context=self.context,
                        agent_name=self.context.agent_id,
                        framework=self.framework,
                        partial=self._status != "completed",
                    )
                    self._last_trace = trace
                    self._raw_trace_bytes = _canonical_json_bytes(trace)
                    self._source_trace_sha256 = _canonical_sha256(trace)
                except NoAuditableEventsError as exc:
                    self._status = "partial"
                    self._audit_status = "skipped"
                    self._audit_issues.append("no_auditable_events")
                    self._audit_error = exc
                except Exception as exc:
                    self._status = "partial"
                    self._audit_status = "failed"
                    self._audit_issues.append("trace_compile_failed")
                    self._audit_error = exc

                if trace is not None:
                    if len(trace.get("steps", [])) < self.policy.min_steps:
                        self._status = "partial"
                        self._audit_status = "skipped"
                        self._audit_issues.append("below_min_steps")
                        self._audit_error = BelowMinStepsError(
                            f"run has fewer than {self.policy.min_steps} auditable steps"
                        )
                    else:
                        try:
                            report = self._run_audit(trace)
                            self._audit_status = "completed"
                        except BinaryNotFoundError as exc:
                            self._status = "partial"
                            self._audit_status = "failed"
                            self._audit_issues.append("auditor_binary_missing")
                            self._audit_error = exc
                        except BelowMinStepsError as exc:
                            self._status = "partial"
                            self._audit_status = "skipped"
                            self._audit_issues.append("below_min_steps")
                            self._audit_error = exc
                        except Exception as exc:
                            self._status = "partial"
                            self._audit_status = "failed"
                            self._audit_issues.append("audit_failed")
                            self._audit_error = exc

                self._persist_parent_artifacts(
                    trace=trace,
                    report=report,
                    findings=findings,
                    error=error,
                )
                self._finalized = True
                self._write_manifest()
            return self._manifest()

    def mark_partial(self, error: BaseException | str | None = None) -> dict[str, Any]:
        return self.finalize(status="partial", error=error)

    def __enter__(self) -> "TraceRazorProcessor":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        if exc is None:
            self.finalize()
        else:
            self.finalize(status="partial", error=exc)
        return False


_DEFAULT_LOCK = threading.Lock()
_DEFAULT_PROCESSOR: TraceRazorProcessor | None = None


def configure(
    *,
    policy: AuditPolicy | Mapping[str, Any] | None = None,
    policy_path: str | os.PathLike[str] | None = None,
    artifact_dir: str | os.PathLike[str] | None = None,
    context: RunContext | None = None,
    **kwargs: Any,
) -> TraceRazorProcessor:
    """Configure and return the process-default runtime processor."""

    if policy is not None and policy_path is not None:
        raise ValueError("pass policy or policy_path, not both")
    resolved_policy: AuditPolicy
    if isinstance(policy, AuditPolicy):
        resolved_policy = policy
    elif policy is not None:
        resolved_policy = AuditPolicy.from_mapping(policy)
    elif policy_path is not None:
        resolved_policy = AuditPolicy.load(policy_path)
    else:
        resolved_policy = AuditPolicy()
    if artifact_dir is not None:
        resolved_policy = replace(resolved_policy, artifact_dir=os.fspath(artifact_dir))
    processor = TraceRazorProcessor(
        context=context,
        policy=resolved_policy,
        artifact_dir=artifact_dir,
        policy_path=policy_path,
        **kwargs,
    )
    global _DEFAULT_PROCESSOR
    with _DEFAULT_LOCK:
        _DEFAULT_PROCESSOR = processor
    return processor


def get_current_processor() -> TraceRazorProcessor | None:
    with _DEFAULT_LOCK:
        return _DEFAULT_PROCESSOR


__all__ = ["RuntimeAuditor", "TraceRazorProcessor", "configure", "get_current_processor"]
