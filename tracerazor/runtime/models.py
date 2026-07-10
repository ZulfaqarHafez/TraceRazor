"""Typed, dependency-free runtime events for TraceRazor.

The runtime event is deliberately richer than the native audit trace.  It keeps
distributed-trace identity, token provenance, capture quality, and verifier
evidence until a run is compiled to the stable native ``steps`` representation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


EVENT_SCHEMA_VERSION = "tracerazor-event/v1"
RUN_SCHEMA_VERSION = "tracerazor-run/v1"

_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ISSUE_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EVENT_TYPES = {
    "reasoning",
    "tool_call",
    "handoff",
    "message",
    "unknown",
    "run_start",
    "run_end",
    "error",
}
_AUDITABLE_EVENT_TYPES = {"reasoning", "tool_call", "handoff", "message", "unknown"}


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp with a stable ``Z`` suffix."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def new_run_id() -> str:
    return str(uuid.uuid4())


def new_trace_id() -> str:
    """Return a non-zero W3C-compatible 16-byte trace identifier."""

    while True:
        value = uuid.uuid4().hex
        if value != "0" * 32:
            return value


def new_span_id() -> str:
    """Return a non-zero W3C-compatible 8-byte span identifier."""

    while True:
        value = os.urandom(8).hex()
        if value != "0" * 16:
            return value


def stable_digest(value: Any) -> str:
    """Hash a value using canonical JSON when possible."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError):
        payload = repr(value).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def arguments_digest(arguments: Any) -> str:
    """Return the only representation of tool arguments persisted by default."""

    return stable_digest(arguments)


def content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _required_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _optional_text(value: str | None, name: str) -> None:
    if value is not None:
        _required_text(value, name)


class TokenProvenance(str, Enum):
    PROVIDER_REPORTED = "provider_reported"
    ESTIMATED = "estimated"
    MISSING = "missing"


class CaptureQuality(str, Enum):
    COMPLETE = "complete"
    DEGRADED = "degraded"
    PARTIAL = "partial"


class PrivacyMode(str, Enum):
    LOCAL_REDACTED = "local-redacted"
    RAW = "raw"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    SKIPPED = "skipped"


class TaskOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TokenUsage:
    """Provider or estimator token accounting for one event."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0
    provenance: TokenProvenance = TokenProvenance.MISSING

    def __post_init__(self) -> None:
        for name in ("input", "output", "cache_read", "cache_write", "reasoning"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"tokens.{name} must be a non-negative integer")
        object.__setattr__(self, "provenance", TokenProvenance(self.provenance))
        if self.provenance is TokenProvenance.MISSING and self.total != 0:
            raise ValueError("missing token provenance cannot carry non-zero token counts")

    @property
    def total(self) -> int:
        # Cache reads/writes and reasoning tokens are subcategories used for
        # diagnostics; provider total is input + output.
        return self.input + self.output

    # Provider SDKs conventionally use the longer names.  Keep ergonomic
    # aliases without duplicating serialized fields.
    @property
    def input_tokens(self) -> int:
        return self.input

    @property
    def output_tokens(self) -> int:
        return self.output

    @property
    def cache_read_tokens(self) -> int:
        return self.cache_read

    @property
    def cache_write_tokens(self) -> int:
        return self.cache_write

    @property
    def reasoning_tokens(self) -> int:
        return self.reasoning

    @classmethod
    def reported(
        cls,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> "TokenUsage":
        return cls(
            input=input_tokens,
            output=output_tokens,
            cache_read=cache_read_tokens,
            cache_write=cache_write_tokens,
            reasoning=reasoning_tokens,
            provenance=TokenProvenance.PROVIDER_REPORTED,
        )

    @classmethod
    def estimated(
        cls,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> "TokenUsage":
        return cls(
            input=input_tokens,
            output=output_tokens,
            cache_read=cache_read_tokens,
            cache_write=cache_write_tokens,
            reasoning=reasoning_tokens,
            provenance=TokenProvenance.ESTIMATED,
        )

    @property
    def enforcement_eligible(self) -> bool:
        return self.provenance is TokenProvenance.PROVIDER_REPORTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "reasoning": self.reasoning,
            "total": self.total,
            "provenance": self.provenance.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "TokenUsage":
        value = value or {}
        return cls(
            input=value.get("input", value.get("input_tokens", 0)),
            output=value.get("output", value.get("output_tokens", 0)),
            cache_read=value.get("cache_read", value.get("cache_read_tokens", 0)),
            cache_write=value.get("cache_write", value.get("cache_write_tokens", 0)),
            reasoning=value.get("reasoning", value.get("reasoning_tokens", 0)),
            provenance=TokenProvenance(value.get("provenance", "missing")),
        )


@dataclass(frozen=True)
class ToolCall:
    """Privacy-preserving tool-call metadata."""

    signature: str
    arguments_digest: str
    status: ToolStatus = ToolStatus.PENDING
    duration_ms: float | None = None
    expected_failure: bool = False
    observation_size: int = 0
    error_type: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.signature, "tool.signature")
        if not isinstance(self.arguments_digest, str) or not _DIGEST_RE.fullmatch(
            self.arguments_digest
        ):
            raise ValueError("tool.arguments_digest must be a lowercase SHA-256 digest")
        object.__setattr__(self, "status", ToolStatus(self.status))
        if not isinstance(self.expected_failure, bool):
            raise ValueError("tool.expected_failure must be a boolean")
        if self.duration_ms is not None and (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, (int, float))
            or self.duration_ms < 0
        ):
            raise ValueError("tool.duration_ms must be a non-negative number")
        if (
            isinstance(self.observation_size, bool)
            or not isinstance(self.observation_size, int)
            or self.observation_size < 0
        ):
            raise ValueError("tool.observation_size must be a non-negative integer")
        _optional_text(self.error_type, "tool.error_type")

    @property
    def arguments_sha256(self) -> str:
        return self.arguments_digest

    @classmethod
    def from_arguments(
        cls,
        signature: str,
        arguments: Any,
        **kwargs: Any,
    ) -> "ToolCall":
        return cls(signature=signature, arguments_digest=arguments_digest(arguments), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "arguments_digest": self.arguments_digest,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "expected_failure": self.expected_failure,
            "observation_size": self.observation_size,
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ToolCall | None":
        if value is None:
            return None
        return cls(
            signature=str(value["signature"]),
            arguments_digest=str(value["arguments_digest"]),
            status=ToolStatus(value.get("status", "pending")),
            duration_ms=value.get("duration_ms"),
            expected_failure=value.get("expected_failure", False),
            observation_size=value.get("observation_size", 0),
            error_type=value.get("error_type"),
        )


@dataclass(frozen=True)
class TaskResult:
    outcome: TaskOutcome = TaskOutcome.UNKNOWN
    verifier: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", TaskOutcome(self.outcome))
        _optional_text(self.verifier, "task.verifier")
        if not isinstance(self.evidence, Mapping):
            raise ValueError("task.evidence must be an object")

    @property
    def verified(self) -> bool:
        return self.outcome in {TaskOutcome.PASSED, TaskOutcome.FAILED} and bool(self.verifier)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "verifier": self.verifier,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "TaskResult | None":
        if value is None:
            return None
        return cls(
            outcome=TaskOutcome(value.get("outcome", "unknown")),
            verifier=value.get("verifier"),
            evidence=value.get("evidence") or {},
        )


@dataclass(frozen=True)
class CaptureInfo:
    quality: CaptureQuality = CaptureQuality.COMPLETE
    privacy: PrivacyMode = PrivacyMode.LOCAL_REDACTED
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "quality", CaptureQuality(self.quality))
        object.__setattr__(self, "privacy", PrivacyMode(self.privacy))
        object.__setattr__(self, "issues", tuple(str(v) for v in self.issues))
        for issue in self.issues:
            _required_text(issue, "capture.issues[]")
            if not _ISSUE_CODE_RE.fullmatch(issue):
                raise ValueError("capture issues must be lowercase machine-readable codes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality": self.quality.value,
            "privacy": self.privacy.value,
            "issues": list(self.issues),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "CaptureInfo":
        value = value or {}
        return cls(
            quality=CaptureQuality(value.get("quality", "complete")),
            privacy=PrivacyMode(value.get("privacy", "local-redacted")),
            issues=tuple(value.get("issues") or ()),
        )


def format_traceparent(trace_id: str, span_id: str, flags: str = "01") -> str:
    """Create a W3C ``traceparent`` header.

    Non-W3C application identifiers are deterministically mapped to the
    required hex width.  New TraceRazor contexts already use native W3C IDs.
    """

    trace_hex = trace_id.lower()
    if not _TRACE_ID_RE.fullmatch(trace_hex):
        trace_hex = hashlib.sha256(trace_id.encode("utf-8")).hexdigest()[:32]
    span_hex = span_id.lower()
    if not _SPAN_ID_RE.fullmatch(span_hex):
        span_hex = hashlib.sha256(span_id.encode("utf-8")).hexdigest()[:16]
    if trace_hex == "0" * 32 or span_hex == "0" * 16:
        raise ValueError("W3C trace and span IDs must not be all zero")
    if not re.fullmatch(r"[0-9a-f]{2}", flags):
        raise ValueError("trace flags must be two lowercase hexadecimal characters")
    return f"00-{trace_hex}-{span_hex}-{flags}"


def parse_traceparent(value: str) -> tuple[str, str, str]:
    parts = value.strip().lower().split("-")
    if len(parts) != 4 or parts[0] != "00":
        raise ValueError("unsupported or malformed W3C traceparent")
    trace_id, span_id, flags = parts[1:]
    if not _TRACE_ID_RE.fullmatch(trace_id) or trace_id == "0" * 32:
        raise ValueError("traceparent has an invalid trace ID")
    if not _SPAN_ID_RE.fullmatch(span_id) or span_id == "0" * 16:
        raise ValueError("traceparent has an invalid parent span ID")
    if not re.fullmatch(r"[0-9a-f]{2}", flags):
        raise ValueError("traceparent has invalid flags")
    return trace_id, span_id, flags


@dataclass(frozen=True)
class RunContext:
    run_id: str
    trace_id: str
    span_id: str
    session_id: str
    agent_id: str
    parent_span_id: str | None = None
    parent_agent_id: str | None = None
    trace_flags: str = "01"

    def __post_init__(self) -> None:
        for field_name in ("run_id", "trace_id", "span_id", "session_id", "agent_id"):
            _required_text(getattr(self, field_name), field_name)
        if not _RUN_ID_RE.fullmatch(self.run_id) or self.run_id in {".", ".."}:
            raise ValueError(
                "run_id must be a single safe path segment of at most 128 characters"
            )
        # Retain the exact W3C identifiers that are propagated. Otherwise a
        # caller-provided application ID would be hashed only when formatting
        # ``traceparent`` and parent/child processes would disagree on trace
        # identity.
        _, canonical_trace, canonical_span, canonical_flags = format_traceparent(
            self.trace_id, self.span_id, self.trace_flags
        ).split("-")
        object.__setattr__(self, "trace_id", canonical_trace)
        object.__setattr__(self, "span_id", canonical_span)
        object.__setattr__(self, "trace_flags", canonical_flags)
        if self.parent_span_id is not None:
            parent = self.parent_span_id.lower()
            if not _SPAN_ID_RE.fullmatch(parent):
                parent = hashlib.sha256(parent.encode("utf-8")).hexdigest()[:16]
            if parent == "0" * 16:
                raise ValueError("parent_span_id must not be all zero")
            object.__setattr__(self, "parent_span_id", parent)
        if self.parent_span_id == self.span_id:
            raise ValueError("parent_span_id must differ from span_id")
        _optional_text(self.parent_span_id, "parent_span_id")
        _optional_text(self.parent_agent_id, "parent_agent_id")
        # Validate that the context can actually propagate using W3C Trace Context.
        format_traceparent(self.trace_id, self.span_id, self.trace_flags)

    @classmethod
    def create(
        cls,
        *,
        agent_id: str = "agent",
        session_id: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        parent_agent_id: str | None = None,
    ) -> "RunContext":
        return cls(
            run_id=run_id or new_run_id(),
            trace_id=trace_id or new_trace_id(),
            span_id=new_span_id(),
            session_id=session_id or str(uuid.uuid4()),
            agent_id=agent_id,
            parent_span_id=parent_span_id,
            parent_agent_id=parent_agent_id,
        )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        agent_id: str | None = None,
    ) -> "RunContext":
        source = os.environ if env is None else env
        trace_id: str | None = source.get("TRACERAZOR_TRACE_ID")
        parent_span_id: str | None = source.get("TRACERAZOR_PARENT_SPAN_ID")
        flags = "01"
        propagated = source.get("TRACEPARENT") or source.get("traceparent")
        if propagated:
            propagated_trace_id, propagated_span_id, flags = parse_traceparent(propagated)
            trace_id = trace_id or propagated_trace_id
            parent_span_id = parent_span_id or propagated_span_id
        return cls(
            run_id=source.get("TRACERAZOR_RUN_ID") or new_run_id(),
            trace_id=trace_id or new_trace_id(),
            span_id=new_span_id(),
            session_id=source.get("TRACERAZOR_SESSION_ID") or str(uuid.uuid4()),
            agent_id=agent_id or source.get("TRACERAZOR_AGENT_ID") or "agent",
            parent_span_id=parent_span_id,
            parent_agent_id=source.get("TRACERAZOR_PARENT_AGENT_ID"),
            trace_flags=flags,
        )

    @property
    def traceparent(self) -> str:
        return format_traceparent(self.trace_id, self.span_id, self.trace_flags)

    def spawn_env(
        self,
        *,
        child_agent_id: str | None = None,
        policy_path: str | os.PathLike[str] | None = None,
        base: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return an environment for a child agent without mutating ``os.environ``."""

        result = dict(os.environ if base is None else base)
        result.update(
            {
                "TRACEPARENT": self.traceparent,
                "TRACERAZOR_RUN_ID": self.run_id,
                "TRACERAZOR_TRACE_ID": self.trace_id,
                "TRACERAZOR_PARENT_SPAN_ID": self.span_id,
                "TRACERAZOR_SESSION_ID": self.session_id,
                "TRACERAZOR_PARENT_AGENT_ID": self.agent_id,
            }
        )
        if child_agent_id:
            result["TRACERAZOR_AGENT_ID"] = child_agent_id
        if policy_path is not None:
            result["TRACERAZOR_POLICY"] = os.fspath(policy_path)
        return result


@dataclass(frozen=True)
class RuntimeEvent:
    """One validated ``tracerazor-event/v1`` event."""

    run_id: str
    trace_id: str
    span_id: str
    session_id: str
    agent_id: str
    event_type: str
    host: str
    framework: str
    tokens: TokenUsage = field(default_factory=TokenUsage)
    parent_span_id: str | None = None
    parent_agent_id: str | None = None
    host_version: str | None = None
    framework_version: str | None = None
    tool: ToolCall | None = None
    task: TaskResult | None = None
    capture: CaptureInfo = field(default_factory=CaptureInfo)
    content: str | None = None
    input_context: str | None = None
    output: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=utc_now)
    sequence: int = 0
    schema_version: str = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EVENT_SCHEMA_VERSION!r}")
        for name in (
            "event_id",
            "run_id",
            "trace_id",
            "span_id",
            "session_id",
            "agent_id",
            "host",
            "framework",
            "event_type",
            "timestamp",
        ):
            _required_text(getattr(self, name), name)
        if self.event_type not in _EVENT_TYPES:
            raise ValueError(f"unsupported event_type: {self.event_type!r}")
        if self.parent_span_id == self.span_id:
            raise ValueError("parent_span_id must differ from span_id")
        for name in ("parent_span_id", "parent_agent_id", "host_version", "framework_version"):
            _optional_text(getattr(self, name), name)
        for name in ("content", "input_context", "output"):
            _optional_text(getattr(self, name), name)
        if not isinstance(self.tokens, TokenUsage):
            raise ValueError("tokens must be a TokenUsage")
        if self.tool is not None and not isinstance(self.tool, ToolCall):
            raise ValueError("tool must be ToolCall metadata")
        if self.task is not None and not isinstance(self.task, TaskResult):
            raise ValueError("task must be a TaskResult")
        if not isinstance(self.capture, CaptureInfo):
            raise ValueError("capture must be CaptureInfo")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if self.event_type == "tool_call" and self.tool is None:
            raise ValueError("tool_call events require tool metadata")
        if self.tool is not None and self.event_type != "tool_call":
            raise ValueError("tool metadata is only valid for tool_call events")
        # Parse the timestamp now so corrupt events fail at their boundary.
        try:
            parsed_timestamp = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be an ISO-8601 timestamp") from exc
        if parsed_timestamp.tzinfo is None:
            raise ValueError("timestamp must include a timezone")

    @property
    def auditable(self) -> bool:
        return self.event_type in _AUDITABLE_EVENT_TYPES

    @property
    def effective_capture(self) -> CaptureInfo:
        quality = self.capture.quality
        issues = list(self.capture.issues)
        if self.auditable and self.tokens.provenance is TokenProvenance.ESTIMATED:
            if quality is CaptureQuality.COMPLETE:
                quality = CaptureQuality.DEGRADED
            if "estimated_token_usage" not in issues:
                issues.append("estimated_token_usage")
        elif self.auditable and self.tokens.provenance is TokenProvenance.MISSING:
            if quality is CaptureQuality.COMPLETE:
                quality = CaptureQuality.DEGRADED
            if "missing_token_usage" not in issues:
                issues.append("missing_token_usage")
        return replace(self.capture, quality=quality, issues=tuple(issues))

    @property
    def enforcement_eligible(self) -> bool:
        return (
            (not self.auditable or self.tokens.enforcement_eligible)
            and self.effective_capture.quality is CaptureQuality.COMPLETE
        )

    @classmethod
    def create(
        cls,
        context: RunContext,
        *,
        event_type: str,
        host: str = "unknown",
        framework: str = "unknown",
        **kwargs: Any,
    ) -> "RuntimeEvent":
        return cls(
            run_id=context.run_id,
            trace_id=context.trace_id,
            span_id=kwargs.pop("span_id", new_span_id()),
            parent_span_id=kwargs.pop("parent_span_id", context.span_id),
            session_id=context.session_id,
            agent_id=context.agent_id,
            parent_agent_id=context.parent_agent_id,
            event_type=event_type,
            host=host,
            framework=framework,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "event_type": self.event_type,
            "host": self.host,
            "host_version": self.host_version,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "tokens": self.tokens.to_dict(),
            "tool": self.tool.to_dict() if self.tool else None,
            "task": self.task.to_dict() if self.task else None,
            "capture": self.effective_capture.to_dict(),
            "content": self.content,
            "content_sha256": content_digest(self.content) if self.content is not None else None,
            "input_context": self.input_context,
            "input_context_sha256": (
                content_digest(self.input_context) if self.input_context is not None else None
            ),
            "output": self.output,
            "output_sha256": content_digest(self.output) if self.output is not None else None,
            "metadata": dict(self.metadata),
        }
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeEvent":
        return cls(
            schema_version=str(value.get("schema_version", EVENT_SCHEMA_VERSION)),
            event_id=str(value.get("event_id") or uuid.uuid4()),
            timestamp=str(value.get("timestamp") or utc_now()),
            sequence=int(value.get("sequence", 0)),
            run_id=str(value["run_id"]),
            trace_id=str(value["trace_id"]),
            span_id=str(value["span_id"]),
            parent_span_id=value.get("parent_span_id"),
            session_id=str(value["session_id"]),
            agent_id=str(value["agent_id"]),
            parent_agent_id=value.get("parent_agent_id"),
            event_type=str(value["event_type"]),
            host=str(value["host"]),
            host_version=value.get("host_version"),
            framework=str(value["framework"]),
            framework_version=value.get("framework_version"),
            tokens=TokenUsage.from_dict(value.get("tokens")),
            tool=ToolCall.from_dict(value.get("tool")),
            task=TaskResult.from_dict(value.get("task")),
            capture=CaptureInfo.from_dict(value.get("capture")),
            content=value.get("content"),
            input_context=value.get("input_context"),
            output=value.get("output"),
            metadata=value.get("metadata") or {},
        )


def run_capture_quality(events: list[RuntimeEvent], *, partial: bool = False) -> CaptureQuality:
    if partial or any(event.effective_capture.quality is CaptureQuality.PARTIAL for event in events):
        return CaptureQuality.PARTIAL
    if any(event.effective_capture.quality is CaptureQuality.DEGRADED for event in events):
        return CaptureQuality.DEGRADED
    return CaptureQuality.COMPLETE


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "CaptureInfo",
    "CaptureQuality",
    "PrivacyMode",
    "RunContext",
    "RuntimeEvent",
    "TaskOutcome",
    "TaskResult",
    "TokenProvenance",
    "TokenUsage",
    "ToolCall",
    "ToolStatus",
    "arguments_digest",
    "content_digest",
    "format_traceparent",
    "new_run_id",
    "new_span_id",
    "new_trace_id",
    "parse_traceparent",
    "run_capture_quality",
    "stable_digest",
    "utc_now",
]
