"""Deterministic, dependency-free streaming guardrails for agent runs.

The detector is deliberately advisory.  It never stops a run or mutates agent
state; ``enforcement_eligible`` only describes whether the evidence for one
finding is exact enough to be considered by a separately configured policy and
task verifier.

All matching uses metadata, sizes, and digests already present in
``tracerazor-event/v1``.  No prompt, tool argument, or observation content is
copied into a finding.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .models import RuntimeEvent, TokenProvenance, ToolStatus, content_digest, stable_digest


FINDING_SCHEMA_VERSION = "tracerazor-guardrail-finding/v1"


class GuardrailSeverity(str, Enum):
    """Stable severity vocabulary for streaming findings."""

    WARNING = "warning"
    HIGH = "high"


@dataclass(frozen=True)
class GuardrailConfig:
    """Thresholds for the bounded streaming state machine.

    ``None`` disables an individual limit.  Token budget findings can be
    evidence-eligible for enforcement only when every observed auditable event
    has provider-reported usage.  Cost findings additionally require the
    explicit ``allow_cost_budget_enforcement`` opt-in because their dollar
    value is derived from a configured rate.
    """

    tool_repeat_threshold: int | None = 3
    tool_repeat_window: int = 8
    max_children_per_parent: int | None = 8
    max_total_children: int | None = 32
    context_repeat_threshold: int | None = 3
    context_repeat_window: int = 8
    min_context_size: int = 512
    max_observation_size: int | None = 32_768
    failed_action_threshold: int | None = 3
    max_failure_subjects: int = 128
    token_budget: int | None = None
    cost_budget_usd: float | None = None
    cost_per_million_tokens_usd: float | None = None
    budget_warning_fraction: float = 0.8
    allow_cost_budget_enforcement: bool = False

    def __post_init__(self) -> None:
        for name in (
            "tool_repeat_threshold",
            "max_children_per_parent",
            "max_total_children",
            "context_repeat_threshold",
            "max_observation_size",
            "failed_action_threshold",
            "token_budget",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"guardrails.{name} must be a positive integer or null")
        for name in (
            "tool_repeat_window",
            "context_repeat_window",
            "max_failure_subjects",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"guardrails.{name} must be a positive integer")
        if (
            isinstance(self.min_context_size, bool)
            or not isinstance(self.min_context_size, int)
            or self.min_context_size < 0
        ):
            raise ValueError("guardrails.min_context_size must be a non-negative integer")
        if (
            self.tool_repeat_threshold is not None
            and self.tool_repeat_window < self.tool_repeat_threshold
        ):
            raise ValueError("tool_repeat_window must be at least tool_repeat_threshold")
        if (
            self.context_repeat_threshold is not None
            and self.context_repeat_window < self.context_repeat_threshold
        ):
            raise ValueError("context_repeat_window must be at least context_repeat_threshold")
        for name in ("cost_budget_usd", "cost_per_million_tokens_usd"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"guardrails.{name} must be a positive finite number or null")
        if (self.cost_budget_usd is None) != (self.cost_per_million_tokens_usd is None):
            raise ValueError(
                "cost_budget_usd and cost_per_million_tokens_usd must be configured together"
            )
        if (
            isinstance(self.budget_warning_fraction, bool)
            or not isinstance(self.budget_warning_fraction, (int, float))
            or not math.isfinite(float(self.budget_warning_fraction))
            or not 0 < float(self.budget_warning_fraction) <= 1
        ):
            raise ValueError("budget_warning_fraction must be in the interval (0, 1]")
        if not isinstance(self.allow_cost_budget_enforcement, bool):
            raise ValueError("allow_cost_budget_enforcement must be a boolean")


@dataclass(frozen=True)
class GuardrailFinding:
    """One privacy-preserving streaming finding."""

    finding_id: str
    signal_id: str
    severity: GuardrailSeverity
    title: str
    summary: str
    first_sequence: int
    last_sequence: int
    occurrence_count: int
    evidence: Mapping[str, Any] = field(default_factory=dict)
    enforcement_eligible: bool = False
    estimate_status: str = "not_applicable"
    schema_version: str = FINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FINDING_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {FINDING_SCHEMA_VERSION!r}")
        object.__setattr__(self, "severity", GuardrailSeverity(self.severity))
        if not re.fullmatch(r"trg_[0-9a-f]{24}", self.finding_id):
            raise ValueError("finding_id must be a stable TraceRazor guardrail ID")
        if not re.fullmatch(r"guardrail\.[a-z0-9_]+", self.signal_id):
            raise ValueError("signal_id must be a stable guardrail signal name")
        for name in ("title", "summary", "estimate_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("first_sequence", "last_sequence", "occurrence_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.last_sequence < self.first_sequence:
            raise ValueError("last_sequence must not precede first_sequence")
        if not isinstance(self.enforcement_eligible, bool):
            raise ValueError("enforcement_eligible must be a boolean")
        if not isinstance(self.evidence, Mapping):
            raise ValueError("evidence must be an object")
        # Freeze a consistently ordered shallow copy.  Evidence values are all
        # JSON scalars produced in this module.
        ordered = {key: self.evidence[key] for key in sorted(self.evidence)}
        object.__setattr__(self, "evidence", MappingProxyType(ordered))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "finding_id": self.finding_id,
            "signal_id": self.signal_id,
            "severity": self.severity.value,
            "title": self.title,
            "summary": self.summary,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "occurrence_count": self.occurrence_count,
            "enforcement_eligible": self.enforcement_eligible,
            "estimate_status": self.estimate_status,
            "evidence": dict(self.evidence),
        }


_SIGNAL_COPY: dict[str, tuple[GuardrailSeverity, str, str]] = {
    "guardrail.tool_call_repeat": (
        GuardrailSeverity.WARNING,
        "Repeated equivalent tool call",
        "Equivalent tool calls repeated within the configured recent-event window.",
    ),
    "guardrail.child_agent_fanout": (
        GuardrailSeverity.HIGH,
        "Child-agent fan-out exceeded",
        "The run spawned more distinct child agents than the configured limit.",
    ),
    "guardrail.context_reinjection": (
        GuardrailSeverity.WARNING,
        "Repeated context injection",
        "The same context payload was injected repeatedly within the configured window.",
    ),
    "guardrail.oversized_observation": (
        GuardrailSeverity.WARNING,
        "Oversized tool observation",
        "A tool observation exceeded the configured size limit.",
    ),
    "guardrail.failed_action_stall": (
        GuardrailSeverity.HIGH,
        "Repeated failed action without state change",
        "The same action failed repeatedly without evidence that state changed.",
    ),
    "guardrail.token_budget": (
        GuardrailSeverity.HIGH,
        "Token budget approaching",
        "Known token usage reached the configured budget warning fraction.",
    ),
    "guardrail.cost_budget": (
        GuardrailSeverity.HIGH,
        "Estimated cost budget approaching",
        "Estimated cost reached the configured budget warning fraction.",
    ),
}


def _normalized_signature(signature: str) -> str:
    value = re.sub(r"[_\W]+", ".", signature.casefold(), flags=re.UNICODE).strip(".")
    return value or stable_digest(signature)[:16]


def _stable_finding_id(run_id: str, signal_id: str, subject: str) -> str:
    payload = f"{FINDING_SCHEMA_VERSION}\0{run_id}\0{signal_id}\0{subject}"
    return "trg_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _subject_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _decimal_number(value: Decimal, *, places: int = 6) -> float:
    quantum = Decimal(1).scaleb(-places)
    return float(value.quantize(quantum))


class StreamingGuardrailDetector:
    """Observe ``RuntimeEvent`` objects in O(1) average work per event.

    Recent tool/context state and failure subjects are bounded by configuration.
    Child identity tracking saturates immediately after the run-wide limit is
    crossed, which prevents an already-unbounded fan-out from growing the
    detector's memory without limit.
    """

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        self.config = config or GuardrailConfig()
        self._lock = threading.RLock()
        self._run_id: str | None = None
        self._observed = 0
        self._findings: dict[str, GuardrailFinding] = {}

        self._tool_window: deque[tuple[tuple[str, str], tuple[str, str]]] = deque()
        self._tool_equivalent_counts: Counter[tuple[str, str]] = Counter()
        self._tool_exact_counts: Counter[tuple[str, str]] = Counter()

        self._context_window: deque[str] = deque()
        self._context_counts: Counter[str] = Counter()

        self._children_by_parent: dict[str, set[str]] = {}
        self._child_pairs: set[tuple[str, str]] = set()
        self._child_tracking_saturated = False

        self._failures: OrderedDict[
            tuple[str, str], tuple[str, str, int]
        ] = OrderedDict()

        self._known_tokens = 0
        self._usage_provenance: Counter[TokenProvenance] = Counter()

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def findings(self) -> tuple[GuardrailFinding, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._findings.values(),
                    key=lambda finding: (
                        finding.first_sequence,
                        finding.signal_id,
                        finding.finding_id,
                    ),
                )
            )

    def to_dicts(self) -> list[dict[str, Any]]:
        return [finding.to_dict() for finding in self.findings]

    def _sequence(self, event: RuntimeEvent) -> int:
        return event.sequence or self._observed

    def _upsert(
        self,
        *,
        event: RuntimeEvent,
        signal_id: str,
        subject: str,
        occurrence_count: int,
        evidence: Mapping[str, Any],
        enforcement_eligible: bool = False,
        estimate_status: str = "not_applicable",
    ) -> GuardrailFinding:
        finding_id = _stable_finding_id(event.run_id, signal_id, subject)
        sequence = self._sequence(event)
        severity, title, summary = _SIGNAL_COPY[signal_id]
        existing = self._findings.get(finding_id)
        finding = GuardrailFinding(
            finding_id=finding_id,
            signal_id=signal_id,
            severity=severity,
            title=title,
            summary=summary,
            first_sequence=existing.first_sequence if existing else sequence,
            last_sequence=sequence,
            occurrence_count=max(
                occurrence_count,
                existing.occurrence_count if existing else 0,
            ),
            evidence=evidence,
            enforcement_eligible=enforcement_eligible,
            estimate_status=estimate_status,
        )
        self._findings[finding_id] = finding
        return finding

    def _observe_tool_repeat(self, event: RuntimeEvent) -> GuardrailFinding | None:
        threshold = self.config.tool_repeat_threshold
        if threshold is None or event.tool is None:
            return None
        normalized = _normalized_signature(event.tool.signature)
        equivalent_key = (normalized, event.tool.arguments_digest)
        exact_key = (event.tool.signature, event.tool.arguments_digest)
        if len(self._tool_window) >= self.config.tool_repeat_window:
            old_equivalent, old_exact = self._tool_window.popleft()
            self._tool_equivalent_counts[old_equivalent] -= 1
            self._tool_exact_counts[old_exact] -= 1
            if self._tool_equivalent_counts[old_equivalent] == 0:
                del self._tool_equivalent_counts[old_equivalent]
            if self._tool_exact_counts[old_exact] == 0:
                del self._tool_exact_counts[old_exact]
        self._tool_window.append((equivalent_key, exact_key))
        self._tool_equivalent_counts[equivalent_key] += 1
        self._tool_exact_counts[exact_key] += 1
        equivalent_count = self._tool_equivalent_counts[equivalent_key]
        if equivalent_count < threshold:
            return None
        exact_count = self._tool_exact_counts[exact_key]
        subject = f"{normalized}:{event.tool.arguments_digest}"
        return self._upsert(
            event=event,
            signal_id="guardrail.tool_call_repeat",
            subject=subject,
            occurrence_count=equivalent_count,
            enforcement_eligible=self._usage_is_exact(),
            evidence={
                "arguments_sha256": event.tool.arguments_digest,
                "equivalent_count": equivalent_count,
                "match": "identical" if exact_count >= threshold else "equivalent",
                "normalized_signature": normalized,
                "threshold": threshold,
                "window_size": self.config.tool_repeat_window,
            },
        )

    def _observe_child_fanout(self, event: RuntimeEvent) -> list[GuardrailFinding]:
        if event.parent_agent_id is None or self._child_tracking_saturated:
            return []
        parent = event.parent_agent_id
        child = event.agent_id
        pair = (parent, child)
        if pair in self._child_pairs:
            return []
        self._child_pairs.add(pair)
        children = self._children_by_parent.setdefault(parent, set())
        children.add(child)
        generated: list[GuardrailFinding] = []
        parent_limit = self.config.max_children_per_parent
        if parent_limit is not None and len(children) > parent_limit:
            parent_digest = _subject_digest(parent)
            generated.append(
                self._upsert(
                    event=event,
                    signal_id="guardrail.child_agent_fanout",
                    subject=f"parent:{parent_digest}",
                    occurrence_count=len(children),
                    enforcement_eligible=self._usage_is_exact(),
                    evidence={
                        "distinct_children_at_least": len(children),
                        "limit": parent_limit,
                        "parent_agent_sha256": parent_digest,
                        "scope": "parent",
                    },
                )
            )
        total_limit = self.config.max_total_children
        if total_limit is not None and len(self._child_pairs) > total_limit:
            generated.append(
                self._upsert(
                    event=event,
                    signal_id="guardrail.child_agent_fanout",
                    subject="run",
                    occurrence_count=len(self._child_pairs),
                    enforcement_eligible=self._usage_is_exact(),
                    evidence={
                        "distinct_parent_child_pairs_at_least": len(self._child_pairs),
                        "limit": total_limit,
                        "scope": "run",
                        "tracking_saturated": True,
                    },
                )
            )
            self._child_tracking_saturated = True
        return generated

    def _observe_context(self, event: RuntimeEvent) -> GuardrailFinding | None:
        threshold = self.config.context_repeat_threshold
        context = event.input_context
        if threshold is None or context is None or len(context) < self.config.min_context_size:
            return None
        digest = content_digest(context)
        if len(self._context_window) >= self.config.context_repeat_window:
            old = self._context_window.popleft()
            self._context_counts[old] -= 1
            if self._context_counts[old] == 0:
                del self._context_counts[old]
        self._context_window.append(digest)
        self._context_counts[digest] += 1
        count = self._context_counts[digest]
        if count < threshold:
            return None
        return self._upsert(
            event=event,
            signal_id="guardrail.context_reinjection",
            subject=digest,
            occurrence_count=count,
            enforcement_eligible=self._usage_is_exact(),
            evidence={
                "context_sha256": digest,
                "context_size": len(context),
                "repeat_count": count,
                "threshold": threshold,
                "window_size": self.config.context_repeat_window,
            },
        )

    def _observe_observation(self, event: RuntimeEvent) -> GuardrailFinding | None:
        limit = self.config.max_observation_size
        if limit is None or event.tool is None or event.tool.observation_size <= limit:
            return None
        normalized = _normalized_signature(event.tool.signature)
        subject = f"{normalized}:{event.tool.arguments_digest}"
        existing_id = _stable_finding_id(
            event.run_id, "guardrail.oversized_observation", subject
        )
        existing = self._findings.get(existing_id)
        previous_max = int(existing.evidence["maximum_observation_size"]) if existing else 0
        occurrence_count = (existing.occurrence_count + 1) if existing else 1
        return self._upsert(
            event=event,
            signal_id="guardrail.oversized_observation",
            subject=subject,
            occurrence_count=occurrence_count,
            enforcement_eligible=self._usage_is_exact(),
            evidence={
                "arguments_sha256": event.tool.arguments_digest,
                "limit": limit,
                "maximum_observation_size": max(
                    previous_max,
                    event.tool.observation_size,
                ),
                "normalized_signature": normalized,
            },
        )

    @staticmethod
    def _state_marker(event: RuntimeEvent) -> tuple[str, str]:
        for key in ("state_digest", "state_sha256", "workspace_digest", "state_version"):
            if key in event.metadata and event.metadata[key] is not None:
                return stable_digest(event.metadata[key]), f"metadata.{key}"
        if event.output is not None:
            return content_digest(event.output), "output_digest_proxy"
        if event.tool is not None and event.tool.error_type is not None:
            return stable_digest(event.tool.error_type), "error_type_proxy"
        if event.tool is not None:
            return event.tool.arguments_digest, "arguments_digest_proxy"
        raise AssertionError("state marker requires a tool event")

    def _observe_failure(self, event: RuntimeEvent) -> GuardrailFinding | None:
        threshold = self.config.failed_action_threshold
        tool = event.tool
        if threshold is None or tool is None:
            return None
        normalized = _normalized_signature(tool.signature)
        tool_key = (normalized, tool.arguments_digest)
        if tool.status is not ToolStatus.ERROR or tool.expected_failure:
            self._failures.pop(tool_key, None)
            return None
        marker, basis = self._state_marker(event)
        previous = self._failures.pop(tool_key, None)
        count = previous[2] + 1 if previous and previous[0] == marker else 1
        self._failures[tool_key] = (marker, basis, count)
        while len(self._failures) > self.config.max_failure_subjects:
            self._failures.popitem(last=False)
        if count < threshold:
            return None
        subject = f"{normalized}:{tool.arguments_digest}:{marker}"
        return self._upsert(
            event=event,
            signal_id="guardrail.failed_action_stall",
            subject=subject,
            occurrence_count=count,
            enforcement_eligible=self._usage_is_exact(),
            evidence={
                "arguments_sha256": tool.arguments_digest,
                "failure_count": count,
                "normalized_signature": normalized,
                "state_basis": basis,
                "state_sha256": marker,
                "threshold": threshold,
            },
        )

    def _provenance_status(self) -> str:
        active = [item.value for item, count in self._usage_provenance.items() if count]
        return active[0] if len(active) == 1 else "mixed"

    def _usage_is_exact(self) -> bool:
        return bool(self._usage_provenance) and set(self._usage_provenance) == {
            TokenProvenance.PROVIDER_REPORTED
        }

    def _observe_budgets(self, event: RuntimeEvent) -> list[GuardrailFinding]:
        if not event.auditable:
            return []
        self._known_tokens += event.tokens.total
        self._usage_provenance[event.tokens.provenance] += 1
        if self._known_tokens == 0:
            return []
        generated: list[GuardrailFinding] = []
        warning_fraction = Decimal(str(self.config.budget_warning_fraction))
        exact = self._usage_is_exact()
        provenance = self._provenance_status()

        if self.config.token_budget is not None:
            budget = self.config.token_budget
            ratio = Decimal(self._known_tokens) / Decimal(budget)
            if ratio >= warning_fraction:
                generated.append(
                    self._upsert(
                        event=event,
                        signal_id="guardrail.token_budget",
                        subject="run",
                        occurrence_count=1,
                        enforcement_eligible=exact,
                        estimate_status="exact" if exact else "estimated_or_incomplete",
                        evidence={
                            "budget_state": (
                                "exceeded" if self._known_tokens > budget else "at_or_approaching"
                            ),
                            "budget_tokens": budget,
                            "known_tokens": self._known_tokens,
                            "token_provenance": provenance,
                            "usage_fraction": _decimal_number(ratio),
                            "warning_fraction": float(warning_fraction),
                        },
                    )
                )

        if self.config.cost_budget_usd is not None:
            cost_budget = Decimal(str(self.config.cost_budget_usd))
            rate = Decimal(str(self.config.cost_per_million_tokens_usd))
            estimated_cost = Decimal(self._known_tokens) * rate / Decimal(1_000_000)
            ratio = estimated_cost / cost_budget
            if ratio >= warning_fraction:
                generated.append(
                    self._upsert(
                        event=event,
                        signal_id="guardrail.cost_budget",
                        subject="run",
                        occurrence_count=1,
                        enforcement_eligible=(
                            exact and self.config.allow_cost_budget_enforcement
                        ),
                        estimate_status="estimated_cost",
                        evidence={
                            "budget_state": (
                                "exceeded" if estimated_cost > cost_budget else "at_or_approaching"
                            ),
                            "budget_usd": float(cost_budget),
                            "cost_per_million_tokens_usd": float(rate),
                            "estimated_cost_usd": _decimal_number(estimated_cost),
                            "known_tokens": self._known_tokens,
                            "token_provenance": provenance,
                            "usage_fraction": _decimal_number(ratio),
                            "warning_fraction": float(warning_fraction),
                        },
                    )
                )
        return generated

    def observe(self, event: RuntimeEvent | Mapping[str, Any]) -> tuple[GuardrailFinding, ...]:
        """Observe one event and return findings created or updated by it."""

        if not isinstance(event, RuntimeEvent):
            event = RuntimeEvent.from_dict(event)
        with self._lock:
            if self._run_id is None:
                self._run_id = event.run_id
            elif event.run_id != self._run_id:
                raise ValueError("one StreamingGuardrailDetector may observe only one run_id")
            self._observed += 1
            generated: list[GuardrailFinding] = []
            # Account provenance first so every finding produced by this event
            # is advisory when any auditable usage is estimated or missing.
            generated.extend(self._observe_budgets(event))
            generated.extend(self._observe_child_fanout(event))
            if event.tool is not None:
                for finding in (
                    self._observe_tool_repeat(event),
                    self._observe_observation(event),
                    self._observe_failure(event),
                ):
                    if finding is not None:
                        generated.append(finding)
            context_finding = self._observe_context(event)
            if context_finding is not None:
                generated.append(context_finding)
            return tuple(generated)

    def observe_many(
        self, events: Iterable[RuntimeEvent | Mapping[str, Any]]
    ) -> tuple[GuardrailFinding, ...]:
        """Observe events in order and return the final finding snapshot."""

        for event in events:
            self.observe(event)
        return self.findings


__all__ = [
    "FINDING_SCHEMA_VERSION",
    "GuardrailConfig",
    "GuardrailFinding",
    "GuardrailSeverity",
    "StreamingGuardrailDetector",
]
