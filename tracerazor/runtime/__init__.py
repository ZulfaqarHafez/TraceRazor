"""Agent-native runtime capture for TraceRazor.

This package has no mandatory dependencies and performs no installation,
network access, or global configuration when imported.
"""

from .compiler import NoAuditableEventsError, compile_native_trace, validate_native_trace_shape
from .instrumentation import (
    InstrumentationResult,
    auto_instrument,
    register_instrumentation,
    registered_instrumentations,
    unregister_instrumentation,
)
from .guardrails import (
    FINDING_SCHEMA_VERSION,
    GuardrailConfig,
    GuardrailFinding,
    GuardrailSeverity,
    StreamingGuardrailDetector,
)
from .models import (
    EVENT_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    CaptureInfo,
    CaptureQuality,
    PrivacyMode,
    RunContext,
    RuntimeEvent,
    TaskOutcome,
    TaskResult,
    TokenProvenance,
    TokenUsage,
    ToolCall,
    ToolStatus,
    arguments_digest,
    content_digest,
    format_traceparent,
    new_run_id,
    new_span_id,
    new_trace_id,
    parse_traceparent,
)
from .persistence import DiskSpoolReceiver, recover_partial_run
from .policy import AuditPolicy
from .processor import RuntimeAuditor, TraceRazorProcessor, configure, get_current_processor


# Short aliases are convenient for typed integrations while the explicit names
# remain canonical in serialized formats and documentation.
Event = RuntimeEvent
Run = RunContext
Tokens = TokenUsage


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "FINDING_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "AuditPolicy",
    "CaptureInfo",
    "CaptureQuality",
    "DiskSpoolReceiver",
    "Event",
    "InstrumentationResult",
    "GuardrailConfig",
    "GuardrailFinding",
    "GuardrailSeverity",
    "NoAuditableEventsError",
    "PrivacyMode",
    "Run",
    "RunContext",
    "RuntimeEvent",
    "RuntimeAuditor",
    "TaskOutcome",
    "TaskResult",
    "TokenProvenance",
    "TokenUsage",
    "Tokens",
    "ToolCall",
    "ToolStatus",
    "TraceRazorProcessor",
    "StreamingGuardrailDetector",
    "arguments_digest",
    "auto_instrument",
    "compile_native_trace",
    "configure",
    "content_digest",
    "format_traceparent",
    "get_current_processor",
    "new_run_id",
    "new_span_id",
    "new_trace_id",
    "parse_traceparent",
    "recover_partial_run",
    "register_instrumentation",
    "registered_instrumentations",
    "unregister_instrumentation",
    "validate_native_trace_shape",
]
