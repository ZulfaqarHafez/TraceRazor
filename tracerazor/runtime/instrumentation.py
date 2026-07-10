"""Optional, lazy runtime instrumentation registry."""

from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, Mapping

from .models import RunContext, TokenProvenance, TokenUsage, ToolCall, ToolStatus
from .processor import TraceRazorProcessor, get_current_processor


Installer = Callable[[TraceRazorProcessor], Any]


@dataclass(frozen=True)
class InstrumentationResult:
    enabled: tuple[str, ...] = ()
    unavailable: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    handles: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": list(self.enabled),
            "unavailable": dict(self.unavailable),
            "errors": dict(self.errors),
        }


_REGISTRY: dict[str, Installer] = {}
_REGISTRY_LOCK = threading.Lock()


def register_instrumentation(name: str, installer: Installer, *, replace: bool = False) -> None:
    """Register a lazy integration without importing its optional SDK."""

    if not name or not name.strip():
        raise ValueError("instrumentation name must be non-empty")
    if not callable(installer):
        raise TypeError("instrumentation installer must be callable")
    normalized = name.strip().lower().replace("-", "_")
    with _REGISTRY_LOCK:
        if normalized in _REGISTRY and not replace:
            raise ValueError(f"instrumentation already registered: {normalized}")
        _REGISTRY[normalized] = installer


def unregister_instrumentation(name: str) -> None:
    with _REGISTRY_LOCK:
        _REGISTRY.pop(name.strip().lower().replace("-", "_"), None)


def registered_instrumentations() -> tuple[str, ...]:
    with _REGISTRY_LOCK:
        return tuple(sorted(_REGISTRY))


def auto_instrument(
    *names: str | Iterable[str],
    processor: TraceRazorProcessor | None = None,
) -> InstrumentationResult:
    """Activate available registered integrations.

    Missing optional SDKs are reported in ``unavailable`` and never raise.
    Installer bugs are isolated in ``errors`` so one framework cannot prevent
    another from being instrumented.
    """

    selected: list[str] = []
    for value in names:
        if isinstance(value, str):
            selected.append(value)
        else:
            selected.extend(str(item) for item in value)
    with _REGISTRY_LOCK:
        registry = dict(_REGISTRY)
    if not selected:
        selected = sorted(registry)
    runtime = processor or get_current_processor()
    if runtime is None:
        return InstrumentationResult(
            unavailable={name: "TraceRazor runtime is not configured" for name in selected}
        )

    enabled: list[str] = []
    unavailable: dict[str, str] = {}
    errors: dict[str, str] = {}
    handles: dict[str, Any] = {}
    for raw_name in selected:
        name = raw_name.strip().lower().replace("-", "_")
        installer = registry.get(name)
        if installer is None:
            unavailable[name] = "integration is not registered"
            continue
        try:
            handle = installer(runtime)
        except (ModuleNotFoundError, ImportError) as exc:
            unavailable[name] = f"optional SDK is unavailable: {exc}"
        except Exception as exc:  # isolate third-party SDK registration failures
            errors[name] = f"{type(exc).__name__}: {exc}"
        else:
            if handle is None or handle is False:
                unavailable[name] = "SDK does not expose automatic runtime registration"
            else:
                enabled.append(name)
                handles[name] = handle
    return InstrumentationResult(tuple(enabled), unavailable, errors, handles)


def _dig(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    exporter = getattr(value, "export", None)
    if callable(exporter):
        exported = exporter()
        if isinstance(exported, dict):
            return exported
    data = getattr(value, "__dict__", None)
    return dict(data) if isinstance(data, dict) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    return None


def _usage_candidates(value: Any) -> list[Mapping[str, Any]]:
    """Return shallow provider-usage mappings without importing an SDK type."""

    result: list[Mapping[str, Any]] = []
    queue = [value]
    seen: set[int] = set()
    for _ in range(12):
        if not queue:
            break
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        mapping = _as_mapping(current)
        if mapping:
            result.append(mapping)
            for key in (
                "usage",
                "token_usage",
                "usage_metadata",
                "llm_output",
                "response_metadata",
            ):
                nested = mapping.get(key)
                if nested is not None:
                    queue.append(nested)
            generations = mapping.get("generations")
            if isinstance(generations, (list, tuple)):
                for group in generations[:4]:
                    if isinstance(group, (list, tuple)):
                        queue.extend(group[:4])
                    else:
                        queue.append(group)
        message = getattr(current, "message", None)
        if message is not None:
            queue.append(message)
        for name in ("usage", "token_usage", "usage_metadata", "llm_output"):
            nested = getattr(current, name, None)
            if nested is not None:
                queue.append(nested)
    return result


def _extract_token_usage(value: Any) -> TokenUsage:
    """Normalize provider usage when an SDK supplies it; never estimate text."""

    input_keys = ("input_tokens", "prompt_tokens")
    output_keys = ("output_tokens", "completion_tokens")
    cache_read_keys = (
        "cache_read_tokens",
        "cache_read_input_tokens",
        "cached_tokens",
        "cache_read",
    )
    cache_write_keys = (
        "cache_write_tokens",
        "cache_creation_input_tokens",
        "cache_write",
    )
    reasoning_keys = ("reasoning_tokens", "thinking_tokens", "reasoning")
    known_keys = set(
        input_keys
        + output_keys
        + cache_read_keys
        + cache_write_keys
        + reasoning_keys
        + ("total_tokens", "total",)
    )

    for mapping in _usage_candidates(value):
        if not known_keys.intersection(mapping):
            continue

        def first(keys: tuple[str, ...]) -> int:
            for key in keys:
                parsed = _non_negative_int(mapping.get(key))
                if parsed is not None:
                    return parsed
            return 0

        has_split = any(key in mapping for key in input_keys + output_keys)
        input_tokens = first(input_keys)
        output_tokens = first(output_keys)
        if not has_split and any(key in mapping for key in ("total_tokens", "total")):
            # The total is provider-supplied, but the schema has no total-only
            # representation. Preserve it while marking the unavailable split
            # as estimated/degraded rather than claiming an exact breakdown.
            input_details = _as_mapping(
                mapping.get("input_token_details") or mapping.get("prompt_tokens_details")
            )
            output_details = _as_mapping(
                mapping.get("output_token_details") or mapping.get("completion_tokens_details")
            )
            detail_cache_read = next(
                (
                    parsed
                    for key in ("cache_read", "cache_read_tokens", "cached_tokens")
                    if (parsed := _non_negative_int(input_details.get(key))) is not None
                ),
                0,
            )
            detail_reasoning = next(
                (
                    parsed
                    for key in ("reasoning", "reasoning_tokens", "thinking_tokens")
                    if (parsed := _non_negative_int(output_details.get(key))) is not None
                ),
                0,
            )
            return TokenUsage.estimated(
                input_tokens=first(("total_tokens", "total")),
                cache_read_tokens=first(cache_read_keys) or detail_cache_read,
                cache_write_tokens=first(cache_write_keys),
                reasoning_tokens=first(reasoning_keys) or detail_reasoning,
            )
        input_details = _as_mapping(
            mapping.get("input_token_details") or mapping.get("prompt_tokens_details")
        )
        output_details = _as_mapping(
            mapping.get("output_token_details") or mapping.get("completion_tokens_details")
        )
        cache_read = first(cache_read_keys)
        if cache_read == 0:
            cache_read = next(
                (
                    parsed
                    for key in ("cache_read", "cache_read_tokens", "cached_tokens")
                    if (parsed := _non_negative_int(input_details.get(key))) is not None
                ),
                0,
            )
        cache_write = first(cache_write_keys)
        if cache_write == 0:
            cache_write = next(
                (
                    parsed
                    for key in ("cache_write", "cache_write_tokens", "cache_creation")
                    if (parsed := _non_negative_int(input_details.get(key))) is not None
                ),
                0,
            )
        reasoning = first(reasoning_keys)
        if reasoning == 0:
            reasoning = next(
                (
                    parsed
                    for key in ("reasoning", "reasoning_tokens", "thinking_tokens")
                    if (parsed := _non_negative_int(output_details.get(key))) is not None
                ),
                0,
            )
        return TokenUsage.reported(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_tokens=reasoning,
        )
    return TokenUsage(provenance=TokenProvenance.MISSING)


def _generation_text(value: Any) -> str:
    mapping = _as_mapping(value)
    generations = mapping.get("generations") or getattr(value, "generations", None)
    if generations:
        try:
            generation = generations[0][0]
        except (IndexError, KeyError, TypeError):
            generation = None
        if generation is not None:
            message = getattr(generation, "message", None)
            for candidate in (
                getattr(generation, "text", None),
                getattr(message, "content", None),
                _as_mapping(generation).get("text"),
                _as_mapping(message).get("content"),
            ):
                if candidate is not None:
                    return _text(candidate)
    for candidate in (
        mapping.get("response"),
        mapping.get("output"),
        getattr(value, "response", None),
        getattr(value, "output", None),
        getattr(value, "content", None),
    ):
        if candidate is not None:
            return _text(candidate)
    return ""


class _RuntimeAdapter:
    """Failure-isolating adapter base shared by callback-driven hosts."""

    def __init__(self, runtime: TraceRazorProcessor) -> None:
        self.runtime = runtime
        self._errors: list[str] = []
        self._lock = threading.RLock()

    @property
    def errors(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._errors)

    def _error(self, exc: BaseException) -> None:
        with self._lock:
            self._errors.append(f"{type(exc).__name__}: {exc}")

    def _record(self, event_type: str, **kwargs: Any) -> None:
        try:
            with self._lock:
                if self.runtime.finalized:
                    return
                self.runtime.record(event_type, **kwargs)
        except Exception as exc:
            self._error(exc)

    def finish(self, *, output: Any = None) -> None:
        try:
            with self._lock:
                if self.runtime.finalized:
                    return
                if output is not None:
                    self.runtime.record(
                        "run_end",
                        output=_text(output),
                        metadata={"source": type(self).__name__},
                    )
                self.runtime.finalize()
        except Exception as exc:
            self._error(exc)

    def fail(self, error: Any) -> None:
        try:
            with self._lock:
                if self.runtime.finalized:
                    return
                self.runtime.record(
                    "error",
                    content=_text(error),
                    metadata={"source": type(self).__name__, "error_type": type(error).__name__},
                )
                self.runtime.finalize(status="error", error=error)
        except Exception as exc:
            self._error(exc)


class _LangGraphCallback(_RuntimeAdapter):
    """LangChain callback protocol implementation used by LangGraph."""

    raise_error = False
    run_inline = True

    def __init__(self, runtime: TraceRazorProcessor) -> None:
        try:
            super().__init__(runtime)
        except TypeError:  # defensive for an SDK base with a different init
            _RuntimeAdapter.__init__(self, runtime)
        self._root_runs: set[str] = set()
        self._llm: dict[str, dict[str, Any]] = {}
        self._tools: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _run_id(value: Any) -> str:
        return _text(value) or f"callback:{time.monotonic_ns()}"

    def on_chain_start(
        self,
        serialized: Mapping[str, Any] | None,
        inputs: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            current = self._run_id(run_id)
            parent = _text(parent_run_id) or self.runtime.context.span_id
            name = _text((serialized or {}).get("name") or kwargs.get("name") or "langgraph")
            if parent_run_id is None:
                with self._lock:
                    self._root_runs.add(current)
                self._record(
                    "run_start",
                    content=name,
                    input_context=_text(inputs),
                    span_id=current,
                    parent_span_id=parent,
                    metadata={"source": "langgraph_callback", "run_name": name},
                )
            else:
                self._record(
                    "handoff",
                    content=name,
                    input_context=_text(inputs),
                    span_id=current,
                    parent_span_id=parent,
                    metadata={"source": "langgraph_callback", "run_name": name},
                )
        except Exception as exc:
            self._error(exc)

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            current = self._run_id(run_id)
            with self._lock:
                is_root = current in self._root_runs
            if parent_run_id is None and is_root:
                self._record(
                    "run_end",
                    output=_text(outputs),
                    span_id=current,
                    parent_span_id=self.runtime.context.span_id,
                    metadata={"source": "langgraph_callback"},
                )
                self.finish()
        except Exception as exc:
            self._error(exc)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self.fail(error)
        else:
            self._record(
                "error",
                content=_text(error),
                span_id=self._run_id(run_id),
                parent_span_id=_text(parent_run_id) or self.runtime.context.span_id,
                metadata={"source": "langgraph_callback", "error_type": type(error).__name__},
            )

    def on_llm_start(
        self,
        serialized: Mapping[str, Any] | None,
        prompts: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            current = self._run_id(run_id)
            serialized_map = _as_mapping(serialized)
            with self._lock:
                self._llm[current] = {
                    "input": _text(prompts),
                    "parent": _text(parent_run_id) or self.runtime.context.span_id,
                    "model": _text(
                        serialized_map.get("name") or kwargs.get("invocation_params", {})
                    ),
                }
        except Exception as exc:
            self._error(exc)

    def on_chat_model_start(
        self,
        serialized: Mapping[str, Any] | None,
        messages: Any,
        **kwargs: Any,
    ) -> None:
        self.on_llm_start(serialized, messages, **kwargs)

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        try:
            current = self._run_id(run_id)
            with self._lock:
                pending = self._llm.pop(current, {})
            self._record(
                "reasoning",
                content=pending.get("model") or "language model call",
                input_context=pending.get("input"),
                output=_generation_text(response),
                tokens=_extract_token_usage(response),
                span_id=current,
                parent_span_id=pending.get("parent") or self.runtime.context.span_id,
                metadata={"source": "langgraph_callback"},
            )
        except Exception as exc:
            self._error(exc)

    def on_llm_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        current = self._run_id(run_id)
        with self._lock:
            pending = self._llm.pop(current, {})
        self._record(
            "error",
            content=_text(error),
            span_id=current,
            parent_span_id=pending.get("parent") or self.runtime.context.span_id,
            metadata={"source": "langgraph_callback", "error_type": type(error).__name__},
        )

    def on_tool_start(
        self,
        serialized: Mapping[str, Any] | None,
        input_str: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            current = self._run_id(run_id)
            serialized_map = _as_mapping(serialized)
            with self._lock:
                self._tools[current] = {
                    "name": _text(serialized_map.get("name") or kwargs.get("name") or "tool"),
                    "arguments": input_str,
                    "parent": _text(parent_run_id) or self.runtime.context.span_id,
                    "started": time.perf_counter(),
                    "expected_failure": bool(kwargs.get("expected_failure", False)),
                }
        except Exception as exc:
            self._error(exc)

    def _finish_tool(self, run_id: Any, output: Any, error: Any = None) -> None:
        current = self._run_id(run_id)
        with self._lock:
            pending = self._tools.pop(current, {})
        output_text = _text(output)
        started = pending.get("started")
        duration_ms = (
            max(0.0, (time.perf_counter() - started) * 1000.0)
            if isinstance(started, (int, float))
            else None
        )
        self._record(
            "tool_call",
            content=f"Calling {pending.get('name') or 'tool'}",
            output=output_text or None,
            tokens=TokenUsage(provenance=TokenProvenance.MISSING),
            tool=ToolCall.from_arguments(
                pending.get("name") or "tool",
                pending.get("arguments"),
                status=ToolStatus.ERROR if error is not None else ToolStatus.SUCCESS,
                duration_ms=duration_ms,
                expected_failure=bool(pending.get("expected_failure", False)),
                observation_size=len(output_text),
                error_type=type(error).__name__ if error is not None else None,
            ),
            span_id=current,
            parent_span_id=pending.get("parent") or self.runtime.context.span_id,
            metadata={"source": "langgraph_callback"},
        )

    def on_tool_end(self, output: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        try:
            self._finish_tool(run_id, output)
        except Exception as exc:
            self._error(exc)

    def on_tool_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        try:
            self._finish_tool(run_id, None, error)
        except Exception as exc:
            self._error(exc)


class LangGraphInstrumentationHandle:
    """Explicit per-invocation callback attachment for LangGraph/LangChain."""

    def __init__(self, runtime: TraceRazorProcessor, callback: _LangGraphCallback) -> None:
        self.runtime = runtime
        self.callback = callback

    @property
    def errors(self) -> tuple[str, ...]:
        return self.callback.errors

    def attach(self, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return a copied RunnableConfig with this callback appended."""

        result = dict(config or {})
        existing = result.get("callbacks")
        if existing is None:
            callbacks: list[Any] = []
        elif isinstance(existing, (list, tuple)):
            callbacks = list(existing)
        else:
            raise TypeError(
                "config['callbacks'] must be a list/tuple for non-mutating attachment; "
                "attach handle.callback to a callback manager explicitly"
            )
        if self.callback not in callbacks:
            callbacks.append(self.callback)
        result["callbacks"] = callbacks
        return result

    config = attach

    def invoke(
        self,
        graph: Any,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke one graph with the callback and fail-safe finalization."""

        try:
            output = graph.invoke(inputs, config=self.attach(config), **kwargs)
        except Exception as exc:
            self.callback.fail(exc)
            raise
        self.callback.finish(output=output)
        return output

    async def ainvoke(
        self,
        graph: Any,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Asynchronously invoke one graph with fail-safe finalization."""

        try:
            output = await graph.ainvoke(inputs, config=self.attach(config), **kwargs)
        except Exception as exc:
            self.callback.fail(exc)
            raise
        self.callback.finish(output=output)
        return output

    def stream(
        self,
        graph: Any,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Stream one graph and finalize on exhaustion, error, or early close."""

        completed = False
        try:
            for item in graph.stream(inputs, config=self.attach(config), **kwargs):
                yield item
            completed = True
        except Exception as exc:
            self.callback.fail(exc)
            raise
        finally:
            if not self.runtime.finalized:
                if completed:
                    self.callback.finish()
                else:
                    try:
                        self.runtime.mark_partial("LangGraph stream closed before exhaustion")
                    except Exception as exc:
                        self.callback._error(exc)

    async def astream(
        self,
        graph: Any,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Asynchronously stream and finalize with the synchronous contract."""

        completed = False
        try:
            async for item in graph.astream(inputs, config=self.attach(config), **kwargs):
                yield item
            completed = True
        except Exception as exc:
            self.callback.fail(exc)
            raise
        finally:
            if not self.runtime.finalized:
                if completed:
                    self.callback.finish()
                else:
                    try:
                        self.runtime.mark_partial("LangGraph async stream closed before exhaustion")
                    except Exception as exc:
                        self.callback._error(exc)

    def finish(self, *, output: Any = None) -> None:
        self.callback.finish(output=output)


class CrewAIInstrumentationHandle(_RuntimeAdapter):
    """Explicit, detachable CrewAI event-bus listener."""

    _EVENT_METHODS = {
        "CrewKickoffStartedEvent": "_on_crew_start",
        "CrewKickoffCompletedEvent": "_on_crew_end",
        "CrewKickoffFailedEvent": "_on_crew_failed",
        "LLMCallCompletedEvent": "_on_llm_end",
        "LLMCallFailedEvent": "_on_llm_failed",
        "ToolUsageFinishedEvent": "_on_tool_end",
        "ToolUsageErrorEvent": "_on_tool_error",
        "ToolExecutionErrorEvent": "_on_tool_error",
        "ToolValidateInputErrorEvent": "_on_tool_error",
        "ToolSelectionErrorEvent": "_on_tool_error",
    }

    def __init__(self, runtime: TraceRazorProcessor, events_module: Any) -> None:
        super().__init__(runtime)
        self._events_module = events_module
        self._bus = getattr(events_module, "crewai_event_bus")
        self._registrations: list[tuple[type[Any], Callable[..., Any]]] = []
        self._scope_ids: set[int] | None = None
        self._scope_text_ids: set[str] | None = None

    @property
    def attached(self) -> bool:
        return bool(self._registrations)

    def attach(self, crew: Any = None) -> "CrewAIInstrumentationHandle":
        """Register official event-bus handlers, optionally scoped to one crew."""

        with self._lock:
            if self._registrations:
                return self
            if crew is not None:
                members = [crew]
                members.extend(list(getattr(crew, "agents", None) or []))
                members.extend(list(getattr(crew, "tasks", None) or []))
                self._scope_ids = {id(item) for item in members}
                self._scope_text_ids = {
                    _text(getattr(item, "id", None)) for item in members if getattr(item, "id", None)
                }
            else:
                self._scope_ids = None
                self._scope_text_ids = None
            for event_name, method_name in self._EVENT_METHODS.items():
                event_type = getattr(self._events_module, event_name, None)
                if not isinstance(event_type, type):
                    continue
                method = getattr(self, method_name)

                def handler(source: Any, event: Any, _method: Callable[..., Any] = method) -> None:
                    try:
                        _method(source, event)
                    except Exception as exc:
                        self._error(exc)

                self._bus.on(event_type)(handler)
                self._registrations.append((event_type, handler))
            if not self._registrations:
                raise RuntimeError("CrewAI SDK exposes no supported event classes")
        return self

    def detach(self) -> None:
        """Remove only the handlers installed by this handle."""

        with self._lock:
            registrations = tuple(self._registrations)
            self._registrations.clear()
        off = getattr(self._bus, "off", None)
        if callable(off):
            for event_type, handler in registrations:
                try:
                    off(event_type, handler)
                except Exception as exc:
                    self._error(exc)

    close = detach

    def _accept(self, source: Any, event: Any) -> bool:
        if self._scope_ids is None:
            return True
        candidates = (
            source,
            getattr(source, "crew", None),
            getattr(event, "crew", None),
            getattr(event, "from_agent", None),
            getattr(event, "from_task", None),
        )
        if any(candidate is not None and id(candidate) in self._scope_ids for candidate in candidates):
            return True
        text_ids = {
            _text(getattr(event, name, None))
            for name in ("crew_id", "agent_id", "task_id")
            if getattr(event, name, None) is not None
        }
        return bool(text_ids.intersection(self._scope_text_ids or set()))

    def _span(self, event: Any) -> dict[str, str]:
        span_id = _text(getattr(event, "event_id", None))
        parent_span_id = _text(getattr(event, "parent_event_id", None))
        result: dict[str, str] = {}
        if span_id:
            result["span_id"] = span_id
        if parent_span_id and parent_span_id != span_id:
            result["parent_span_id"] = parent_span_id
        return result

    def _on_crew_start(self, source: Any, event: Any) -> None:
        if not self._accept(source, event):
            return
        self._record(
            "run_start",
            content=_text(getattr(event, "crew_name", None) or "CrewAI kickoff"),
            input_context=_text(getattr(event, "inputs", None)),
            metadata={"source": "crewai_event_bus"},
            **self._span(event),
        )

    def _on_crew_end(self, source: Any, event: Any) -> None:
        if not self._accept(source, event):
            return
        self._record(
            "run_end",
            output=_text(getattr(event, "output", None)),
            tokens=_extract_token_usage(event),
            metadata={"source": "crewai_event_bus"},
            **self._span(event),
        )
        self.finish()

    def _on_crew_failed(self, source: Any, event: Any) -> None:
        if self._accept(source, event):
            self.fail(getattr(event, "error", "CrewAI kickoff failed"))

    def _on_llm_end(self, source: Any, event: Any) -> None:
        if not self._accept(source, event):
            return
        self._record(
            "reasoning",
            content=_text(getattr(event, "model", None) or "CrewAI LLM call"),
            input_context=_text(getattr(event, "messages", None)),
            output=_generation_text(event),
            tokens=_extract_token_usage(event),
            metadata={
                "source": "crewai_event_bus",
                "call_id": _text(getattr(event, "call_id", None)),
                "agent_id": _text(getattr(event, "agent_id", None)),
                "task_id": _text(getattr(event, "task_id", None)),
            },
            **self._span(event),
        )

    def _on_llm_failed(self, source: Any, event: Any) -> None:
        if not self._accept(source, event):
            return
        self._record(
            "error",
            content=_text(getattr(event, "error", "CrewAI LLM call failed")),
            metadata={"source": "crewai_event_bus", "phase": "llm"},
            **self._span(event),
        )

    @staticmethod
    def _duration_ms(event: Any) -> float | None:
        started = getattr(event, "started_at", None)
        finished = getattr(event, "finished_at", None)
        if isinstance(started, datetime) and isinstance(finished, datetime):
            try:
                return max(0.0, (finished - started).total_seconds() * 1000.0)
            except (TypeError, ValueError):
                return None
        return None

    def _on_tool_end(self, source: Any, event: Any) -> None:
        if not self._accept(source, event):
            return
        output = _text(getattr(event, "output", None))
        name = _text(getattr(event, "tool_name", None) or "tool")
        self._record(
            "tool_call",
            content=f"Calling {name}",
            output=output or None,
            tokens=TokenUsage(provenance=TokenProvenance.MISSING),
            tool=ToolCall.from_arguments(
                name,
                getattr(event, "tool_args", None),
                status=ToolStatus.SUCCESS,
                duration_ms=self._duration_ms(event),
                observation_size=len(output),
            ),
            metadata={"source": "crewai_event_bus"},
            **self._span(event),
        )

    def _on_tool_error(self, source: Any, event: Any) -> None:
        if not self._accept(source, event):
            return
        name = _text(getattr(event, "tool_name", None) or "tool")
        error = getattr(event, "error", "CrewAI tool failed")
        self._record(
            "tool_call",
            content=f"Calling {name}",
            tokens=TokenUsage(provenance=TokenProvenance.MISSING),
            tool=ToolCall.from_arguments(
                name,
                getattr(event, "tool_args", None),
                status=ToolStatus.ERROR,
                observation_size=0,
                error_type=type(error).__name__,
            ),
            metadata={"source": "crewai_event_bus", "error": _text(error)},
            **self._span(event),
        )


class _OpenAIAgentsProcessor:
    """Process-isolated adapter for the OpenAI Agents tracing API.

    One SDK processor is process-global, but each SDK trace owns a distinct
    TraceRazorProcessor.  Callback failures are retained for diagnostics and
    never raised into the host agent.
    """

    def __init__(self, runtime: TraceRazorProcessor) -> None:
        self.runtime = runtime
        self._processors: dict[str, TraceRazorProcessor] = {}
        self._base_claimed = False
        self._lock = threading.RLock()
        self._errors: list[str] = []
        self._unregister: Callable[[], Any] | None = None

    @property
    def processors(self) -> tuple[TraceRazorProcessor, ...]:
        with self._lock:
            return tuple(self._processors.values())

    @property
    def errors(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._errors)

    @staticmethod
    def _identifier(value: Any, *names: str) -> str | None:
        mapping = _as_mapping(value)
        for name in names:
            item = mapping.get(name)
            if item is None and hasattr(value, name):
                item = getattr(value, name)
            if item is not None and str(item).strip():
                return str(item)
        return None

    def _trace_key(self, trace: Any) -> str:
        return self._identifier(trace, "trace_id", "id") or f"object:{id(trace)}"

    def _clone_runtime(self, sdk_trace_id: str | None) -> TraceRazorProcessor:
        context = RunContext.create(
            agent_id=self.runtime.context.agent_id,
            session_id=self.runtime.context.session_id,
            trace_id=sdk_trace_id,
        )
        return TraceRazorProcessor(
            context=context,
            policy=self.runtime.policy,
            artifact_dir=self.runtime.run_dir.parent,
            host=self.runtime.host,
            host_version=self.runtime.host_version,
            framework=self.runtime.framework,
            framework_version=self.runtime.framework_version,
            guardrails=self.runtime.guardrails.config,
            auditor=self.runtime._auditor,
        )

    def _processor_for_key(
        self,
        key: str,
        *,
        sdk_trace_id: str | None,
        create: bool,
    ) -> TraceRazorProcessor | None:
        with self._lock:
            existing = self._processors.get(key)
            if existing is not None or not create:
                return existing
            if not self._base_claimed and not self.runtime.finalized and not self.runtime.events:
                processor = self.runtime
                self._base_claimed = True
            else:
                processor = self._clone_runtime(sdk_trace_id)
            self._processors[key] = processor
            return processor

    def _record_error(self, exc: BaseException) -> None:
        with self._lock:
            self._errors.append(type(exc).__name__)

    def on_trace_start(self, trace: Any) -> None:  # API protocol method
        try:
            key = self._trace_key(trace)
            sdk_trace_id = self._identifier(trace, "trace_id", "id")
            self._processor_for_key(key, sdk_trace_id=sdk_trace_id, create=True)
        except Exception as exc:  # never break the host tracing pipeline
            self._record_error(exc)

    def on_trace_end(self, trace: Any) -> None:  # API protocol method
        try:
            key = self._trace_key(trace)
            processor = self._processor_for_key(key, sdk_trace_id=None, create=False)
            if processor is not None and not processor.finalized:
                processor.finalize()
        except Exception as exc:
            self._record_error(exc)

    def on_span_start(self, span: Any) -> None:  # API protocol method
        return None

    def on_span_end(self, span: Any) -> None:  # API protocol method
        try:
            payload = _as_mapping(span)
            span_data = payload.get("span_data") or payload.get("data") or payload
            data = _as_mapping(span_data)
            sdk_trace_id = (
                self._identifier(span, "trace_id")
                or self._identifier(span_data, "trace_id")
            )
            if sdk_trace_id is not None:
                key = sdk_trace_id
            else:
                with self._lock:
                    active = [
                        item for item in self._processors.items() if not item[1].finalized
                    ]
                if len(active) != 1:
                    return
                key = active[0][0]
            processor = self._processor_for_key(
                key,
                sdk_trace_id=sdk_trace_id,
                create=True,
            )
            if processor is None or processor.finalized:
                return
            kind_text = str(
                data.get("type")
                or data.get("span_type")
                or type(span_data).__name__
            ).lower()
            event_type = (
                "tool_call"
                if any(item in kind_text for item in ("tool", "function"))
                else "reasoning"
            )

            usage = data.get("usage") or payload.get("usage") or {}
            usage_map = _as_mapping(usage)
            input_tokens = int(usage_map.get("input_tokens") or usage_map.get("input") or 0)
            output_tokens = int(usage_map.get("output_tokens") or usage_map.get("output") or 0)
            has_usage = bool(usage_map) and any(
                item in usage_map
                for item in (
                    "input_tokens",
                    "input",
                    "output_tokens",
                    "output",
                    "cached_tokens",
                    "cache_read",
                    "cache_write",
                    "reasoning_tokens",
                )
            )
            tokens = TokenUsage(
                input=input_tokens,
                output=output_tokens,
                cache_read=int(usage_map.get("cached_tokens") or usage_map.get("cache_read") or 0),
                cache_write=int(usage_map.get("cache_write") or 0),
                reasoning=int(usage_map.get("reasoning_tokens") or 0),
                provenance=(
                    TokenProvenance.PROVIDER_REPORTED
                    if has_usage
                    else TokenProvenance.MISSING
                ),
            )
            content = data.get("input") or data.get("content") or data.get("name") or kind_text
            output = data.get("output")
            sdk_span_id = self._identifier(span, "span_id", "id")
            sdk_parent_id = self._identifier(span, "parent_id", "parent_span_id")
            kwargs: dict[str, Any] = {
                "tokens": tokens,
                "content": str(content),
                "output": str(output) if output is not None else None,
                "span_id": sdk_span_id,
                "parent_span_id": sdk_parent_id,
                "metadata": {
                    "source": "openai_agents_trace_processor",
                    "sdk_trace_id": sdk_trace_id,
                    "sdk_span_id": sdk_span_id,
                    "sdk_parent_span_id": sdk_parent_id,
                },
            }
            if event_type == "tool_call":
                from .models import ToolCall, ToolStatus

                name = str(data.get("name") or data.get("tool_name") or "tool")
                arguments = data.get("arguments") or data.get("input") or {}
                error = data.get("error")
                kwargs["tool"] = ToolCall.from_arguments(
                    name,
                    arguments,
                    status=ToolStatus.ERROR if error else ToolStatus.SUCCESS,
                    error_type=type(error).__name__ if error else None,
                    observation_size=len(str(output)) if output is not None else 0,
                )
            processor.record(event_type, **kwargs)
        except Exception as exc:
            self._record_error(exc)

    def force_flush(self) -> None:  # API protocol method
        return None

    def shutdown(self) -> None:  # API protocol method
        with self._lock:
            processors = tuple(self._processors.values())
        for processor in processors:
            try:
                if not processor.finalized:
                    processor.mark_partial("trace processor shut down before trace end")
            except Exception as exc:
                self._record_error(exc)
        unregister = self._unregister
        self._unregister = None
        if unregister is not None:
            try:
                unregister()
            except Exception as exc:
                self._record_error(exc)


def _install_openai_agents(runtime: TraceRazorProcessor) -> Any:
    agents = importlib.import_module("agents")
    tracing = getattr(agents, "tracing", None)
    register = getattr(tracing, "add_trace_processor", None) if tracing else None
    if register is None:
        register = getattr(agents, "add_trace_processor", None)
    if not callable(register):
        return None
    adapter = _OpenAIAgentsProcessor(runtime)
    register(adapter)
    remove = getattr(tracing, "remove_trace_processor", None) if tracing else None
    if remove is None:
        remove = getattr(agents, "remove_trace_processor", None)
    if callable(remove):
        adapter._unregister = lambda: remove(adapter)
    return adapter


def _install_langgraph(runtime: TraceRazorProcessor) -> LangGraphInstrumentationHandle:
    langgraph = importlib.import_module("langgraph")
    callbacks = importlib.import_module("langchain_core.callbacks")
    base_callback = getattr(callbacks, "BaseCallbackHandler", None)
    if not isinstance(base_callback, type):
        raise ImportError("langchain_core.callbacks.BaseCallbackHandler is unavailable")
    callback_type = type(
        "TraceRazorLangGraphCallback",
        (_LangGraphCallback, base_callback),
        {"__module__": __name__},
    )
    callback = callback_type(runtime)
    if runtime.framework == "unknown":
        runtime.framework = "langgraph"
    if runtime.framework_version is None:
        runtime.framework_version = _text(getattr(langgraph, "__version__", None)) or None
    return LangGraphInstrumentationHandle(runtime, callback)


def _install_crewai(runtime: TraceRazorProcessor) -> CrewAIInstrumentationHandle:
    crewai = importlib.import_module("crewai")
    events = importlib.import_module("crewai.events")
    bus = getattr(events, "crewai_event_bus", None)
    if bus is None or not callable(getattr(bus, "on", None)) or not callable(
        getattr(bus, "off", None)
    ):
        raise ImportError("CrewAI event bus needs callable on/off registration methods")
    if runtime.framework == "unknown":
        runtime.framework = "crewai"
    if runtime.framework_version is None:
        runtime.framework_version = _text(getattr(crewai, "__version__", None)) or None
    # Registration is intentionally deferred to handle.attach(). CrewAI's bus
    # is process-global, so discovering the SDK must not start capturing other
    # crews without a second explicit action.
    return CrewAIInstrumentationHandle(runtime, events)


register_instrumentation("openai_agents", _install_openai_agents)
register_instrumentation("langgraph", _install_langgraph)
register_instrumentation("crewai", _install_crewai)


__all__ = [
    "InstrumentationResult",
    "CrewAIInstrumentationHandle",
    "LangGraphInstrumentationHandle",
    "auto_instrument",
    "register_instrumentation",
    "registered_instrumentations",
    "unregister_instrumentation",
]
