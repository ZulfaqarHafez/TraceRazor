from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from types import ModuleType, SimpleNamespace

import pytest

from tracerazor.runtime import AuditPolicy, TraceRazorProcessor, auto_instrument


class _BaseCallbackHandler:
    pass


def _install_fake_langgraph(monkeypatch):
    langgraph = ModuleType("langgraph")
    langgraph.__version__ = "1.2.3"
    langchain_core = ModuleType("langchain_core")
    callbacks = ModuleType("langchain_core.callbacks")
    callbacks.BaseCallbackHandler = _BaseCallbackHandler
    monkeypatch.setitem(sys.modules, "langgraph", langgraph)
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", callbacks)


class _FakeEventBus:
    def __init__(self):
        self.handlers = defaultdict(list)

    def on(self, event_type):
        def register(handler):
            self.handlers[event_type].append(handler)
            return handler

        return register

    def off(self, event_type, handler):
        self.handlers[event_type].remove(handler)

    def emit(self, source, event):
        for event_type, handlers in list(self.handlers.items()):
            if isinstance(event, event_type):
                for handler in list(handlers):
                    handler(source, event)


class _Event:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _install_fake_crewai(monkeypatch):
    crewai = ModuleType("crewai")
    crewai.__version__ = "1.14.7"
    events = ModuleType("crewai.events")
    bus = _FakeEventBus()
    events.crewai_event_bus = bus
    names = (
        "CrewKickoffStartedEvent",
        "CrewKickoffCompletedEvent",
        "CrewKickoffFailedEvent",
        "LLMCallCompletedEvent",
        "LLMCallFailedEvent",
        "ToolUsageFinishedEvent",
        "ToolUsageErrorEvent",
        "ToolExecutionErrorEvent",
        "ToolValidateInputErrorEvent",
        "ToolSelectionErrorEvent",
    )
    classes = {}
    for name in names:
        event_type = type(name, (_Event,), {})
        setattr(events, name, event_type)
        classes[name] = event_type
    monkeypatch.setitem(sys.modules, "crewai", crewai)
    monkeypatch.setitem(sys.modules, "crewai.events", events)
    return bus, classes


def _runtime(tmp_path):
    return TraceRazorProcessor(
        policy=AuditPolicy(mode="off", artifact_dir=str(tmp_path)),
    )


def test_langgraph_handle_attaches_callback_without_mutating_config_and_captures_run(
    tmp_path, monkeypatch
):
    _install_fake_langgraph(monkeypatch)
    runtime = _runtime(tmp_path)

    result = auto_instrument("langgraph", processor=runtime)

    assert result.enabled == ("langgraph",)
    handle = result.handles["langgraph"]
    sentinel = object()
    original = {"callbacks": [sentinel], "metadata": {"tenant": "test"}}
    attached = handle.attach(original)
    assert original["callbacks"] == [sentinel]
    assert attached["callbacks"] == [sentinel, handle.callback]
    assert attached["metadata"] == original["metadata"]

    callback = handle.callback
    callback.on_chain_start(
        {"name": "CompiledGraph"},
        {"messages": ["hello"]},
        run_id="root-run",
        parent_run_id=None,
    )
    callback.on_llm_start(
        {"name": "provider-model"},
        ["hello"],
        run_id="llm-run",
        parent_run_id="root-run",
    )
    response = SimpleNamespace(
        generations=[[SimpleNamespace(text="answer")]],
        llm_output={
            "token_usage": {
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "cached_tokens": 2,
            }
        },
    )
    callback.on_llm_end(response, run_id="llm-run")
    callback.on_chat_model_start(
        {"name": "chat-model"},
        [[SimpleNamespace(content="follow up")]],
        run_id="chat-run",
        parent_run_id="root-run",
    )
    callback.on_llm_end(
        SimpleNamespace(
            generations=[[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="chat answer",
                        usage_metadata={
                            "input_tokens": 7,
                            "output_tokens": 3,
                            "input_token_details": {"cache_read": 2},
                            "output_token_details": {"reasoning": 1},
                        },
                    )
                )
            ]],
            llm_output=None,
        ),
        run_id="chat-run",
    )
    callback.on_tool_start(
        {"name": "search"},
        '{"query":"answer"}',
        run_id="tool-run",
        parent_run_id="root-run",
    )
    callback.on_tool_end("result", run_id="tool-run")
    callback.on_chain_end({"answer": "done"}, run_id="root-run", parent_run_id=None)

    assert runtime.finalized
    assert runtime.framework == "langgraph"
    assert runtime.framework_version == "1.2.3"
    assert [event.event_type for event in runtime.events] == [
        "run_start",
        "reasoning",
        "reasoning",
        "tool_call",
        "run_end",
    ]
    reasoning = runtime.events[1]
    assert reasoning.tokens.input == 11
    assert reasoning.tokens.output == 4
    assert reasoning.tokens.cache_read == 2
    assert reasoning.tokens.provenance.value == "provider_reported"
    assert runtime.events[2].tokens.total == 10
    assert runtime.events[2].tokens.cache_read == 2
    assert runtime.events[2].tokens.reasoning == 1
    assert runtime.events[2].tokens.provenance.value == "provider_reported"
    tool = runtime.events[3]
    assert tool.tool is not None
    assert tool.tool.signature == "search"
    assert tool.tool.status.value == "success"
    assert tool.tool.observation_size == len("result")
    assert handle.errors == ()


def test_langgraph_root_failure_is_recorded_and_never_raised_from_callback(tmp_path, monkeypatch):
    _install_fake_langgraph(monkeypatch)
    runtime = _runtime(tmp_path)
    callback = auto_instrument("langgraph", processor=runtime).handles["langgraph"].callback
    callback.on_chain_start({}, {}, run_id="root", parent_run_id=None)

    callback.on_chain_error(RuntimeError("graph failed"), run_id="root", parent_run_id=None)

    assert runtime.finalized
    assert runtime.status == "error"
    assert runtime.events[-1].event_type == "error"
    assert "graph failed" in (runtime.events[-1].content or "")


def test_crewai_handle_requires_explicit_attach_and_captures_official_bus_events(
    tmp_path, monkeypatch
):
    bus, events = _install_fake_crewai(monkeypatch)
    runtime = _runtime(tmp_path)

    result = auto_instrument("crewai", processor=runtime)

    assert result.enabled == ("crewai",)
    handle = result.handles["crewai"]
    assert not handle.attached
    assert not bus.handlers

    crew = SimpleNamespace(id="crew-1", agents=[], tasks=[])
    handle.attach(crew)
    assert handle.attached
    assert sum(len(handlers) for handlers in bus.handlers.values()) == 10

    bus.emit(
        crew,
        events["CrewKickoffStartedEvent"](
            crew_name="research-crew",
            inputs={"topic": "agents"},
            crew=crew,
        ),
    )
    bus.emit(
        crew,
        events["LLMCallCompletedEvent"](
            model="provider-model",
            messages=[{"role": "user", "content": "research"}],
            response="draft",
            usage={"input_tokens": 20, "output_tokens": 6},
            call_id="call-1",
            event_id="event-llm",
            parent_event_id="event-root",
            crew=crew,
        ),
    )
    started = datetime.now(timezone.utc)
    bus.emit(
        crew,
        events["ToolUsageFinishedEvent"](
            tool_name="web_search",
            tool_args={"query": "agents"},
            output="evidence",
            started_at=started,
            finished_at=started + timedelta(milliseconds=25),
            crew=crew,
        ),
    )
    bus.emit(
        crew,
        events["CrewKickoffCompletedEvent"](
            output="final answer",
            total_tokens=26,
            crew=crew,
        ),
    )

    assert runtime.finalized
    assert runtime.framework == "crewai"
    assert runtime.framework_version == "1.14.7"
    assert [event.event_type for event in runtime.events] == [
        "run_start",
        "reasoning",
        "tool_call",
        "run_end",
    ]
    assert runtime.events[1].tokens.total == 26
    assert runtime.events[1].tokens.provenance.value == "provider_reported"
    assert runtime.events[1].span_id == "event-llm"
    assert runtime.events[1].parent_span_id == "event-root"
    assert runtime.events[3].tokens.total == 26
    assert runtime.events[3].tokens.provenance.value == "estimated"
    tool = runtime.events[2].tool
    assert tool is not None
    assert tool.signature == "web_search"
    assert tool.duration_ms == 25.0
    assert handle.errors == ()

    handle.detach()
    assert not handle.attached
    assert not any(bus.handlers.values())


def test_crewai_scope_filter_and_failure_mapping(tmp_path, monkeypatch):
    bus, events = _install_fake_crewai(monkeypatch)
    runtime = _runtime(tmp_path)
    handle = auto_instrument("crewai", processor=runtime).handles["crewai"]
    crew = SimpleNamespace(id="crew-1", agents=[], tasks=[])
    handle.attach(crew)

    other = SimpleNamespace(id="crew-2")
    bus.emit(
        other,
        events["LLMCallCompletedEvent"](
            response="must be ignored",
            usage={"input_tokens": 100, "output_tokens": 100},
            crew=other,
        ),
    )
    assert runtime.events == ()

    bus.emit(
        crew,
        events["CrewKickoffFailedEvent"](error="crew failed", crew=crew),
    )
    assert runtime.finalized
    assert runtime.status == "error"
    assert len(runtime.events) == 1
    assert runtime.events[0].event_type == "error"


def test_current_langgraph_invocation_contract_when_optional_sdk_is_installed(tmp_path):
    pytest.importorskip("langgraph")
    graph_module = pytest.importorskip("langgraph.graph")

    def increment(state):
        return {"value": state["value"] + 1}

    builder = graph_module.StateGraph(dict)
    builder.add_node("increment", increment)
    builder.add_edge(graph_module.START, "increment")
    builder.add_edge("increment", graph_module.END)
    graph = builder.compile()
    runtime = _runtime(tmp_path)
    handle = auto_instrument("langgraph", processor=runtime).handles["langgraph"]

    assert handle.invoke(graph, {"value": 1}) == {"value": 2}
    assert runtime.finalized
    assert runtime.events[0].event_type == "run_start"
    assert runtime.events[-1].event_type == "run_end"
    assert handle.errors == ()
