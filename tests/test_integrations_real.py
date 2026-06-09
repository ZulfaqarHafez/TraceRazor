"""Real-framework integration tests.

The hermetic tests in test_integrations_smoke.py use a fake client and never
touch the real framework libraries. These drive our adapters with the ACTUAL
framework base classes / event objects installed, then audit against the real
`tracerazor` binary. They catch contract/signature drift against
`langchain_core` and the openai-agents SDK that hermetic tests cannot (that is
the exact class of bug that once shipped: a wrong `analyse(threshold=...)` call).

Each test is gated on (a) the framework lib being importable and (b) the binary
being available, so it runs in CI (which installs the extras and builds the
binary) and skips cleanly otherwise.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


def _require_binary() -> str:
    b = os.environ.get("TRACERAZOR_BIN") or shutil.which("tracerazor")
    if not b:
        repo = Path(__file__).resolve().parent.parent
        for rel in ("target/release/tracerazor", "target/debug/tracerazor"):
            cand = repo / rel
            if cand.is_file():
                b = str(cand)
                break
    if not b or not Path(b).is_file():
        pytest.skip("tracerazor binary not built")
    return b


def test_langgraph_real_callback_events():
    """Drive the LangGraph callback with real langchain_core event objects."""
    pytest.importorskip("langchain_core")
    from langchain_core.outputs import Generation, LLMResult

    from tracerazor.integrations.langgraph import TraceRazorCallback

    binary = _require_binary()
    cb = TraceRazorCallback(agent_name="lg-real", tracerazor_bin=binary)
    for i in range(4):
        cb.on_llm_start({"name": "llm"}, [f"Decide the next action for refund task {i}"])
        cb.on_llm_end(LLMResult(
            generations=[[Generation(text=f"Reasoning about step {i} of the task")]],
            llm_output={"usage": {"total_tokens": 180}},
        ))
        cb.on_tool_start({"name": "get_order"}, '{"order_id": "ORD-%d"}' % i)
        cb.on_tool_end(f"order details for record {i}")

    report = cb.analyse()
    assert 0.0 <= report.tas_score <= 100.0
    assert report.total_steps >= 5


def test_openai_agents_real_runhooks():
    """Drive the OpenAI-Agents hooks bound to the real RunHooks base class."""
    agents = pytest.importorskip("agents")
    import asyncio

    from tracerazor.integrations.openai_agents import TraceRazorHooks

    binary = _require_binary()
    hooks = TraceRazorHooks(agent_name="oa-real", tracerazor_bin=binary)

    # When the SDK is installed, our hooks should actually subclass RunHooks.
    if hasattr(agents, "RunHooks"):
        assert any(b.__name__ == "RunHooks" for b in type(hooks).__mro__)

    class _Named:
        def __init__(self, name):
            self.name = name

    async def drive():
        await hooks.on_agent_start(None, _Named("agent"))
        for i in range(5):
            await hooks.on_tool_start(None, _Named("agent"), _Named(f"tool{i}"))
            await hooks.on_tool_end(None, _Named("agent"), _Named(f"tool{i}"), f"result {i}")
        await hooks.on_agent_end(None, _Named("agent"), "final answer text for the task")

    asyncio.run(drive())
    report = hooks.analyse()
    assert 0.0 <= report.tas_score <= 100.0
    assert report.total_steps >= 5
