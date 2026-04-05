"""
TraceRazor OpenAI Agents SDK adapter.

Hooks into every LLM call and tool execution from an OpenAI Agents SDK
``Runner.run()`` invocation and submits the trace for efficiency analysis.

Usage::

    from tracerazor_openai_agents import TraceRazorHooks
    from agents import Agent, Runner

    hooks = TraceRazorHooks(agent_name="my-agent", threshold=70)
    result = await Runner.run(agent, "What is the refund status?", hooks=hooks)

    report = hooks.analyse()
    print(report.markdown())
    hooks.assert_passes()
"""

from .hooks import TraceRazorHooks

__all__ = ["TraceRazorHooks"]
