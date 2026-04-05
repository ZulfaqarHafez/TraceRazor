"""
TraceRazor CrewAI adapter.

Automatically captures every LLM call and tool execution from a CrewAI crew
and submits the trace to TraceRazor for efficiency analysis.

Usage::

    from tracerazor_crewai import TraceRazorCallback

    callback = TraceRazorCallback(agent_name="my-crew", threshold=70)
    crew = Crew(agents=[...], tasks=[...], callbacks=[callback])
    crew.kickoff()
    report = callback.analyse()
    print(report.markdown())
"""

from .callback import TraceRazorCallback

__all__ = ["TraceRazorCallback"]
