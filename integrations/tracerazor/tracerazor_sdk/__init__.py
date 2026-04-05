"""
TraceRazor Python SDK — framework-agnostic token efficiency auditing.

Works with any Python agent: OpenAI, Anthropic, CrewAI, AutoGen, or raw code.

Two modes:
  - CLI mode (default): calls the local tracerazor binary — no server needed.
  - HTTP mode: POSTs to a running tracerazor-server — no binary needed.

Quickstart (CLI mode):
    from tracerazor_sdk import Tracer

    with Tracer(agent_name="my-agent") as t:
        response = llm.invoke(prompt)
        t.reasoning(response.text, tokens=response.usage.total_tokens)

        result = my_tool(arg)
        t.tool("my_tool", params={"arg": arg}, output=result, success=True, tokens=120)

    report = t.analyse()
    print(report.summary())

Quickstart (HTTP mode):
    from tracerazor_sdk import Tracer

    with Tracer(agent_name="my-agent", server="http://localhost:8080") as t:
        ...

    report = t.analyse()
"""

from .tracer import Tracer
from .client import TraceRazorClient, TraceRazorReport
from .trace import TraceStep

__all__ = ["Tracer", "TraceRazorClient", "TraceRazorReport", "TraceStep"]
__version__ = "0.1.0"
