"""
TraceRazor LangGraph integration.

Provides a BaseCallbackHandler that captures LangGraph/LangChain execution events
and streams them into the TraceRazor auditor for real-time and post-hoc analysis.

Usage:
    from tracerazor_langgraph import TraceRazorCallback

    callback = TraceRazorCallback(agent_name="my-agent", threshold=75)

    result = graph.invoke(inputs, config={"callbacks": [callback]})

    # Get the efficiency report after execution
    report = callback.analyse()
    print(report.markdown())

    # Or check inline during CI/CD
    callback.assert_passes()  # raises AssertionError if TAS < threshold
"""

from .callback import TraceRazorCallback
from .client import TraceRazorClient, TraceRazorReport

__all__ = ["TraceRazorCallback", "TraceRazorClient", "TraceRazorReport"]
__version__ = "0.1.0"
