"""LangGraph / LangChain integration for TraceRazor.

Install the optional dependencies with ``pip install "tracerazor[langgraph]"``.
"""

try:
    from .callback import TraceRazorCallback
    from .trace_builder import TraceBuilder
except ImportError as _e:
    raise ImportError(
        "LangGraph integration requires langchain-core and langgraph.\n"
        "Install with: pip install \"tracerazor[langgraph]\"\n"
        f"(missing: {_e})"
    ) from _e

__all__ = ["TraceRazorCallback", "TraceBuilder"]
