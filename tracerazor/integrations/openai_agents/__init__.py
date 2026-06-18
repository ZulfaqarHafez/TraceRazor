"""OpenAI Agents SDK integration for TraceRazor.

Install the optional dependencies with ``pip install "tracerazor[agents]"``.
"""

try:
    from .hooks import TraceRazorHooks
except ImportError as _e:
    raise ImportError(
        "OpenAI Agents integration requires openai-agents.\n"
        "Install with: pip install \"tracerazor[agents]\"\n"
        f"(missing: {_e})"
    ) from _e

__all__ = ["TraceRazorHooks"]
