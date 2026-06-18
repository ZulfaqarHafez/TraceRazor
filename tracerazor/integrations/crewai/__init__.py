"""CrewAI integration for TraceRazor.

Install the optional dependencies with ``pip install "tracerazor[crewai]"``.
"""

try:
    from .callback import TraceRazorCallback
except ImportError as _e:
    raise ImportError(
        "CrewAI integration requires crewai.\n"
        "Install with: pip install \"tracerazor[crewai]\"\n"
        f"(missing: {_e})"
    ) from _e

__all__ = ["TraceRazorCallback"]
