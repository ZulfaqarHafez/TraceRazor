"""Framework adapters for TraceRazor (deprecated in favour of the runtime).

These trace-builder callbacks predate ``tracerazor.runtime.auto_instrument``,
which covers the same frameworks (LangGraph/LangChain, CrewAI, OpenAI Agents)
without estimating token counts from message length. They keep working through
1.x; new code should use the runtime handles (see docs/python_api.md).

Each subpackage is opt-in and is only importable when its framework
dependency is installed. Install the relevant extra:

    pip install "tracerazor[langgraph]"
    pip install "tracerazor[crewai]"
    pip install "tracerazor[agents]"
"""

import warnings

warnings.warn(
    "tracerazor.integrations.* is deprecated; use "
    "tracerazor.runtime.auto_instrument(...) instead (see docs/python_api.md).",
    DeprecationWarning,
    stacklevel=2,
)
