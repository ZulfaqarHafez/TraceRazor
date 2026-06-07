"""Framework adapters -- ingest real agent runs into Teacher-consumable traces.

* ``LangGraphAdapter`` -- records LangGraph/LangChain runs (or replays existing
  ``tracerazor`` LangGraph callback output) into auditor-schema trace dicts.
"""
from .base import FrameworkAdapter
from .langgraph import LangGraphAdapter, RunRecorder

__all__ = ["FrameworkAdapter", "LangGraphAdapter", "RunRecorder"]
