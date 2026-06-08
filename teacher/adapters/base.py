"""Adapter protocol -- the seam between a framework and the Teacher."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Minimal contract every framework adapter satisfies."""

    def collect_traces(self) -> list[dict]:
        """Return recorded runs as auditor-schema trace dicts."""
        ...

    def reset(self) -> None:
        """Drop any buffered traces."""
        ...
