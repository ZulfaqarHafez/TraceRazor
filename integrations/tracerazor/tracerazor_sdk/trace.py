"""
Lightweight trace data structures. No external dependencies.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceStep:
    id: int
    type: str                          # "reasoning" or "tool_call"
    content: str
    tokens: int
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    tool_success: Optional[bool] = None
    tool_error: Optional[str] = None
    input_context: Optional[str] = None
    output: Optional[str] = None
    agent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "tokens": self.tokens,
        }
        if self.tool_name is not None:
            d["tool_name"] = self.tool_name
        if self.tool_params is not None:
            d["tool_params"] = self.tool_params
        if self.tool_success is not None:
            d["tool_success"] = self.tool_success
        if self.tool_error is not None:
            d["tool_error"] = self.tool_error
        if self.input_context is not None:
            d["input_context"] = self.input_context
        if self.output is not None:
            d["output"] = self.output
        if self.agent_id is not None:
            d["agent_id"] = self.agent_id
        return d


@dataclass
class Trace:
    agent_name: str
    framework: str
    task_value_score: float = 1.0
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[TraceStep] = field(default_factory=list)

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "framework": self.framework,
            "task_value_score": self.task_value_score,
            "steps": [s.to_dict() for s in self.steps],
        }
