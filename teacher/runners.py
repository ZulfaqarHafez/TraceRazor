"""Pluggable runners for the Teacher curriculum.

A ``Runner`` is the seam between ``Teacher.improve`` and *how* a candidate
config is exercised + measured:

  * ``OfflineRunner`` -- the deterministic mock agent (no network, CI-safe);
  * ``teacher.online.OnlineRunner`` -- a real HTTP tool-calling agent loop.

Both expose the same two methods, so ``Teacher.improve(runner=...)`` is agnostic
to which one drives the loop.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .diagnose import Diagnoser
from .runner import Task, evaluate as _evaluate_offline, run_task
from .schemas import AgentConfig, EvalResult


@runtime_checkable
class Runner(Protocol):
    name: str

    def evaluate(self, cfg: AgentConfig) -> EvalResult:
        """Exercise ``cfg`` over the holdout; return tokens + success (+ tas)."""
        ...

    def sample_trace(self, cfg: AgentConfig) -> dict:
        """One representative trace of ``cfg`` (for diagnosis)."""
        ...


class OfflineRunner:
    """Deterministic mock-agent runner (the default; no network)."""
    name = "offline"

    def __init__(self, tasks: list[Task], diagnoser: Diagnoser | None = None):
        if not tasks:
            raise ValueError("OfflineRunner needs at least one Task")
        self.tasks = tasks
        self.diagnoser = diagnoser or Diagnoser()

    def evaluate(self, cfg: AgentConfig) -> EvalResult:
        return _evaluate_offline(cfg, self.tasks, self.diagnoser)

    def sample_trace(self, cfg: AgentConfig) -> dict:
        return run_task(cfg, self.tasks[0])["trace"]
