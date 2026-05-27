"""Phase 1 baseline runners.

Three configurations per PRD §5.3:
  K1Baseline          — single run (k_max=k_min=1), the reference
  NaiveK5Baseline     — 5 independent runs, majority vote
  SelfConsistencyBaseline — K=1 intermediate, k_sc samples on final step

Each runner exposes a ``run_task(task, seed)`` coroutine and a
``run_domain(tasks, seeds)`` coroutine that returns a ``BaselineResult``.
"""
from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from tracerazor._adaptive_k import AdaptiveKNode
from tracerazor._consensus import ExactMatchConsensus
from tracerazor._mutation import MutationMetadata
from tracerazor._naive_ensemble import NaiveKEnsemble
from tracerazor._report import ConsensusReport, TokenCounts
from tracerazor._self_consistency import SelfConsistencyBaseline
from benchmark.tau2_loader import Tau2Task


# ── per-task result ────────────────────────────────────────────────────────

@dataclass
class TaskRunResult:
    task_id: str
    seed: int
    passed: bool
    tokens: TokenCounts
    elapsed_s: float
    avg_consensus: float   # 1.0 for K=1 runs
    divergences: int       # 0 for K=1 runs
    config_label: str      # e.g. "k1", "naive_k5", "sc_k5"


# ── pass^k aggregation ─────────────────────────────────────────────────────

def pass_at_k(per_task_runs: List[List[bool]]) -> Tuple[float, float]:
    """Mean ± std of pass^k across tasks.

    Parameters
    ----------
    per_task_runs:
        ``[[True, False, True], [True, True, True], ...]`` — one inner list
        per task, each element is whether that seed's run passed.

    Returns (mean, std).
    """
    scores = [1.0 if all(runs) else 0.0 for runs in per_task_runs]
    mean = statistics.mean(scores) if scores else 0.0
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return mean, std


# ── domain-level result ────────────────────────────────────────────────────

@dataclass
class BaselineResult:
    config_label: str
    domain: str
    runs: List[TaskRunResult] = field(default_factory=list)

    def pass_at_k_score(self) -> Tuple[float, float]:
        by_task: Dict[str, List[bool]] = {}
        for r in self.runs:
            by_task.setdefault(r.task_id, []).append(r.passed)
        return pass_at_k(list(by_task.values()))

    def mean_tokens(self) -> float:
        totals = [r.tokens.total for r in self.runs]
        return statistics.mean(totals) if totals else 0.0

    def mean_fresh_tokens(self) -> float:
        totals = [r.tokens.fresh for r in self.runs]
        return statistics.mean(totals) if totals else 0.0

    def summary_line(self) -> str:
        pk, std = self.pass_at_k_score()
        return (
            f"{self.config_label:<14}  domain={self.domain}  "
            f"pass^k={pk:.1%} ±{std:.3f}  "
            f"mean_tokens={self.mean_tokens():,.0f}  "
            f"mean_fresh={self.mean_fresh_tokens():,.0f}"
        )


# ── K=1 baseline ──────────────────────────────────────────────────────────

class K1Baseline:
    """Single-run agent with no branching."""

    def __init__(
        self,
        llm_factory: Callable[[], Callable],
        tools: Dict[str, Callable],
        evaluator: Callable,
        mutation_metadata: Optional[Any] = None,
        max_steps: int = 30,
    ) -> None:
        self.llm_factory = llm_factory
        self.tools = tools
        self.evaluator = evaluator
        self.max_steps = max_steps
        if isinstance(mutation_metadata, dict):
            mutation_metadata = MutationMetadata(overrides=mutation_metadata)
        self.mutation = mutation_metadata or MutationMetadata()

    async def run_task(self, task: Tau2Task, seed: int) -> TaskRunResult:
        node = AdaptiveKNode(
            llm=self.llm_factory(),
            tools=self.tools,
            k_max=1, k_min=1,
            mutation_metadata=self.mutation,
            max_steps=self.max_steps,
        )
        t0 = time.monotonic()
        state = await node({"messages": task.initial_messages()})
        elapsed = time.monotonic() - t0
        report: ConsensusReport = state["consensus_report"]
        passed = self.evaluator(report.trajectory, task)
        return TaskRunResult(
            task_id=task.task_id,
            seed=seed,
            passed=passed,
            tokens=report.tokens,
            elapsed_s=elapsed,
            avg_consensus=1.0,
            divergences=0,
            config_label="k1",
        )

    async def run_domain(
        self, tasks: List[Tau2Task], seeds: int = 3
    ) -> BaselineResult:
        result = BaselineResult(config_label="k1", domain=tasks[0].domain if tasks else "")
        for task in tasks:
            for seed in range(seeds):
                r = await self.run_task(task, seed)
                result.runs.append(r)
        return result


# ── Naive K=5 baseline ────────────────────────────────────────────────────

class NaiveK5Baseline:
    """K independent runs, majority vote on action sequence."""

    def __init__(
        self,
        llm_factory: Callable[[], Callable],
        tools: Dict[str, Callable],
        evaluator: Callable,
        k: int = 5,
        mutation_metadata: Optional[Any] = None,
        max_steps: int = 30,
    ) -> None:
        self.llm_factory = llm_factory
        self.tools = tools
        self.evaluator = evaluator
        self.k = k
        self.max_steps = max_steps
        if isinstance(mutation_metadata, dict):
            mutation_metadata = MutationMetadata(overrides=mutation_metadata)
        self.mutation = mutation_metadata or MutationMetadata()
        self._label = f"naive_k{k}"

    async def run_task(self, task: Tau2Task, seed: int) -> TaskRunResult:
        ensemble = NaiveKEnsemble(
            llm_factory=self.llm_factory,
            tools=self.tools,
            k=self.k,
            mutation_metadata=self.mutation,
            max_steps=self.max_steps,
        )
        t0 = time.monotonic()
        run = await ensemble.run(task.initial_messages())
        elapsed = time.monotonic() - t0
        passed = self.evaluator(run.trajectory, task)
        return TaskRunResult(
            task_id=task.task_id,
            seed=seed,
            passed=passed,
            tokens=run.tokens,
            elapsed_s=elapsed,
            avg_consensus=run.voted_by / self.k,
            divergences=0,
            config_label=self._label,
        )

    async def run_domain(
        self, tasks: List[Tau2Task], seeds: int = 3
    ) -> BaselineResult:
        result = BaselineResult(config_label=self._label, domain=tasks[0].domain if tasks else "")
        for task in tasks:
            for seed in range(seeds):
                r = await self.run_task(task, seed)
                result.runs.append(r)
        return result


# ── Self-consistency baseline ─────────────────────────────────────────────

class SCBaseline:
    """K=1 intermediate steps, k_sc samples on final answer."""

    def __init__(
        self,
        llm_factory: Callable[[], Callable],
        tools: Dict[str, Callable],
        evaluator: Callable,
        k_sc: int = 5,
        mutation_metadata: Optional[Any] = None,
        max_steps: int = 30,
    ) -> None:
        self.llm_factory = llm_factory
        self.tools = tools
        self.evaluator = evaluator
        self.k_sc = k_sc
        self.max_steps = max_steps
        if isinstance(mutation_metadata, dict):
            mutation_metadata = MutationMetadata(overrides=mutation_metadata)
        self.mutation = mutation_metadata or MutationMetadata()
        self._label = f"sc_k{k_sc}"

    async def run_task(self, task: Tau2Task, seed: int) -> TaskRunResult:
        sc = SelfConsistencyBaseline(
            llm_factory=self.llm_factory,
            tools=self.tools,
            k_sc=self.k_sc,
            mutation_metadata=self.mutation,
            max_steps=self.max_steps,
        )
        t0 = time.monotonic()
        run = await sc.run(task.initial_messages())
        elapsed = time.monotonic() - t0
        passed = self.evaluator(run.trajectory, task)
        return TaskRunResult(
            task_id=task.task_id,
            seed=seed,
            passed=passed,
            tokens=run.tokens,
            elapsed_s=elapsed,
            avg_consensus=run.consensus_rate,
            divergences=0,
            config_label=self._label,
        )

    async def run_domain(
        self, tasks: List[Tau2Task], seeds: int = 3
    ) -> BaselineResult:
        result = BaselineResult(config_label=self._label, domain=tasks[0].domain if tasks else "")
        for task in tasks:
            for seed in range(seeds):
                r = await self.run_task(task, seed)
                result.runs.append(r)
        return result
