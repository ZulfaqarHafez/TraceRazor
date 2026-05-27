"""Naive K-sample majority-vote ensemble (PRD §5.3 baseline 2).

Each "attempt" runs K independent full-run agents (each K=1 internally) then
casts a majority vote over the complete canonicalized action sequence.
Cost = K × single-run cost.  Adaptive-K must beat this by ≥30% on tokens.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ._adaptive_k import AdaptiveKNode
from ._canonicalize import canonical_key as _ckey
from ._consensus import ExactMatchConsensus
from ._mutation import MutationMetadata
from ._report import ConsensusReport, TokenCounts


@dataclass
class NaiveRunResult:
    """Result of one K-run majority-vote attempt."""
    trajectory: List[dict]          # the voted trajectory
    trajectory_key: str             # canonical key of the voted trajectory
    voted_by: int                   # how many of K runs agreed
    k: int                          # K used
    tokens: TokenCounts             # aggregate across all K runs
    all_reports: List[ConsensusReport]
    passed: Optional[bool] = None   # set by evaluator after the fact


def _trajectory_key(trajectory: List[dict]) -> str:
    """Canonical string for a complete action sequence (for voting)."""
    parts = []
    for step in trajectory:
        if step.get("type") == "tool_call":
            parts.append(_ckey(step["tool"], step.get("args", {})))
        else:
            text = (step.get("content") or "").strip()
            parts.append(f"answer:{text}")
    return "|".join(parts)


class NaiveKEnsemble:
    """K independent single-run agents, majority vote on full action sequence.

    Each of the K runs is an ``AdaptiveKNode`` with ``k_max=k_min=1``
    (pure K=1, no branching).  After all K runs complete, trajectories are
    canonicalised and the plurality winner is returned.

    Parameters
    ----------
    llm_factory:
        Callable that returns a **fresh** async LLM callable each time it is
        called.  A fresh callable is needed because ``mock_llm`` tracks
        per-instance state; sharing one instance across K independent runs
        causes step-counter corruption.  Pass ``lambda: mock_llm(responses)``
        for tests, or ``lambda: openai_llm(client, model)`` for real runs.
    tools:
        ``{name: callable}`` tool map.
    k:
        Number of independent runs per attempt.  Default 5.
    mutation_metadata:
        ``MutationMetadata`` or override dict.
    max_steps:
        Per-run step limit.
    """

    def __init__(
        self,
        llm_factory: Callable[[], Callable],
        tools: Dict[str, Callable],
        *,
        k: int = 5,
        mutation_metadata: Optional[Any] = None,
        max_steps: int = 30,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.llm_factory = llm_factory
        self.tools = tools
        self.k = k
        self.max_steps = max_steps
        if isinstance(mutation_metadata, dict):
            mutation_metadata = MutationMetadata(overrides=mutation_metadata)
        self.mutation = mutation_metadata or MutationMetadata()

    def _make_node(self) -> AdaptiveKNode:
        return AdaptiveKNode(
            llm=self.llm_factory(),
            tools=self.tools,
            k_max=1,
            k_min=1,
            consensus=ExactMatchConsensus(),
            mutation_metadata=self.mutation,
            max_steps=self.max_steps,
        )

    async def run(self, messages: List[dict]) -> NaiveRunResult:
        """Run K independent agents and return majority-voted result."""
        tasks = [
            self._make_node()({"messages": list(messages)})
            for _ in range(self.k)
        ]
        states = await asyncio.gather(*tasks)
        reports: List[ConsensusReport] = [s["consensus_report"] for s in states]

        keys = [_trajectory_key(r.trajectory) for r in reports]
        counts: Counter[str] = Counter(keys)
        winning_key, winning_count = counts.most_common(1)[0]

        winning_report = next(r for r, k in zip(reports, keys) if k == winning_key)

        total = TokenCounts()
        for r in reports:
            total += r.tokens

        return NaiveRunResult(
            trajectory=winning_report.trajectory,
            trajectory_key=winning_key,
            voted_by=winning_count,
            k=self.k,
            tokens=total,
            all_reports=reports,
        )
