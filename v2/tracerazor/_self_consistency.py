"""Self-consistency baseline at the final step (PRD §5.3 baseline 3).

All intermediate steps run K=1.  When the agent is about to give a final
answer, k_sc samples are drawn and the majority answer is returned.

This approximates Wang et al. (2022) for tool-using agents: sampling variance
matters most at the terminal reasoning step, not during deterministic tool calls.

Cost: slightly above K=1 (k_sc extra tokens on the last step only).
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ._adaptive_k import AdaptiveKNode
from ._consensus import BranchProposal, ExactMatchConsensus
from ._mutation import MutationMetadata
from ._report import ConsensusReport, TokenCounts


@dataclass
class SCResult:
    """Result of a self-consistency run."""
    trajectory: List[dict]          # tool-call steps from the K=1 run
    final_answer: str               # majority-voted final answer
    voted_by: int                   # how many of k_sc agreed on winner
    k_sc: int
    tokens: TokenCounts
    consensus_rate: float           # agreement rate on final-step vote


class SelfConsistencyBaseline:
    """Standard K=1 agent loop then majority-vote on the terminal answer.

    Parameters
    ----------
    llm_factory:
        Callable that returns a fresh async LLM callable.  Needed so the K=1
        intermediate run and the k_sc final-step samples each get independent
        state (important for ``mock_llm`` which tracks message history).
    tools:
        ``{name: callable}`` tool map.
    k_sc:
        Number of samples drawn at the final step.  Default 5.
    mutation_metadata:
        ``MutationMetadata`` or override dict.
    max_steps:
        Step limit for the intermediate K=1 run.
    """

    def __init__(
        self,
        llm_factory: Callable[[], Callable],
        tools: Dict[str, Callable],
        *,
        k_sc: int = 5,
        mutation_metadata: Optional[Any] = None,
        max_steps: int = 30,
    ) -> None:
        if k_sc < 1:
            raise ValueError("k_sc must be >= 1")
        self.llm_factory = llm_factory
        self.tools = tools
        self.k_sc = k_sc
        self.max_steps = max_steps
        self._consensus = ExactMatchConsensus()
        if isinstance(mutation_metadata, dict):
            mutation_metadata = MutationMetadata(overrides=mutation_metadata)
        self.mutation = mutation_metadata or MutationMetadata()

    async def run(self, messages: List[dict]) -> SCResult:
        """K=1 through all tool calls, then k_sc samples for the final answer."""

        # ── Phase 1: K=1 run to completion ───────────────────────────────
        k1_node = AdaptiveKNode(
            llm=self.llm_factory(),
            tools=self.tools,
            k_max=1,
            k_min=1,
            mutation_metadata=self.mutation,
            max_steps=self.max_steps,
        )
        state = await k1_node({"messages": list(messages)})
        k1_report: ConsensusReport = state["consensus_report"]
        k1_messages: List[dict] = state["messages"]

        # ── Phase 2: identify messages before final answer ────────────────
        # AdaptiveKNode appends {"role": "assistant", "content": answer} last.
        # Strip it to get the pre-final-answer context.
        if k1_messages and k1_messages[-1].get("role") == "assistant" and k1_messages[-1].get("content"):
            pre_final_msgs = k1_messages[:-1]
        else:
            # max_steps hit without final answer — sample from current state
            pre_final_msgs = k1_messages

        # ── Phase 3: k_sc samples at the final step ───────────────────────
        tool_schema = [
            {"name": name, "description": getattr(fn, "__doc__", "") or ""}
            for name, fn in self.tools.items()
        ]
        sample_tasks = [
            self.llm_factory()(list(pre_final_msgs), tool_schema)
            for _ in range(self.k_sc)
        ]
        raw_samples = await asyncio.gather(*sample_tasks)

        proposals = [
            BranchProposal(
                branch_id=i,
                tool_name=r.get("tool_name"),
                arguments=r.get("arguments"),
                final_answer=r.get("final_answer", ""),
                input_tokens=r.get("input_tokens", 0),
                output_tokens=r.get("output_tokens", 0),
                cached_tokens=r.get("cached_tokens", 0),
            )
            for i, r in enumerate(raw_samples)
        ]

        result = self._consensus.aggregate(proposals)
        winner = result.winning_proposal

        # ── Token totals ──────────────────────────────────────────────────
        sc_tokens = TokenCounts(
            input=sum(p.input_tokens for p in proposals),
            output=sum(p.output_tokens for p in proposals),
            cached=sum(p.cached_tokens for p in proposals),
        )
        total = TokenCounts(
            input=k1_report.tokens.input + sc_tokens.input,
            output=k1_report.tokens.output + sc_tokens.output,
            cached=k1_report.tokens.cached + sc_tokens.cached,
        )

        # ── Build trajectory (tool calls only — final answer replaced) ────
        tool_traj = [s for s in k1_report.trajectory if s.get("type") == "tool_call"]
        final_answer = winner.final_answer or ""
        tool_traj.append({"type": "final_answer", "content": final_answer})

        return SCResult(
            trajectory=tool_traj,
            final_answer=final_answer,
            voted_by=int(result.consensus_rate * self.k_sc),
            k_sc=self.k_sc,
            tokens=total,
            consensus_rate=result.consensus_rate,
        )
