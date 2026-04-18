"""AdaptiveKNode: core adaptive-K speculative ensemble.

The node runs the agent loop with K parallel LLM samples per step.
Consensus collapses K → 1 for tool execution (preventing duplicate side effects
at mutation boundaries) then re-fans. K shrinks under sustained full consensus
and resets on divergence or after a mutating call.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from ._consensus import BranchProposal, ConsensusResult, ExactMatchConsensus, Outcome
from ._mutation import MutationMetadata
from ._report import ConsensusReport, StepRecord, TokenCounts


class AdaptiveKNode:
    """Adaptive-K speculative ensemble as a LangGraph-compatible node.

    Drop-in replacement for a ReAct-style agent node. Wrap the same LLM call
    and tool schema; the node handles K-sampling, consensus, and mutation
    boundary collapsing transparently.

    Parameters
    ----------
    llm:
        Async callable ``(messages, tool_schema) -> dict`` where the returned
        dict must contain at least one of:
          - ``"tool_name"`` + ``"arguments"`` (tool call proposal)
          - ``"final_answer"`` (terminal step)
        Optional token fields: ``"input_tokens"``, ``"output_tokens"``,
        ``"cached_tokens"``.
    tools:
        ``{tool_name: async_callable}`` mapping.  Each callable receives the
        tool arguments as keyword arguments and returns a string observation.
    k_max:
        Maximum parallel branches per step.  Default 5 (matches the K=5 naive
        ensemble baseline in the benchmark).
    k_min:
        Floor for K under sustained full consensus.  Default 2.
    consensus:
        Aggregator instance.  Defaults to ``ExactMatchConsensus()``.
    mutation_metadata:
        ``MutationMetadata`` instance or ``{tool_name: bool}`` override dict.
        Unknown tools are treated as mutating (safe default).
    max_steps:
        Hard ceiling on agent loop iterations.  Default 20.
    """

    def __init__(
        self,
        llm: Callable,
        tools: Dict[str, Callable],
        *,
        k_max: int = 5,
        k_min: int = 2,
        consensus: Optional[ExactMatchConsensus] = None,
        mutation_metadata: Optional[Any] = None,
        max_steps: int = 20,
    ) -> None:
        if k_min < 1:
            raise ValueError("k_min must be >= 1")
        # clamp k_min to k_max so callers can set k_max=1 without pain
        k_min = min(k_min, k_max)

        self.llm = llm
        self.tools = tools
        self.k_max = k_max
        self.k_min = k_min
        self.consensus = consensus or ExactMatchConsensus()
        self.max_steps = max_steps

        if isinstance(mutation_metadata, dict):
            mutation_metadata = MutationMetadata(overrides=mutation_metadata)
        self.mutation: MutationMetadata = mutation_metadata or MutationMetadata()

        self._last_report: Optional[ConsensusReport] = None

    # ── public ─────────────────────────────────────────────────────────────

    @property
    def report(self) -> Optional[ConsensusReport]:
        """ConsensusReport from the most recent ``__call__`` invocation."""
        return self._last_report

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node entry point.

        Reads ``state["messages"]``, runs the adaptive-K agent loop, and
        returns an updated state dict containing:
        - ``"messages"``: final message list (shared history)
        - ``"consensus_report"``: the ``ConsensusReport`` for this run
        """
        messages: List[dict] = list(state.get("messages", []))
        tool_schema = self._build_tool_schema()

        trajectory: List[dict] = []
        step_records: List[StepRecord] = []
        consensus_rates: List[float] = []
        divergences: List[int] = []
        total_tokens = TokenCounts()

        k = self.k_max

        for step in range(self.max_steps):
            # ── 1. sample K completions in parallel ───────────────────────
            proposals = await self._sample(messages, tool_schema, k)

            # ── 2. aggregate consensus ────────────────────────────────────
            result: ConsensusResult = self.consensus.aggregate(proposals)
            winning = result.winning_proposal

            # ── 3. token accounting ───────────────────────────────────────
            step_tokens = TokenCounts(
                input=sum(p.input_tokens for p in proposals),
                output=sum(p.output_tokens for p in proposals),
                cached=sum(p.cached_tokens for p in proposals),
            )
            total_tokens += step_tokens

            consensus_rates.append(result.consensus_rate)
            if result.outcome != Outcome.FULL:
                divergences.append(step)

            # ── 4. build divergent-alternatives record ────────────────────
            key_counts: Counter[str] = Counter(p._key for p in proposals)
            alts = [
                {"tool": p.tool_name, "args": p.arguments, "answer": p.final_answer}
                for p in proposals
                if p._key != result.winning_key
                and p.branch_id
                == min(
                    (q.branch_id for q in proposals if q._key == p._key),
                    default=p.branch_id,
                )
            ]

            # ── 5. adapt K for next step ──────────────────────────────────
            if result.outcome == Outcome.FULL:
                k = max(self.k_min, k - 1)
            elif result.outcome == Outcome.DIVERGENT:
                k = self.k_max

            # ── 6a. final answer → break ──────────────────────────────────
            if winning.tool_name is None:
                answer = winning.final_answer or ""
                trajectory.append({"type": "final_answer", "content": answer})
                step_records.append(
                    StepRecord(
                        step_number=step,
                        outcome=result.outcome.value,
                        consensus_rate=result.consensus_rate,
                        k_used=result.k,
                        executed_tool=None,
                        executed_args=None,
                        final_answer=answer,
                        is_mutating=False,
                        tokens=step_tokens,
                        divergent_alternatives=alts,
                    )
                )
                messages.append({"role": "assistant", "content": answer})
                break

            # ── 6b. tool call ─────────────────────────────────────────────
            tool_name = winning.tool_name
            tool_args = winning.arguments or {}
            is_mutating = self.mutation.is_mutating(tool_name)

            observation = await self._execute_tool(tool_name, tool_args)

            trajectory.append(
                {
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": tool_args,
                    "observation": observation,
                    "mutating": is_mutating,
                    "consensus_rate": result.consensus_rate,
                }
            )
            step_records.append(
                StepRecord(
                    step_number=step,
                    outcome=result.outcome.value,
                    consensus_rate=result.consensus_rate,
                    k_used=result.k,
                    executed_tool=tool_name,
                    executed_args=tool_args,
                    final_answer=None,
                    is_mutating=is_mutating,
                    tokens=step_tokens,
                    divergent_alternatives=alts,
                )
            )

            # append call + observation to shared history
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"name": tool_name, "arguments": tool_args}],
                }
            )
            messages.append({"role": "tool", "name": tool_name, "content": observation})

            # re-fan after mutation boundary
            if is_mutating:
                k = self.k_max

        self._last_report = ConsensusReport(
            trajectory=trajectory,
            consensus_rate=consensus_rates,
            divergences=divergences,
            tokens=total_tokens,
            steps=step_records,
        )

        return {**state, "messages": messages, "consensus_report": self._last_report}

    # ── private helpers ─────────────────────────────────────────────────────

    def _build_tool_schema(self) -> List[dict]:
        return [
            {
                "name": name,
                "description": getattr(fn, "__doc__", "") or "",
                "parameters": getattr(fn, "_schema", {}),
            }
            for name, fn in self.tools.items()
        ]

    async def _sample(
        self,
        messages: List[dict],
        tool_schema: List[dict],
        k: int,
    ) -> List[BranchProposal]:
        """Call the LLM k times in parallel with identical inputs."""
        tasks = [self.llm(list(messages), tool_schema) for _ in range(k)]
        raw: list[dict] = await asyncio.gather(*tasks)
        return [
            BranchProposal(
                branch_id=i,
                tool_name=r.get("tool_name"),
                arguments=r.get("arguments"),
                final_answer=r.get("final_answer"),
                input_tokens=r.get("input_tokens", 0),
                output_tokens=r.get("output_tokens", 0),
                cached_tokens=r.get("cached_tokens", 0),
            )
            for i, r in enumerate(raw)
        ]

    async def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a registered tool once and return the observation string."""
        fn = self.tools.get(tool_name)
        if fn is None:
            return f"[error: tool '{tool_name}' not registered]"
        try:
            result = fn(**args)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as exc:
            return f"[error: {exc}]"
