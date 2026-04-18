"""ExactMatchConsensus: aggregate K branch proposals into a single decision."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from ._canonicalize import canonical_key as _ckey


class Outcome(str, Enum):
    """Consensus outcome for a single step."""
    FULL = "full"        # all K branches agree
    PARTIAL = "partial"  # strict majority (> K/2) agree
    DIVERGENT = "divergent"  # no majority


@dataclass
class BranchProposal:
    """Output from one LLM branch at a single step."""
    branch_id: int
    tool_name: Optional[str]    # None if this branch proposes a final answer
    arguments: Optional[dict]
    final_answer: Optional[str]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    _key: str = field(default="", repr=False)  # set by ExactMatchConsensus


@dataclass
class ConsensusResult:
    outcome: Outcome
    winning_proposal: BranchProposal
    winning_key: str
    consensus_rate: float        # fraction of branches that matched the winner
    all_proposals: List[BranchProposal]
    k: int


class ExactMatchConsensus:
    """Aggregate K branch proposals using exact-match comparison.

    Parameters
    ----------
    null_equiv:
        Passed to the canonicalizer; treats None, empty string, and missing
        keys as equivalent during comparison.
    """

    def __init__(self, *, null_equiv: bool = True) -> None:
        self._null_equiv = null_equiv

    def _assign_keys(self, proposals: List[BranchProposal]) -> None:
        """Stamp canonical keys onto each proposal in-place."""
        for p in proposals:
            if p.tool_name is not None:
                p._key = _ckey(p.tool_name, p.arguments or {}, null_equiv=self._null_equiv)
            else:
                # Final-answer branch: normalise whitespace for comparison
                text = (p.final_answer or "").strip()
                p._key = f"answer:{text}"

    def aggregate(self, proposals: List[BranchProposal]) -> ConsensusResult:
        """Return the consensus decision for one step.

        The winning proposal is the plurality winner (most-voted canonical key).
        When there is a tie, the proposal with the lowest branch_id wins
        (deterministic tiebreak).
        """
        if not proposals:
            raise ValueError("No proposals to aggregate")

        self._assign_keys(proposals)
        k = len(proposals)

        counts: Counter[str] = Counter(p._key for p in proposals)
        # deterministic order: sort by (-count, key_string) so ties are stable
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        winning_key, winning_count = ranked[0]

        consensus_rate = winning_count / k

        if winning_count == k:
            outcome = Outcome.FULL
        elif winning_count > k / 2:
            outcome = Outcome.PARTIAL
        else:
            outcome = Outcome.DIVERGENT

        # Deterministic winner: lowest branch_id among those with the winning key
        winning = min(
            (p for p in proposals if p._key == winning_key),
            key=lambda p: p.branch_id,
        )

        return ConsensusResult(
            outcome=outcome,
            winning_proposal=winning,
            winning_key=winning_key,
            consensus_rate=consensus_rate,
            all_proposals=proposals,
            k=k,
        )
