"""Tests for ExactMatchConsensus aggregation logic."""
import pytest
from tracerazor._consensus import BranchProposal, ExactMatchConsensus, Outcome


def make_tool(branch_id, name, args, *, input_tokens=50, output_tokens=10):
    return BranchProposal(
        branch_id=branch_id,
        tool_name=name,
        arguments=args,
        final_answer=None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def make_answer(branch_id, text, *, input_tokens=50, output_tokens=10):
    return BranchProposal(
        branch_id=branch_id,
        tool_name=None,
        arguments=None,
        final_answer=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


C = ExactMatchConsensus()


# ── FULL consensus ──────────────────────────────────────────────────────────

class TestFullConsensus:
    def test_k5_all_agree(self):
        proposals = [make_tool(i, "search", {"q": "flights"}) for i in range(5)]
        r = C.aggregate(proposals)
        assert r.outcome == Outcome.FULL
        assert r.consensus_rate == 1.0
        assert r.k == 5

    def test_k1_trivially_full(self):
        r = C.aggregate([make_tool(0, "search", {"q": "x"})])
        assert r.outcome == Outcome.FULL
        assert r.consensus_rate == 1.0

    def test_full_consensus_winning_key_matches(self):
        proposals = [make_tool(i, "book", {"id": 1}) for i in range(3)]
        r = C.aggregate(proposals)
        assert r.winning_proposal.tool_name == "book"
        assert r.consensus_rate == 1.0

    def test_full_consensus_on_final_answer(self):
        proposals = [make_answer(i, "The answer is 42") for i in range(4)]
        r = C.aggregate(proposals)
        assert r.outcome == Outcome.FULL
        assert r.winning_proposal.final_answer == "The answer is 42"

    def test_whitespace_normalised_to_full(self):
        # Two branches return the same tool but with whitespace in args
        p1 = make_tool(0, "search", {"q": "  flights  "})
        p2 = make_tool(1, "search", {"q": "flights"})
        r = C.aggregate([p1, p2])
        assert r.outcome == Outcome.FULL

    def test_numeric_string_normalised_to_full(self):
        p1 = make_tool(0, "book", {"seats": "2"})
        p2 = make_tool(1, "book", {"seats": 2})
        r = C.aggregate([p1, p2])
        assert r.outcome == Outcome.FULL

    def test_key_order_normalised_to_full(self):
        p1 = make_tool(0, "book", {"b": 1, "a": 2})
        p2 = make_tool(1, "book", {"a": 2, "b": 1})
        r = C.aggregate([p1, p2])
        assert r.outcome == Outcome.FULL


# ── PARTIAL consensus ───────────────────────────────────────────────────────

class TestPartialConsensus:
    def test_k3_two_agree(self):
        proposals = [
            make_tool(0, "search", {"q": "NYC"}),
            make_tool(1, "search", {"q": "NYC"}),
            make_tool(2, "lookup", {"id": 1}),
        ]
        r = C.aggregate(proposals)
        assert r.outcome == Outcome.PARTIAL
        assert r.winning_proposal.tool_name == "search"
        assert pytest.approx(r.consensus_rate) == 2 / 3

    def test_k5_three_agree(self):
        proposals = [
            make_tool(i, "search", {"q": "x"}) for i in range(3)
        ] + [
            make_tool(3, "lookup", {"id": 1}),
            make_tool(4, "cancel", {"id": 2}),
        ]
        r = C.aggregate(proposals)
        assert r.outcome == Outcome.PARTIAL
        assert r.consensus_rate == pytest.approx(3 / 5)

    def test_partial_winner_is_majority(self):
        proposals = [
            make_tool(0, "a", {}),
            make_tool(1, "a", {}),
            make_tool(2, "b", {}),
        ]
        r = C.aggregate(proposals)
        assert r.winning_proposal.tool_name == "a"


# ── DIVERGENT consensus ─────────────────────────────────────────────────────

class TestDivergentConsensus:
    def test_k4_two_two_split(self):
        proposals = [
            make_tool(0, "a", {}),
            make_tool(1, "a", {}),
            make_tool(2, "b", {}),
            make_tool(3, "b", {}),
        ]
        r = C.aggregate(proposals)
        assert r.outcome == Outcome.DIVERGENT

    def test_k5_all_different(self):
        proposals = [make_tool(i, f"tool_{i}", {}) for i in range(5)]
        r = C.aggregate(proposals)
        assert r.outcome == Outcome.DIVERGENT
        assert r.consensus_rate == pytest.approx(1 / 5)

    def test_divergent_tiebreak_deterministic(self):
        # Two-way tie → same result every time (lowest branch_id wins)
        proposals = [
            make_tool(0, "a", {}),
            make_tool(1, "a", {}),
            make_tool(2, "b", {}),
            make_tool(3, "b", {}),
        ]
        r1 = C.aggregate(proposals)
        r2 = C.aggregate(proposals[::-1])  # reversed order
        # winner key should be the same (alphabetically sorted: "a" < "b")
        assert r1.winning_key == r2.winning_key


# ── edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_proposals_raises(self):
        with pytest.raises(ValueError):
            C.aggregate([])

    def test_all_proposals_in_result(self):
        proposals = [make_tool(i, "f", {}) for i in range(3)]
        r = C.aggregate(proposals)
        assert len(r.all_proposals) == 3

    def test_final_answer_vs_tool_call_different(self):
        proposals = [
            make_tool(0, "search", {"q": "x"}),
            make_answer(1, "The answer"),
        ]
        r = C.aggregate(proposals)
        # one of each — divergent
        assert r.outcome == Outcome.DIVERGENT

    def test_winning_proposal_is_from_input_list(self):
        proposals = [make_tool(i, "f", {"n": i}) for i in range(3)]
        r = C.aggregate(proposals)
        assert r.winning_proposal in proposals

    def test_null_equiv_normalises_missing_key(self):
        p1 = make_tool(0, "f", {"a": 1, "b": None})
        p2 = make_tool(1, "f", {"a": 1})
        r = C.aggregate([p1, p2])
        assert r.outcome == Outcome.FULL

    def test_token_counts_preserved(self):
        p = make_tool(0, "f", {}, input_tokens=200, output_tokens=30)
        r = C.aggregate([p])
        assert r.winning_proposal.input_tokens == 200
        assert r.winning_proposal.output_tokens == 30
