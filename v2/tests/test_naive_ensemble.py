"""Tests for NaiveKEnsemble majority-vote baseline."""
import pytest
from tracerazor import mock_llm
from tracerazor._naive_ensemble import NaiveKEnsemble, _trajectory_key


# ── helpers ────────────────────────────────────────────────────────────────

def two_step_llm(tool_name, tool_args, answer="Done"):
    """Stateless mock: tool call on step 0, final answer on step 1."""
    async def _call(messages, tools):
        tool_obs = sum(1 for m in messages if m.get("role") == "tool")
        if tool_obs == 0:
            return {"tool_name": tool_name, "arguments": tool_args,
                    "input_tokens": 100, "output_tokens": 20}
        return {"final_answer": answer, "input_tokens": 120, "output_tokens": 15}
    return _call


def final_only_llm(answer="Done"):
    async def _call(messages, tools):
        return {"final_answer": answer, "input_tokens": 80, "output_tokens": 10}
    return _call


# ── trajectory key ──────────────────────────────────────────────────────────

class TestTrajectoryKey:
    def test_empty_trajectory(self):
        assert _trajectory_key([]) == ""

    def test_single_tool_call(self):
        traj = [{"type": "tool_call", "tool": "search", "args": {"q": "x"}}]
        key = _trajectory_key(traj)
        assert key.startswith("tool:search:")

    def test_final_answer(self):
        traj = [{"type": "final_answer", "content": "hello"}]
        assert _trajectory_key(traj) == "answer:hello"

    def test_tool_then_answer(self):
        traj = [
            {"type": "tool_call", "tool": "search", "args": {"q": "x"}},
            {"type": "final_answer", "content": "done"},
        ]
        key = _trajectory_key(traj)
        assert "|" in key
        assert key.endswith("|answer:done")

    def test_canonicalisation_makes_equivalent_keys_equal(self):
        t1 = [{"type": "tool_call", "tool": "search", "args": {"q": "x"}}]
        t2 = [{"type": "tool_call", "tool": "SEARCH", "args": {"q": " x "}}]
        assert _trajectory_key(t1) == _trajectory_key(t2)

    def test_different_trajectories_have_different_keys(self):
        t1 = [{"type": "tool_call", "tool": "search", "args": {"q": "x"}}]
        t2 = [{"type": "tool_call", "tool": "lookup", "args": {"id": 1}}]
        assert _trajectory_key(t1) != _trajectory_key(t2)


# ── NaiveKEnsemble ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_k1_returns_single_run():
    llm = two_step_llm("search", {"q": "flights"})
    ens = NaiveKEnsemble(llm_factory=lambda: llm, tools={"search": lambda q: "r"}, k=1)
    result = await ens.run([{"role": "user", "content": "find flights"}])
    assert result.k == 1
    assert result.voted_by == 1


@pytest.mark.asyncio
async def test_majority_vote_when_all_agree():
    llm = two_step_llm("search", {"q": "flights"}, answer="Found 5")
    ens = NaiveKEnsemble(
        llm_factory=lambda: llm,
        tools={"search": lambda q: "results"},
        k=3,
    )
    result = await ens.run([{"role": "user", "content": "find flights"}])
    assert result.voted_by == 3
    assert result.k == 3


@pytest.mark.asyncio
async def test_divergent_vote_returns_plurality():
    """When K=3 runs disagree 2:1, the majority (2) wins."""
    counter = [0]

    def varied_llm():
        run_idx = counter[0]
        counter[0] += 1
        # run 0 and 1: search for LAX; run 2: search for SFO
        dest = "LAX" if run_idx < 2 else "SFO"
        async def _call(messages, tools):
            n_obs = sum(1 for m in messages if m.get("role") == "tool")
            if n_obs == 0:
                return {"tool_name": "search", "arguments": {"dest": dest},
                        "input_tokens": 100, "output_tokens": 20}
            return {"final_answer": "done", "input_tokens": 110, "output_tokens": 10}
        return _call

    ens = NaiveKEnsemble(
        llm_factory=varied_llm,
        tools={"search": lambda dest: "r"},
        k=3,
    )
    result = await ens.run([{"role": "user", "content": "x"}])
    assert result.voted_by == 2
    # winning trajectory uses LAX
    assert "lax" in result.trajectory_key.lower()


@pytest.mark.asyncio
async def test_token_aggregation():
    """Total tokens = sum across K independent runs."""
    llm = final_only_llm()
    ens = NaiveKEnsemble(llm_factory=lambda: llm, tools={}, k=4)
    result = await ens.run([{"role": "user", "content": "x"}])
    # each run: 80 input + 10 output = 90 total; k=4
    assert result.tokens.total == 4 * 90


@pytest.mark.asyncio
async def test_all_reports_collected():
    llm = final_only_llm()
    ens = NaiveKEnsemble(llm_factory=lambda: llm, tools={}, k=5)
    result = await ens.run([{"role": "user", "content": "x"}])
    assert len(result.all_reports) == 5


@pytest.mark.asyncio
async def test_k_zero_raises():
    with pytest.raises(ValueError):
        NaiveKEnsemble(llm_factory=lambda: final_only_llm(), tools={}, k=0)


@pytest.mark.asyncio
async def test_trajectory_in_result():
    llm = two_step_llm("book", {"id": 1})
    ens = NaiveKEnsemble(
        llm_factory=lambda: llm,
        tools={"book": lambda id: "booked"},
        k=2,
    )
    result = await ens.run([{"role": "user", "content": "book"}])
    assert any(s.get("type") == "tool_call" for s in result.trajectory)


@pytest.mark.asyncio
async def test_mutation_tool_called_once_per_run():
    """Mutating tools should only execute once per K=1 run."""
    call_log = []

    def book(flight_id):
        call_log.append(flight_id)
        return "booked"

    llm = two_step_llm("book_reservation", {"flight_id": "AA100"})
    ens = NaiveKEnsemble(
        llm_factory=lambda: llm,
        tools={"book_reservation": book},
        k=3,
        mutation_metadata={"book_reservation": True},
    )
    await ens.run([{"role": "user", "content": "book"}])
    # 3 independent runs, 1 book call each = 3 total
    assert len(call_log) == 3
