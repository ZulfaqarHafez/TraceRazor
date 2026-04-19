"""Tests for SelfConsistencyBaseline."""
import pytest
from tracerazor._self_consistency import SelfConsistencyBaseline


# ── helpers ────────────────────────────────────────────────────────────────

def two_step_llm(tool_name, tool_args, answer="Done"):
    """Stateless: tool call on first step, final answer on second."""
    async def _call(messages, tools):
        n_obs = sum(1 for m in messages if m.get("role") == "tool")
        if n_obs == 0:
            return {"tool_name": tool_name, "arguments": tool_args,
                    "input_tokens": 200, "output_tokens": 30, "cached_tokens": 0}
        return {"final_answer": answer,
                "input_tokens": 250, "output_tokens": 25, "cached_tokens": 200}
    return _call


def direct_answer_llm(answer="Answer"):
    async def _call(messages, tools):
        return {"final_answer": answer,
                "input_tokens": 100, "output_tokens": 20, "cached_tokens": 0}
    return _call


# ── basic behaviour ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_single_step_answer_majority_voted():
    """When agent gives a direct answer, SC still samples k_sc times."""
    llm = direct_answer_llm("Paris")
    sc = SelfConsistencyBaseline(
        llm_factory=lambda: llm,
        tools={},
        k_sc=5,
    )
    result = await sc.run([{"role": "user", "content": "capital of France?"}])
    assert result.final_answer == "Paris"
    assert result.k_sc == 5


@pytest.mark.asyncio
async def test_majority_answer_wins():
    """k_sc samples return 2 different answers — majority wins."""
    answers = ["Paris", "Paris", "London"]
    call_count = [0]

    def varying_llm():
        async def _call(messages, tools):
            n_obs = sum(1 for m in messages if m.get("role") == "tool")
            if n_obs == 0:
                return {"tool_name": "search", "arguments": {"q": "x"},
                        "input_tokens": 100, "output_tokens": 20}
            i = call_count[0] % len(answers)
            call_count[0] += 1
            return {"final_answer": answers[i], "input_tokens": 110, "output_tokens": 15}
        return _call

    sc = SelfConsistencyBaseline(
        llm_factory=varying_llm,
        tools={"search": lambda q: "results"},
        k_sc=3,
    )
    result = await sc.run([{"role": "user", "content": "x"}])
    # "Paris" appears 2×, "London" 1× — Paris should win
    assert result.final_answer == "Paris"
    assert result.voted_by == 2


@pytest.mark.asyncio
async def test_tool_calls_preserved_in_trajectory():
    """Intermediate tool calls should appear in trajectory."""
    llm = two_step_llm("search", {"q": "flights"})
    sc = SelfConsistencyBaseline(
        llm_factory=lambda: llm,
        tools={"search": lambda q: "results"},
        k_sc=3,
    )
    result = await sc.run([{"role": "user", "content": "find flights"}])
    tool_steps = [s for s in result.trajectory if s.get("type") == "tool_call"]
    assert len(tool_steps) >= 1
    assert tool_steps[0]["tool"] == "search"


@pytest.mark.asyncio
async def test_final_answer_in_trajectory():
    llm = direct_answer_llm("42")
    sc = SelfConsistencyBaseline(llm_factory=lambda: llm, tools={}, k_sc=2)
    result = await sc.run([{"role": "user", "content": "x"}])
    final_steps = [s for s in result.trajectory if s.get("type") == "final_answer"]
    assert len(final_steps) == 1
    assert final_steps[0]["content"] == "42"


# ── token accounting ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tokens_include_both_phases():
    """Total tokens = K=1 run + k_sc final samples."""
    llm = direct_answer_llm()
    k_sc = 4
    sc = SelfConsistencyBaseline(llm_factory=lambda: llm, tools={}, k_sc=k_sc)
    result = await sc.run([{"role": "user", "content": "x"}])
    # K=1 run: 1 call × (100in + 20out) = 120
    # k_sc samples: 4 calls × (100in + 20out) = 480
    # total = 600
    assert result.tokens.total == (1 + k_sc) * 120


@pytest.mark.asyncio
async def test_k_sc_one_works():
    """k_sc=1 degenerates to a single final-step sample (= K=1 behaviour)."""
    llm = direct_answer_llm("solo")
    sc = SelfConsistencyBaseline(llm_factory=lambda: llm, tools={}, k_sc=1)
    result = await sc.run([{"role": "user", "content": "x"}])
    assert result.final_answer == "solo"
    assert result.voted_by == 1


# ── consensus rate ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_agreement_gives_rate_one():
    llm = direct_answer_llm("yes")
    sc = SelfConsistencyBaseline(llm_factory=lambda: llm, tools={}, k_sc=5)
    result = await sc.run([{"role": "user", "content": "x"}])
    assert result.consensus_rate == 1.0


# ── guard rail ───────────────────────────────────────────────────────────────

def test_k_sc_zero_raises():
    with pytest.raises(ValueError):
        SelfConsistencyBaseline(llm_factory=lambda: direct_answer_llm(), tools={}, k_sc=0)
