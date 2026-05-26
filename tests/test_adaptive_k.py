"""Tests for AdaptiveKNode end-to-end behavior.

All tests use mock_llm to avoid real API calls. The mock cycles through a
pre-defined list of responses, so we can script exactly what the LLM "returns"
at each step and assert on the resulting ConsensusReport.
"""
import pytest
from tracerazor import AdaptiveKNode, mock_llm, ConsensusReport
from tracerazor._consensus import Outcome


# ── helpers ─────────────────────────────────────────────────────────────────

def tool_resp(name, args, *, it=50, ot=10, ct=0):
    return {"tool_name": name, "arguments": args, "input_tokens": it, "output_tokens": ot, "cached_tokens": ct}


def answer_resp(text, *, it=50, ot=10):
    return {"final_answer": text, "input_tokens": it, "output_tokens": ot, "cached_tokens": 0}


def _make_tools(**fns):
    return fns


async def run(node, messages=None):
    state = {"messages": messages or [{"role": "user", "content": "test"}]}
    return await node(state)


# ── single-step final answer ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_single_step_final_answer():
    """Agent returns a final answer on the first step."""
    llm = mock_llm([answer_resp("hello")])
    node = AdaptiveKNode(llm=llm, tools={}, k_max=3)

    result = await run(node)

    report: ConsensusReport = result["consensus_report"]
    assert len(report.trajectory) == 1
    assert report.trajectory[0]["type"] == "final_answer"
    assert report.trajectory[0]["content"] == "hello"
    assert len(report.consensus_rate) == 1
    assert report.consensus_rate[0] == 1.0  # all 3 branches agreed


@pytest.mark.asyncio
async def test_tool_call_then_answer():
    """One tool call followed by a final answer."""
    def search(q):
        return f"results for {q}"

    llm = mock_llm([
        tool_resp("search", {"q": "flights"}),   # step 0: search (3 samples)
        answer_resp("Found 5 flights"),           # step 1: answer (3 samples)
    ])
    node = AdaptiveKNode(llm=llm, tools={"search": search}, k_max=3)

    result = await run(node)
    report = result["consensus_report"]

    assert len(report.trajectory) == 2
    assert report.trajectory[0]["type"] == "tool_call"
    assert report.trajectory[0]["tool"] == "search"
    assert report.trajectory[1]["type"] == "final_answer"


# ── K management ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_k_decrements_on_full_consensus():
    """K drops by 1 each step while consensus is full (floor = k_min)."""
    llm = mock_llm([answer_resp("done")])
    node = AdaptiveKNode(llm=llm, tools={}, k_max=5, k_min=2)

    # Step 0 uses k=5 (initial)
    result = await run(node)
    report = result["consensus_report"]
    assert report.steps[0].k_used == 5


@pytest.mark.asyncio
async def test_k_resets_after_mutation():
    """K resets to k_max after a mutating tool call."""
    def book(flight_id):
        return "booked"

    # step 0: read-only search (k=5, stays at 5 unless consensus drops it)
    # step 1: mutating book (k decremented to 4 from full consensus) → resets to 5
    # step 2: final answer
    llm = mock_llm([
        tool_resp("search", {"q": "x"}),   # step 0
        tool_resp("book_reservation", {"flight_id": "AA100"}),  # step 1
        answer_resp("booked"),             # step 2
    ])
    node = AdaptiveKNode(
        llm=llm,
        tools={
            "search": lambda q: "results",
            "book_reservation": book,
        },
        k_max=5,
        k_min=2,
        mutation_metadata={"book_reservation": True, "search": False},
    )

    result = await run(node)
    report = result["consensus_report"]

    # After mutation boundary at step 1, step 2 should use k_max=5
    assert report.steps[2].k_used == 5


@pytest.mark.asyncio
async def test_k_resets_on_divergence():
    """K resets to k_max when branches diverge."""
    call_count = [0]

    async def diverse_llm(messages, tools):
        i = call_count[0]
        call_count[0] += 1
        # Branches 0,1 agree; branch 2 differs (partial consensus at step 0)
        if i < 2:
            return tool_resp("search", {"q": "x"})
        elif i == 2:
            return tool_resp("lookup", {"id": 1})
        else:
            # step 1: all agree → final answer
            return answer_resp("done")

    node = AdaptiveKNode(llm=diverse_llm, tools={"search": lambda q: "r", "lookup": lambda id: "r2"}, k_max=3, k_min=2)
    result = await run(node)
    report = result["consensus_report"]

    # Step 0 is PARTIAL (2/3 agree on "search") → divergences contains 0
    assert 0 in report.divergences


# ── token accounting ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_token_accounting():
    """Total tokens = sum across all K*steps calls."""
    llm = mock_llm([answer_resp("done", it=100, ot=20)])
    node = AdaptiveKNode(llm=llm, tools={}, k_max=3)

    result = await run(node)
    report = result["consensus_report"]

    # k_max=3, 1 step → 3 LLM calls, each 100 in + 20 out
    assert report.tokens.input == 300
    assert report.tokens.output == 60


@pytest.mark.asyncio
async def test_cached_tokens_tracked():
    """Cached token counts are accumulated separately."""
    r = {"final_answer": "ok", "input_tokens": 80, "output_tokens": 20, "cached_tokens": 60}
    node = AdaptiveKNode(llm=mock_llm([r]), tools={}, k_max=2)

    result = await run(node)
    report = result["consensus_report"]

    assert report.tokens.cached == 120  # 2 branches × 60
    assert report.tokens.fresh == 40    # 2 × (80-60)


# ── mutation boundary (branch-and-prune) ────────────────────────────────────

@pytest.mark.asyncio
async def test_mutating_tool_executed_once():
    """Mutating tools are called exactly once regardless of K."""
    call_log = []

    def book_reservation(flight_id):
        call_log.append(flight_id)
        return "booked"

    llm = mock_llm([
        tool_resp("book_reservation", {"flight_id": "AA100"}),
        answer_resp("Reservation confirmed"),
    ])
    node = AdaptiveKNode(
        llm=llm,
        tools={"book_reservation": book_reservation},
        k_max=5,
        mutation_metadata={"book_reservation": True},
    )

    await run(node)
    assert call_log == ["AA100"], f"Expected 1 call, got {call_log}"


@pytest.mark.asyncio
async def test_read_only_tool_executed_once():
    """Read-only tools are also executed once (consensus winner only)."""
    call_log = []

    def search(q):
        call_log.append(q)
        return "results"

    llm = mock_llm([
        tool_resp("search", {"q": "test"}),
        answer_resp("done"),
    ])
    node = AdaptiveKNode(
        llm=llm,
        tools={"search": search},
        k_max=4,
        mutation_metadata={"search": False},
    )

    await run(node)
    assert len(call_log) == 1


# ── state passthrough ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extra_state_fields_preserved():
    """Non-message state fields pass through unchanged."""
    llm = mock_llm([answer_resp("done")])
    node = AdaptiveKNode(llm=llm, tools={}, k_max=2)

    state = {"messages": [{"role": "user", "content": "hi"}], "session_id": "abc-123"}
    result = await node(state)

    assert result["session_id"] == "abc-123"
    assert "consensus_report" in result


@pytest.mark.asyncio
async def test_messages_updated_in_state():
    """Final message list in state includes the assistant's answer."""
    llm = mock_llm([answer_resp("The capital is Paris")])
    node = AdaptiveKNode(llm=llm, tools={}, k_max=2)

    result = await run(node, messages=[{"role": "user", "content": "capital of France?"}])
    msgs = result["messages"]
    last = msgs[-1]
    assert last["role"] == "assistant"
    assert "Paris" in last["content"]


# ── report structure ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_report_accessible_via_property():
    """node.report is set after __call__."""
    llm = mock_llm([answer_resp("ok")])
    node = AdaptiveKNode(llm=llm, tools={}, k_max=2)

    await run(node)
    assert node.report is not None
    assert isinstance(node.report, ConsensusReport)


@pytest.mark.asyncio
async def test_report_summary_runs():
    llm = mock_llm([answer_resp("ok")])
    node = AdaptiveKNode(llm=llm, tools={}, k_max=2)
    await run(node)
    s = node.report.summary()
    assert "steps=1" in s
    assert "avg_consensus=" in s


@pytest.mark.asyncio
async def test_report_to_json():
    llm = mock_llm([answer_resp("ok")])
    node = AdaptiveKNode(llm=llm, tools={}, k_max=2)
    await run(node)
    import json
    data = json.loads(node.report.to_json())
    assert "trajectory" in data
    assert "tokens" in data


# ── guard rails ──────────────────────────────────────────────────────────────

def test_k_min_gt_k_max_clamped():
    # k_min is silently clamped down to k_max — no error
    node = AdaptiveKNode(llm=mock_llm([]), tools={}, k_max=2, k_min=5)
    assert node.k_min == 2


def test_k_min_zero_raises():
    with pytest.raises(ValueError):
        AdaptiveKNode(llm=mock_llm([]), tools={}, k_min=0)


@pytest.mark.asyncio
async def test_max_steps_limits_loop():
    """max_steps hard-stops the loop even without a final answer."""
    llm = mock_llm([tool_resp("search", {"q": "x"})])
    node = AdaptiveKNode(
        llm=llm,
        tools={"search": lambda q: "r"},
        k_max=2,
        max_steps=3,
    )
    result = await run(node)
    assert len(result["consensus_report"].trajectory) <= 3


# ── unknown tool ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_tool_returns_error_observation():
    llm = mock_llm([
        tool_resp("nonexistent_tool", {}),
        answer_resp("fallback"),
    ])
    node = AdaptiveKNode(llm=llm, tools={}, k_max=1)
    result = await run(node)
    traj = result["consensus_report"].trajectory
    assert traj[0]["observation"].startswith("[error:")
