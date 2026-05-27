"""Integration tests for K1Baseline, NaiveK5Baseline, SCBaseline, pass^k."""
import pytest
from benchmark.baselines import (
    BaselineResult,
    K1Baseline,
    NaiveK5Baseline,
    SCBaseline,
    TaskRunResult,
    pass_at_k,
)
from benchmark.tau2_evaluator import ActionMatchEvaluator
from benchmark.tau2_loader import MOCK_AIRLINE_TASKS, Tau2Task
from tracerazor._report import TokenCounts


# ── pass^k unit tests ────────────────────────────────────────────────────────

class TestPassAtK:
    def test_all_pass(self):
        mean, std = pass_at_k([[True, True, True], [True, True, True]])
        assert mean == 1.0

    def test_all_fail(self):
        mean, std = pass_at_k([[False, False], [False]])
        assert mean == 0.0

    def test_one_task_mixed(self):
        mean, std = pass_at_k([[True, False, True]])
        # Not all k runs passed → 0.0
        assert mean == 0.0

    def test_partial_tasks(self):
        # task 1: all 3 seeds pass; task 2: one seed fails
        mean, std = pass_at_k([[True, True, True], [True, False, True]])
        assert mean == 0.5  # 1/2 tasks achieved pass^k

    def test_empty_returns_zero(self):
        mean, std = pass_at_k([])
        assert mean == 0.0

    def test_single_seed(self):
        mean, std = pass_at_k([[True], [False]])
        assert mean == 0.5

    def test_std_zero_when_all_same(self):
        mean, std = pass_at_k([[True, True], [True, True]])
        assert std == 0.0


# ── helpers ────────────────────────────────────────────────────────────────

def make_llm_for_task(task: Tau2Task):
    """Stateless LLM that executes expected actions in order then answers."""
    expected = task.expected_actions

    async def _call(messages, tools):
        n_obs = sum(1 for m in messages if m.get("role") == "tool")
        if n_obs < len(expected):
            action = expected[n_obs]
            return {"tool_name": action["name"], "arguments": action["arguments"],
                    "input_tokens": 200, "output_tokens": 30, "cached_tokens": 0}
        return {"final_answer": "Done.", "input_tokens": 250, "output_tokens": 20,
                "cached_tokens": 200}

    return _call


def _tools_for_domain(domain="airline"):
    if domain == "airline":
        return {
            "search_direct_flight": lambda origin, destination, date: "[AA100]",
            "book_reservation": lambda flight_id, passenger_name: "CONF-001",
            "cancel_reservation": lambda confirmation_id: "cancelled",
        }
    return {
        "get_order_details": lambda order_id: "{'status': 'processing'}",
        "cancel_order": lambda order_id: "cancelled",
    }


# ── K1Baseline ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_k1_baseline_passes_correct_trajectory():
    task = MOCK_AIRLINE_TASKS[0]  # search + book
    evaluator = ActionMatchEvaluator()
    b = K1Baseline(
        llm_factory=lambda: make_llm_for_task(task),
        tools=_tools_for_domain(),
        evaluator=evaluator,
    )
    r = await b.run_task(task, seed=0)
    assert r.passed is True
    assert r.config_label == "k1"
    assert r.tokens.total > 0


@pytest.mark.asyncio
async def test_k1_baseline_domain_run():
    tasks = MOCK_AIRLINE_TASKS[:2]
    evaluator = ActionMatchEvaluator()

    def factory():
        # factory must produce a new llm each time; we use a generic one
        async def _call(messages, tools):
            n_obs = sum(1 for m in messages if m.get("role") == "tool")
            task_idx = 0  # same task used for all seeds in this test
            if n_obs == 0:
                return {"tool_name": "search_direct_flight",
                        "arguments": {"origin": "JFK", "destination": "LAX", "date": "2024-06-15"},
                        "input_tokens": 200, "output_tokens": 30}
            if n_obs == 1:
                return {"tool_name": "book_reservation",
                        "arguments": {"flight_id": "AA100", "passenger_name": "Alice Smith"},
                        "input_tokens": 240, "output_tokens": 25}
            return {"final_answer": "Booked.", "input_tokens": 260, "output_tokens": 20}
        return _call

    b = K1Baseline(llm_factory=factory, tools=_tools_for_domain(), evaluator=evaluator)
    result = await b.run_domain(tasks[:1], seeds=2)
    assert len(result.runs) == 2
    assert result.config_label == "k1"


# ── NaiveK5Baseline ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_naive_k5_passes_correct_trajectory():
    task = MOCK_AIRLINE_TASKS[0]
    evaluator = ActionMatchEvaluator()
    b = NaiveK5Baseline(
        llm_factory=lambda: make_llm_for_task(task),
        tools=_tools_for_domain(),
        evaluator=evaluator,
        k=3,
    )
    r = await b.run_task(task, seed=0)
    assert r.passed is True
    assert r.config_label == "naive_k3"


@pytest.mark.asyncio
async def test_naive_k5_token_cost_is_k_times_single():
    task = MOCK_AIRLINE_TASKS[2]  # single action: search only
    evaluator = ActionMatchEvaluator()

    k1 = K1Baseline(
        llm_factory=lambda: make_llm_for_task(task),
        tools=_tools_for_domain(),
        evaluator=evaluator,
    )
    naive = NaiveK5Baseline(
        llm_factory=lambda: make_llm_for_task(task),
        tools=_tools_for_domain(),
        evaluator=evaluator,
        k=3,
    )
    r_k1 = await k1.run_task(task, seed=0)
    r_naive = await naive.run_task(task, seed=0)
    # Naive with k=3 should cost ~3× more than K=1
    assert r_naive.tokens.total == pytest.approx(r_k1.tokens.total * 3, rel=0.2)


# ── SCBaseline ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sc_baseline_passes_correct_trajectory():
    task = MOCK_AIRLINE_TASKS[0]
    evaluator = ActionMatchEvaluator()
    b = SCBaseline(
        llm_factory=lambda: make_llm_for_task(task),
        tools=_tools_for_domain(),
        evaluator=evaluator,
        k_sc=3,
    )
    r = await b.run_task(task, seed=0)
    assert r.passed is True
    assert r.config_label == "sc_k3"


# ── BaselineResult aggregation ────────────────────────────────────────────────

class TestBaselineResult:
    def _make_result(self, label, passed_list):
        result = BaselineResult(config_label=label, domain="airline")
        for i, passed in enumerate(passed_list):
            result.runs.append(TaskRunResult(
                task_id=f"t{i // 3}",  # 3 seeds per task
                seed=i % 3,
                passed=passed,
                tokens=TokenCounts(input=300, output=40),
                elapsed_s=0.5,
                avg_consensus=1.0,
                divergences=0,
                config_label=label,
            ))
        return result

    def test_pass_at_k_all_pass(self):
        r = self._make_result("k1", [True, True, True, True, True, True])
        pk, std = r.pass_at_k_score()
        assert pk == 1.0

    def test_pass_at_k_one_seed_fails(self):
        # task 0: seeds [T, F, T] → fails; task 1: seeds [T, T, T] → passes
        r = self._make_result("k1", [True, False, True, True, True, True])
        pk, std = r.pass_at_k_score()
        assert pk == 0.5

    def test_mean_tokens(self):
        r = self._make_result("k1", [True] * 3)
        assert r.mean_tokens() == pytest.approx(340.0)  # 300+40

    def test_summary_line_contains_config(self):
        r = self._make_result("naive_k5", [True] * 3)
        line = r.summary_line()
        assert "naive_k5" in line
        assert "pass^k=" in line
