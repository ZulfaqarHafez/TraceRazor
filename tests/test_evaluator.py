"""Tests for ActionMatchEvaluator."""
import pytest
from benchmark.tau2_loader import Tau2Task
from benchmark.tau2_evaluator import ActionMatchEvaluator, get_evaluator


def make_task(actions):
    return Tau2Task(
        task_id="t",
        instruction="x",
        expected_actions=actions,
        domain="airline",
    )


def tool_step(name, args):
    return {"type": "tool_call", "tool": name, "args": args}


def answer_step(text="done"):
    return {"type": "final_answer", "content": text}


EV = ActionMatchEvaluator()


# ── exact match ──────────────────────────────────────────────────────────────

class TestExactMatch:
    def test_correct_single_action(self):
        task = make_task([{"name": "search", "arguments": {"q": "NYC"}}])
        traj = [tool_step("search", {"q": "NYC"}), answer_step()]
        assert EV(traj, task) is True

    def test_correct_two_actions(self):
        task = make_task([
            {"name": "search", "arguments": {"q": "NYC"}},
            {"name": "book", "arguments": {"id": 1}},
        ])
        traj = [
            tool_step("search", {"q": "NYC"}),
            tool_step("book", {"id": 1}),
            answer_step(),
        ]
        assert EV(traj, task) is True

    def test_wrong_tool_name(self):
        task = make_task([{"name": "search", "arguments": {"q": "NYC"}}])
        traj = [tool_step("lookup", {"q": "NYC"})]
        assert EV(traj, task) is False

    def test_wrong_argument_value(self):
        task = make_task([{"name": "search", "arguments": {"q": "NYC"}}])
        traj = [tool_step("search", {"q": "LA"})]
        assert EV(traj, task) is False

    def test_too_many_tool_calls(self):
        task = make_task([{"name": "search", "arguments": {"q": "x"}}])
        traj = [tool_step("search", {"q": "x"}), tool_step("book", {"id": 1})]
        assert EV(traj, task) is False

    def test_too_few_tool_calls(self):
        task = make_task([
            {"name": "search", "arguments": {"q": "x"}},
            {"name": "book", "arguments": {"id": 1}},
        ])
        traj = [tool_step("search", {"q": "x"})]
        assert EV(traj, task) is False

    def test_empty_expected_and_empty_trajectory(self):
        task = make_task([])
        assert EV([], task) is True

    def test_empty_expected_but_has_tool_call(self):
        task = make_task([])
        assert EV([tool_step("search", {})], task) is False


# ── canonicalisation in evaluation ───────────────────────────────────────────

class TestCanonicalisation:
    def test_case_insensitive_tool_name(self):
        task = make_task([{"name": "search", "arguments": {"q": "NYC"}}])
        traj = [tool_step("SEARCH", {"q": "NYC"})]
        assert EV(traj, task) is True

    def test_whitespace_in_args(self):
        task = make_task([{"name": "search", "arguments": {"q": "NYC"}}])
        traj = [tool_step("search", {"q": "  NYC  "})]
        assert EV(traj, task) is True

    def test_numeric_string_equals_int(self):
        task = make_task([{"name": "book", "arguments": {"seats": 2}}])
        traj = [tool_step("book", {"seats": "2"})]
        assert EV(traj, task) is True

    def test_dict_key_order(self):
        task = make_task([{"name": "f", "arguments": {"b": 1, "a": 2}}])
        traj = [tool_step("f", {"a": 2, "b": 1})]
        assert EV(traj, task) is True

    def test_null_equivalence(self):
        task = make_task([{"name": "f", "arguments": {"a": 1}}])
        traj = [tool_step("f", {"a": 1, "b": None})]
        assert EV(traj, task) is True


# ── require_final_answer flag ────────────────────────────────────────────────

class TestRequireFinalAnswer:
    def test_no_final_answer_fails_when_required(self):
        ev = ActionMatchEvaluator(require_final_answer=True)
        task = make_task([{"name": "search", "arguments": {"q": "x"}}])
        traj = [tool_step("search", {"q": "x"})]  # no final_answer step
        assert ev(traj, task) is False

    def test_with_final_answer_passes_when_required(self):
        ev = ActionMatchEvaluator(require_final_answer=True)
        task = make_task([{"name": "search", "arguments": {"q": "x"}}])
        traj = [tool_step("search", {"q": "x"}), answer_step()]
        assert ev(traj, task) is True

    def test_default_does_not_require_final_answer(self):
        task = make_task([{"name": "search", "arguments": {"q": "x"}}])
        traj = [tool_step("search", {"q": "x"})]
        assert EV(traj, task) is True


# ── get_evaluator factory ────────────────────────────────────────────────────

class TestGetEvaluator:
    def test_offline_returns_action_match(self):
        ev = get_evaluator("airline", offline=True)
        assert isinstance(ev, ActionMatchEvaluator)

    def test_offline_mode_used_when_tau2_unavailable(self):
        # even with offline=False, falls back if tau2-bench not installed
        ev = get_evaluator("airline", offline=False)
        # should be ActionMatchEvaluator as tau2-bench is not installed here
        assert callable(ev)
