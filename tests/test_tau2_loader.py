"""Tests for Tau2Loader and Tau2Task."""
import json
import pytest
from pathlib import Path

from benchmark.tau2_loader import (
    MOCK_AIRLINE_TASKS,
    MOCK_RETAIL_TASKS,
    Tau2Loader,
    Tau2Task,
)


# ── Tau2Task ────────────────────────────────────────────────────────────────

class TestTau2Task:
    def test_initial_messages_user_only(self):
        task = Tau2Task(
            task_id="t1",
            instruction="Book a flight",
            expected_actions=[],
            domain="airline",
        )
        msgs = task.initial_messages()
        assert msgs[-1]["role"] == "user"
        assert "Book a flight" in msgs[-1]["content"]

    def test_initial_messages_with_system_prompt(self):
        task = Tau2Task(
            task_id="t1",
            instruction="Book a flight",
            expected_actions=[],
            domain="airline",
            metadata={"system_prompt": "You are an airline agent."},
        )
        msgs = task.initial_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_no_system_prompt_skipped(self):
        task = Tau2Task("t1", "x", [], "airline")
        msgs = task.initial_messages()
        assert all(m["role"] != "system" for m in msgs)

    def test_expected_actions_structure(self):
        task = MOCK_AIRLINE_TASKS[0]
        assert len(task.expected_actions) >= 1
        for action in task.expected_actions:
            assert "name" in action
            assert "arguments" in action


# ── Tau2Loader mock mode ─────────────────────────────────────────────────────

class TestTau2LoaderMock:
    def test_airline_mock_loads(self):
        loader = Tau2Loader(domain="airline")
        tasks = loader.load()
        assert len(tasks) >= 1
        assert all(t.domain == "airline" for t in tasks)

    def test_retail_mock_loads(self):
        loader = Tau2Loader(domain="retail")
        tasks = loader.load()
        assert len(tasks) >= 1
        assert all(t.domain == "retail" for t in tasks)

    def test_max_tasks_truncates(self):
        loader = Tau2Loader(domain="airline", max_tasks=1)
        tasks = loader.load()
        assert len(tasks) == 1

    def test_max_tasks_larger_than_available(self):
        loader = Tau2Loader(domain="airline", max_tasks=9999)
        tasks = loader.load()
        assert len(tasks) == len(MOCK_AIRLINE_TASKS)

    def test_unknown_domain_raises(self):
        loader = Tau2Loader(domain="banking")
        with pytest.raises(ValueError):
            loader.load()

    def test_task_ids_are_unique(self):
        tasks = Tau2Loader(domain="airline").load()
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))


# ── Tau2Loader disk mode ─────────────────────────────────────────────────────

class TestTau2LoaderDisk:
    def test_loads_from_json_file(self, tmp_path):
        data = [
            {
                "task_id": "disk-001",
                "user_instruction": "Find flights",
                "expected_actions": [
                    {"name": "search_direct_flight",
                     "arguments": {"origin": "JFK", "destination": "LAX", "date": "2024-01-01"}}
                ],
            }
        ]
        # Write in the first candidate path pattern
        p = tmp_path / "tau_bench" / "data" / "airline"
        p.mkdir(parents=True)
        (p / "tasks.json").write_text(json.dumps(data))

        loader = Tau2Loader(domain="airline", tau2_path=tmp_path)
        tasks = loader.load()
        assert len(tasks) == 1
        assert tasks[0].task_id == "disk-001"
        assert tasks[0].instruction == "Find flights"

    def test_missing_path_raises(self, tmp_path):
        loader = Tau2Loader(domain="airline", tau2_path=tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_parses_alternate_key_instruction(self, tmp_path):
        data = [{"task_id": "alt-001", "instruction": "Cancel", "expected_actions": []}]
        p = tmp_path / "data" / "airline"
        p.mkdir(parents=True)
        (p / "tasks.json").write_text(json.dumps(data))
        tasks = Tau2Loader(domain="airline", tau2_path=tmp_path).load()
        assert tasks[0].instruction == "Cancel"

    def test_nested_tasks_key(self, tmp_path):
        data = {"tasks": [
            {"task_id": "n-001", "user_instruction": "x", "expected_actions": []}
        ]}
        p = tmp_path / "tau_bench" / "data" / "airline"
        p.mkdir(parents=True)
        (p / "tasks.json").write_text(json.dumps(data))
        tasks = Tau2Loader(domain="airline", tau2_path=tmp_path).load()
        assert tasks[0].task_id == "n-001"


# ── mock task content ────────────────────────────────────────────────────────

class TestMockTaskContent:
    def test_airline_task_001_has_search_and_book(self):
        t = MOCK_AIRLINE_TASKS[0]
        names = [a["name"] for a in t.expected_actions]
        assert "search_direct_flight" in names
        assert "book_reservation" in names

    def test_retail_task_001_has_cancel(self):
        t = MOCK_RETAIL_TASKS[0]
        names = [a["name"] for a in t.expected_actions]
        assert "cancel_order" in names
