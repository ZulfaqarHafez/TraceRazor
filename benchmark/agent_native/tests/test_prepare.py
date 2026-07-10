from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest

from benchmark.agent_native.evaluate import CONDITIONS, EvaluationError, load_protocol
from benchmark.agent_native.prepare import build_task_manifest, load_catalog, write_manifest


HOSTS = ("codex", "claude_code", "gemini_cli")
STRATA = ("coding", "tool_heavy_research", "support")


def _catalog(tasks_per_cell: int = 6) -> dict:
    tasks = []
    for host in HOSTS:
        for stratum in STRATA:
            for number in range(tasks_per_cell):
                task_id = f"{host}-{stratum}-{number}"
                tasks.append(
                    {
                        "task_id": task_id,
                        "content_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
                        "host_category": host,
                        "workload_stratum": stratum,
                        "verifier_id": f"verifier-{stratum}",
                    }
                )
    return {"schema_version": "tracerazor-agent-task-catalog/v1", "tasks": tasks}


def test_prepare_is_deterministic_and_position_balanced(tmp_path):
    protocol = load_protocol("benchmark/agent_native/protocol.json")
    catalog = _catalog()

    first = build_task_manifest(
        catalog,
        protocol,
        seed=91,
        generated_at="2026-07-10T00:00:00Z",
    )
    second = build_task_manifest(
        catalog,
        protocol,
        seed=91,
        generated_at="2026-07-10T00:00:00Z",
    )

    assert first == second
    assert len(first["tasks"]) == 54
    positions = Counter()
    for task in first["tasks"]:
        assert {row["repetition"] for row in task["randomization"]} == {1, 2, 3}
        assert len({row["block_id"] for row in task["randomization"]}) == 3
        for row in task["randomization"]:
            for position, condition in enumerate(row["condition_order"], 1):
                positions[(condition, position)] += 1
    assert len(set(positions.values())) == 1
    assert set(condition for condition, _ in positions) == set(CONDITIONS)

    output = tmp_path / "manifest.json"
    write_manifest(output, first)
    assert b"\r\n" not in output.read_bytes()


def test_prepare_rejects_duplicate_content():
    protocol = load_protocol("benchmark/agent_native/protocol.json")
    catalog = _catalog()
    catalog["tasks"][1]["content_sha256"] = catalog["tasks"][0]["content_sha256"]

    with pytest.raises(EvaluationError, match="duplicate task content digest"):
        build_task_manifest(catalog, protocol, generated_at="2026-07-10T00:00:00Z")


def test_prepare_rejects_underfilled_cells():
    protocol = load_protocol("benchmark/agent_native/protocol.json")

    with pytest.raises(EvaluationError, match="tasks; 50 required"):
        build_task_manifest(
            _catalog(tasks_per_cell=4),
            protocol,
            generated_at="2026-07-10T00:00:00Z",
        )


def test_prepare_requires_timezone_aware_lock_time():
    protocol = load_protocol("benchmark/agent_native/protocol.json")

    with pytest.raises(EvaluationError, match="timezone offset"):
        build_task_manifest(
            _catalog(),
            protocol,
            generated_at="2026-07-10T00:00:00",
        )


def test_load_catalog_rejects_unknown_fields(tmp_path):
    catalog = _catalog()
    catalog["tasks"][0]["prompt"] = "private prompt must not enter the planner"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(EvaluationError, match="fields must be exactly"):
        load_catalog(path)
