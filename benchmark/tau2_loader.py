"""tau2-bench task loader.

Provides:
- ``Tau2Task``   dataclass representing one benchmark task
- ``Tau2Loader`` loads tasks from disk (tau2-bench repo) or a mock list
- ``MOCK_AIRLINE_TASKS`` / ``MOCK_RETAIL_TASKS`` for offline testing

Expected on-disk format (tau2-bench repo):
    <tau2_path>/tau_bench/data/<domain>/tasks.json
    or
    <tau2_path>/data/<domain>/tasks.json

Each task JSON object:
{
    "task_id":          "airline-001",
    "user_instruction": "Book a direct flight from JFK to LAX ...",
    "expected_actions": [
        {"name": "search_direct_flight",
         "arguments": {"origin": "JFK", "destination": "LAX", "date": "2024-06-15"}},
        {"name": "book_reservation",
         "arguments": {"flight_id": "AA100", "passenger_name": "Alice Smith"}}
    ],
    "metadata": {}   // optional
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Tau2Task:
    task_id: str
    instruction: str
    expected_actions: List[Dict[str, Any]]   # [{name, arguments}]
    domain: str                               # "airline" | "retail"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def initial_messages(self) -> List[Dict[str, str]]:
        """Build the opening message list for this task."""
        msgs: List[Dict[str, str]] = []
        sys = self.metadata.get("system_prompt", "")
        if sys:
            msgs.append({"role": "system", "content": sys})
        msgs.append({"role": "user", "content": self.instruction})
        return msgs


# ── candidate paths inside a tau2-bench checkout ──────────────────────────

_CANDIDATE_SUBPATHS = [
    "tau_bench/data/{domain}/tasks.json",
    "data/{domain}/tasks.json",
    "tasks/{domain}.json",
    "{domain}/tasks.json",
]


class Tau2Loader:
    """Load tau2-bench tasks from disk or from an in-memory mock list.

    Parameters
    ----------
    tau2_path:
        Root of the tau2-bench checkout.  ``None`` → use mock tasks only.
    domain:
        ``"airline"`` or ``"retail"``.
    max_tasks:
        Truncate to first N tasks (useful for quick smoke tests).
    """

    def __init__(
        self,
        domain: str,
        tau2_path: Optional[Path] = None,
        max_tasks: Optional[int] = None,
    ) -> None:
        self.domain = domain
        self.tau2_path = Path(tau2_path) if tau2_path else None
        self.max_tasks = max_tasks

    def load(self) -> List[Tau2Task]:
        if self.tau2_path is not None:
            tasks = self._load_from_disk()
        else:
            tasks = self._load_mock()
        if self.max_tasks is not None:
            tasks = tasks[: self.max_tasks]
        return tasks

    def _load_from_disk(self) -> List[Tau2Task]:
        assert self.tau2_path is not None
        for pattern in _CANDIDATE_SUBPATHS:
            path = self.tau2_path / pattern.format(domain=self.domain)
            if path.exists():
                return self._parse_file(path)
        raise FileNotFoundError(
            f"Could not find tasks for domain '{self.domain}' under {self.tau2_path}.\n"
            f"Tried: {[str(self.tau2_path / p.format(domain=self.domain)) for p in _CANDIDATE_SUBPATHS]}"
        )

    def _parse_file(self, path: Path) -> List[Tau2Task]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            # some versions nest tasks under a key
            raw = raw.get("tasks", raw.get(self.domain, list(raw.values())[0]))
        tasks = []
        for item in raw:
            tasks.append(
                Tau2Task(
                    task_id=item.get("task_id", item.get("id", f"{self.domain}-{len(tasks)}")),
                    instruction=item.get("user_instruction", item.get("instruction", "")),
                    expected_actions=item.get("expected_actions", []),
                    domain=self.domain,
                    metadata=item.get("metadata", {}),
                )
            )
        return tasks

    def _load_mock(self) -> List[Tau2Task]:
        if self.domain == "airline":
            return list(MOCK_AIRLINE_TASKS)
        if self.domain == "retail":
            return list(MOCK_RETAIL_TASKS)
        raise ValueError(f"No mock tasks for domain '{self.domain}'")


# ── offline mock tasks (used when tau2_path is None) ──────────────────────

MOCK_AIRLINE_TASKS: List[Tau2Task] = [
    Tau2Task(
        task_id="mock-airline-001",
        instruction="Book a direct flight from JFK to LAX on 2024-06-15 for Alice Smith.",
        domain="airline",
        expected_actions=[
            {"name": "search_direct_flight",
             "arguments": {"origin": "JFK", "destination": "LAX", "date": "2024-06-15"}},
            {"name": "book_reservation",
             "arguments": {"flight_id": "AA100", "passenger_name": "Alice Smith"}},
        ],
    ),
    Tau2Task(
        task_id="mock-airline-002",
        instruction="Cancel reservation CONF-AA100-9182.",
        domain="airline",
        expected_actions=[
            {"name": "cancel_reservation",
             "arguments": {"confirmation_id": "CONF-AA100-9182"}},
        ],
    ),
    Tau2Task(
        task_id="mock-airline-003",
        instruction="What direct flights are available from ORD to MIA on 2024-07-04?",
        domain="airline",
        expected_actions=[
            {"name": "search_direct_flight",
             "arguments": {"origin": "ORD", "destination": "MIA", "date": "2024-07-04"}},
        ],
    ),
]

MOCK_RETAIL_TASKS: List[Tau2Task] = [
    Tau2Task(
        task_id="mock-retail-001",
        instruction="Cancel order ORD-9182.",
        domain="retail",
        expected_actions=[
            {"name": "get_order_details", "arguments": {"order_id": "ORD-9182"}},
            {"name": "cancel_order", "arguments": {"order_id": "ORD-9182"}},
        ],
    ),
    Tau2Task(
        task_id="mock-retail-002",
        instruction="What is the status of order ORD-5555?",
        domain="retail",
        expected_actions=[
            {"name": "get_order_details", "arguments": {"order_id": "ORD-5555"}},
        ],
    ),
]
