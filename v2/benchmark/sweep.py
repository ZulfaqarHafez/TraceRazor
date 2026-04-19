"""Full benchmark sweep — all configurations across seeds, Pareto output.

Runs every combination of (config, domain, seed) and reports:
  - pass^k ± std
  - mean token cost (total and fresh)
  - Pareto table: pass^k vs mean tokens

Usage (real run, requires API key + tau2-bench):
    python -m benchmark.sweep --domain airline --seeds 3 --model gpt-4.1

Usage (offline dry-run, no API key):
    python -m benchmark.sweep --dry-run
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.baselines import (
    BaselineResult,
    K1Baseline,
    NaiveK5Baseline,
    SCBaseline,
    pass_at_k,
)
from benchmark.tau2_evaluator import ActionMatchEvaluator, get_evaluator
from benchmark.tau2_loader import Tau2Loader, Tau2Task
from tracerazor import AdaptiveKNode, mock_llm
from tracerazor._mutation import MutationMetadata


# ── Pareto output ─────────────────────────────────────────────────────────

def print_pareto(results: List[BaselineResult]) -> None:
    print("\n" + "=" * 72)
    print(f"{'Config':<14}  {'pass^k':>8}  {'std':>6}  {'mean_tokens':>12}  {'mean_fresh':>11}")
    print("-" * 72)
    for r in results:
        pk, std = r.pass_at_k_score()
        print(
            f"{r.config_label:<14}  {pk:>8.1%}  {std:>6.3f}  "
            f"{r.mean_tokens():>12,.0f}  {r.mean_fresh_tokens():>11,.0f}"
        )
    print("=" * 72)


def save_results(results: List[BaselineResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for r in results:
        pk, std = r.pass_at_k_score()
        data.append({
            "config": r.config_label,
            "domain": r.domain,
            "pass_k": pk,
            "std": std,
            "mean_tokens": r.mean_tokens(),
            "mean_fresh_tokens": r.mean_fresh_tokens(),
            "n_runs": len(r.runs),
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")


# ── sweep runner ──────────────────────────────────────────────────────────

async def run_sweep(
    domain: str,
    llm_factory: Callable[[], Callable],
    tools: Dict[str, Callable],
    evaluator: Callable,
    seeds: int = 3,
    k_naive: int = 5,
    k_sc: int = 5,
    k_adaptive_max: int = 5,
    mutation_metadata: Optional[Any] = None,
    max_steps: int = 30,
    tasks: Optional[List[Tau2Task]] = None,
    tau2_path: Optional[Path] = None,
) -> List[BaselineResult]:
    """Run all four configurations and return results list."""

    if tasks is None:
        tasks = Tau2Loader(domain=domain, tau2_path=tau2_path).load()

    common = dict(
        llm_factory=llm_factory,
        tools=tools,
        evaluator=evaluator,
        mutation_metadata=mutation_metadata,
        max_steps=max_steps,
    )

    runners = [
        K1Baseline(**common),
        NaiveK5Baseline(**common, k=k_naive),
        SCBaseline(**common, k_sc=k_sc),
    ]

    # Adaptive-K runner (wraps AdaptiveKNode directly)
    adaptive_runner = _AdaptiveKBaseline(
        k_max=k_adaptive_max,
        **common,
    )

    all_runners = runners + [adaptive_runner]
    results: List[BaselineResult] = []
    for runner in all_runners:
        r = await runner.run_domain(tasks, seeds=seeds)
        results.append(r)
        print(r.summary_line())

    return results


# ── AdaptiveK baseline wrapper ────────────────────────────────────────────

class _AdaptiveKBaseline:
    """Thin wrapper so AdaptiveKNode fits the baseline runner interface."""

    def __init__(
        self,
        llm_factory, tools, evaluator,
        k_max=5, mutation_metadata=None, max_steps=30,
    ):
        self.llm_factory = llm_factory
        self.tools = tools
        self.evaluator = evaluator
        self.k_max = k_max
        self.max_steps = max_steps
        if isinstance(mutation_metadata, dict):
            mutation_metadata = MutationMetadata(overrides=mutation_metadata)
        self.mutation = mutation_metadata or MutationMetadata()

    async def run_domain(self, tasks, seeds=3) -> BaselineResult:
        import time
        label = f"adaptive_k{self.k_max}"
        result = BaselineResult(config_label=label, domain=tasks[0].domain if tasks else "")
        for task in tasks:
            for seed in range(seeds):
                node = AdaptiveKNode(
                    llm=self.llm_factory(),
                    tools=self.tools,
                    k_max=self.k_max,
                    k_min=2,
                    mutation_metadata=self.mutation,
                    max_steps=self.max_steps,
                )
                t0 = time.monotonic()
                state = await node({"messages": task.initial_messages()})
                elapsed = time.monotonic() - t0
                report = state["consensus_report"]
                passed = self.evaluator(report.trajectory, task)
                from benchmark.baselines import TaskRunResult
                result.runs.append(TaskRunResult(
                    task_id=task.task_id,
                    seed=seed,
                    passed=passed,
                    tokens=report.tokens,
                    elapsed_s=elapsed,
                    avg_consensus=report.avg_consensus(),
                    divergences=len(report.divergences),
                    config_label=label,
                ))
        return result


# ── dry-run with mock data ────────────────────────────────────────────────

async def dry_run(domain: str = "airline", seeds: int = 1) -> None:
    """Validate the full sweep harness offline using mock LLM and mock tasks."""

    from benchmark.tau2_loader import MOCK_AIRLINE_TASKS, MOCK_RETAIL_TASKS
    from tracerazor._mutation import MutationMetadata

    tasks = MOCK_AIRLINE_TASKS if domain == "airline" else MOCK_RETAIL_TASKS

    # Tools that satisfy mock task expected actions
    def search_direct_flight(origin, destination, date):
        return f"[{{'flight_id': 'AA100', 'origin': '{origin}', 'dest': '{destination}'}}]"

    def book_reservation(flight_id, passenger_name):
        return f"{{'confirmation': 'CONF-{flight_id}-0001'}}"

    def cancel_reservation(confirmation_id):
        return f"{{'status': 'cancelled', 'id': '{confirmation_id}'}}"

    def get_order_details(order_id):
        return f"{{'order_id': '{order_id}', 'status': 'processing'}}"

    def cancel_order(order_id):
        return f"{{'order_id': '{order_id}', 'status': 'cancelled'}}"

    tools = {
        "search_direct_flight": search_direct_flight,
        "book_reservation": book_reservation,
        "cancel_reservation": cancel_reservation,
        "get_order_details": get_order_details,
        "cancel_order": cancel_order,
    }

    mutation_meta = {
        "book_reservation": True, "cancel_reservation": True,
        "cancel_order": True, "modify_order": True,
        "search_direct_flight": False, "get_order_details": False,
    }

    evaluator = ActionMatchEvaluator()

    def _make_llm_for_task(task: Tau2Task):
        """Build a mock that returns each expected action in order, then answers."""
        responses = []
        for action in task.expected_actions:
            responses.append({
                "tool_name": action["name"],
                "arguments": action["arguments"],
                "input_tokens": 300, "output_tokens": 40, "cached_tokens": 0,
            })
        responses.append({
            "final_answer": "Done.",
            "input_tokens": 350, "output_tokens": 20, "cached_tokens": 300,
        })
        return lambda: mock_llm(responses)

    print(f"\nDRY RUN — domain={domain}, seeds={seeds}, {len(tasks)} tasks")
    results: List[BaselineResult] = []

    for task in tasks:
        llm_factory = _make_llm_for_task(task)
        common = dict(
            tools=tools,
            evaluator=evaluator,
            mutation_metadata=mutation_meta,
            max_steps=10,
        )
        runners = [
            K1Baseline(llm_factory=llm_factory, **common),
            NaiveK5Baseline(llm_factory=llm_factory, k=3, **common),
            SCBaseline(llm_factory=llm_factory, k_sc=3, **common),
        ]
        for runner in runners:
            r = await runner.run_domain([task], seeds=seeds)
            # merge into aggregate
            existing = next((x for x in results if x.config_label == r.config_label), None)
            if existing:
                existing.runs.extend(r.runs)
            else:
                results.append(r)

    print_pareto(results)
    save_results(results, Path("results") / f"dry_run_{domain}.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="airline", choices=["airline", "retail"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        asyncio.run(dry_run(domain=args.domain, seeds=args.seeds))
    else:
        print("Pass --dry-run for offline validation, or wire a real LLM + tau2-bench.")
