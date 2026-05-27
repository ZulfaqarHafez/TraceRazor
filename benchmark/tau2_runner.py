"""tau2-bench runner stub — Phase 1 baseline harness.

Structure follows PRD Section 5 (The Benchmark):
- Wraps a ReAct agent in the AdaptiveKNode
- Measures pass^k at k=1, 4, 8
- Reports (pass^k, tokens) Pareto point per configuration
- Works with either OpenAI (primary) or Anthropic (secondary) as LLM

IMPORTANT: This module is a STUB. The tau2-bench task loader and expected-action
evaluator are NOT implemented here — those require the tau2-bench repo
(https://github.com/sierra-research/tau2-bench) to be cloned and installed.

This file provides:
1. The configuration / harness wiring
2. Token accounting wrapper
3. Pass^k evaluation logic
4. The run() entry point that will be filled in once tau2-bench is available

Usage (once tau2-bench is installed):
    python -m benchmark.tau2_runner \
        --domain airline \
        --k 1 4 5 \
        --seeds 3 \
        --model gpt-4.1
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracerazor import AdaptiveKNode, ConsensusReport, openai_llm, mock_llm
from tracerazor._mutation import MutationMetadata

# ── tau2-bench tool mutation table (manually declared per PRD §4.4) ───────────

TAU2_MUTATION_MAP: Dict[str, bool] = {
    # airline domain — mutating
    "book_reservation": True,
    "cancel_reservation": True,
    "update_reservation": True,
    "send_certificate": True,
    # airline domain — read-only
    "search_direct_flight": False,
    "search_onestop_flight": False,
    "search_direct_flight_simple": False,
    "get_flight_status": False,
    "get_airport_info": False,
    "calculate_fare": False,
    "list_all_airports": False,
    # retail domain — mutating
    "cancel_order": True,
    "modify_order": True,
    "return_delivered_order_items": True,
    "exchange_delivered_order_items": True,
    # retail domain — read-only
    "get_order_details": False,
    "get_product_details": False,
    "list_orders": False,
    "find_user_id_by_email": False,
    "find_user_id_by_name_zip": False,
    "get_user_details": False,
    "get_product_details_by_name": False,
}


# ── pass^k evaluator ──────────────────────────────────────────────────────────

def pass_at_k(results: List[bool]) -> float:
    """Return fraction of tasks where ALL k runs succeeded.

    For a single task run k times, pass^k = 1.0 if all k succeeded.
    """
    return 1.0 if all(results) else 0.0


def aggregate_pass_at_k(per_task_results: List[List[bool]]) -> Tuple[float, float]:
    """Mean ± std of pass^k across all tasks.

    Parameters
    ----------
    per_task_results:
        List of [k booleans], one per task.

    Returns
    -------
    (mean, std)
    """
    scores = [pass_at_k(runs) for runs in per_task_results]
    mean = statistics.mean(scores) if scores else 0.0
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return mean, std


# ── run configuration ────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    domain: str                 # "airline" or "retail"
    k_max: int                  # K to evaluate (1, 4, or 5)
    k_min: int = 2
    seeds: int = 3              # independent repetitions per task
    model: str = "gpt-4.1"
    temperature: float = 1.0
    max_steps: int = 30
    output_dir: Path = Path("results")


@dataclass
class TaskResult:
    task_id: str
    seed: int
    passed: bool
    tokens: Dict[str, int]
    consensus_rates: List[float]
    divergence_count: int


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    task_results: List[TaskResult] = field(default_factory=list)

    def pass_at_k_scores(self) -> Dict[str, Any]:
        """Group by task_id → compute pass^k."""
        by_task: Dict[str, List[bool]] = {}
        for r in self.task_results:
            by_task.setdefault(r.task_id, []).append(r.passed)
        mean, std = aggregate_pass_at_k(list(by_task.values()))
        return {
            "pass_k": mean,
            "std": std,
            "k": self.config.k_max,
            "tasks": len(by_task),
            "seeds": self.config.seeds,
        }

    def token_stats(self) -> Dict[str, float]:
        totals = [sum(r.tokens.values()) for r in self.task_results]
        if not totals:
            return {}
        return {
            "mean_tokens": statistics.mean(totals),
            "std_tokens": statistics.stdev(totals) if len(totals) > 1 else 0.0,
            "total_tokens": sum(totals),
        }

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (self.config.output_dir / f"{self.config.domain}_k{self.config.k_max}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config": {
                "domain": self.config.domain,
                "k_max": self.config.k_max,
                "k_min": self.config.k_min,
                "seeds": self.config.seeds,
                "model": self.config.model,
                "temperature": self.config.temperature,
            },
            "pass_at_k": self.pass_at_k_scores(),
            "tokens": self.token_stats(),
            "task_results": [
                {
                    "task_id": r.task_id,
                    "seed": r.seed,
                    "passed": r.passed,
                    "tokens": r.tokens,
                    "avg_consensus": statistics.mean(r.consensus_rates) if r.consensus_rates else 0.0,
                    "divergences": r.divergence_count,
                }
                for r in self.task_results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path


# ── task runner ───────────────────────────────────────────────────────────────

async def run_task(
    task: Dict[str, Any],
    llm: Callable,
    tools: Dict[str, Callable],
    config: BenchmarkConfig,
    seed: int,
    evaluator: Callable[[Dict], bool],
) -> TaskResult:
    """Run one tau2-bench task and return a TaskResult."""
    node = AdaptiveKNode(
        llm=llm,
        tools=tools,
        k_max=config.k_max,
        k_min=config.k_min,
        mutation_metadata=MutationMetadata(overrides=TAU2_MUTATION_MAP),
        max_steps=config.max_steps,
    )

    initial_messages = task.get("messages", [
        {"role": "user", "content": task.get("instruction", "")}
    ])
    state = {"messages": initial_messages}

    result = await node(state)
    report: ConsensusReport = result["consensus_report"]

    passed = evaluator({"state": result, "task": task, "report": report})

    return TaskResult(
        task_id=task["task_id"],
        seed=seed,
        passed=passed,
        tokens=report.tokens.as_dict(),
        consensus_rates=report.consensus_rate,
        divergence_count=len(report.divergences),
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="tau2-bench adaptive-K runner")
    parser.add_argument("--domain", choices=["airline", "retail"], default="airline")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5],
                        help="K values to sweep (e.g. --k 1 4 5)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--tau2-path", type=Path, default=None,
                        help="Path to tau2-bench repo. Required for real runs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 3 mock tasks to validate the harness wiring.")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — using mock LLM (3 synthetic tasks)")
        await _dry_run(args)
        return

    if args.tau2_path is None:
        print(
            "ERROR: --tau2-path is required for real benchmark runs.\n"
            "Clone tau2-bench and pass the repo path:\n"
            "  git clone https://github.com/sierra-research/tau2-bench\n"
            "  python tau2_runner.py --tau2-path ./tau2-bench --domain airline"
        )
        sys.exit(1)

    print("tau2-bench integration not yet wired — Phase 1 deliverable.")
    print("Once tau2-bench is installed, implement _load_tasks() and _make_evaluator().")


async def _dry_run(args: Any) -> None:
    """Validate harness wiring with 3 deterministic mock tasks."""
    from tracerazor._adapters import mock_llm as _mock_llm

    MOCK_TASKS = [
        {"task_id": "mock-001", "instruction": "Find and book AA100 JFK→LAX June 15"},
        {"task_id": "mock-002", "instruction": "Cancel order ORD-9182"},
        {"task_id": "mock-003", "instruction": "What flights are available JFK→ORD?"},
    ]

    MOCK_RESPONSES = [
        {"tool_name": "search_direct_flight",
         "arguments": {"origin": "JFK", "destination": "LAX", "date": "2024-06-15"},
         "input_tokens": 300, "output_tokens": 40},
        {"tool_name": "book_reservation",
         "arguments": {"flight_id": "AA100", "passenger_name": "Test"},
         "input_tokens": 450, "output_tokens": 35, "cached_tokens": 300},
        {"final_answer": "Booked: CONF-AA100-0001",
         "input_tokens": 500, "output_tokens": 45, "cached_tokens": 450},
    ]

    def _tools():
        return {
            "search_direct_flight": lambda origin, destination, date: "[AA100 $450]",
            "book_reservation": lambda flight_id, passenger_name: "CONF-AA100-0001",
            "cancel_order": lambda order_id: "cancelled",
        }

    all_results = []
    for k_val in args.k:
        print(f"\n--- K={k_val} dry-run ---")
        config = BenchmarkConfig(
            domain=args.domain, k_max=k_val, seeds=1, model="mock"
        )
        benchmark = BenchmarkResult(config=config)

        for task in MOCK_TASKS:
            llm = _mock_llm(MOCK_RESPONSES)
            r = await run_task(
                task=task,
                llm=llm,
                tools=_tools(),
                config=config,
                seed=0,
                evaluator=lambda _: True,  # all pass in dry-run
            )
            benchmark.task_results.append(r)
            print(f"  {r.task_id}  tokens={sum(r.tokens.values()):,}  "
                  f"avg_consensus={statistics.mean(r.consensus_rates) if r.consensus_rates else 0:.1%}")

        stats = benchmark.pass_at_k_scores()
        token_stats = benchmark.token_stats()
        print(f"  pass^{k_val} = {stats['pass_k']:.1%}  "
              f"mean_tokens = {token_stats['mean_tokens']:,.0f}")
        all_results.append((k_val, stats["pass_k"], token_stats["mean_tokens"]))

    print("\n=== Pareto summary (pass^k vs tokens) ===")
    print(f"{'K':>4}  {'pass^k':>8}  {'mean_tokens':>12}")
    for k_val, pask, tok in all_results:
        print(f"{k_val:>4}  {pask:>8.1%}  {tok:>12,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
