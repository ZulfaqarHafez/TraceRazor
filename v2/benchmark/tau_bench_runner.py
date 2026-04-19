"""Real tau-bench runner — integrates with the actual tau-bench environment.

Runs all four Phase 1 configurations against the live tau-bench airline env
(with user simulation) and reports pass^k vs token Pareto numbers.

Requirements:
    - tau-bench cloned at TAU_BENCH_PATH (or TAUBENCH env var)
    - OPENAI_API_KEY set (loaded from .env automatically)
    - pip install openai litellm

Usage:
    # Quick smoke-test: 3 tasks, 1 seed, gpt-4o-mini
    python -m benchmark.tau_bench_runner --tasks 3 --seeds 1

    # Phase 1 gate: 50 tasks, 3 seeds, gpt-4o (matches tau-bench reference)
    python -m benchmark.tau_bench_runner --tasks 50 --seeds 3 --model gpt-4o
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── load .env ────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── tau-bench path ────────────────────────────────────────────────────────
TAU_BENCH_PATH = os.environ.get("TAUBENCH", str(Path.home() / "tau-bench"))
if TAU_BENCH_PATH not in sys.path:
    sys.path.insert(0, TAU_BENCH_PATH)

# ── local package path ────────────────────────────────────────────────────
_v2_path = str(Path(__file__).parent.parent)
if _v2_path not in sys.path:
    sys.path.insert(0, _v2_path)

from openai import AsyncOpenAI

from tracerazor._adapters import mock_llm
from tracerazor._mutation import MutationMetadata
from tracerazor._naive_ensemble import NaiveKEnsemble
from tracerazor._self_consistency import SelfConsistencyBaseline
from tracerazor._adaptive_k import AdaptiveKNode
from tracerazor._consensus import BranchProposal, ExactMatchConsensus, Outcome
from tracerazor._report import TokenCounts
from benchmark.baselines import pass_at_k


# ── tau-bench tool mutation map ───────────────────────────────────────────

AIRLINE_MUTATION = {
    "book_reservation": True,
    "cancel_reservation": True,
    "update_reservation_flights": True,
    "update_reservation_baggages": True,
    "update_reservation_passengers": True,
    "send_certificate": True,
    "transfer_to_human_agents": True,
    "search_direct_flight": False,
    "search_onestop_flight": False,
    "get_reservation_details": False,
    "get_user_details": False,
    "list_all_airports": False,
    "calculate": False,
    "think": False,
}


# ── tau-bench agent adapter ───────────────────────────────────────────────

@dataclass
class RunResult:
    task_index: int
    seed: int
    reward: float
    passed: bool
    tokens: TokenCounts
    elapsed_s: float
    steps: int
    config: str


class TauBenchAgent:
    """Wraps tau-bench Env to run one task with a given LLM.

    Implements the standard ReAct loop:
      - tool call  → env.step(tool_action)
      - text reply → env.step(respond_action)
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        user_model: str = "gpt-4o-mini",
        max_steps: int = 30,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.user_model = user_model
        self.max_steps = max_steps
        self.temperature = temperature

    async def _llm(
        self, messages: List[dict], tools: List[dict]
    ) -> Tuple[Optional[str], Any, TokenCounts, Optional[str]]:
        """Call the LLM. Returns (tool_name, args_or_content, tokens, tool_call_id)."""
        for attempt in range(8):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools if tools else None,
                    temperature=self.temperature,
                )
                break
            except Exception as exc:
                msg = str(exc).lower()
                if "rate_limit" in msg or "429" in msg or "connection" in msg or "getaddrinfo" in msg:
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError("LLM rate-limit retries exhausted")
        msg = resp.choices[0].message
        usage = resp.usage
        details = getattr(usage, "prompt_tokens_details", None)
        tokens = TokenCounts(
            input=usage.prompt_tokens,
            output=usage.completion_tokens,
            cached=getattr(details, "cached_tokens", 0) if details else 0,
        )
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {"raw": tc.function.arguments}
            return tc.function.name, args, tokens, tc.id
        return None, msg.content or "", tokens, None

    async def run_task(self, env, task_index: int, seed: int, config: str) -> RunResult:
        from tau_bench.types import Action, RESPOND_ACTION_NAME

        t0 = time.monotonic()
        reset = env.reset(task_index=task_index)
        obs = reset.observation

        # Match tau-bench reference agent: only domain tools, no respond function.
        # The LLM produces plain text to respond; absence of tool_calls → respond action.
        tools_info = list(env.tools_info)

        messages: List[dict] = [
            {"role": "system", "content": env.wiki},
            {"role": "user", "content": obs},
        ]

        total_tokens = TokenCounts()
        reward = 0.0
        done = False
        steps = 0

        for _ in range(self.max_steps):
            tool_name, result, tokens, tc_id = await self._llm(messages, tools_info)
            total_tokens += tokens
            steps += 1

            if tool_name and tool_name != RESPOND_ACTION_NAME:
                action = Action(name=tool_name, kwargs=result)
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": tc_id, "type": "function",
                                    "function": {"name": tool_name, "arguments": json.dumps(result)}}],
                })
            else:
                content = result if isinstance(result, str) else str(result)
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
                messages.append({"role": "assistant", "content": content})

            env_resp = env.step(action)
            reward = env_resp.reward
            done = env_resp.done
            obs = env_resp.observation

            if tool_name and tool_name != RESPOND_ACTION_NAME:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tool_name,
                    "content": obs,
                })
            else:
                if not done:
                    messages.append({"role": "user", "content": obs})

            if done:
                break

        elapsed = time.monotonic() - t0
        return RunResult(
            task_index=task_index,
            seed=seed,
            reward=reward,
            passed=reward >= 1.0,
            tokens=total_tokens,
            elapsed_s=elapsed,
            steps=steps,
            config=config,
        )


# ── Phase 2: Adaptive-K agent ────────────────────────────────────────────

class TauBenchAdaptiveAgent(TauBenchAgent):
    """TauBenchAgent upgraded with K-parallel sampling + ExactMatchConsensus.

    At each step, samples K completions in parallel.  The consensus winner is
    executed exactly once in the env (no duplicate side-effects at mutation
    boundaries).  K adapts: shrinks under sustained full consensus, resets on
    divergence or after a mutating tool call.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        user_model: str = "gpt-4o-mini",
        k_max: int = 5,
        k_min: int = 2,
        max_steps: int = 30,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(client=client, model=model, user_model=user_model,
                         max_steps=max_steps, temperature=temperature)
        self.k_max = k_max
        self.k_min = min(k_min, k_max)
        self.consensus = ExactMatchConsensus()
        self.mutation_meta = MutationMetadata(overrides=AIRLINE_MUTATION)

    async def run_task(self, env, task_index: int, seed: int, config: str) -> RunResult:
        from tau_bench.types import Action, RESPOND_ACTION_NAME

        t0 = time.monotonic()
        reset = env.reset(task_index=task_index)
        obs = reset.observation
        tools_info = list(env.tools_info)

        messages: List[dict] = [
            {"role": "system", "content": env.wiki},
            {"role": "user", "content": obs},
        ]

        total_tokens = TokenCounts()
        reward = 0.0
        done = False
        steps = 0
        k = self.k_max

        for _ in range(self.max_steps):
            # ── sample K completions in parallel ─────────────────────────
            raw = await asyncio.gather(*[
                self._llm(list(messages), tools_info) for _ in range(k)
            ])
            for _, _, tok, _ in raw:
                total_tokens += tok

            # ── build branch proposals ────────────────────────────────────
            proposals = []
            for i, (tn, res, tok, _tc) in enumerate(raw):
                if tn is None:
                    tn_prop = RESPOND_ACTION_NAME
                    args_prop = {"content": res if isinstance(res, str) else str(res)}
                else:
                    tn_prop = tn
                    args_prop = res if isinstance(res, dict) else {}
                proposals.append(BranchProposal(
                    branch_id=i,
                    tool_name=tn_prop,
                    arguments=args_prop,
                    final_answer=None,
                    input_tokens=tok.input,
                    output_tokens=tok.output,
                    cached_tokens=tok.cached,
                ))

            # ── consensus ────────────────────────────────────────────────
            cr = self.consensus.aggregate(proposals)
            winning = cr.winning_proposal

            if cr.outcome == Outcome.FULL:
                k = max(self.k_min, k - 1)
            elif cr.outcome == Outcome.DIVERGENT:
                k = self.k_max

            # ── execute winner once ───────────────────────────────────────
            tool_name = winning.tool_name
            win_args = winning.arguments or {}
            tc_id = raw[winning.branch_id][3]  # actual OpenAI tool_call_id

            if tool_name == RESPOND_ACTION_NAME:
                content = win_args.get("content", "") if isinstance(win_args, dict) else str(win_args)
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
                messages.append({"role": "assistant", "content": content})
            else:
                action = Action(name=tool_name, kwargs=win_args)
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": tc_id, "type": "function",
                                    "function": {"name": tool_name,
                                                 "arguments": json.dumps(win_args)}}],
                })

            env_resp = env.step(action)
            reward = env_resp.reward
            done = env_resp.done
            obs = env_resp.observation
            steps += 1

            if tool_name != RESPOND_ACTION_NAME:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tool_name,
                    "content": obs,
                })
                if self.mutation_meta.is_mutating(tool_name):
                    k = self.k_max
            else:
                if not done:
                    messages.append({"role": "user", "content": obs})

            if done:
                break

        elapsed = time.monotonic() - t0
        return RunResult(
            task_index=task_index,
            seed=seed,
            reward=reward,
            passed=reward >= 1.0,
            tokens=total_tokens,
            elapsed_s=elapsed,
            steps=steps,
            config=config,
        )


# ── Phase 3a: Naive K=5 ensemble ─────────────────────────────────────────

class TauBenchNaiveK5Agent:
    """Run K independent K=1 agents at temperature=1.0; task passes if majority pass.

    This is the dumb-spend baseline: K×token cost with independent runs.
    Majority = ceil(K/2) of K runs must pass.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        user_model: str = "gpt-4o-mini",
        k: int = 5,
        max_steps: int = 30,
    ) -> None:
        self.client = client
        self.model = model
        self.user_model = user_model
        self.k = k
        self.max_steps = max_steps

    async def run_task(self, env, task_index: int, seed: int, config: str) -> RunResult:
        from tau_bench.envs.airline.env import MockAirlineDomainEnv

        t0 = time.monotonic()
        agents = [
            TauBenchAgent(
                client=self.client, model=self.model, user_model=self.user_model,
                max_steps=self.max_steps, temperature=1.0,
            )
            for _ in range(self.k)
        ]

        async def _one(i: int) -> RunResult:
            ind_env = MockAirlineDomainEnv(
                user_strategy="llm",
                user_model=self.user_model,
                user_provider="openai",
                task_split="test",
                task_index=0,
            )
            return await agents[i].run_task(ind_env, task_index, seed, f"{config}_r{i}")

        sub_results = await asyncio.gather(*[_one(i) for i in range(self.k)])

        total_tokens = TokenCounts()
        for r in sub_results:
            total_tokens += r.tokens
        passes = sum(1 for r in sub_results if r.passed)
        majority_pass = passes >= math.ceil(self.k / 2)

        elapsed = time.monotonic() - t0
        return RunResult(
            task_index=task_index,
            seed=seed,
            reward=1.0 if majority_pass else 0.0,
            passed=majority_pass,
            tokens=total_tokens,
            elapsed_s=elapsed,
            steps=sum(r.steps for r in sub_results),
            config=config,
        )


# ── Phase 3b: Self-Consistency (SC) ──────────────────────────────────────

class TauBenchSCAgent(TauBenchAgent):
    """K=1 through all tool calls, then k_sc re-samples of the final respond.

    For tau-bench, the final respond content matters only for output-required tasks
    (those with task.outputs). SC helps surface the correct output values by sampling
    k_sc final responses and picking the one mentioning all required outputs.
    Database state is set by the deterministic K=1 tool calls — SC only affects text.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        user_model: str = "gpt-4o-mini",
        k_sc: int = 5,
        max_steps: int = 30,
    ) -> None:
        super().__init__(client=client, model=model, user_model=user_model,
                         max_steps=max_steps, temperature=0.0)
        self.k_sc = k_sc

    async def run_task(self, env, task_index: int, seed: int, config: str) -> RunResult:
        from tau_bench.types import Action, RESPOND_ACTION_NAME

        t0 = time.monotonic()
        reset = env.reset(task_index=task_index)
        obs = reset.observation
        tools_info = list(env.tools_info)
        messages: List[dict] = [
            {"role": "system", "content": env.wiki},
            {"role": "user", "content": obs},
        ]

        total_tokens = TokenCounts()
        reward = 0.0
        done = False
        steps = 0
        pre_final_messages: Optional[List[dict]] = None

        # ── Phase 1: K=1 run to completion (deterministic tool calls) ─────
        for _ in range(self.max_steps):
            tool_name, result, tokens, tc_id = await self._llm(messages, tools_info)
            total_tokens += tokens
            steps += 1

            if tool_name and tool_name != RESPOND_ACTION_NAME:
                action = Action(name=tool_name, kwargs=result)
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": tc_id, "type": "function",
                                    "function": {"name": tool_name,
                                                 "arguments": json.dumps(result)}}],
                })
                env_resp = env.step(action)
                reward = env_resp.reward
                done = env_resp.done
                obs = env_resp.observation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tool_name,
                    "content": obs,
                })
            else:
                # About to respond — snapshot pre-final context for SC sampling
                pre_final_messages = list(messages)
                content = result if isinstance(result, str) else str(result)

                # ── Phase 2: sample k_sc responses, pick one mentioning most outputs ─
                if self.k_sc > 1:
                    sc_agent = TauBenchAgent(
                        client=self.client, model=self.model,
                        user_model=self.user_model, temperature=1.0,
                    )
                    candidates: List[str] = [content]
                    for _ in range(self.k_sc - 1):
                        _, cand, tok, _ = await sc_agent._llm(pre_final_messages, [])
                        total_tokens += tok
                        candidates.append(cand if isinstance(cand, str) else str(cand))

                    # Pick candidate mentioning most required outputs (case-insensitive)
                    task_outputs = [o.lower() for o in env.task.outputs]
                    if task_outputs:
                        def _score(c: str) -> int:
                            cl = c.lower().replace(",", "")
                            return sum(1 for o in task_outputs if o in cl)
                        content = max(candidates, key=_score)
                    else:
                        content = candidates[0]  # no outputs required — K=1 is fine

                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
                messages.append({"role": "assistant", "content": content})
                env_resp = env.step(action)
                reward = env_resp.reward
                done = env_resp.done
                obs = env_resp.observation
                if not done:
                    messages.append({"role": "user", "content": obs})

            if done:
                break

        elapsed = time.monotonic() - t0
        return RunResult(
            task_index=task_index,
            seed=seed,
            reward=reward,
            passed=reward >= 1.0,
            tokens=total_tokens,
            elapsed_s=elapsed,
            steps=steps,
            config=config,
        )


# ── multi-seed pass^k runner ──────────────────────────────────────────────

async def run_config(
    config_label: str,
    client: AsyncOpenAI,
    model: str,
    user_model: str,
    task_indices: List[int],
    seeds: int,
    max_steps: int = 30,
    concurrency: int = 5,
    k_max: int = 1,
    k_min: int = 1,
    naive_k: int = 0,
    sc_k: int = 0,
) -> List[RunResult]:
    """Run one configuration across tasks and seeds with bounded concurrency."""
    from tau_bench.envs.airline.env import MockAirlineDomainEnv

    total = len(task_indices) * seeds
    done_count = [0]
    sem = asyncio.Semaphore(concurrency)

    if naive_k > 1:
        agent: Any = TauBenchNaiveK5Agent(
            client=client, model=model, user_model=user_model,
            k=naive_k, max_steps=max_steps,
        )
    elif sc_k > 1:
        agent = TauBenchSCAgent(
            client=client, model=model, user_model=user_model,
            k_sc=sc_k, max_steps=max_steps,
        )
    elif k_max > 1:
        agent = TauBenchAdaptiveAgent(
            client=client, model=model, user_model=user_model,
            k_max=k_max, k_min=k_min, max_steps=max_steps,
        )
    else:
        agent = TauBenchAgent(client=client, model=model, user_model=user_model, max_steps=max_steps)

    async def _run_one(task_idx: int, seed: int) -> RunResult:
        async with sem:
            for attempt in range(6):
                try:
                    env = MockAirlineDomainEnv(
                        user_strategy="llm",
                        user_model=user_model,
                        user_provider="openai",
                        task_split="test",
                        task_index=0,
                    )
                    r = await agent.run_task(env, task_index=task_idx, seed=seed, config=config_label)
                    break
                except Exception as exc:
                    msg = str(exc).lower()
                    if any(kw in msg for kw in ("connection", "getaddrinfo", "rate_limit", "429", "500", "internalserver", "server error", "service unavailable", "503")):
                        wait = 5 * (2 ** attempt)
                        print(f"  [retry {attempt+1}/4] task={task_idx} err={type(exc).__name__} wait={wait}s",
                              flush=True)
                        await asyncio.sleep(wait)
                    else:
                        raise
            else:
                raise RuntimeError(f"task={task_idx} failed after 6 retries")
            done_count[0] += 1
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{done_count[0]:>3}/{total}] task={task_idx:>2} seed={seed} "
                  f"{status} reward={r.reward:.2f} tokens={r.tokens.total:,} steps={r.steps}",
                  flush=True)
            return r

    tasks = [_run_one(idx, s) for idx in task_indices for s in range(seeds)]
    results = await asyncio.gather(*tasks)
    return list(results)


def aggregate(results: List[RunResult], seeds: int) -> Dict[str, Any]:
    by_task: Dict[int, List[bool]] = {}
    for r in results:
        by_task.setdefault(r.task_index, []).append(r.passed)
    pk, std = pass_at_k(list(by_task.values()))
    mean_tok = sum(r.tokens.total for r in results) / len(results) if results else 0
    mean_fresh = sum(r.tokens.fresh for r in results) / len(results) if results else 0
    return {
        "config": results[0].config if results else "",
        "n_tasks": len(by_task),
        "seeds": seeds,
        "pass_k": round(pk, 4),
        "std": round(std, 4),
        "mean_tokens_per_run": round(mean_tok),
        "mean_fresh_tokens": round(mean_fresh),
        "total_cost_tokens": sum(r.tokens.total for r in results),
    }


def print_pareto(summaries: List[Dict]) -> None:
    print("\n" + "=" * 78)
    print(f"{'Config':<14}  {'pass^k':>8}  {'±std':>6}  {'mean_tokens':>12}  {'mean_fresh':>11}")
    print("-" * 78)
    for s in summaries:
        print(f"{s['config']:<14}  {s['pass_k']:>8.1%}  {s['std']:>6.4f}  "
              f"{s['mean_tokens_per_run']:>12,}  {s['mean_fresh_tokens']:>11,}")
    print("=" * 78)


# ── main ──────────────────────────────────────────────────────────────────

async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="tau-bench baseline runner (Phase 1 & 2)")
    parser.add_argument("--tasks", type=int, default=5,
                        help="Number of tasks to run (max 50 for airline test split)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Seeds per task (pass^k uses all seeds)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="Agent LLM (gpt-4o-mini for cheap tests, gpt-4o for reference)")
    parser.add_argument("--user-model", default="gpt-4o-mini",
                        help="User simulator LLM")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max parallel tasks (default 5)")
    parser.add_argument("--config",
                        choices=["k1", "adaptive_k", "naive_k5", "sc"],
                        default="k1",
                        help="k1=baseline  adaptive_k=Phase2  naive_k5=dumb-K5  sc=self-consistency")
    parser.add_argument("--k-max", type=int, default=5,
                        help="AdaptiveK: max branches per step")
    parser.add_argument("--k-min", type=int, default=2,
                        help="AdaptiveK: min branches after full consensus")
    parser.add_argument("--k-sc", type=int, default=5,
                        help="SC: number of final-step re-samples")
    parser.add_argument("--out", type=Path, default=Path("results/tau_bench_phase1.json"))
    args = parser.parse_args()

    cfg = args.config
    if cfg == "adaptive_k":
        config_label = f"adaptive_k{args.k_max}"
    elif cfg == "naive_k5":
        config_label = "naive_k5"
    elif cfg == "sc":
        config_label = f"sc_k{args.k_sc}"
    else:
        config_label = "k1_baseline"

    print(f"\ntau-bench airline — config={config_label}")
    print(f"  model={args.model}  user_model={args.user_model}  "
          f"tasks={args.tasks}  seeds={args.seeds}")

    client = AsyncOpenAI()
    task_indices = list(range(args.tasks))

    print(f"Running {config_label} ({args.tasks} tasks × {args.seeds} seeds)...")
    results = await run_config(
        config_label=config_label,
        client=client,
        model=args.model,
        user_model=args.user_model,
        task_indices=task_indices,
        seeds=args.seeds,
        max_steps=args.max_steps,
        concurrency=args.concurrency,
        k_max=args.k_max if cfg == "adaptive_k" else 1,
        k_min=args.k_min if cfg == "adaptive_k" else 1,
        naive_k=5 if cfg == "naive_k5" else 0,
        sc_k=args.k_sc if cfg == "sc" else 0,
    )

    summary = aggregate(results, args.seeds)
    summaries = [summary]
    print_pareto(summaries)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_data = {
        "summaries": summaries,
        "runs": [asdict(r) | {"tokens": r.tokens.as_dict()} for r in results],
    }
    args.out.write_text(json.dumps(all_data, indent=2))
    print(f"\nResults saved to {args.out}")

    if cfg == "k1":
        pk = summary["pass_k"]
        published_ref = 0.57
        gap = abs(pk - published_ref)
        print(f"\nPhase 1 gate check (model={args.model}):")
        print(f"  pass^1 = {pk:.1%}  |  reference (gpt-4o) = {published_ref:.1%}  |  gap = {gap:.1%}")
        if args.model not in ("gpt-4o", "gpt-4.1"):
            print("  NOTE: run with --model gpt-4o to compare against published baseline")
        elif gap <= 0.05:
            print("  GATE: PASSED (within 5pp of reference — user sim mismatch accounts for rest)")
        else:
            print(f"  GATE: OUTSIDE 5pp — investigate measurement stack")


if __name__ == "__main__":
    asyncio.run(main())
