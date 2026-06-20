# TRICE S-Tier Product Report

Date: 2026-06-21

## What Was Implemented

TRICE is now present as an offline prototype and CLI surface.

- Python prototype package: `benchmark/trice/`
- CLI policy generation: `tracerazor optimize --trace TRACE --budget-ratio 0.40 --out context_policy.json`
- CLI replay: `tracerazor replay --trace TRACE --policy context_policy.json --format json`
- Bench JSON additions: `input_tokens_before`, `input_tokens_after`, `cache_adjusted_cost`, `quality_delta`, `pass_noninferior`, and `evidence_recall`
- Research base: `docs/trice_research_ledger.md`

This is not yet an S-tier measured result. It is the machinery needed to run
the S-tier loop honestly.

## TRICE Method

TRICE optimizes context as a portfolio:

```text
maximize sum_i [ U_i(a_i) - lambda R_i(a_i) - mu C_i(a_i) + rho K_i(a_i) - gamma H_i(a_i) ]

subject to:
  total policy tokens <= budget
  evidence recall >= 0.95
  locked anchors unchanged
  pass-rate noninferiority >= -2pp
```

Where:

- `U`: next-action utility from task relevance, identifiers, tool/error relevance, and goal anchoring.
- `R`: deletion risk from active errors, mutating calls, user/task anchors, and unknown state.
- `C`: input-token and approximate prefill/KV pressure.
- `K`: cache continuity reward for stable reusable context.
- `H`: hallucination or reconstruction risk after summarization/masking.

Every segment receives one state:

```text
essential | rehydratable | expired | redundant | distractor | unknown
```

Every segment receives one action:

```text
keep | extract | summarize | mask_with_receipt | anchor_prefix | lazy_recall
```

The policy solver uses 128-token buckets and a multi-choice knapsack: exactly
one action per segment. Essential segments are locked and kept byte-for-byte,
usually as `anchor_prefix` for early task anchors or `keep` for later mutating
state.

## Current Engineering Shape

The Python prototype is the research workbench:

- `segment.py`: labels trace steps and creates receipts.
- `score.py`: computes utility, risk, cost, cache, and hallucination features.
- `policy.py`: solves the budgeted action assignment.
- `render.py`: emits policy JSON and compressed context text.
- `replay.py`: evaluates evidence recall, action divergence, stale-info retention, rehydration success, and overhead.
- `learn.py`: performs the online policy-weight update.
- `report.py`: renders a Markdown report with token flow and context portfolio tables.

The Rust CLI is the user-facing v0.1:

- It generates the same policy concept directly from TraceRazor traces.
- It replays policies without requiring Python.
- It extends `bench` so measured savings can be quality-aware.

## How To Run The Loop

First create a policy:

```sh
tracerazor optimize --trace traces/support-agent-run-2847.json --budget-ratio 0.40 --out context_policy.json --format json
```

Then replay it:

```sh
tracerazor replay --trace traces/support-agent-run-2847.json --policy context_policy.json --format json
```

Then use the managed-agent loop:

```text
audit -> optimize/apply context policy -> rerun agent -> bench -> accept/rollback -> update weights -> report
```

The rule is strict: if `pass_noninferior` is false or the rerun loses task
success, the token reduction is not counted as a saving.

## S-Tier Benchmark Plan

TR-Replay:

- Use recorded traces only.
- Measure next-action preservation, evidence recall, stale-context removal, and receipt/rehydration validity.
- Iterate quickly across many policies and budget ratios.

TR-Rollout:

- Use real repositories with clean checkouts and fixed tool envelopes.
- Pilot on 20 SWE-bench Verified/Lite tasks with 3 replicates.
- Claim run on 50 held-out tasks clustered by repository.
- Report pass-rate delta, `pass^k`, tokens-to-solve, cost-to-solve, and bootstrap confidence intervals.

TR-Stress:

- Inject duplicated logs, stale failed hypotheses, irrelevant files, giant test output, renamed APIs, and early hidden evidence.
- Confirm TRICE removes distractors without losing the evidence needed to solve.

Baselines:

- No optimization
- Last-N truncation
- Summarization-only
- Retrieval-only
- Cache-only accounting
- AgentDiet-style trajectory pruning
- Existing TraceRazor safe patches
- TRICE

## Acceptance Gate

TRICE can claim S-tier only when all of these hold on held-out real reruns:

- Mean input-token reduction is at least 60%.
- Pass-rate lower confidence bound is no worse than -2 percentage points.
- `pass^k` does not regress.
- Evidence recall is at least 95% on solved traces.
- Cache/prefill adjusted cost improves after compressor overhead.
- All negative/null results are reported next to wins.

## Current Risks

- Static trace replay is only a proxy for next-action fidelity.
- The first CLI implementation uses deterministic heuristics rather than a learned policy.
- Closed-source CLI agents can only receive prompt/context deltas; full runtime compression needs managed adapters.
- The 60% target may require combining TRICE with cache-aware prefix stabilization and tool-output ingestion filters.

## Next Iterations

1. Add a manifest-driven real-repo runner that can apply TRICE context policies to managed agents.
2. Store policy/replay results as a savings ledger: projected, replay-verified, and rerun-measured.
3. Add budget sweeps and Pareto AUC reports.
4. Feed accepted/rejected rerun outcomes into `benchmark.trice.learn`.
5. Promote high-confidence policy weights into the Rust CLI once held-out evidence stabilizes.

The product becomes S-tier when it behaves like a disciplined teammate: it
shrinks context aggressively where recovery is safe, refuses to hide uncertainty,
and never celebrates token savings that make the task worse.
