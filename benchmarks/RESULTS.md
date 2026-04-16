# TraceRazor Benchmark Results (v0.2.0)

Measured by running `tracerazor audit` on every trace under `benchmarks/traces/`. Each trace is a synthetic scenario that isolates a specific class of token waste. Reproduce with `python benchmarks/run_benchmarks.py`.

**Updated:** 2026-04-16  
**New in v0.2.0:** Semantic Continuity (CSD) metric detects reasoning drift; Adherence Scoring (IAR) validates optimization effectiveness.

| Trace | Agent | TAS | Grade | Tokens | Waste | Est. savings | Fixes |
|---|---|---:|:-:|---:|---:|---:|---:|
| `bloated-agent` | bloated-agent | 87.9 | Good | 2,320 | 30% | 693 | 2 |
| `clean-agent` | clean-agent | 89.4 | Good | 860 | 34% | 289 | 1 |
| `looping-agent` | looping-agent | 69.4 | Fair | 1,710 | 35% | 603 | 3 |
| `reformulator-agent` | reformulator-agent | 86.9 | Good | 1,340 | 32% | 433 | 1 |
| `verbose-agent` | verbose-agent | 77.1 | Good | 2,760 | 53% | 1,455 | 2 |

## Summary

- Traces benchmarked: **5**
- Average TAS: **82.1**
- Total tokens: **8,990**
- Total estimated savings: **3,473 tokens (39%)**

Estimated savings are the sum of per-fix `estimated_token_savings` from the report. To validate a specific patch set against a real re-run, use `tracerazor bench --before <old>.json --after <new>.json --fixes <fixes>.json`.

## New in v0.2.0

**Semantic Continuity (CSD)** — Measures mean cosine similarity between consecutive reasoning steps. Detects agents whose reasoning drifts topic mid-trace ("wandering agents"). Target: ≥0.60.

**Adherence Scoring (IAR)** — After applying fixes from `tracerazor optimize`, re-audit to measure % of fix types that improved their target metrics. Validates that optimizer recommendations work in practice. Target: ≥0.75.

These metrics are included in all v0.2.0+ audit reports automatically.
