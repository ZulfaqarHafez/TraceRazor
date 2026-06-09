# TraceRazor Benchmark Results
# TraceRazor Benchmark Results (crate v0.1.0)

Measured by running `tracerazor audit` on every trace under `benchmarks/traces/`. Each trace is a **synthetic scenario authored to isolate a specific class of token waste** — this table is a *smoke test* proving each metric fires on the pattern it was built to detect, **not** a generalisation benchmark on real-world traces. The "Est. savings" column is the sum of per-fix heuristic estimates (see `estimated_token_savings`), not a measured reduction from re-running an agent; validate real savings with `tracerazor bench` on a captured before/after trace pair. Reproduce with `python benchmarks/run_benchmarks.py`.

**Updated:** 2026-04-16  
**Metric set:** Semantic Continuity (CSD) detects reasoning drift; Adherence Scoring (IAR) validates optimization effectiveness; Trajectory Path Entropy (TPE) reports on-path directedness.

| Trace | Agent | TAS | Grade | Tokens | Waste | Est. savings | Fixes |
|---|---|---:|:-:|---:|---:|---:|---:|
| `bloated-agent` | bloated-agent | 79.3 | Good | 2,320 | 30% | 693 | 2 |
| `clean-agent` | clean-agent | 84.3 | Good | 860 | 34% | 289 | 1 |
| `looping-agent` | looping-agent | 55.4 | Fair | 1,710 | 35% | 603 | 3 |
| `reformulator-agent` | reformulator-agent | 84.4 | Good | 1,340 | 32% | 433 | 1 |
| `verbose-agent` | verbose-agent | 68.1 | Fair | 2,760 | 53% | 1,455 | 2 |

## Summary

- Traces benchmarked: **5**
- Average TAS: **74.3**
- Total tokens: **8,990**
- Total estimated savings: **3,473 tokens (39%)**

Estimated savings are the sum of per-fix `estimated_token_savings` from the report. To validate a specific patch set against a real re-run, use `tracerazor bench --before <old>.json --after <new>.json --fixes <fixes>.json`.
