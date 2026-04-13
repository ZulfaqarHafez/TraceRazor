# TraceRazor Benchmark Results

Measured by running `tracerazor audit` on every trace under `benchmarks/traces/`. Each trace is a synthetic scenario that isolates a specific class of token waste. Reproduce with `python benchmarks/run_benchmarks.py`.

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
