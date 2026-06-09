# TraceRazor Benchmark Results

Measured by running `tracerazor audit` on every trace under `benchmarks/traces/`. Each trace is a synthetic scenario that isolates a specific class of token waste. Reproduce with `python benchmarks/run_benchmarks.py`.

| Trace | Agent | TAS | Grade | Tokens | Waste | Est. savings | Fixes |
|---|---|---:|:-:|---:|---:|---:|---:|
| `bloated-agent` | bloated-agent | 77.5 | Good | 2,320 | 30% | 693 | 3 |
| `clean-agent` | clean-agent | 81.9 | Good | 860 | 34% | 289 | 2 |
| `looping-agent` | looping-agent | 54.4 | Fair | 1,710 | 35% | 603 | 4 |
| `reformulator-agent` | reformulator-agent | 80.8 | Good | 1,340 | 32% | 433 | 2 |
| `verbose-agent` | verbose-agent | 64.8 | Fair | 2,760 | 53% | 1,455 | 3 |

## Summary

- Traces benchmarked: **5**
- Average TAS: **71.9**
- Total tokens: **8,990**
- Total estimated savings: **3,473 tokens (39%)**

Estimated savings are the sum of per-fix `estimated_token_savings` from the report. To validate a specific patch set against a real re-run, use `tracerazor bench --before <old>.json --after <new>.json --fixes <fixes>.json`.
