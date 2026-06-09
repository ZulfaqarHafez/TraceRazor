# TraceRazor Benchmark Results

Measured by running `tracerazor audit` on every real public agent trace under `traces/external/` (tau-bench airline/retail; SWE-agent edit-format variants). These are real trajectories, not synthetic scenarios. Reproduce with `python benchmarks/run_benchmarks.py`.

| Source | Trace | TAS | Grade | Tokens | Waste | Est. savings | Fixes |
|---|---|---:|:-:|---:|---:|---:|---:|
| swe_agent | `marshmallow_cursors` | 68.8 | Fair | 7,553 | 79% | 5,972 | 9 |
| swe_agent | `marshmallow_default` | 72.8 | Good | 6,447 | 56% | 3,623 | 9 |
| swe_agent | `marshmallow_fn_calling` | 69.6 | Fair | 5,471 | 82% | 4,510 | 7 |
| swe_agent | `marshmallow_xml` | 70.9 | Good | 3,636 | 76% | 2,770 | 7 |
| tau_bench | `claude-sonnet-3.5-new_airline_task0` | 47.3 | Poor | 6,215 | 79% | 4,892 | 6 |
| tau_bench | `claude-sonnet-3.5-new_airline_task1` | 83.8 | Good | 1,477 | 25% | 373 | 2 |
| tau_bench | `claude-sonnet-3.5-new_airline_task2` | 68.6 | Fair | 3,070 | 71% | 2,181 | 3 |
| tau_bench | `claude-sonnet-3.5-new_airline_task3` | 54.5 | Fair | 3,613 | 46% | 1,647 | 4 |
| tau_bench | `claude-sonnet-3.5-new_airline_task4` | 78.5 | Good | 2,256 | 50% | 1,134 | 2 |
| tau_bench | `claude-sonnet-3.5-new_retail_task0` | 85.0 | Good | 2,621 | 17% | 439 | 2 |
| tau_bench | `claude-sonnet-3.5-new_retail_task1` | 88.4 | Good | 2,990 | 14% | 413 | 1 |
| tau_bench | `claude-sonnet-3.5-new_retail_task2` | 84.4 | Good | 3,160 | 32% | 1,009 | 2 |
| tau_bench | `claude-sonnet-3.5-new_retail_task3` | 60.0 | Fair | 3,030 | 32% | 969 | 2 |
| tau_bench | `claude-sonnet-3.5-new_retail_task4` | 82.8 | Good | 3,893 | 38% | 1,488 | 3 |
| tau_bench | `gpt-4o_airline_task0` | 55.5 | Fair | 2,306 | 66% | 1,520 | 3 |
| tau_bench | `gpt-4o_airline_task1` | 48.3 | Poor | 313 | 76% | 239 | 2 |
| tau_bench | `gpt-4o_airline_task2` | 55.4 | Fair | 1,726 | 55% | 957 | 2 |
| tau_bench | `gpt-4o_airline_task3` | 47.1 | Poor | 4,414 | 55% | 2,420 | 8 |
| tau_bench | `gpt-4o_airline_task4` | 56.6 | Fair | 1,517 | 44% | 670 | 2 |
| tau_bench | `gpt-4o_retail_task0` | 84.3 | Good | 3,249 | 34% | 1,109 | 2 |
| tau_bench | `gpt-4o_retail_task1` | 82.8 | Good | 3,352 | 30% | 994 | 4 |
| tau_bench | `gpt-4o_retail_task2` | 83.1 | Good | 2,999 | 41% | 1,244 | 2 |
| tau_bench | `gpt-4o_retail_task3` | 87.9 | Good | 2,792 | 29% | 810 | 2 |
| tau_bench | `gpt-4o_retail_task4` | 87.5 | Good | 1,932 | 10% | 196 | 1 |

## Summary

- Real traces benchmarked: **24**
- Average TAS: **71.0**
- Total tokens: **80,032**
- Total estimated savings: **41,579 tokens (52%)**

Estimated savings are the sum of per-fix `estimated_token_savings`; they are projections, not a measured re-run. Token counts for external sources are approximated where the source did not record them, so read the relative ordering rather than absolute totals.
