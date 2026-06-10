# TraceRazor Benchmark Results

Measured by running `tracerazor audit` on every real public agent trace under `traces/external/` (tau-bench airline/retail; SWE-agent edit-format variants). These are real trajectories, not synthetic scenarios. Reproduce with `python benchmarks/run_benchmarks.py`.

| Source | Trace | TAS | Grade | Tokens | Waste | Est. savings | Fixes |
|---|---|---:|:-:|---:|---:|---:|---:|
| agentinstruct | `agentinstruct-os_0` | 82.2 | Good | 414 | 28% | 117 | 3 |
| agentinstruct | `agentinstruct-os_11` | 81.1 | Good | 638 | 26% | 167 | 1 |
| agentinstruct | `agentinstruct-os_5` | 74.8 | Good | 691 | 46% | 316 | 4 |
| agentinstruct | `agentinstruct-os_6` | 73.7 | Good | 305 | 46% | 139 | 3 |
| swe_agent | `marshmallow_cursors` | 68.8 | Fair | 7,553 | 79% | 5,967 | 8 |
| swe_agent | `marshmallow_default` | 74.7 | Good | 6,447 | 53% | 3,418 | 7 |
| swe_agent | `marshmallow_fn_calling` | 71.4 | Good | 5,471 | 82% | 4,505 | 6 |
| swe_agent | `marshmallow_xml` | 72.2 | Good | 3,636 | 72% | 2,623 | 6 |
| tau_bench | `claude-sonnet-3.5-new_airline_task0` | 54.1 | Fair | 6,215 | 53% | 3,280 | 4 |
| tau_bench | `claude-sonnet-3.5-new_airline_task1` | 85.8 | Good | 1,477 | 13% | 189 | 2 |
| tau_bench | `claude-sonnet-3.5-new_airline_task2` | 72.0 | Good | 3,070 | 31% | 954 | 3 |
| tau_bench | `claude-sonnet-3.5-new_airline_task3` | 57.0 | Fair | 3,613 | 30% | 1,077 | 5 |
| tau_bench | `claude-sonnet-3.5-new_airline_task4` | 83.7 | Good | 2,256 | 23% | 523 | 2 |
| tau_bench | `claude-sonnet-3.5-new_retail_task0` | 85.9 | Good | 2,621 | 0% | 12 | 2 |
| tau_bench | `claude-sonnet-3.5-new_retail_task1` | 89.5 | Good | 2,990 | 0% | 0 | 1 |
| tau_bench | `claude-sonnet-3.5-new_retail_task2` | 86.8 | Good | 3,160 | 15% | 464 | 2 |
| tau_bench | `claude-sonnet-3.5-new_retail_task3` | 61.7 | Fair | 3,030 | 19% | 570 | 2 |
| tau_bench | `claude-sonnet-3.5-new_retail_task4` | 85.6 | Good | 3,893 | 26% | 1,006 | 4 |
| tau_bench | `gpt-4o_airline_task0` | 61.1 | Fair | 2,306 | 7% | 169 | 3 |
| tau_bench | `gpt-4o_airline_task1` | 50.2 | Fair | 313 | 76% | 239 | 2 |
| tau_bench | `gpt-4o_airline_task2` | 57.8 | Fair | 1,726 | 25% | 429 | 2 |
| tau_bench | `gpt-4o_airline_task3` | 51.9 | Fair | 4,414 | 43% | 1,888 | 8 |
| tau_bench | `gpt-4o_airline_task4` | 60.3 | Fair | 1,517 | 23% | 342 | 2 |
| tau_bench | `gpt-4o_retail_task0` | 85.9 | Good | 3,249 | 18% | 570 | 2 |
| tau_bench | `gpt-4o_retail_task1` | 84.8 | Good | 3,352 | 17% | 583 | 4 |
| tau_bench | `gpt-4o_retail_task2` | 84.6 | Good | 2,999 | 28% | 854 | 2 |
| tau_bench | `gpt-4o_retail_task3` | 87.4 | Good | 2,792 | 20% | 570 | 2 |
| tau_bench | `gpt-4o_retail_task4` | 90.6 | Excellent | 1,932 | 0% | 0 | 1 |

## Summary

- Real traces benchmarked: **28**
- Average TAS: **74.1**
- Total tokens: **82,080**
- Total estimated savings: **30,971 tokens (38%)**

Estimated savings are the sum of per-fix `estimated_token_savings`; they are projections, not a measured re-run. Token counts for external sources are approximated where the source did not record them, so read the relative ordering rather than absolute totals.
