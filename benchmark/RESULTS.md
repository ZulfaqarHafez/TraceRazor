# TraceRazor Benchmark Results

Measured by running `tracerazor audit` on every real public agent trace under `traces/external/` (tau-bench airline/retail; SWE-agent edit-format variants). These are real trajectories, not synthetic scenarios. Reproduce with `python -m benchmark.run_benchmarks`.

| Source | Trace | TAS | Grade | Tokens | Waste | Est. savings | Fixes |
|---|---|---:|:-:|---:|---:|---:|---:|
| agentinstruct | `agentinstruct-os_0` | 87.9 | Good | 414 | 21% | 87 | 3 |
| agentinstruct | `agentinstruct-os_11` | 89.9 | Good | 638 | 0% | 0 | 1 |
| agentinstruct | `agentinstruct-os_5` | 83.8 | Good | 691 | 46% | 316 | 4 |
| agentinstruct | `agentinstruct-os_6` | 78.9 | Good | 305 | 46% | 139 | 3 |
| swe_agent | `marshmallow_cursors` | 80.3 | Good | 7,553 | 78% | 5,900 | 8 |
| swe_agent | `marshmallow_default` | 84.7 | Good | 6,447 | 52% | 3,351 | 7 |
| swe_agent | `marshmallow_fn_calling` | 83.4 | Good | 5,471 | 81% | 4,438 | 6 |
| swe_agent | `marshmallow_xml` | 85.0 | Good | 3,636 | 70% | 2,556 | 6 |
| tau_bench | `claude-sonnet-3.5-new_airline_task0` | 58.5 | Fair | 6,215 | 27% | 1,691 | 4 |
| tau_bench | `claude-sonnet-3.5-new_airline_task1` | 91.7 | Excellent | 1,477 | 13% | 189 | 2 |
| tau_bench | `claude-sonnet-3.5-new_airline_task2` | 85.8 | Good | 3,070 | 17% | 536 | 3 |
| tau_bench | `claude-sonnet-3.5-new_airline_task3` | 62.4 | Fair | 3,613 | 19% | 686 | 5 |
| tau_bench | `claude-sonnet-3.5-new_airline_task4` | 92.4 | Excellent | 2,256 | 13% | 295 | 2 |
| tau_bench | `claude-sonnet-3.5-new_retail_task0` | 93.1 | Excellent | 2,621 | 0% | 12 | 2 |
| tau_bench | `claude-sonnet-3.5-new_retail_task1` | 96.1 | Excellent | 2,990 | 0% | 0 | 1 |
| tau_bench | `claude-sonnet-3.5-new_retail_task2` | 91.7 | Excellent | 3,160 | 15% | 464 | 2 |
| tau_bench | `claude-sonnet-3.5-new_retail_task3` | 64.8 | Fair | 3,030 | 19% | 570 | 2 |
| tau_bench | `claude-sonnet-3.5-new_retail_task4` | 89.0 | Good | 3,893 | 26% | 1,006 | 4 |
| tau_bench | `gpt-4o_airline_task0` | 65.0 | Fair | 2,306 | 7% | 169 | 3 |
| tau_bench | `gpt-4o_airline_task1` | 54.0 | Fair | 313 | 76% | 239 | 2 |
| tau_bench | `gpt-4o_airline_task2` | 65.7 | Fair | 1,726 | 14% | 241 | 2 |
| tau_bench | `gpt-4o_airline_task3` | 56.6 | Fair | 4,414 | 28% | 1,216 | 8 |
| tau_bench | `gpt-4o_airline_task4` | 66.7 | Fair | 1,517 | 13% | 192 | 2 |
| tau_bench | `gpt-4o_retail_task0` | 91.8 | Excellent | 3,249 | 18% | 570 | 2 |
| tau_bench | `gpt-4o_retail_task1` | 89.3 | Good | 3,352 | 17% | 583 | 4 |
| tau_bench | `gpt-4o_retail_task2` | 91.5 | Excellent | 2,999 | 19% | 570 | 2 |
| tau_bench | `gpt-4o_retail_task3` | 92.3 | Excellent | 2,792 | 20% | 570 | 2 |
| tau_bench | `gpt-4o_retail_task4` | 97.5 | Excellent | 1,932 | 0% | 0 | 1 |

## Summary

- Real traces benchmarked: **28**
- Average TAS: **81.1**
- Total tokens: **82,080**
- Total estimated savings: **26,586 tokens (32%)**

Estimated savings are the sum of per-fix `estimated_token_savings`; they are projections, not a measured re-run. Token counts for external sources are approximated where the source did not record them, so read the relative ordering rather than absolute totals.
