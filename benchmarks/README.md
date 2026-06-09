# TraceRazor Benchmarks

The benchmark runs `tracerazor audit` over **real public agent trajectories** and
records the measured TAS, grade, tokens, waste, and estimated savings for each.
Results are in [`RESULTS.md`](RESULTS.md).

## Data

The traces live in [`traces/external/`](../traces/external/) and are real runs,
not synthetic scenarios:

| Source | What it is |
|---|---|
| `tau_bench` | tau-bench airline + retail episodes for GPT-4o and Claude 3.5 Sonnet |
| `swe_agent` | SWE-agent solving the same SWE-bench task under different edit formats (cursors / default / fn_calling / xml) |

The SWE-agent edit-format variants are a useful within-task comparison: the same
task solved with very different token costs (e.g. `xml` ~3.6k vs `cursors` ~7.6k).

## Run

```bash
cargo build --release -p tracerazor
python benchmarks/run_benchmarks.py
```

The script audits every `.json` under `traces/external/` and writes the markdown
table to `RESULTS.md`. Traces under the 5-step minimum are skipped.

## Adding your own traces

Drop any [trace JSON](../traces/support-agent-run-2847.json) (raw, LangSmith, or
OTEL format) under `traces/external/<source>/` and re-run; the runner picks it up.

## Caveat

Estimated savings are per-fix projections, not measured re-runs, and token counts
for some external sources are approximated where the source did not record them.
Read the relative ordering rather than absolute totals. For a measured before/after
saving use `tracerazor bench`.
