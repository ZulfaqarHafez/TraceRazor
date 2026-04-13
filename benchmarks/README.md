# TraceRazor Benchmarks

A small, reproducible suite of synthetic traces that each isolate a specific class of token waste. The goal is not to match real-world agent distributions — it is to prove that every TraceRazor metric fires on the pattern it was built to detect, and to give users a concrete baseline they can re-run locally.

## Scenarios

| Trace | What it isolates | Expected dominant metric |
|---|---|---|
| `clean-agent.json` | Baseline — nothing wrong | High TAS, no fixes |
| `looping-agent.json` | Three identical tool calls with the same params | LDI, `termination_guard` fix |
| `verbose-agent.json` | Hedging, preamble, over-explained reasoning | VDI / SHL / CCR, AVS alert |
| `reformulator-agent.json` | Re-states the user's request verbatim before answering | `reformulation_guard` fix |
| `bloated-agent.json` | Ever-growing context window plus a tool misfire | CCE + TCA, `tool_schema` + `context_compression` fixes |

## Run

```bash
cargo build --release -p tracerazor
python benchmarks/run_benchmarks.py
```

The script runs `tracerazor audit --format json` on every `.json` under `benchmarks/traces/`, compiles a markdown table of TAS / grade / tokens / estimated savings / fix count, and writes it to `benchmarks/RESULTS.md`.

## Adding your own traces

Drop any [trace JSON](../traces/support-agent-run-2847.json) into `benchmarks/traces/`. The benchmark runner picks it up on the next invocation — no script changes needed. Traces under 2 steps are skipped by the core analyser.

## Why synthetic

Published agent datasets (AgentBench, ToolBench, SWE-Bench agents) each ship in different trace formats and licenses, and many don't record tool-level parameters. The synthetic suite here lets anyone reproduce the numbers in `RESULTS.md` with zero setup. A second benchmark pass over real external datasets is on the roadmap once the ingest layer gets adapters for each.