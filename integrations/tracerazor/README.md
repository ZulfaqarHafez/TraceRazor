# tracerazor

Python SDK for [TraceRazor](../../README.md) — token efficiency auditing for AI agents.

Works with any Python agent: OpenAI, Anthropic, LangGraph, CrewAI, AutoGen, or raw code.

## Install

```bash
pip install tracerazor
```

Requires the `tracerazor` binary to be built and accessible. Either:

```bash
# Option A: build from source (one-time)
cargo build --release
export TRACERAZOR_BIN=/path/to/TraceRazor/target/release/tracerazor

# Option B: use HTTP mode against a running server (no binary on client)
# docker compose up  (in the TraceRazor repo)
```

## Quickstart

```python
from tracerazor_sdk import Tracer

tracer = Tracer(agent_name="my-agent", framework="openai")

# After each LLM call, record the reasoning step:
tracer.reasoning(
    content=llm_response.text,
    tokens=llm_response.usage.total_tokens,
    input_context=prompt,  # optional, improves CCE detection
)

# After each tool call:
tracer.tool(
    name="get_order_details",
    params={"order_id": "ORD-9182"},
    output=tool_result,
    success=True,
    tokens=120,
)

# After the agent finishes:
report = tracer.analyse()
print(report.summary())
# → TAS 74.3/100 [Good] | 6 steps, 3200 tokens | Saved 1100 tokens (34%)

print(report.markdown())  # full formatted report
report.assert_passes()    # raises AssertionError if TAS < threshold (CI use)
```

## HTTP mode

If you'd rather not put the binary on every machine, run the server once and POST from anywhere:

```python
from tracerazor_sdk import Tracer

tracer = Tracer(
    agent_name="my-agent",
    server="http://localhost:8080",  # tracerazor-server URL
)
# Record steps the same way, then:
report = tracer.analyse()
```

Install with HTTP support:

```bash
pip install tracerazor[http]
```

## Context manager

```python
with Tracer(agent_name="my-agent") as t:
    t.reasoning("...", tokens=500)
    t.tool("search", params={}, output="...", success=True, tokens=100)

report = t.analyse()
```

## API

### `Tracer(agent_name, framework, threshold, task_value_score, bin_path, server)`

| param | default | description |
|---|---|---|
| `agent_name` | required | shown in reports and used for baseline tracking |
| `framework` | `"custom"` | any string: `"openai"`, `"anthropic"`, `"crewai"`, etc. |
| `threshold` | `70.0` | minimum TAS for `assert_passes()` |
| `task_value_score` | `1.0` | answer quality (0–1), update with `set_task_value()` |
| `bin_path` | auto | path to `tracerazor` binary; falls back to `TRACERAZOR_BIN` env var |
| `server` | `None` | if set, use HTTP mode |

### `tracer.reasoning(content, tokens, input_context, output)`

Record one LLM reasoning step. `input_context` is the full prompt — include it for accurate CCE bloat detection.

### `tracer.tool(name, params, output, success, error, tokens, input_context)`

Record one tool call. `success=False` triggers misfire detection (TCA) and auto-fix generation.

### `tracer.set_task_value(score: float)`

Update the task quality score after validating the agent's answer. Call before `analyse()`.

### `tracer.analyse() → TraceRazorReport`

Submit the trace and return the report.

### `TraceRazorReport`

| attribute | type | description |
|---|---|---|
| `tas_score` | `float` | 0–100 composite score |
| `grade` | `str` | `Excellent`, `Good`, `Fair`, `Poor` |
| `passes` | `bool` | `tas_score >= threshold` |
| `savings` | `dict` | `tokens_saved`, `reduction_pct`, `monthly_savings_usd` |
| `fixes` | `list` | auto-generated fix patches |
| `anomalies` | `list` | z-score alerts vs. agent baseline (after 5+ runs) |
| `metrics` | `dict` | raw per-metric scores (SRR, LDI, TCA, RDA, ISR, TUR, CCE, DBO) |
| `.summary()` | method | one-line string |
| `.markdown()` | method | full formatted report |
| `.assert_passes()` | method | raises `AssertionError` if TAS < threshold |
