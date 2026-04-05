# tracerazor-crewai

CrewAI callback adapter for [TraceRazor](../../README.md).

Automatically captures every task execution and tool call from your CrewAI crew with zero manual instrumentation.

## Install

```bash
pip install tracerazor-crewai
pip install tracerazor-crewai[crewai]  # includes crewai
```

Requires the `tracerazor` binary:

```bash
cargo build --release
export TRACERAZOR_BIN=/path/to/TraceRazor/target/release/tracerazor
```

## Usage

```python
from tracerazor_crewai import TraceRazorCallback
from crewai import Agent, Task, Crew

callback = TraceRazorCallback(
    agent_name="support-crew",
    threshold=70,
)

crew = Crew(
    agents=[...],
    tasks=[...],
    callbacks=[callback],
)

crew.kickoff()

# After the crew finishes:
report = callback.analyse()
print(report.markdown())

# CI/CD gate — raises AssertionError if TAS < threshold:
callback.assert_passes()
```

## API

### `TraceRazorCallback(agent_name, framework, threshold, task_value_score, tracerazor_bin)`

| param | default | description |
|---|---|---|
| `agent_name` | `"crewai-crew"` | shown in all reports |
| `framework` | `"crewai"` | framework label |
| `threshold` | `70.0` | minimum TAS for `assert_passes()` |
| `task_value_score` | `1.0` | answer quality (0–1) |
| `tracerazor_bin` | auto | path to binary; falls back to `TRACERAZOR_BIN` env var |

### `callback.analyse() → TraceRazorReport`

Finalise and submit the trace. Returns the report.

### `callback.assert_passes()`

Raise `AssertionError` if TAS < threshold.

### `callback.set_task_value_score(score: float)`

Update quality score before calling `analyse()`.

## Captured events

| CrewAI event | TraceRazor step type |
|---|---|
| `on_task_start` / `on_task_end` | `reasoning` |
| `on_agent_action` | `reasoning` or `tool_call` |
| `on_tool_use_start` / `on_tool_use_end` | `tool_call` (success) |
| `on_tool_error` | `tool_call` (failure → TCA misfire) |
