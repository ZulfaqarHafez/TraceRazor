# tracerazor-openai-agents

OpenAI Agents SDK hooks adapter for [TraceRazor](../../README.md).

Automatically captures every LLM call, tool execution, and agent handoff
from a `Runner.run()` invocation — zero manual instrumentation required.

**v0.2.0 — New Metrics:**
- ✨ **Semantic Continuity (CSD)** — Detects when your agent's reasoning drifts topic mid-execution
- ✨ **Adherence Scoring (IAR)** — After optimizing, validates that fixes actually improved metrics

## Install

```bash
pip install tracerazor-openai-agents
pip install tracerazor-openai-agents[agents]  # includes openai-agents
```

Requires the `tracerazor` binary:

```bash
cargo build --release
export TRACERAZOR_BIN=/path/to/TraceRazor/target/release/tracerazor
```

## Usage

```python
from tracerazor_openai_agents import TraceRazorHooks
from agents import Agent, Runner

hooks = TraceRazorHooks(
    agent_name="support-agent",
    threshold=70,
)

result = await Runner.run(
    agent,
    "I need a refund for order ORD-9182",
    hooks=hooks,
)

# After the run:
report = hooks.analyse()
print(report.markdown())

# CI/CD gate — raises AssertionError if TAS < threshold:
hooks.assert_passes()
```

## API

### `TraceRazorHooks(agent_name, framework, threshold, task_value_score, tracerazor_bin)`

| param | default | description |
|---|---|---|
| `agent_name` | `"openai-agent"` | shown in all reports |
| `framework` | `"openai-agents"` | framework label |
| `threshold` | `70.0` | minimum TAS for `assert_passes()` |
| `task_value_score` | `1.0` | answer quality (0–1) |
| `tracerazor_bin` | auto | path to binary; falls back to `TRACERAZOR_BIN` env var |

### `hooks.analyse() → TraceRazorReport`

Finalise and submit the trace. Returns the report.

### `hooks.assert_passes()`

Raise `AssertionError` if TAS < threshold.

### `hooks.set_task_value_score(score: float)`

Update quality score before calling `analyse()`.

## Captured events

| SDK hook | TraceRazor step type |
|---|---|
| `on_agent_end` | `reasoning` (agent output) |
| `on_tool_start` / `on_tool_end` | `tool_call` (success) |
| `on_handoff` | `reasoning` (handoff marker) |

## Multi-agent traces

When your workflow uses multiple agents via handoffs, TraceRazor automatically
produces a per-agent breakdown alongside the composite score:

```
MULTI-AGENT BREAKDOWN
Agent                    Steps   Tokens   Share      TAS  Grade
TriageAgent                  4    1,200   28.6%     82.5  GOOD
SupportAgent                 7    2,600   61.9%     61.2  FAIR
EscalationAgent              2      400    9.5%      N/A  N/A
```

The composite TAS is weighted by each agent's token consumption.

See [Multi-Agent Guide](../tracerazor/examples/MULTI_AGENT_GUIDE.md) for detailed workflow examples with cost analysis and optimization validation.
