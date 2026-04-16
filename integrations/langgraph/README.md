# tracerazor-langgraph

LangGraph/LangChain callback adapter for [TraceRazor](../../README.md).

Automatically captures every LLM call and tool call from your LangGraph graph with zero manual instrumentation.

**v0.2.0 — New Metrics:**
- ✨ **Semantic Continuity (CSD)** — Detects when your agent's reasoning drifts topic mid-execution
- ✨ **Adherence Scoring (IAR)** — After optimizing, validates that fixes actually improved metrics

## Install

```bash
pip install tracerazor-langgraph
pip install tracerazor-langgraph[langgraph]  # includes langgraph
```

Requires the `tracerazor` binary:

```bash
cargo build --release
export TRACERAZOR_BIN=/path/to/TraceRazor/target/release/tracerazor
```

## Usage

```python
from tracerazor_langgraph import TraceRazorCallback
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

callback = TraceRazorCallback(
    agent_name="support-agent",
    threshold=70,
)

model = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(model, tools=[...])

result = agent.invoke(
    {"messages": [HumanMessage(content="I want a refund")]},
    config={"callbacks": [callback]},
)

# After the agent finishes:
report = callback.analyse()
print(report.markdown())

# CI/CD gate — raises AssertionError if TAS < threshold:
callback.assert_passes()
```

## API

### `TraceRazorCallback(agent_name, framework, threshold, task_value_score, tracerazor_bin)`

| param | default | description |
|---|---|---|
| `agent_name` | `"langgraph-agent"` | shown in all reports |
| `framework` | `"langgraph"` | framework label |
| `threshold` | `70.0` | minimum TAS for `assert_passes()` |
| `task_value_score` | `1.0` | answer quality (0–1) |
| `tracerazor_bin` | auto | path to binary; falls back to `TRACERAZOR_BIN` env var |

### `callback.analyse() → TraceRazorReport`

Finalise and submit the trace. Returns the report.

### `callback.assert_passes()`

Raise `AssertionError` if TAS < threshold.

### `callback.set_task_value_score(score: float)`

Update quality score before calling `analyse()`.

## Multi-Agent Workflows

For workflows with multiple graphs or nodes, use a separate callback for each agent:

```python
from tracerazor_langgraph import TraceRazorCallback

# Agent 1: Triage
triage_callback = TraceRazorCallback(agent_name="triage-agent")
triage_result = triage_graph.invoke(input, config={"callbacks": [triage_callback]})
triage_report = triage_callback.analyse()

# Agent 2: Resolution
resolution_callback = TraceRazorCallback(agent_name="resolution-agent")
resolution_result = resolution_graph.invoke(triage_result, config={"callbacks": [resolution_callback]})
resolution_report = resolution_callback.analyse()

# Aggregate metrics
total_tokens = triage_report.total_tokens + resolution_report.total_tokens
avg_efficiency = (triage_report.tas_score + resolution_report.tas_score) / 2
```

Each agent is audited independently, enabling per-agent optimization. See [Multi-Agent Guide](../tracerazor/examples/MULTI_AGENT_GUIDE.md) for complete example with cost analysis.
