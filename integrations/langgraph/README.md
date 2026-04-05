# tracerazor-langgraph

LangGraph/LangChain callback adapter for [TraceRazor](../../README.md).

Automatically captures every LLM call and tool call from your LangGraph graph with zero manual instrumentation.

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
