# TraceRazor

Token efficiency auditing and adaptive sampling for production AI agents.

TraceRazor does two things:

**Audit** your agent's traces to find wasted tokens, detect tool misfires and
reasoning loops, generate fix patches, and estimate cost savings.

**Sample** more reliably by running K parallel LLM candidates per step and
picking the consensus winner. Improves task pass rates without changing your
agent's logic.

Both features are independent. Use one, the other, or both.

---

## Install

```bash
pip install tracerazor
```

Install with optional dependencies as needed:

```bash
pip install "tracerazor[openai]"        # OpenAI adapter
pip install "tracerazor[anthropic]"     # Anthropic adapter
pip install "tracerazor[langgraph]"     # LangGraph integration
pip install "tracerazor[http]"          # HTTP mode for remote server
pip install "tracerazor[all]"           # Everything
```

---

## Audit quickstart

Record steps manually with `Tracer`, then call `analyse()` to get a report:

```python
from tracerazor import Tracer

with Tracer(agent_name="support-agent", framework="openai") as t:
    response = llm.invoke(prompt)
    t.reasoning(response.text, tokens=response.usage.total_tokens)

    result = lookup_order(order_id="ORD-123")
    t.tool("lookup_order", params={"order_id": "ORD-123"},
           output=str(result), success=True, tokens=80)

report = t.analyse()
print(report.summary())
# TAS 81.4/100 [Good] | 2 steps, 900 tokens | Saved 140 tokens (16%)

report.assert_passes()  # raises AssertionError in CI if TAS < 70
```

The `Tracer` submits the trace to the local `tracerazor` binary (CLI mode) or
to a running `tracerazor-server` (HTTP mode). Build the binary with:

```bash
cargo build --release
```

Or point to an existing binary:

```bash
export TRACERAZOR_BIN=/path/to/tracerazor
```

---

## Sampling quickstart

`AdaptiveKNode` is a drop-in replacement for a LangGraph ReAct node. It samples
K parallel LLM candidates at each step and picks the consensus winner.

```python
from tracerazor import AdaptiveKNode, openai_llm
from openai import AsyncOpenAI
from langgraph.graph import StateGraph

llm = openai_llm(AsyncOpenAI(), model="gpt-4.1")
node = AdaptiveKNode(llm=llm, tools=my_tools, k_max=5, k_min=2)

graph = StateGraph(AgentState)
graph.add_node("agent", node)
# ... add edges and compile as usual ...

result = await graph.ainvoke({"messages": [HumanMessage(content="...")]})
print(result["consensus_report"].summary())
```

K adapts automatically: it shrinks toward `k_min` when all candidates agree
(saving tokens), and resets to `k_max` after a divergent vote or a
state-mutating tool call (e.g. booking a flight, cancelling an order).

---

## Baselines

Use `NaiveKEnsemble` and `SelfConsistencyBaseline` to benchmark your setup:

```python
from tracerazor import NaiveKEnsemble, SelfConsistencyBaseline
```

`NaiveKEnsemble` runs K independent full-task agents and picks the majority
result. `SelfConsistencyBaseline` uses a single deterministic tool-calling
pass, then re-samples the final response K times.

In tau-bench airline benchmarks (50 tasks, gpt-4o):

| Strategy | pass^1 | mean tokens | vs baseline |
|---|---|---|---|
| K=1 baseline | 38% | 63k | 1.0x |
| NaiveKEnsemble (K=5) | 40% | 282k | 4.5x |
| AdaptiveKNode (K=5) | 46% | 246k | 3.9x |
| SelfConsistency (K=5) | 48% | 137k | 2.2x |

---

## Audit API

| Name | Description |
|---|---|
| `Tracer` | Context manager for recording steps and submitting for analysis |
| `TraceRazorClient` | Lower-level client for submitting trace dicts directly |
| `TraceRazorReport` | Parsed audit result with TAS score, metrics, fixes, and savings |
| `TraceStep` | Data class for a single recorded step |

## Sampling API

| Name | Description |
|---|---|
| `AdaptiveKNode` | LangGraph node with per-step adaptive parallel sampling |
| `ExactMatchConsensus` | Aggregates K branch proposals by exact-match comparison |
| `MutationMetadata` | Classifies tools as mutating vs read-only |
| `NaiveKEnsemble` | K independent full-task agents, majority vote |
| `SelfConsistencyBaseline` | K re-samples of the final response only |

## LLM adapters

| Name | Description |
|---|---|
| `openai_llm` | Adapter factory for `AsyncOpenAI` |
| `anthropic_llm` | Adapter factory for `AsyncAnthropic` |
| `mock_llm` | Deterministic mock for tests and offline demos |

---

## License

MIT. Copyright (c) 2024 Zulfaqar Hafez.
