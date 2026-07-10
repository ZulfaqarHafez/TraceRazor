# TraceRazor

Local-first efficiency supervision for production AI agents.

TraceRazor has one stable product surface and one independent Labs surface:

**Audit** your agent's traces to find wasted tokens, detect tool misfires and
reasoning loops, generate fix patches, and estimate cost savings.

**Labs / Sample** K parallel LLM candidates per step and select a consensus
result. This experimental surface has no general pass-rate or efficiency claim.

Both features are independent. Use one, the other, or both.

---

## Install

```bash
pip install "tracerazor[mcp]>=1.1,<2"
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
# TAS 81.4/100 [Good] | 2 steps, 900 tokens | Estimated 140 tokens (16%)

# For CI, compare a candidate against a declared baseline for the same workload.
```

The `Tracer` submits the trace to the bundled local `tracerazor` binary (CLI
mode) or to a running `tracerazor-server` (HTTP mode). Platform wheels include
the binary. Source-development builds can use:

```bash
cargo build --release
```

Or point to an existing binary:

```bash
export TRACERAZOR_BIN=/path/to/tracerazor
```

---

## Agent-native runtime

The dependency-free runtime records provenance-aware events and persists a
privacy-safe run envelope:

```python
from tracerazor.runtime import TokenUsage, configure

runtime = configure(
    policy_path="tracerazor.toml",
    host="openai-agents",
    framework="openai-agents",
    agent_id="planner",
)
runtime.record(
    "reasoning",
    content="Choose the smallest applicable tool.",
    tokens=TokenUsage.reported(input_tokens=120, output_tokens=32),
)
runtime.finalize()
```

Use `runtime.spawn_env(child_agent_id="researcher")` when launching children;
it propagates W3C `traceparent` plus the TraceRazor run, parent-agent, and policy
identifiers. Estimated or missing token counts are marked degraded and cannot
drive enforcement. See [agent-native.md](agent-native.md) for the complete
policy, event, privacy, and host-bootstrap contract.

### Framework callback handles

`auto_instrument()` imports optional SDKs only when requested and returns host
handles in `InstrumentationResult.handles`:

```python
from tracerazor.runtime import auto_instrument, configure

runtime = configure(policy_path="tracerazor.toml", framework="langgraph")
result = auto_instrument("langgraph", processor=runtime)
langgraph = result.handles["langgraph"]
answer = langgraph.invoke(graph, graph_input)
```

For an existing LangGraph invocation, use
`graph.invoke(graph_input, config=langgraph.attach(existing_config))`; the input
configuration is copied, not mutated. The returned callback captures root and
node lifecycle, LLM responses and provider usage, tools, and failures. Use
`langgraph.ainvoke(...)`, `langgraph.stream(...)`, or `langgraph.astream(...)`
rather than attaching manually when the wrapper should own finalization.

CrewAI event-bus attachment requires a second explicit action because its bus is
process-global:

```python
runtime = configure(policy_path="tracerazor.toml", framework="crewai")
crewai = auto_instrument("crewai", processor=runtime).handles["crewai"]
crewai.attach(crew)
try:
    output = crew.kickoff()
    crewai.finish(output=output)
finally:
    crewai.detach()
```

Both adapters isolate callback errors from the host and expose them through
`handle.errors`. They never estimate token counts from message length. Missing
framework usage remains `missing`; a provider-only total is preserved but marked
`estimated` because an exact input/output split is unavailable. CrewAI source filtering
is best-effort when upstream events omit crew/agent/task identifiers, so do not
run multiple unscoped listeners concurrently. These runtime handles are distinct
from the older `tracerazor.integrations.*` trace-builder callbacks.

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

MIT. Copyright (c) 2025-2026 Zulfaqar Hafez. See [LICENSE](../LICENSE).
