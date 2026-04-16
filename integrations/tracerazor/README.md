# tracerazor

Python SDK for [TraceRazor](../../README.md) — comprehensive token efficiency auditing & optimization for AI agents.

Works with any Python agent: OpenAI, Anthropic, LangGraph, CrewAI, AutoGen, or raw code.

**v0.2.0** — New metrics:
- **Semantic Continuity (CSD)** — detects reasoning drift across steps
- **Adherence Scoring (IAR)** — validates that optimization fixes improve metrics
- **Multi-agent reporting** — audit 2+ agents in a workflow, aggregate efficiency

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

## Multi-Agent Workflows

TraceRazor handles multi-agent systems natively. Each agent in your workflow gets its own tracer and independent report:

```python
from tracerazor_sdk import Tracer

# Agent 1: Triage
with Tracer(agent_name="triage") as t:
    category = classify_request(user_query)
    t.reasoning(f"Classified as: {category}", tokens=120)
    triage_result = t.analyse()

# Agent 2: Resolution
with Tracer(agent_name="resolution") as t:
    answer = resolve_by_category(category)
    t.tool("search_kb", params={"q": category}, output=answer, success=True, tokens=180)
    resolution_result = t.analyse()

# Agent 3: Escalation (if needed)
if not resolution_result.passes:
    with Tracer(agent_name="escalation") as t:
        ticket = escalate_to_human()
        t.tool("create_ticket", params={}, output=ticket, success=True, tokens=100)
        escalation_result = t.analyse()

# Aggregate results
total_tokens = sum([
    triage_result.total_tokens,
    resolution_result.total_tokens,
    escalation_result.total_tokens,
])
avg_efficiency = (
    triage_result.tas_score +
    resolution_result.tas_score +
    escalation_result.tas_score
) / 3

print(f"Workflow: {total_tokens} tokens | Efficiency: {avg_efficiency:.1f}/100")
```

See [`examples/multi_agent_workflow.py`](examples/multi_agent_workflow.py) for a complete working example with 4 agents, tool calling, and cost analysis.

## What's Audited

Each trace is analyzed across **13 independent signals**:

**Structural Efficiency:**
- Step Redundancy (SRR) — near-duplicate steps
- Loop Detection (LDI) — repeated tool calls
- Tool Accuracy (TCA) — failed tool calls
- Reasoning Depth (RDA) — over-complex reasoning
- Information Sufficiency (ISR) — wasted steps
- Token Utilisation (TUR) — off-task content
- Context Efficiency (CCE) — duplicate context
- Decision Optimality (DBO) — suboptimal tool sequences
- **Semantic Continuity (CSD)** — reasoning drift [NEW in v0.2]

**Verbosity & Presentation:**
- Verbosity Density (VDI) — filler words, low-substance content
- Sycophancy/Hedging (SHL) — over-polite phrasing
- Compression Ratio (CCR) — highly compressible output

**Optimization Validation:**
- **Adherence Score (IAR)** — % of fixes that improved metrics on re-audit [NEW in v0.2]

## Examples

- **Multi-agent customer support system** — [`examples/multi_agent_workflow.py`](examples/multi_agent_workflow.py)
- **LangGraph integration** — [`../langgraph/examples/`](../langgraph/examples/)
- **CrewAI integration** — [`../crewai/examples/`](../crewai/examples/)

## Troubleshooting

**Q: How do I optimize an agent?**  
Use the Rust CLI:
```bash
tracerazor optimize trace.json --output optimized_prompt.txt --target-tas 85
```

**Q: Can I compare two runs?**  
Yes:
```bash
tracerazor bench --before trace_v1.json --after trace_v2.json
```

**Q: What if I don't have the `tracerazor` binary?**  
Use HTTP mode with a running server:
```bash
pip install tracerazor[http]
docker compose up  # in the TraceRazor repo
```

Then pass `server="http://localhost:8080"` to `Tracer()`.

## License

Apache 2.0. See [LICENSE](../../LICENSE).
