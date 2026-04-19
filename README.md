# TraceRazor

**The token efficiency audit layer for production AI agents.**

[![CI](https://github.com/ZulfaqarHafez/tracerazor/actions/workflows/tracerazor.yml/badge.svg)](https://github.com/ZulfaqarHafez/tracerazor/actions)
&nbsp;·&nbsp; Apache 2.0 &nbsp;·&nbsp; Rust &nbsp;·&nbsp; Author: Zulfaqar Hafez

---

## Abstract

Recent research (ACL 2025, NeurIPS 2024, KDD 2025) shows **40-70% of agent tokens are structurally redundant**: wasted on repeated steps, sycophantic preamble, reformulated context, and unnecessary reasoning loops [[1][3][10]](#research-foundation).

Current observability tools (LangSmith, Langfuse, Arize) record agent activity but don't measure efficiency or suggest fixes. The gap is not instrumentation but analysis.

**TraceRazor** is an offline audit engine that scores completed traces across thirteen efficiency metrics, producing a 0-100 Token Audit Score (TAS), optimal path diffs, and fix patches. No agent modification, API keys, or inference latency required. At 50K runs/month, a 30% efficiency gain saves six figures annually.

---

## The Problem

A production support agent with 8 tool calls across 3 loops typically consumes **15,000-40,000 tokens per resolution**:

| Pattern | Observed Frequency | Token Cost |
|---|---|---|
| Redundant reasoning steps | 18-35% of traces | ~20% of tokens |
| Sycophantic/hedging preamble | >60% of outputs | 5-15% per step |
| Input context reformulation | 1-3 steps per trace | 300-800 tokens each |
| Unnecessary reasoning depth | ~25% of traces | 10-30% of tokens |
| Repeated tool-call loops | ~15% of traces | Full loop cost |

*Sources: Han et al. [[1]](#research-foundation), Shi et al. [[11]](#research-foundation), Mohammadi et al. [[10]](#research-foundation)*

Existing tools show that runs happened, not which steps wasted tokens or what efficiency looks like.

---

## What TraceRazor Measures

TraceRazor decomposes efficiency into thirteen signals targeting specific waste categories. All analysis runs offline in under 5ms, no model weights or API keys needed.

### Structural Efficiency

| Metric | Weight | What It Detects |
|--------|--------|--------|
| **Step Redundancy Rate** (SRR) | 17% | Near-duplicate steps wasting tokens |
| **Loop Detection Index** (LDI) | 13% | Repeated tool calls re-attempting actions |
| **Tool Call Accuracy** (TCA) | 13% | Failed tool calls and retries |
| **Reasoning Depth** (RDA) | 10% | Over-deep reasoning for simple tasks |
| **Information Sufficiency** (ISR) | 10% | Steps lacking novel information |
| **Token Utilisation** (TUR) | 10% | Off-task token spending |
| **Context Efficiency** (CCE) | 10% | Duplicate context across steps |
| **Decision Optimality** (DBO) | 9% | Sub-optimal tool sequences |
| **Semantic Continuity** (CSD) | 5% | Reasoning drift mid-trace |

### Verbosity & Presentation

LLM outputs often include sycophantic openers, hedging, and compressible filler. This waste compounds token costs without improving reasoning.

| Metric | Weight | What It Detects |
|--------|--------|--------|
| **Verbosity Density** (VDI) | 9% | Filler words and low-substance content |
| **Sycophancy/Hedging** (SHL) | 5% | Excessive politeness and caution |
| **Compression Ratio** (CCR) | 4% | Highly compressible text |

**Verbosity Alert:** When combined signals exceed 40%, the report flags the primary driver.

### Content Reformulation Detection

Steps that restate the user's request add no information. TraceRazor detects this by comparing opening sentences to input context via bigram overlap. Overlap ≥70% triggers a fix.

### Optimization Validation

After applying fixes, re-audit to measure actual improvement. The **Adherence Score** (target ≥75%) validates that fixes improved metrics, not just projections.

Workflow:
- Run `tracerazor audit trace.json` to identify waste and get fixes
- Apply fixes to agent configuration or system prompt
- Re-run your test case with optimized agent
- Run `tracerazor bench --before trace.json --after trace_v2.json` to validate
- Adherence score shows % of fix types that improved

---

## Sample Output

```bash
tracerazor audit traces/support-agent-run-2847.json
```

```
TRACERAZOR REPORT
------------------------------------------------------
Trace:     support-agent-run-2847
Agent:     support-agent
Framework: langgraph
Steps:     9   Tokens: 18420
------------------------------------------------------
TRACERAZOR SCORE:  64 / 100  [FAIR]
------------------------------------------------------
!! VERBOSITY ALERT  AVS: 0.52  Primary driver: SHL (sycophancy/hedging)
   Est. verbose tokens: 9578
------------------------------------------------------
METRIC BREAKDOWN
Code   Metric                         Score    Target   Status
SRR    Step Redundancy Rate           18.2%    <15%     FAIL
LDI    Loop Detection Index           0.182    <0.10    FAIL
TCA    Tool Call Accuracy             83.3%    >85%     FAIL
RDA    Reasoning Depth Approp.        0.820    >0.75    PASS [hist]
ISR    Info Sufficiency Rate          88.0%    >80%     PASS
TUR    Token Utilisation Ratio        0.714    >0.35    PASS
CCE    Context Carry-over Eff.        0.880    >0.60    PASS
DBO    Decision Branch Optimality     0.700    >0.70    PASS [cold]
-- Verbosity Metrics ----------------------------------
VDI    Verbosity Density Index        0.512    >0.60    FAIL
SHL    Sycophancy/Hedging Level       0.380    <0.20    FAIL
CCR    Caveman Compression Ratio      0.412    <0.30    FAIL
------------------------------------------------------
SAVINGS ESTIMATE
Tokens saved:      9,840  (53.4% reduction)
Cost saved:        $0.0295 per run
At 50K runs/month: $1,477.20/month saved
```

---

## Getting Started

### Docker

```bash
git clone https://github.com/ZulfaqarHafez/tracerazor
cd tracerazor
docker compose up --build
# http://localhost:8080
```

### Build from source

```bash
cargo build --release
./target/release/tracerazor audit traces/support-agent-run-2847.json
```

### Python

```bash
pip install tracerazor                      # audit + adaptive sampling
pip install "tracerazor[openai]"            # OpenAI adapter
pip install "tracerazor[anthropic]"         # Anthropic adapter
pip install "tracerazor[langgraph]"         # LangGraph integration
pip install "tracerazor[all]"               # everything
```

### CI gate

```bash
tracerazor audit trace.json --threshold 75
# exits non-zero if TAS < 75
```

---

## End-to-end example: LangGraph customer-support agent

Walkthrough using `tracerazor-langgraph` to measure and optimize an agent.

### Step 1: Instrument your agent

```python
# pip install tracerazor-langgraph langgraph langchain-openai
from tracerazor_langgraph import TraceRazorCallback
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_order_status(order_id: str) -> str:
    """Look up current order status."""
    return f"Order {order_id}: shipped 2026-04-10, arriving 2026-04-15."

@tool
def get_refund_policy(order_id: str) -> str:
    """Return the refund policy for an order."""
    return "Refund eligible within 30 days of delivery."

callback = TraceRazorCallback(agent_name="support-agent", threshold=75)
agent   = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), [get_order_status, get_refund_policy])

agent.invoke(
    {"messages": [{"role": "user", "content": "Status of ORD-1001? Can I still get a refund?"}]},
    config={"callbacks": [callback]},
)

# Writes trace to disk and prints the audit report
callback.analyse()
```

### Step 2: Audit the trace

```
$ tracerazor audit trace.json

╔══════════════════════════════════════╗
║  TRACERAZOR EFFICIENCY REPORT        ║
╚══════════════════════════════════════╝
Agent:   support-agent
TAS:     69.5 / 100   [FAIR]
Tokens:  1 710 total  |  603 wasted (35%)

Issues:
  ✗  LDI  0.43 : 1 reasoning loop (steps 2 → 4 → 6 repeat identical tool call)
  ✗  RDA  0.21 : 7 steps used for a trivial task (expected ≤ 2)
  ✗  CCE  0.53 : 805 duplicate tokens across context windows

Fixes:
  1. [termination_guard]  "Once search_products returns results, do not
                           call it again for the same query."   est. 420 tokens/run
  2. [context_compression] "Summarise conversation to last 3 facts before
                            each tool call."                    est. 183 tokens/run

Est. savings: 603 tokens/run  ·  $90/month at 50 K runs
```

### Step 3: Optimize the system prompt

```bash
export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY, or TRACERAZOR_LLM_*
tracerazor optimize trace.json --output system_prompt_v2.txt --target-tas 82
```

```
Optimizing 'support-agent' (TAS 69.5 → target 82.0) using gpt-4o-mini…
  Iteration 1/3: calling LLM… projected TAS 83.7 (+14.2), tokens -440
  Target reached: stopping early.
Wrote optimised prompt → system_prompt_v2.txt
```

The new `system_prompt_v2.txt` contains directives such as:

```
EFFICIENCY RULES
• Call each tool at most once per unique input. If a tool already returned
  results for this query, use those results directly.
• Keep reasoning to one sentence. Do not restate the user's question.
• Summarise prior context to the last three facts before any tool call.
• Reply immediately once the answer is known: no closing preamble.
```

### Step 4: Re-run and verify

Set `system_prompt_v2.txt` as your agent's system prompt, re-run the same
conversation, then confirm the improvement with `tracerazor bench`:

```bash
tracerazor bench --before trace.json --after trace_v2.json --fixes fixes.json
```

```
Before → After
  TAS      69.5 → 83.7   (+14.2)   ✓ MATCH estimated
  Tokens    1710 →  1270   (−440)
  Cost/run  $0.0051 → $0.0038   (−25.7%)
  Verdict   MATCH: actual savings within 10% of estimate
```

| | Before | After | Delta |
|---|---:|---:|---:|
| TAS | 69.5 | 83.7 | **+14.2** |
| Tokens | 1 710 | 1 270 | **−440** |
| Waste | 35% | 9% | **−26 pp** |
| Est. monthly cost (50 K runs) | $255 | $190 | **−$65** |

The full example code lives in
[`integrations/langgraph/examples/customer_support.py`](integrations/langgraph/examples/customer_support.py).

---

## Scoring Pipeline

```mermaid
flowchart TD
    T[Trace JSON] --> P[Parse & Ingest]
    P --> M["Compute All Signals"]

    subgraph M["Structural & Semantic Analysis"]
        direction LR
        S1["Step Redundancy\n17%"]
        S2["Loop Detection\n13%"]
        S3["Tool Accuracy\n13%"]
        S4["Reasoning Depth\n10%"]
        S5["Info Sufficiency\n10%"]
        S6["Token Utilisation\n10%"]
        S7["Context Efficiency\n10%"]
        S8["Decision Optimality\n9%"]
        S9["Semantic Continuity\n5%"]
    end

    subgraph V["Verbosity & Presentation"]
        direction LR
        V1["Density\n9%"]
        V2["Sycophancy/Hedging\n5%"]
        V3["Compression\n4%"]
    end

    M --> W["Weighted Score\n(normalised 0–100)"]
    V --> W
    W --> TAS["TAS\nToken Audit Score"]
    TAS --> G["Grade\nExcellent/Good/Fair/Poor"]

    V1 & V2 & V3 --> AVS["Verbosity Alert\nif combined > 0.40"]
    AVS -.->|detects| VA["!! VERBOSITY ALERT\nwith remediation"]
```

| Grade | TAS | Meaning |
|-------|-----|---------|
| **Excellent** | 90–100 | Minimal recoverable waste |
| **Good** | 70–89 | Addressable inefficiency |
| **Fair** | 50–69 | Significant structural waste |
| **Poor** | 0–49 | Fundamental reasoning issues |

---

## Automated Remediation

Every audit produces machine-applicable fix patches tied to the specific metrics that failed. Fixes include estimated token savings.

```json
"fixes": [
  {
    "fix_type": "tool_schema",
    "target": "check_refund_eligibility",
    "patch": "Mark `order_id` as required in the tool schema...",
    "estimated_token_savings": 580
  },
  {
    "fix_type": "hedge_reduction",
    "target": "system_prompt",
    "patch": "Do not begin responses with preamble phrases (let me, I'd be happy to, certainly)...",
    "estimated_token_savings": 740
  },
  {
    "fix_type": "reformulation_guard",
    "target": "system_prompt",
    "patch": "Do not re-state the user's request at the start of reasoning. Proceed directly to analysis. (Steps [2, 5] detected as reformulating input context.)",
    "estimated_token_savings": 360
  }
]
```

| Fix Type | Trigger | Target |
|---|---|---|
| `tool_schema` | TCA misfire | Tool's required parameter schema |
| `prompt_insert` | RDA over-depth | Step-count instruction |
| `termination_guard` | LDI loop | Loop-breaking condition |
| `context_compression` | CCE bloat | Context summarisation instruction |
| `verbosity_reduction` | VDI fail + AVS > 0.40 | Filler-word elimination |
| `hedge_reduction` | SHL fail + AVS > 0.40 | Sycophancy/hedging directive |
| `caveman_prompt_insert` | CCR fail + AVS > 0.40 | Maximal conciseness directive |
| `reformulation_guard` | Reformulation flag | Skip re-stating input context |

---

## Anomaly Detection

Rolling baselines detect metric regressions after 5+ traces. Fires at |z| > 2.0. Each metric checked independently.

```json
"anomalies": [
  { "metric": "shl", "value": 0.45, "z_score": -2.3, "baseline_mean": 0.12, "baseline_std": 0.14 }
]
```

---

## Live Guardrail Proxy

Four-layer request interceptor applies efficiency guardrails before tokens are consumed.

```mermaid
flowchart TD
    REQ[ProxyRequest] --> L1

    subgraph L1[Layer 1: Semantic Preservation]
        S1{similarity ≥ threshold?}
    end
    L1 -->|No: drift| B1[Blocked / layer: 1]
    L1 -->|Yes| L2

    subgraph L2[Layer 2: Scope Whitelist]
        S2{tool in whitelist?}
    end
    L2 -->|No| B2[Blocked / layer: 2]
    L2 -->|Yes| L3

    subgraph L3[Layer 3: Budget Injection]
        S3{tokens > 75% of budget?}
    end
    S3 -->|Yes| BI[Inject budget directive]
    S3 -->|No| L4
    BI --> L4

    subgraph L4[Layer 4: Verbosity Directive]
        S4{rolling_ccr ≥ 0.35?}
    end
    S4 -->|0.35–0.50| VI1[Standard conciseness directive]
    S4 -->|> 0.50| VI2[Ultra-concise directive]
    S4 -->|No| APP
    VI1 --> APP
    VI2 --> APP

    APP[Approved: modified system_prompt]
```

```rust
use tracerazor_proxy::{ProxyConfig, ProxyRequest, ProxyResponse};

let proxy = ProxyConfig::default();
let req = ProxyRequest {
    system_prompt: "You are a support agent.".into(),
    rolling_ccr: Some(0.38),   // Layer 4 triggers standard directive
    tokens_used: 4200,
    ..
};

match proxy.intercept(&req) {
    ProxyResponse::Approved { system_prompt, .. } => { /* call LLM */ }
    ProxyResponse::Blocked  { reason, layer }     => { /* log and abort */ }
}
```

---

## Integrations

### Python SDK — Audit

Record steps manually and submit for analysis:

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
report.assert_passes()   # raises AssertionError in CI if TAS < 70
```

Or submit a trace dict directly:

```python
from tracerazor import TraceRazorClient

client = TraceRazorClient(server="http://localhost:8080")
report = client.analyse(trace_dict)
print(report.tas_score, report.grade)
```

### Python SDK — Adaptive Sampling

Replace your LangGraph ReAct node with `AdaptiveKNode` to sample K parallel
candidates per step and pick the consensus winner:

```python
from tracerazor import AdaptiveKNode, openai_llm
from openai import AsyncOpenAI

llm = openai_llm(AsyncOpenAI(), model="gpt-4.1")
node = AdaptiveKNode(llm=llm, tools=my_tools, k_max=5, k_min=2)

graph = StateGraph(AgentState)
graph.add_node("agent", node)
```

Benchmark results on tau-bench airline (50 tasks, gpt-4o):

| Strategy | pass^1 | mean tokens | vs baseline |
|---|---|---|---|
| K=1 baseline | 38% | 63k | 1.0x |
| AdaptiveKNode (K=5) | 46% | 246k | 3.9x |
| SelfConsistencyBaseline (K=5) | 48% | 137k | 2.2x |

### LangGraph / LangChain

```python
from tracerazor_langgraph import TraceRazorCallback

callback = TraceRazorCallback(agent_name="support-graph", threshold=70)
result = graph.invoke({"messages": [...]}, config={"callbacks": [callback]})
callback.analyse().markdown()
```

### CrewAI

```python
from tracerazor_crewai import TraceRazorCallback

callback = TraceRazorCallback(agent_name="support-crew", threshold=70)
crew = Crew(agents=[...], tasks=[...], callbacks=[callback])
crew.kickoff()
callback.assert_passes()
```

### OpenAI Agents SDK

```python
from tracerazor_openai_agents import TraceRazorHooks

hooks = TraceRazorHooks(agent_name="support-agent", threshold=70)
await Runner.run(agent, "I need a refund for order ORD-9182", hooks=hooks)
hooks.assert_passes()
```

### GitHub Action

```yaml
- uses: ./.github/actions/tracerazor
  with:
    trace-file: traces/latest.json
    threshold: '75'
```

Outputs: `tas-score`, `grade`, `passes`, `report`. Exits 1 if TAS < threshold.

| Framework | Adapter |
|---|---|
| LangGraph / LangChain | Native callback + LangSmith / OTEL ingest |
| OpenAI Agents SDK | Native `RunHooks` |
| CrewAI | Native `CrewCallbackHandler` |
| OTEL-instrumented agents | OTEL JSON ingest |
| Raw / custom | Python SDK or JSON |

---

## CLI Reference

```
tracerazor <COMMAND>

Commands:
  audit      Score a trace file; optionally gate on --threshold <N>
  optimize   Rewrite the system prompt with an LLM to eliminate detected waste
  apply      Patch a system prompt file with safe, non-functional fixes
  bench      Compare before/after traces and verify actual savings
  compare    Per-metric delta table between two trace files
  simulate   Project TAS impact of removing or merging steps
  cost       Monthly savings estimate across a set of traces
  export     Forward a stored trace to OTEL or a webhook

Options (audit):
  --threshold <N>         Exit non-zero if TAS < N
  --format markdown|json
  --trace-format auto|raw|langsmith|otel
  --enhanced              LLM embeddings for SRR/ISR (OpenAI / OpenAI-compatible; Anthropic chat-only)

Options (optimize):
  --system-prompt <FILE>  Existing system prompt to rewrite (creates one if absent)
  --output <FILE>         Write optimised prompt here (stdout if omitted)
  --iterations <N>        Max LLM calls per run (default: 3)
  --target-tas <N>        Stop early when projected TAS ≥ N (default: 85.0)
  --format markdown|json
```

```bash
tracerazor compare before.json after.json --regression-threshold 5.0
tracerazor simulate trace.json --remove 3,8 --merge 6,7
tracerazor cost trace*.json --provider anthropic-claude-3-5-sonnet --runs-per-month 50000
tracerazor optimize trace.json --system-prompt agent.txt --output agent_v2.txt
```

LLM backend selection is environment-driven (used by `optimize` and `--enhanced`):

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic (chat completion; embeddings fall back to BoW)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI-compatible (Ollama, vLLM, OpenRouter, Groq, Together, LM Studio, ...)
export TRACERAZOR_LLM_PROVIDER=openai-compatible
export TRACERAZOR_LLM_BASE_URL=http://localhost:11434/v1
export TRACERAZOR_LLM_MODEL=llama3.1
# optional auth:
export TRACERAZOR_LLM_API_KEY=...
```

---

## REST API

Start: `./target/release/tracerazor-server`

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/audit` | Score a trace; auto-captures to KB if TAS ≥ 85 |
| `GET` | `/api/traces` | List stored traces |
| `GET/DELETE` | `/api/traces/:id` | Full trace + report / delete |
| `GET` | `/api/dashboard` | Aggregate stats |
| `GET` | `/api/agents` | Per-agent stats, worst-first |
| `GET` | `/api/compare?a=:id&b=:id` | Metric diff between two traces |
| `GET/DELETE` | `/api/kb/:id` | Known-Good-Paths entries |
| `GET` | `/api/metrics` | Prometheus exposition |
| `WS` | `/ws` | Live audit event stream |

Audit response includes `tas_score`, `grade`, `avs`, `fixes`, `tokens_saved`, `anomalies`, `per_agent`, `kb_match`, and `report_markdown`.

### Prometheus

```yaml
scrape_configs:
  - job_name: tracerazor
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: /api/metrics
```

---

## Deployment

```bash
docker compose up --build   # http://localhost:8080
PORT=9090 docker compose up
```

| Variable | Default | Description |
|---|---|---|
| `TRACERAZOR_DB_PATH` | `./tracerazor.db` | Persistent database path |
| `PORT` | `8080` | HTTP server port |
| `TRACERAZOR_CORS_ORIGINS` | *(permissive)* | Comma-separated allowed origins |

---

## Architecture

```
tracerazor/
├── crates/
│   ├── tracerazor-core/      # 13 metrics, AVS, reformulation, scoring, fixes, IAR, reports
│   ├── tracerazor-ingest/    # Parsers: raw JSON, LangSmith, OpenTelemetry
│   ├── tracerazor-semantic/  # BoW similarity + pluggable LLM backend (OpenAI/Anthropic/OpenAI-compatible)
│   ├── tracerazor-store/     # SurrealDB: traces, KB, baselines, anomaly detection
│   ├── tracerazor-server/    # Axum REST + WebSocket + embedded dashboard
│   ├── tracerazor-proxy/     # Four-layer guardrail proxy
│   └── tracerazor-cli/       # CLI entry point; persistent store at ~/.tracerazor/
├── v2/
│   └── tracerazor/           # Python package v1.0.0  (pip install tracerazor)
│                             # Audit: Tracer, TraceRazorClient, TraceRazorReport
│                             # Sampling: AdaptiveKNode, SelfConsistencyBaseline,
│                             #           NaiveKEnsemble, ExactMatchConsensus
├── integrations/
│   ├── crewai/               # CrewAI adapter
│   ├── openai-agents/        # OpenAI Agents SDK adapter
│   └── langgraph/            # LangGraph / LangChain callback adapter
└── .github/                  # CI workflow + composite GitHub Action
```

`tracerazor-core` has zero network dependencies. Semantic crate is separate, so offline analysis never pulls in `reqwest`. `--enhanced` activates at runtime without recompiling.

---

## Test Coverage

| Crate | Tests |
|---|---|
| tracerazor-core | 116 |
| tracerazor-ingest | 3 |
| tracerazor-semantic | 12 |
| tracerazor-store | 21 |
| tracerazor-server | 13 |
| tracerazor-proxy | 9 |
| **Total** | **183, all pass** |

Comprehensive coverage including:
- Structural metric detection (redundancy, loops, tool failures, depth)
- Verbosity analysis (density, hedging, compression)
- Semantic drift and continuity validation
- Optimization validation and adherence scoring
- End-to-end integration tests across all crates

---

## Research Foundation

TraceRazor's metrics are grounded in peer-reviewed findings on LLM reasoning efficiency. The weight distribution and detection thresholds were calibrated against the empirical results in these papers.

| # | Paper | Informs |
|---|---|---|
| [1] | Han et al. (2024). **Token-Budget-Aware LLM Reasoning (TALE)**. ACL 2025. | TUR, CCE |
| [2] | Zhao et al. (2025). **SelfBudgeter: Adaptive Token Allocation**. | Proxy Layer 3 |
| [3] | Lee et al. (2025). **Evaluating Step-by-step Reasoning Traces: A Survey**. | Framework basis |
| [4] | Su et al. (2024). **Dualformer: Controllable Fast and Slow Thinking**. | RDA |
| [5] | Wu et al. (2025). **Step Pruner: Efficient Reasoning in LLMs**. | Optimal path diff |
| [6] | Feng et al. (2025). **Efficient Reasoning Models: A Survey**. | Metric validation |
| [7] | Pan et al. (2024). **ToolChain\*: A\* Search for Tool Sequences**. NeurIPS 2024. | DBO, KB design |
| [8] | Hassid et al. (2025). **Reasoning on a Budget**. | VAE score, proxy |
| [9] | (2025). **Balanced Thinking (SCALe-SFT)**. | Efficiency without accuracy loss |
| [10] | Mohammadi et al. (2025). **Evaluation and Benchmarking of LLM Agents**. KDD 2025. | Composite scoring |
| [11] | Shi et al. (2024). **Verbosity Bias in LLM Responses**. | VDI, SHL, CCR design |

---

## License

Apache 2.0. Copyright 2025 Zulfaqar Hafez. See [LICENSE](LICENSE).