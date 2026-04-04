# TraceRazor

**Agentic Reasoning Path Efficiency Auditor**

> "Lighthouse score for AI agents. Audit reasoning traces and tell you exactly where your agent wastes tokens, why, and how to fix it."

Version 1.3 · Apache 2.0 · Author: Zulfaqar Hafez

---

## What It Does

TraceRazor is a framework-agnostic auditor for AI agent reasoning traces. It analyses completed agentic execution paths and produces:

- A composite **TraceRazor Score (TAS)** from 0–100 (like Google Lighthouse, but for agents)
- Per-step annotations identifying **redundant steps, loops, tool misfires, and context bloat**
- An **optimal path diff** showing exactly which steps to remove or trim
- **Token and cost savings estimates** at enterprise scale

Industry research shows 40–70% of reasoning tokens in typical chain-of-thought traces are redundant. TraceRazor makes that waste visible, quantifiable, and fixable.

---

## The Problem

Token costs scale with agent complexity. A single customer-support resolution requiring 8 tool calls and 3 reasoning loops can consume 15,000–40,000 tokens per interaction. At 50,000 interactions/month, a 30% reduction in waste translates to six-figure annual savings.

Existing tools either passively log what happened (LangSmith, Langfuse, Arize) or modify inference behaviour (TALE, SelfBudgeter, Step Pruner). **No existing product audits traces post-hoc and recommends concrete optimisations.** TraceRazor fills that gap.

---

## Quick Start

### Requirements

- Rust 1.70+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Node.js 18+ (for dashboard only)

### Build and Run

```bash
git clone https://github.com/ZulfaqarHafez/tracerazor
cd tracerazor
cargo build --release
# Binary at: target/release/tracerazor
```

### Audit a trace (CLI)

```bash
# Phase 1: structural analysis (no API key needed)
./target/release/tracerazor audit ./traces/support-agent-run-2847.json

# Phase 2: full analysis with OpenAI embeddings + LLM metrics
./target/release/tracerazor audit ./traces/support-agent-run-2847.json --semantic
```

```
TRACERAZOR REPORT
------------------------------------------------------
Trace:     support-agent-run-2847
Agent:     customer-support-v3
Framework: langgraph
Steps:     11   Tokens: 14,280
Analysed:  3ms (structural)
------------------------------------------------------
TRACERAZOR SCORE:  82 / 100  [GOOD]
------------------------------------------------------
METRIC BREAKDOWN
SRR    Step Redundancy Rate      18.2%    <15%     FAIL
LDI    Loop Detection Index      0.182    <0.10    FAIL
TCA    Tool Call Accuracy        83.3%    >85%     FAIL
TUR    Token Utilisation Ratio   71.4%    >35%     PASS
CCE    Context Carry-over        100.0%   >60%     PASS
------------------------------------------------------
SAVINGS ESTIMATE
Tokens saved:      7,006  (49.1% reduction)
At 50K runs/month: $1,050.90/month saved
```

### Output formats

```bash
./target/release/tracerazor audit trace.json --format json      # machine-readable
./target/release/tracerazor audit trace.json --format markdown  # human-readable (default)
```

### CI/CD gating

```bash
./target/release/tracerazor audit trace.json --threshold 75
# Exits non-zero if TAS < 75 — blocks the PR
```

---

## Phase 3: Web Dashboard + Server

### Start the server

```bash
# Build the dashboard first
cd dashboard && npm install && npm run build && cd ..

# Set database path (optional — defaults to ./tracerazor.db)
export TRACERAZOR_DB_PATH=./tracerazor.db

# Start the server
./target/release/tracerazor-server
# Listening on http://localhost:8080
```

### Server endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/audit` | Ingest and analyse a trace |
| `GET` | `/api/traces` | List all stored traces |
| `GET` | `/api/traces/:id` | Full trace + report |
| `DELETE` | `/api/traces/:id` | Delete a trace |
| `GET` | `/api/dashboard` | Aggregate dashboard data |
| `GET` | `/api/agents` | Per-agent statistics |
| `GET` | `/api/agents/:name` | Single agent stats |
| `WS` | `/ws` | Real-time events (WebSocket) |
| `GET` | `/` | React dashboard |

### Audit via REST API

```bash
curl -s -X POST http://localhost:8080/api/audit \
  -H "Content-Type: application/json" \
  -d '{"trace": '"$(cat traces/support-agent-run-2847.json)"'}' | jq .
```

### WebSocket live events

Connect to `ws://localhost:8080/ws` to receive real-time JSON events:

```json
{"type":"trace_analysed","trace_id":"run-001","agent_name":"support-v3","tas_score":82.1,"grade":"Good","tokens_saved":7006}
{"type":"loop_detected","trace_id":"run-002","step_id":5,"cycle":"check_eligibility→check_eligibility"}
```

### Dashboard

Open `http://localhost:8080` to see:
- **Overview** — total traces, agents, avg TAS, tokens saved
- **TAS trend chart** — Recharts line chart of score over time
- **Agent rankings** — worst-to-best table
- **Trace list** — click any trace for the full report
- **Audit** — paste trace JSON and get a report in-browser
- **Live** — real-time event feed from the WebSocket

---

## Phase 3: Proxy Guardrail System

The `tracerazor-proxy` crate intercepts LLM calls and applies three guardrail layers:

### Layer 1 — Semantic Preservation

Blocks requests where the combined prompt has drifted too far from the original task description (default threshold: cosine similarity < 0.55).

### Layer 2 — Scope Whitelist

Validates tool names against a configurable allowlist. Blocks calls to disallowed tools before they reach the LLM.

```rust
let scope = ScopeConfig::whitelist(["get_order", "process_refund"]);
// Blocks "delete_database", "drop_table", etc.
```

### Layer 3 — Token Budget Injection

When token usage exceeds 75% of the budget, prepends a `<budget>` directive to the system prompt nudging conciseness — without hard-blocking the request.

```
<budget remaining="2000" total="8000">
Be concise. Avoid repeating context already established in this conversation.
</budget>
[original system prompt...]
```

### Usage

```rust
use tracerazor_proxy::{ProxyConfig, ProxyRequest, ProxyResponse};

let proxy = ProxyConfig::default();
let req = ProxyRequest {
    task_description: "Process refund for order ORD-9182".into(),
    system_prompt: "You are a helpful assistant.".into(),
    user_message: "Check the refund status.".into(),
    requested_tools: vec!["get_order".into()],
    tokens_used: 1200,
};

match proxy.intercept(&req) {
    ProxyResponse::Approved { system_prompt, .. } => { /* proceed */ }
    ProxyResponse::Blocked { reason, layer } => { /* log and abort */ }
}
```

---

## LangGraph Integration

```python
from tracerazor_langgraph import TraceRazorCallback

callback = TraceRazorCallback(
    agent_name="my-agent",
    threshold=75,       # CI/CD gating
    semantic=True,      # OpenAI embeddings + RDA/DBO
)

result = graph.invoke(inputs, config={"callbacks": [callback]})
report = callback.analyse()
print(report.markdown())

# Raise AssertionError if TAS < 75 (for tests/CI)
callback.assert_passes()
```

Install the adapter:
```bash
pip install -e integrations/langgraph
```

---

## Metrics Framework

### Phase 1: Structural Metrics

| Code | Metric | Weight | Formula | Target |
|------|--------|--------|---------|--------|
| **SRR** | Step Redundancy Rate | 20% | `redundant_steps / total_steps × 100` | < 15% |
| **LDI** | Loop Detection Index | 15% | `max_cycle_length / total_steps` | < 0.10 |
| **TCA** | Tool Call Accuracy | 15% | `successful_first_attempts / total_tool_calls × 100` | > 85% |
| **TUR** | Token Utilisation Ratio | 10% | `useful_output_tokens / total_tokens` | > 0.35 |
| **CCE** | Context Carry-over Efficiency | 10% | `1 − (duplicate_context_tokens / total_input_tokens)` | > 0.60 |

### Phase 2: Semantic Metrics (requires `--semantic` + `OPENAI_API_KEY`)

| Code | Metric | Weight | Method |
|------|--------|--------|--------|
| **RDA** | Reasoning Depth Appropriateness | 10% | GPT-4o-mini task complexity classifier |
| **ISR** | Information Sufficiency Rate | 10% | Cosine distance; steps with < 10% novelty flagged |
| **DBO** | Decision Branch Optimality | 10% | GPT-4o-mini retrospective branch judge |

### TAS Score (0–100)

Weighted composite re-normalised over available metrics.

| Grade | Range | Meaning |
|-------|-------|---------|
| Excellent | 90–100 | Highly optimised |
| Good | 70–89 | Minor inefficiencies |
| Fair | 50–69 | Actionable waste |
| Poor | 0–49 | Significant restructuring needed |

---

## Architecture

```
tracerazor/
├── crates/
│   ├── tracerazor-core/      # Graph engine, metrics (SRR/LDI/TCA/TUR/CCE/RDA/ISR/DBO), scoring, reports
│   ├── tracerazor-ingest/    # Parsers: raw JSON, LangSmith, OpenTelemetry
│   ├── tracerazor-semantic/  # BoW + OpenAI embeddings + LLM chat client
│   ├── tracerazor-store/     # SurrealDB: in-memory (CLI) + persistent kv-surrealkv (server)
│   ├── tracerazor-server/    # Axum HTTP/WebSocket server + REST API
│   ├── tracerazor-proxy/     # LLM proxy with 3-layer guardrail system
│   └── tracerazor-cli/       # CLI entry point (clap 4)
├── dashboard/                # React + Vite dashboard (Recharts)
├── integrations/
│   └── langgraph/            # Python LangGraph callback adapter
├── traces/                   # Sample trace files
└── .github/workflows/        # GitHub Actions CI/CD
```

---

## CI/CD: GitHub Action

`.github/workflows/tracerazor.yml` runs on every push:

1. **`cargo check`** — all crates compile
2. **`cargo test`** — 42 unit + integration tests pass
3. **`cargo clippy`** — no warnings treated as errors
4. **Dashboard build** — `npm ci && npm run build`
5. **TAS gate** — audit `traces/support-agent-run-2847.json`, fail if TAS < 60

```yaml
- name: Run TAS gate on sample trace
  run: ./target/release/tracerazor audit \
         --file traces/support-agent-run-2847.json \
         --threshold 60
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for `--semantic` (Phase 2) |
| `TRACERAZOR_LLM_MODEL` | `gpt-4o-mini` | Chat model for RDA / DBO |
| `TRACERAZOR_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for SRR / ISR |
| `TRACERAZOR_DB_PATH` | `./tracerazor.db` | Server persistent DB path |
| `PORT` | `8080` | Server port |
| `TRACERAZOR_BIN` | auto-detected | Path to `tracerazor` binary (Python adapter) |

Store secrets in `.env` at the repo root (already in `.gitignore`).

---

## Test Coverage

```
cargo test --workspace
```

| Crate | Tests | Coverage |
|-------|-------|---------|
| tracerazor-core | 20 | SRR, LDI, TCA, TUR, CCE, scoring, report, graph |
| tracerazor-ingest | 3 | raw JSON, LangSmith, OTEL parsing |
| tracerazor-semantic | 5 | BoW similarity edge cases |
| tracerazor-store | 5 | save/retrieve, list, baseline, dashboard, delete |
| tracerazor-server | 4 | index, audit+list, dashboard, 404 |
| tracerazor-proxy | 5 | scope whitelist, budget injection |
| **Total** | **42** | **all pass** |

---

## Trace Format

### Raw JSON

```json
{
  "trace_id": "run-001",
  "agent_name": "my-agent",
  "framework": "langgraph",
  "task_value_score": 1.0,
  "steps": [
    {
      "id": 1,
      "step_type": "reasoning",
      "content": "Parse the user request...",
      "tokens": 820
    },
    {
      "id": 2,
      "step_type": "tool_call",
      "content": "Fetching order details",
      "tokens": 340,
      "tool_name": "get_order_details",
      "tool_params": {"order_id": "ORD-9182"},
      "tool_success": true,
      "input_context": "full prompt sent to LLM"
    }
  ]
}
```

**Step fields:**
| Field | Required | Description |
|-------|----------|-------------|
| `id` | ✓ | 1-based step index |
| `step_type` | ✓ | `reasoning`, `tool_call`, or `handoff` |
| `content` | ✓ | Primary text content |
| `tokens` | ✓ | Total tokens consumed |
| `tool_name` | — | Tool identifier |
| `tool_params` | — | Parameters passed to tool |
| `tool_success` | — | Whether the call succeeded |
| `tool_error` | — | Error message if failed |
| `input_context` | — | Full input context (for CCE) |
| `agent_id` | — | Agent ID for multi-agent traces |

### LangSmith, OpenTelemetry

Auto-detected by the parser (checks for `run_type`, `child_runs`, or `resourceSpans` fields). Pass `--trace-format langsmith|otel|raw` to override.

---

## Research Background

| # | Paper | Relevance |
|---|-------|-----------|
| [1] | Han et al. (2024). **Token-Budget-Aware LLM Reasoning (TALE)**. ACL 2025. | Motivates TUR and CCE |
| [2] | Zhao et al. (2025). **SelfBudgeter: Adaptive Token Allocation**. | Proxy Layer 3 budget injection |
| [3] | Lee et al. (2025). **Evaluating Step-by-step Reasoning Traces: A Survey**. | 8-metric framework basis |
| [4] | Su et al. (2024). **Dualformer: Controllable Fast and Slow Thinking**. | RDA metric design |
| [5] | Wu et al. (2025). **Step Pruner: Efficient Reasoning in LLMs**. | Optimal path recommendation |
| [6] | Feng et al. (2025). **Efficient Reasoning Models: A Survey**. | Metric selection validation |
| [7] | Pan et al. (2024). **ToolChain*: A* Search for Tool Sequences**. NeurIPS 2024. | DBO metric + path search |
| [8] | Hassid et al. (2025). **Reasoning on a Budget**. | VAE scoring + proxy design |
| [9] | (2025). **Balanced Thinking (SCALe-SFT)**. | Validates efficiency without accuracy loss |
| [10] | Mohammadi et al. (2025). **Evaluation and Benchmarking of LLM Agents**. KDD 2025. | Composite scoring validation |

**Key finding:** Studies [1], [2], [5], [8] all show 40–70% of reasoning tokens are redundant in typical CoT traces. Study [3] confirms step-by-step trace evaluation is tractable from trace data alone — no model internals required.

---

## Compatibility

| Framework | Format | Status |
|-----------|--------|--------|
| LangGraph / LangChain | LangSmith JSON, OTEL | Implemented |
| OpenAI Agents SDK | OTEL spans | Implemented |
| CrewAI | Task logs | Parser included |
| Raw / Custom | User-defined JSON | Implemented |
| OpenTelemetry (generic) | OTEL JSON | Implemented |
| AutoGen | Conversation JSON | Planned |

---

## Licence

Apache 2.0.

All Rust crates, the CLI, Python SDK, and the web dashboard are open-source. Monetisation path: managed cloud hosting (TraceRazor Cloud) with SLA, SSO, and team management — mirrors the Langfuse / PostHog model.
