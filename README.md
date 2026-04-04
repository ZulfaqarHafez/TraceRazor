# TraceRazor

**Token efficiency for AI agents.**

> Audit your agent's reasoning traces, score them like Lighthouse scores a webpage, and get a step-by-step plan to cut token waste. No changes to your agent code required.

[![CI](https://github.com/ZulfaqarHafez/tracerazor/actions/workflows/tracerazor.yml/badge.svg)](https://github.com/ZulfaqarHafez/tracerazor/actions)
&nbsp;Apache 2.0 &nbsp;·&nbsp; Rust + Alpine.js &nbsp;·&nbsp; Author: Zulfaqar Hafez

---

## Why TraceRazor

Production AI agents are expensive because they reason too much. Five academic research groups across ACL 2025, NeurIPS 2024, and KDD 2025 independently measured **40-70% of reasoning tokens as redundant** in typical chain-of-thought traces [[1](#research-foundation)-[6](#research-foundation)]. That redundancy is invisible until it shows up on the invoice.

A customer-support agent requiring 8 tool calls and 3 reasoning loops can consume 15,000-40,000 tokens per resolution. At 50,000 interactions/month, a 30% efficiency improvement pays for six figures annually.

Existing observability tools (LangSmith, Langfuse, Arize) tell you *what happened*. They don't tell you *what was unnecessary* or *what the efficient version looks like*. TraceRazor is a post-hoc auditor: it reads the trace after execution, scores it, and produces a concrete diff.

---

## What You Get

**TAS Score.** A composite 0-100 efficiency grade derived from eight research-backed metrics, weighted and normalised consistently. One number your team can track, gate CI/CD on, and regression-test against.

**Optimal path diff.** A step-by-step breakdown of which reasoning steps to remove, which tool calls are redundant, and which context windows are being re-transmitted for no reason.

**Savings projection.** Tokens saved multiplied by cost per token multiplied by your run volume, extrapolated to monthly and annual figures.

**Known-Good-Paths KB.** Every trace scoring >= 85 is automatically stored as a reference. Future traces by the same agent are matched against it and the audit response surfaces the closest prior run: its score, its path, and its token count.

**Live guardrails.** An interceptor layer that blocks semantically-drifted prompts, enforces tool scope whitelists, and injects token budget directives before an agent blows its budget.

---

## Quickstart

### Docker (recommended)

```bash
git clone https://github.com/ZulfaqarHafez/tracerazor
cd tracerazor
echo "OPENAI_API_KEY=sk-..." > .env   # optional, Phase 2 only
docker compose up --build
```

Open `http://localhost:8080`. No Node.js or Rust toolchain required on the host.

### Binary

```bash
cargo build --release
./target/release/tracerazor audit traces/support-agent-run-2847.json
```

```
TRACERAZOR SCORE:  82 / 100  [GOOD]
──────────────────────────────────────
SRR  Step Redundancy Rate    18.2%   FAIL  (target < 15%)
LDI  Loop Detection Index    0.182   FAIL  (target < 0.10)
TCA  Tool Call Accuracy      83.3%   FAIL  (target > 85%)
TUR  Token Utilisation       71.4%   PASS
CCE  Context Carry-over     100.0%   PASS
──────────────────────────────────────
Tokens saved:  7,006  (49.1%)
At 50K runs/month:  $1,050.90/month saved
```

### CI/CD gate

```bash
# Exits non-zero if TAS drops below 75, blocking the build
tracerazor audit trace.json --threshold 75
```

---

## Dashboard

The web interface ships as a single HTML file compiled into the server binary. There is no separate frontend deployment and no Node.js required in production.

| Tab | What it shows |
|-----|---------------|
| **Dashboard** | TAS trend chart, agent rankings worst-first, savings totals |
| **Traces** | Full trace history with drill-down to the report |
| **Audit** | Paste any trace JSON, get a full report in-browser |
| **Compare** | Side-by-side diff of two trace IDs: TAS delta, tokens saved delta, verdict |
| **KB** | Known-Good-Paths library, browse optimal paths from high-scoring traces |
| **Live** | Real-time WebSocket feed of every audit event |

Light and dark themes, localStorage persistence, auto-reconnecting WebSocket.

---

## API Reference

Start the server: `./target/release/tracerazor-server`

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/audit` | Ingest and analyse a trace; auto-captures to KB if TAS >= 85 |
| `GET` | `/api/traces` | List all stored traces |
| `GET` | `/api/traces/:id` | Full trace and report |
| `DELETE` | `/api/traces/:id` | Remove a trace |
| `GET` | `/api/dashboard` | Aggregate stats for the dashboard |
| `GET` | `/api/agents` | Per-agent statistics, sorted worst-first |
| `GET` | `/api/agents/:name` | Single agent stats |
| `GET` | `/api/compare?a=:id&b=:id` | Score two traces against each other |
| `GET` | `/api/kb` | List Known-Good-Paths entries |
| `GET` | `/api/kb/:id` | Full optimal path for one KB entry |
| `DELETE` | `/api/kb/:id` | Remove a KB entry |
| `GET` | `/api/metrics` | Prometheus exposition format |
| `WS` | `/ws` | Live audit events |

### Audit response

```json
{
  "trace_id": "run-001",
  "tas_score": 91.2,
  "grade": "Excellent",
  "tokens_saved": 3400,
  "captured_to_kb": true,
  "kb_match": {
    "entry": { "source_trace_id": "run-098", "tas_score": 94.1, "optimal_tokens": 2800 },
    "similarity": 0.81
  },
  "report_markdown": "..."
}
```

`captured_to_kb: true` means this trace's optimal path was added to the KB. `kb_match` is present when a similar prior run exists.

### Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: tracerazor
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: /api/metrics
```

Exposes `tracerazor_traces_total`, `tracerazor_avg_tas_score`, `tracerazor_tokens_saved_total`, `tracerazor_cost_saved_usd_total`.

---

## Known-Good-Paths KB

The KB stores the optimal execution paths of high-scoring traces so future runs by the same agent can be compared against a validated reference.

**Capture.** Any trace scoring >= 85 TAS has its KEEP/TRIM steps extracted from the diff and written to a `kb_entries` table in SurrealDB, along with the task hint, token counts, and grade.

**Match.** When a new trace is submitted, TraceRazor computes bag-of-words cosine similarity between the incoming trace's first reasoning step and every KB entry for the same agent. Matches above the 0.45 threshold are returned in the audit response.

**Over time.** The KB accumulates the proven-efficient patterns for each agent. Unlike passive trace logging, the KB only stores *optimal* paths, not what actually happened.

---

## Guardrail Proxy

`tracerazor-proxy` sits between your agent orchestrator and the model and applies three checks in order.

**Layer 1: Semantic Preservation.** Computes cosine similarity between the incoming prompt and the original task description. If the agent has drifted beyond the 0.55 threshold, the request is blocked. This catches runaway reasoning loops before they consume tokens.

**Layer 2: Scope Whitelist.** Validates every requested tool name against a configured allowlist. Tools not on the list never reach the model.

```rust
let scope = ScopeConfig::whitelist(["get_order", "check_eligibility", "process_refund"]);
```

**Layer 3: Budget Injection.** When cumulative token usage crosses 75% of the configured budget, a `<budget>` directive is prepended to the system prompt.

```
<budget remaining="2000" total="8000">
Be concise. Avoid repeating context already established in this conversation.
</budget>
```

```rust
use tracerazor_proxy::{ProxyConfig, ProxyRequest, ProxyResponse};

let proxy = ProxyConfig::default();
match proxy.intercept(&req) {
    ProxyResponse::Approved { system_prompt, .. } => { /* call LLM */ }
    ProxyResponse::Blocked { reason, layer }      => { /* log and abort */ }
}
```

---

## LangGraph Integration

```python
from tracerazor_langgraph import TraceRazorCallback

callback = TraceRazorCallback(
    agent_name="support-agent",
    threshold=75,    # block deployment if TAS falls below this
    semantic=True,   # use OpenAI embeddings for SRR/ISR and GPT-4o-mini for RDA/DBO
)

result = graph.invoke(inputs, config={"callbacks": [callback]})

report = callback.analyse()
print(report.markdown())

# Raises AssertionError if TAS < threshold, useful in pytest
callback.assert_passes()
```

```bash
pip install -e integrations/langgraph
```

---

## Metrics

All eight metrics derive from trace data alone. No model weights, no inference hooks, no prompt modification.

### Structural (Phase 1, offline, no API key)

| Code | Metric | Weight | What it measures | Target |
|------|--------|--------|-----------------|--------|
| **SRR** | Step Redundancy Rate | 20% | Fraction of reasoning steps that are near-duplicates of prior steps | < 15% |
| **LDI** | Loop Detection Index | 15% | Longest repeated tool-call cycle as a fraction of total steps | < 0.10 |
| **TCA** | Tool Call Accuracy | 15% | Rate of tool calls that succeed on the first attempt | > 85% |
| **TUR** | Token Utilisation Ratio | 10% | Fraction of tokens attributed to useful work | > 35% |
| **CCE** | Context Carry-over Efficiency | 10% | How much of the input context is novel vs. already seen | > 60% |

### Semantic (Phase 2, requires `OPENAI_API_KEY`)

| Code | Metric | Weight | Method |
|------|--------|--------|--------|
| **RDA** | Reasoning Depth Appropriateness | 10% | GPT-4o-mini classifies task complexity and compares it to actual reasoning depth |
| **ISR** | Information Sufficiency Rate | 10% | Embedding novelty per step; steps below 10% novelty are flagged |
| **DBO** | Decision Branch Optimality | 10% | GPT-4o-mini retrospectively judges each decision point |

Activate with `--semantic`. Weights re-normalise automatically when Phase 2 metrics are unavailable, so the score stays interpretable in both modes.

### TAS Score

| Grade | Range | Meaning |
|-------|-------|---------|
| **Excellent** | 90-100 | Optimised, minor gains only |
| **Good** | 70-89 | Some inefficiency present |
| **Fair** | 50-69 | Significant waste, restructuring recommended |
| **Poor** | 0-49 | Fundamental reasoning pattern issues |

---

## Architecture

Seven Rust crates, one embedded Alpine.js dashboard, one Python integration layer.

```
tracerazor/
├── crates/
│   ├── tracerazor-core/      # Metrics, scoring, report generation, DAG engine
│   ├── tracerazor-ingest/    # Format parsers: raw JSON, LangSmith, OpenTelemetry
│   ├── tracerazor-semantic/  # BoW similarity, OpenAI embeddings, LLM client
│   ├── tracerazor-store/     # SurrealDB persistence: traces and KB entries
│   ├── tracerazor-server/    # Axum REST, WebSocket, embedded dashboard
│   ├── tracerazor-proxy/     # Three-layer LLM guardrail interceptor
│   └── tracerazor-cli/       # CLI entry point (clap 4)
├── dashboard/                # Alpine.js + Chart.js (embedded) / React build (optional)
├── integrations/
│   └── langgraph/            # Python callback adapter
├── traces/                   # Sample traces
├── Dockerfile                # Multi-stage: node -> rust -> debian-slim
└── .github/workflows/        # CI: check, test, clippy, dashboard build, TAS gate
```

A few decisions worth noting:

- `tracerazor-core` has zero network dependencies. All structural metrics run offline in under 5ms.
- `tracerazor-semantic` is a separate crate so the offline path never pulls in `reqwest` or async runtimes.
- The dashboard is compiled into the server binary via `include_str!`. The React build in `dashboard/` is an optional alternative for teams that want to extend the UI.
- SurrealDB runs in-memory for the CLI and `kv-surrealkv` for the server. The store API is identical in both modes.

---

## Deployment

### Docker Compose

```bash
docker compose up --build
# Server at http://localhost:8080
# Data persisted to a named volume
```

Override port or database path:

```bash
PORT=9090 docker compose up
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | none | Required for Phase 2 semantic metrics |
| `TRACERAZOR_LLM_MODEL` | `gpt-4o-mini` | Chat model for RDA and DBO |
| `TRACERAZOR_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for SRR and ISR |
| `TRACERAZOR_DB_PATH` | `./tracerazor.db` | Persistent database path |
| `PORT` | `8080` | HTTP server port |
| `TRACERAZOR_BIN` | auto-detected | Path to CLI binary (Python adapter) |

Store secrets in `.env` at the repo root. It is already in `.gitignore`.

---

## CI/CD

`.github/workflows/tracerazor.yml` runs on every push to `main`:

1. `cargo check` — workspace compiles
2. `cargo test` — 45 tests pass
3. `cargo clippy` — zero warnings
4. Dashboard build — `npm ci && npm run build`
5. **TAS gate** — audits the included sample trace, exits non-zero if TAS < 60

Replace the sample trace with your agent's most recent production trace and the gate becomes a regression test for efficiency, not just correctness.

---

## Test Coverage

| Crate | Tests |
|-------|-------|
| tracerazor-core | 20: all eight metrics, scoring, report generation, graph engine |
| tracerazor-ingest | 3: raw JSON, LangSmith, OTEL parsers |
| tracerazor-semantic | 5: BoW similarity edge cases |
| tracerazor-store | 6: traces and KB: save, retrieve, list, baseline, dashboard, delete |
| tracerazor-server | 6: audit, list, dashboard, 404, metrics, compare |
| tracerazor-proxy | 5: scope whitelist, budget injection |
| **Total** | **45, all pass** |

---

## Trace Format

```json
{
  "trace_id": "run-001",
  "agent_name": "support-agent",
  "framework": "langgraph",
  "task_value_score": 1.0,
  "steps": [
    {
      "id": 1,
      "step_type": "reasoning",
      "content": "Parse the user request about order refund",
      "tokens": 820
    },
    {
      "id": 2,
      "step_type": "tool_call",
      "content": "Fetch order details",
      "tokens": 340,
      "tool_name": "get_order_details",
      "tool_params": { "order_id": "ORD-9182" },
      "tool_success": true,
      "input_context": "full prompt sent to LLM"
    }
  ]
}
```

LangSmith exports and OpenTelemetry JSON spans are auto-detected. Pass `--trace-format langsmith|otel|raw` to override.

| Field | Required | Notes |
|-------|----------|-------|
| `id` | yes | 1-based step index |
| `step_type` | yes | `reasoning`, `tool_call`, or `handoff` |
| `content` | yes | Primary text of the step |
| `tokens` | yes | Token count for this step |
| `tool_name` | no | Required for accurate TCA |
| `tool_success` | no | `false` triggers misfire detection |
| `input_context` | no | Full LLM input, used by CCE |
| `agent_id` | no | For multi-agent traces |

---

## Framework Support

| Framework | Ingestion format | Status |
|-----------|-----------------|--------|
| LangGraph / LangChain | LangSmith JSON, OTEL | Supported |
| OpenAI Agents SDK | OTEL spans | Supported |
| CrewAI | Task logs | Parser included |
| Any OTEL-instrumented agent | OTEL JSON | Supported |
| Raw / custom | User-defined JSON | Supported |
| AutoGen | Conversation JSON | Planned |

---

## Research Foundation

The eight TAS metrics map directly to failure modes identified in peer-reviewed work from 2024-2025. The metric selection follows from the literature rather than from intuition.

| # | Paper | Metric |
|---|-------|--------|
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

The 40-70% redundancy figure is independently replicated in studies [1], [2], [5], and [8]. Study [3] establishes that trace-level evaluation is tractable from execution logs alone, without model internals or ground-truth labels.

---

## Licence

Apache 2.0. The CLI, server, dashboard, proxy, and Python integration are all open-source.

The commercial path is managed hosting with persistent storage, SSO, team management, and SLA support, following the Langfuse and PostHog model.
