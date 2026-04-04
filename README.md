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
- No other dependencies — all analysis runs fully offline

### Build

```bash
git clone https://github.com/ZulfaqarHafez/tracerazor
cd tracerazor
cargo build --release
# Binary at: target/release/tracerazor
```

### Audit a trace

```bash
tracerazor audit ./traces/support-agent-run-2847.json
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
VAE SCORE:         0.74
------------------------------------------------------
METRIC BREAKDOWN
Code   Metric                    Score    Target   Status
SRR    Step Redundancy Rate      18.2%    <15%     FAIL
LDI    Loop Detection Index      0.182    <0.10    FAIL
TCA    Tool Call Accuracy        83.3%    >85%     FAIL
...
------------------------------------------------------
SAVINGS ESTIMATE
Tokens saved:      7,006  (49.1% reduction)
At 50K runs/month: $1,050.90/month saved
```

### Output formats

```bash
tracerazor audit trace.json --format json      # machine-readable
tracerazor audit trace.json --format markdown  # human-readable (default)
```

### CI/CD gating

```bash
tracerazor audit trace.json --threshold 75
# Exits non-zero if TAS < 75 — blocks the PR
```

### Compare two traces

```bash
tracerazor compare baseline.json new-version.json
```

---

## Metrics Framework

TraceRazor computes a composite efficiency score from eight component metrics across three categories. Phase 1 (this release) delivers the five Rust-only structural metrics. Phase 2 adds the three LLM-powered metrics.

### Phase 1: Structural Metrics (implemented)

| Code | Metric | Weight | Formula | Target |
|------|--------|--------|---------|--------|
| **SRR** | Step Redundancy Rate | 20% | `redundant_steps / total_steps × 100` | < 15% |
| **LDI** | Loop Detection Index | 15% | `max_cycle_length / total_steps` | < 0.10 |
| **TCA** | Tool Call Accuracy | 15% | `successful_first_attempts / total_tool_calls × 100` | > 85% |
| **TUR** | Token Utilisation Ratio | 10% | `useful_output_tokens / total_tokens` | > 0.35 |
| **CCE** | Context Carry-over Efficiency | 10% | `1 − (duplicate_context_tokens / total_input_tokens)` | > 0.60 |

### Phase 2: Semantic Metrics (planned)

| Code | Metric | Weight | Detection |
|------|--------|--------|-----------|
| **RDA** | Reasoning Depth Appropriateness | 10% | LLM task complexity classifier |
| **ISR** | Information Sufficiency Rate | 10% | Embedding drift / novelty |
| **DBO** | Decision Branch Optimality | 10% | LLM retrospective judge |

### Composite Score (TAS)

The **TraceRazor Score** is a weighted composite normalised to 0–100. In Phase 1, weights are re-normalised over the five available metrics.

**Grade bands** (mirrors Google Lighthouse):
- 90–100: Excellent — agent is highly optimised
- 70–89: Good — minor inefficiencies present
- 50–69: Fair — actionable waste present
- 0–49: Poor — significant restructuring recommended

### Value-Adjusted Efficiency (VAE)

```
VAE = (task_value_score × raw_efficiency) / normalised_token_cost
```

VAE is a multiplier that prevents high-efficiency scores for agents that fail to complete tasks. A failed agent (task_value_score = 0) scores VAE = 0 regardless of token efficiency. Provide `task_value_score` in your trace JSON, or default to 1.0.

---

## Trace Format

### Raw JSON (native format)

```json
{
  "trace_id": "run-001",
  "agent_name": "my-agent",
  "framework": "langgraph",
  "task_value_score": 1.0,
  "steps": [
    {
      "id": 1,
      "type": "reasoning",
      "content": "Parse the user request...",
      "tokens": 820
    },
    {
      "id": 2,
      "type": "tool_call",
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
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | integer | ✓ | 1-based step index |
| `type` | string | ✓ | `reasoning`, `tool_call`, or `handoff` |
| `content` | string | ✓ | Primary text content of the step |
| `tokens` | integer | ✓ | Total tokens consumed |
| `tool_name` | string | — | Tool identifier (tool_call steps) |
| `tool_params` | object | — | Parameters passed to tool |
| `tool_success` | boolean | — | Whether the call succeeded |
| `tool_error` | string | — | Error message if failed |
| `input_context` | string | — | Full input context (for CCE analysis) |
| `agent_id` | string | — | Agent ID for multi-agent traces |

### LangSmith format

Pass `--trace-format langsmith` or let auto-detection handle it (the parser checks for `run_type` or `child_runs` fields). Supports nested run trees; chains are flattened.

### OpenTelemetry format

Supports OTEL JSON span exports from OpenAI Agents SDK, Semantic Kernel, and any OTEL-instrumented framework. Auto-detected via `resourceSpans` field.

---

## Architecture

TraceRazor is organised as a Rust workspace with focused crates:

```
tracerazor/
├── crates/
│   ├── tracerazor-core/      # Graph engine, metrics, scoring, reports
│   ├── tracerazor-ingest/    # Trace format parsers
│   ├── tracerazor-semantic/  # Similarity computation (BoW → ONNX)
│   ├── tracerazor-store/     # SurrealDB persistence
│   └── tracerazor-cli/       # CLI entry point
└── traces/                   # Sample trace files
```

### Layered architecture

| Layer | Responsibility | Technology | Performance |
|-------|---------------|------------|-------------|
| Ingestion | Parse traces from multiple formats | Rust (serde, custom parsers) | 10,000 traces/sec |
| Graph Engine | Construct DAG, detect cycles, compute paths | Rust (petgraph) | Sub-millisecond |
| Structural Analyser | SRR, LDI, TCA, CCE, TUR computation | Rust | Sub-millisecond |
| Graph Storage | Persist trace graphs, historical benchmarking | SurrealDB (embedded) | Write < 10ms |
| Semantic Analyser | RDA, ISR, DBO (Phase 2) | Async LLM API | Batched, < 2s |
| Scoring Engine | Weighted composite score, grade, trend | Rust | Microsecond |
| CLI | Commands: audit, list, compare | Rust + clap | Negligible overhead |

### Core principles

1. **Rust engine handles all compute-intensive work** — graph construction, structural analysis, scoring. Zero external dependencies beyond `petgraph` and `serde` in `tracerazor-core`.

2. **LLM calls reserved for semantic tasks only** (Phase 2) — redundancy verification, task complexity classification, branch optimality judgement.

3. **Framework agnostic** — TraceRazor never calls your agent or LLM. It reads the trace after execution.

4. **Fully offline** — Phase 1 structural analysis requires no API keys, no internet, no model downloads.

### Multi-agent traces

In multi-agent traces, each agent is assigned to a named thread within the DAG. Steps within a thread are connected sequentially; handoffs between agents are cross-thread edges. Each thread receives its own TAS score; the overall score is a token-weighted average.

---

## Detection Algorithms

### SRR — Step Redundancy Rate

**Phase 1 (BoW cosine similarity):**
1. Tokenise each step's content (lowercased, stop words removed, n > 1 chars)
2. Build TF (term-frequency) vectors
3. Compute cosine similarity between every step pair (O(n²))
4. Flag pairs above the 0.65 threshold (calibrated for BoW; equivalent to 0.85 with sentence embeddings)

**Confidence tiers:**
- High (≥ 0.85): Almost certainly redundant — shown in all modes
- Medium (0.65–0.84): Likely redundant — shown in default mode
- Low (0.55–0.64): Possibly redundant — shown in `--verbose` mode only

**Phase 2 upgrade:** Swap BoW for `all-MiniLM-L6-v2` sentence embeddings via ONNX Runtime (`ort` crate). Raises threshold back to PRD-specified 0.85. Handles paraphrases and semantic near-duplicates that BoW misses.

### LDI — Loop Detection Index

Two complementary methods:
1. **State hashing**: Hash each tool call step by `(tool_name, params)`. Repeated hashes identify the same tool called with identical parameters.
2. **Sequence repeat detection**: Sliding window of length 2–5 looking for consecutive repeated patterns.

Uses `kosaraju_scc` from `petgraph` for SCC (strongly connected component) detection. Loop start steps are kept in the optimal path; repeat occurrences are deleted.

### TCA — Tool Call Accuracy

Pattern matching: scan for `tool_call → failure → retry` sequences using `tool_success: false` or non-null `tool_error` fields. Each misfire counts as one failed first-attempt. Wasted tokens = failed call tokens + retry tokens.

### TUR — Token Utilisation Ratio

Heuristic attribution: steps flagged as REDUNDANT or LOOP contribute 0 useful tokens; OVER_DEPTH steps contribute 30%; CONTEXT_BLOAT steps contribute 50%; clean steps contribute 100% of their tokens.

### CCE — Context Carry-over Efficiency

4-gram overlap ratio between each step's `input_context` (or `content` as fallback) and the cumulative text of all prior steps. Steps with > 40% overlap are flagged as context bloat. CCE = 1 − (duplicate_tokens / total_input_tokens).

---

## Phasing

### Phase 1: CLI Auditor (this release, Weeks 1–6)

- [x] Rust workspace with 5 crates
- [x] LangSmith JSON + raw JSON + OTEL ingestion
- [x] SRR, LDI, TCA, TUR, CCE structural metrics
- [x] TAS composite score + VAE multiplier
- [x] JSON and Markdown report output
- [x] CLI: `audit`, `list`, `compare` commands
- [x] CI/CD gating via `--threshold` flag
- [x] SurrealDB in-memory store for within-session historical benchmarking
- [x] 23 unit tests across all crates
- [x] Sample trace: `traces/support-agent-run-2847.json`

### Phase 2: Semantic Analysis and Integrations (Weeks 7–12)

- [ ] ONNX Runtime (`ort`) + all-MiniLM-L6-v2 bundled embeddings for true SRR
- [ ] LLM-powered metrics: RDA, ISR, DBO (Anthropic + OpenAI API)
- [ ] Callback adapters: LangGraph, CrewAI, OpenAI Agents SDK
- [ ] Real-time loop detection with configurable abort
- [ ] SurrealDB `kv-surrealkv` for persistent cross-session storage
- [ ] GitHub Action for CI/CD gating
- [ ] PyO3 Python bindings (`pip install tracerazor`)

### Phase 3: Dashboard and Proxy (Weeks 13–20)

- [ ] Web dashboard (React + Axum backend) with aggregate analytics
- [ ] Proxy layer with active prompt rewriting and guardrail system
- [ ] Known-good-paths knowledge base (SurrealDB graph storage)
- [ ] Historical benchmarking and trend analysis
- [ ] Team and project management features
- [ ] WASM compilation target for browser/TypeScript usage

---

## Questions to Address Before Phase 2

**API keys needed for Phase 2:**
- `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) for LLM-powered metrics (RDA, ISR, DBO)
- No API key needed for Phase 1 — fully offline

**Open questions:**
- Should the Python SDK expose the full API or a simplified high-level interface?
- What's the preferred deployment for the Phase 3 hosted dashboard (self-hosted Docker vs managed cloud)?
- Should the GitHub Action produce PR comments with the report inline, or post to a separate status check?

---

## Research Background

TraceRazor's design is grounded in the following research:

| # | Paper | Relevance |
|---|-------|-----------|
| [1] | Han et al. (2024). **Token-Budget-Aware LLM Reasoning (TALE)**. ACL 2025. | Budget-constrained reasoning; motivates TUR and CCE metrics |
| [2] | Zhao et al. (2025). **SelfBudgeter: Adaptive Token Allocation**. | Adaptive budget injection; informs Phase 3 proxy layer |
| [3] | Lee et al. (2025). **Evaluating Step-by-step Reasoning Traces: A Survey**. | Taxonomy of reasoning trace quality metrics; basis for 8-metric framework |
| [4] | Su et al. (2024). **Dualformer: Controllable Fast and Slow Thinking**. | Fast vs slow reasoning modes; informs RDA metric design |
| [5] | Wu et al. (2025). **Step Pruner: Efficient Reasoning in LLMs**. | Step pruning algorithms; basis for optimal path recommendation engine |
| [6] | Feng et al. (2025). **Efficient Reasoning Models: A Survey**. | Comprehensive survey of token efficiency techniques; validates metric selection |
| [7] | Pan et al. (2024). **ToolChain*: A* Search for Tool Sequences**. NeurIPS 2024. | Graph-based optimal tool sequence search; informs DBO metric and path recommendations |
| [8] | Hassid et al. (2025). **Reasoning on a Budget: Survey of Adaptive Test-Time Compute**. | Budget adaptation strategies; informs VAE scoring and Phase 3 proxy design |
| [9] | (2025). **Balanced Thinking: CoT Training in VLMs (SCALe-SFT)**. | Avoids over-reasoning; validates that efficiency doesn't require accuracy sacrifice |
| [10] | Mohammadi et al. (2025). **Evaluation and Benchmarking of LLM Agents**. KDD 2025. | Agent evaluation methodology; validates composite scoring approach |

**Key insight from the research:** Studies [1], [2], [5], [8] all show that 40–70% of reasoning tokens are redundant in typical CoT traces, validating TraceRazor's core premise. Study [3] confirms that step-by-step trace evaluation is a tractable problem from trace data alone (no model internals needed), directly supporting the PRD's metric selection rationale.

---

## Compatibility

| Framework | Trace Format | Status |
|-----------|-------------|--------|
| LangGraph / LangChain | LangSmith JSON, OTEL | P0 — implemented |
| CrewAI | Task logs | P0 — parser included |
| OpenAI Agents SDK | OTEL spans | P0 — OTEL parser included |
| AutoGen | Conversation JSON | P1 — planned Phase 2 |
| Semantic Kernel | OTEL spans | P2 — planned Phase 2 |
| Raw / Custom | User-defined JSON | P0 — implemented |
| OpenTelemetry (generic) | OTEL JSON/protobuf | P0 — implemented |

---

## Licence

Apache 2.0. See [LICENCE](LICENCE).

All Rust crates, the CLI, Python SDK, WASM bindings, and the web dashboard are open-source. Monetisation path (future): managed cloud hosting (TraceRazor Cloud) with SLA, SSO, and team management — the open-source tool is the funnel, the hosted service is the product (mirrors Langfuse and PostHog model).
