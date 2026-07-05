# TraceRazor native trace format

TraceRazor ingests these formats (auto-detected, or forced with `-F`):

| Format | Flag | How to get it |
|---|---|---|
| **Native JSON** (this document) | `-F raw` | Write it directly, use the Python `Tracer`, or convert a chat log with [`tools/convert_openai.py`](../tools/convert_openai.py) |
| **LangSmith** run export | `-F langsmith` | `tools/fetch_langsmith.py`, or `client.list_runs()` dumped to JSON (flat arrays are re-treed via `parent_run_id`) |
| **OpenTelemetry GenAI** spans | `-F otel` | OTLP-JSON with `gen_ai.*` semconv attributes (OpenLLMetry et al.) |
| **Claude Code transcript** | `-F claude-code` | Local Claude Code session JSONL, or `tracerazor claude convert <session.jsonl>` |
| **Langfuse** trace export | `-F langfuse` | JSON with `observations`, `traces[].observations`, or a bare observation array |
| **Arize Phoenix** export | `-F phoenix` | Phoenix/OTel-shaped JSON; parsed through the OTel path |

The universal importer normalizes these formats and can immediately audit them:

```bash
tracerazor import run.json --from langfuse --out trace.json --audit
tracerazor import ./exports --from auto --out ./normalized --audit
```

Claude Code can be wired automatically with a local hook:

```bash
tracerazor claude install --scope local --mode coach
tracerazor claude convert ~/.claude/projects/.../session.jsonl --out trace.json
```

Plain OpenAI/Anthropic `messages` arrays are **not** a trace format — convert
them first: `python tools/convert_openai.py conversation.json -o trace.json`.

A machine-readable JSON Schema lives at
[`schemas/trace.schema.json`](../schemas/trace.schema.json).

## Shape

```json
{
  "trace_id": "run-2847",
  "agent_name": "support-agent",
  "framework": "openai",
  "steps": [
    {
      "id": 1,
      "type": "reasoning",
      "content": "Parse the refund request for order ORD-123.",
      "tokens": 180
    },
    {
      "id": 2,
      "type": "tool_call",
      "content": "lookup_order(order_id=ORD-123)",
      "tokens": 90,
      "tool_name": "lookup_order",
      "tool_params": {"order_id": "ORD-123"},
      "tool_success": true,
      "output": "Order found: blue jacket, shipped"
    }
  ],
  "task_value_score": 1.0,
  "metadata": {"task": "Process the refund for ORD-123 if eligible."}
}
```

## Trace fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `trace_id` | string | **yes** | Non-empty. |
| `agent_name` | string | **yes** | Used for store baselines and fleet grouping. |
| `framework` | string | **yes** | Free-form provenance label (`"openai"`, `"langgraph"`, …). |
| `steps` | array | **yes** | ≥ 1 step to parse; **≥ 5 steps to audit** (shorter traces are skipped with a notice). |
| `total_tokens` | integer | no | Summed from steps when absent or zero. |
| `task_value_score` | number 0.0–1.0 | no (default 1.0) | Task-completion quality; gates the TAS ceiling (×0.7 at 0.0). |
| `metadata` | object | no | Free-form. **`metadata.task` (or `goal`/`objective`) anchors the goal-oriented metrics (GAR, TPE)** — traces that carry their objective produce strictly more trustworthy path metrics. |

## Step fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | integer ≥ 1 | **yes** | Unique within the trace (duplicates are rejected). |
| `type` | string | **yes** | One of `reasoning`, `tool_call`, `handoff`, `unknown`. Invalid values are validation failures and the CLI exits `2`. |
| `content` | string | **yes** | The step's text. Steps whose content is empty or a bare tool name degrade content-derived metrics — the audit warns when > 50% of steps look like placeholders. |
| `tokens` | integer | **yes** | Input + output tokens for the step. Use real provider counts when you have them; `0` on many steps triggers the degraded-ingest warning. |
| `tool_name` | string | tool calls | |
| `tool_params` | object | no | Drives loop detection (LDI matches repeated tool+params skeletons). |
| `tool_success` | boolean | no | `false` (or `tool_error` set) feeds TCA's misfire/retry analysis. |
| `tool_error` | string | no | |
| `agent_id` | string | no | ≥ 2 distinct values enable the per-agent breakdown. |
| `input_context` | string | no | New user/environment input this step responds to. **SRR never marks a step redundant against pre-context steps when this is set** — include it for multi-turn traces or re-searches get flagged as waste. |
| `output` | string | no | Tool/step output text. |

## Validation rules

- `trace_id` non-empty; at least one step.
- Step `id`s are ≥ 1 and unique (metrics resolve steps by id).
- Every step must include `type`; values outside `reasoning`, `tool_call`,
  `handoff`, or `unknown` are rejected.
- Audits need at least five steps by default. Shorter traces parse, print a
  notice, emit no report body in text mode, and exit `0`; pass `--min-steps`
  (clamped to at least `2`) for deliberately short smoke traces.
- Exit codes: parse/validation failures exit `2`; only an explicit
  `--threshold` can produce exit `1`.
