# TraceRazor agent guide — end-to-end recipe

The single machine-oriented walkthrough for driving TraceRazor from an agent: get
the tool, get a trace, audit it, read every field of the report, act on the fixes,
and (only then) claim savings. Every command here was run against the shipped binary
(`tracerazor 1.1.0`). Numbers shift between scorer versions — the field *shape* is the
contract, not the exact value.

For the short convention see [../AGENTS.md](../AGENTS.md); for the Claude Code skill
see [../skills/tracerazor/SKILL.md](../skills/tracerazor/SKILL.md).

## Golden rules

- Always audit with `--hermetic --format json`. Without `--hermetic` the audit reads
  and writes `~/.tracerazor` store baselines, so the score depends on machine history
  and is not reproducible. JSON is the source of truth; the text/markdown output is
  for humans.
- Traces need **≥ 5 steps**. Fewer is a **skip** (exit 0, `Notice:` on stderr, no
  report) — pass `--min-steps N` (clamped ≥ 2) to audit short runs.
- The TAS is an **ordinal heuristic**; `savings.*` and `estimated_token_savings` are
  **projections**. A number is real only after `tracerazor bench`. Say "estimated".

## 0. Resolve the binary

```bash
tracerazor --version            # -> "tracerazor 1.1.0" if the native binary is present
```

If that errors, or `audit` exits 2 with a missing-binary message, the wheel shipped
without the native auditor. Fix it one of three ways:

```bash
cargo build --release -p tracerazor      # binary at target/release/tracerazor[.exe]
export TRACERAZOR_BIN=/abs/path/to/tracerazor   # or reuse an existing binary
docker compose up                        # local HTTPS dashboard; see container.md
```

Compose requires a random `TRACERAZOR_API_TOKEN` in `.env` and serves through
the loopback-only Caddy TLS gateway. Follow [container.md](container.md) to
provision and trust its local CA. A bare `docker run -p` remains intentionally
unreachable because the standalone image keeps the backend on loopback.

On Windows PowerShell, put cargo on PATH first:
`$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"`.

The pure-Python `tracerazor-trice` CLI works without the native binary.

## 1. Locate and convert the input

TraceRazor audits its **native trace JSON**. Get one per source.

### Claude Code transcripts

Sessions are stored as JSONL at:

```
~/.claude/projects/<munged-cwd>/<session-id>.jsonl
```

`<munged-cwd>` is the project's working directory with path separators replaced by
`-` (e.g. `-Users-me-proj`). Audit the JSONL directly (auto-detected) or convert it:

```bash
tracerazor claude convert ~/.claude/projects/-Users-me-proj/<session-id>.jsonl --out trace.json
tracerazor audit trace.json --hermetic --format json
```

To capture every future session automatically, install the SessionEnd hook (see §8).

### LangSmith / Langfuse / Arize Phoenix / OpenTelemetry GenAI

Use the universal importer; it normalizes and can audit in one step:

```bash
tracerazor import run.json --from auto --out trace.json --audit
```

- `--from` accepts `auto | raw | langsmith | otel | claude-code | langfuse | phoenix`.
  Auto-detection is reliable; force it only when detection is wrong.
- `--audit` writes `trace.json`, `<name>.report.json`, `<name>.fixes.json`,
  `<name>.coach.md`, and `<name>.summary.json` next to the output.
- Pass a directory (with `--out <dir>`) to batch-import a whole export folder.
- LangSmith flat run arrays are re-treed via `parent_run_id`; OTel/Phoenix parse the
  `gen_ai.*` semconv attributes; Langfuse reads `observations` / `traces[].observations`.

Plain OpenAI/Anthropic `messages` arrays are **not** a trace format — convert first:
`python tools/convert_openai.py chat.json -o trace.json`.

### Raw construction (author a native trace directly)

Required trace fields: `trace_id`, `agent_name`, `framework`, `steps`. Required step
fields: `id` (unique, ≥ 1), `content`, `tokens`. Set `metadata.task` so the
goal-oriented metrics (GAR, path entropy) have an anchor. Full field reference:
[trace-format.md](trace-format.md) and [../schemas/trace.schema.json](../schemas/trace.schema.json).

A minimal **valid, 5-step** trace (this exact JSON audits to ~92/100 [Excellent],
1 safe fix, `ingest_quality.degraded: false`):

```json
{
  "trace_id": "demo-1",
  "agent_name": "demo-agent",
  "framework": "raw",
  "task_value_score": 1.0,
  "metadata": { "task": "Look up order ORD-9 and refund it if eligible." },
  "steps": [
    { "id": 1, "type": "reasoning", "content": "The user wants a refund for order ORD-9. First look it up.", "tokens": 120 },
    { "id": 2, "type": "tool_call", "content": "get_order(order_id=ORD-9)", "tokens": 60, "tool_name": "get_order", "tool_params": {"order_id": "ORD-9"}, "tool_success": true, "output": "Order ORD-9: blue jacket, delivered 3 days ago, eligible." },
    { "id": 3, "type": "reasoning", "content": "It is within the return window and eligible, so process the refund.", "tokens": 90 },
    { "id": 4, "type": "tool_call", "content": "refund(order_id=ORD-9)", "tokens": 55, "tool_name": "refund", "tool_params": {"order_id": "ORD-9"}, "tool_success": true, "output": "Refund of $89 issued to original payment method." },
    { "id": 5, "type": "reasoning", "content": "Refund issued. Confirm to the customer and close the ticket.", "tokens": 70 }
  ]
}
```

Always verify a hand-authored trace by running the binary on it before trusting the
output:

```bash
tracerazor audit demo-1.json --hermetic --format json
```

Use **real provider token counts** in `tokens`. Many `0`-token or bare-tool-name
steps trigger the degraded-ingest warning and make token/content metrics unreliable.

## 2. Audit invocation patterns

```bash
# Canonical machine run (reproducible)
tracerazor audit trace.json --hermetic --format json

# Short real-world trajectory (ReAct runs are often 3–4 steps)
tracerazor audit trace.json --hermetic --format json --min-steps 2

# Optional project-local floor — use only when the project explicitly declares it
tracerazor audit trace.json --hermetic --threshold 75

# Preferred CI gate — same workload against its declared baseline
tracerazor compare baseline.json candidate.json --format json --regression-threshold 10

# Force a source format, override cost model
tracerazor audit export.json --hermetic --format json -F langsmith --cost-per-million 3.0

# Batch: a directory or several files -> one aggregate fleet report
tracerazor audit ./traces --hermetic --format json
```

## 3. Read the report, field by field

Field names below are verbatim from
`tracerazor audit traces/support-agent-run-2847.json --hermetic --format json`.

Top level:

- `trace_id`, `agent_name`, `framework` — echoed provenance.
- `total_steps`, `total_tokens` — headline size (11 steps / 14280 tokens in the sample).
- `analysis_duration_ms` — audit wall time.
- `summary`, `summary_oneliner` — human-readable headline. The one-liner already spells
  out "heuristic projection" and the assumed run volume — reuse that framing.

`score` — the TAS block:

- `score.score` — the TAS, 0–100 (83.1 in the sample). `score.grade` — Excellent ≥ 90 /
  Good ≥ 70 / Fair ≥ 50 / Poor.
- `score.raw_tas` — the pre-task-value score before the `task_value_score` ceiling.
- `score.task_value_score` — the task-quality multiplier carried from the trace.
- `score.passes_threshold` — whether the run met the configured `--threshold`.
- `score.avs`, `score.vae` — verbosity/anti-value diagnostics.
- `score.metric_normalised` — the per-metric 0–1 map (**1.0 = clean, lower = more
  waste**). Keys: `srr` (step redundancy), `ldi` (tool-call loops), `tca` (tool-call
  accuracy), `tur` (token utility), `cce` (context/cache efficiency), `rda` (reasoning
  depth), `isr` (info-gain / novelty), `dbo` (decision branch optimality), `vdi`
  (verbosity density), `shl` (sentence hedge/length), `ccr` (compressibility), `gar`
  (goal advancement), `csd` (context/semantic drift), `obs` (observation share). **The
  lowest entries are the biggest waste** — in the sample, `obs` 0.377, `gar` 0.403,
  `csd` 0.438.
- `score.<metric>` — a detail object per metric alongside the normalised score, e.g.:
  - `score.tca.misfires[]` — `{failed_step, retry_step, tool_name, error, wasted_tokens}`.
  - `score.cce.bloated_steps[]` — `{step_id, duplicate_pct, duplicate_tokens}` and
    `score.cce.duplicate_tokens` total.
  - `score.srr.redundant_steps`, `score.ldi.loops`, `score.tur.wasted_tokens`,
    `score.gar.low_advancement_steps`, `score.csd.high_drift_pairs`, etc.
  - each detail carries `pass` and its `target` so you can see which metrics failed.

`diff[]` — the optimal-path plan, one entry per step: `{action, step_id, step_type,
description, justification, tokens_actual, tokens_suggested}`. `action` is `keep` /
`trim` / `delete`; `justification` explains a `trim`/`delete` (e.g. "Misfired: wrong
params …, retried at step 5"). `tokens_suggested` is the post-fix estimate.

`savings` — **projected**, per the honesty rules:

- `tokens_saved`, `reduction_pct` — estimated recoverable tokens and the % of total.
- `cost_saved_per_run_usd`, `monthly_savings_usd`, `latency_saved_seconds`.
- `monthly_runs` + `monthly_runs_assumed: true` — the monthly/annual dollars assume a
  default run volume; never present them as measured.

Other blocks: `path_entropy` (`focus_score`, `path_entropy`, `advances/stalls/regresses`
— the "staying on the path" signal), `features` (raw scalar features), `agf` (action-
grounding factuality), `anomalies`, `per_agent` (populated when ≥ 2 distinct `agent_id`s
exist), `mvtg`, `iar`.

`manifest` — provenance and reproducibility. Check these before trusting a score:

- `trace_sha256`, `tool_version`, `created_at`, `similarity_backend` (`bow` unless
  `--enhanced`), `hermetic`, `min_steps`, `threshold`, `cost_per_million_tokens`.
- `weights` + `weights_sha256` — the exact composite weights used (some metrics carry
  weight 0.0 — they are diagnostics, not part of the composite).
- `ingest_quality` — `{format, token_coverage, content_coverage, step_count,
  zero_token_pct, placeholder_content_pct, degraded, degraded_ingest, warnings}`.
  **If `degraded_ingest` is true or coverage < 1.0, token/content metrics are
  unreliable** — say so.
- `signature`, `signing_key_pub` — present only for signed runs (see §7).

The report JSON validates against [../schemas/report.schema.json](../schemas/report.schema.json).

## 4. Fixes and `apply`

`fixes[]` — each fix is `{fix_type, target, patch, estimated_token_savings, risk}`.
`risk` is one of three tiers, and `apply` treats them differently:

- **`safe`** — system-prompt-only, non-functional (`hedge_reduction`,
  `verbosity_reduction`, `caveman_prompt_insert`, `reformulation_guard`, `goal_anchor`,
  `context_compression`). Applied by default.
- **`needs_review`** — may change behavior (e.g. `tool_schema` edits). Applied only
  with `--all`.
- **`dangerous`** — can suppress legitimate behavior (e.g. termination guards). Applied
  only with `--all --force`.

`apply` **appends** patches to a target prompt file. Always preview first:

```bash
# Preview only — writes nothing
tracerazor apply report.json --to system_prompt.txt --dry-run

# Apply the safe subset
tracerazor apply report.json --to system_prompt.txt

# Include needs_review; add --force to also include dangerous
tracerazor apply report.json --to system_prompt.txt --all
```

The `<fixes>` argument accepts either the full audit report or a raw `[Fix, …]` array
(the `fixes.json` written by the coach/importer). Present each applied fix with its
`risk` label and *estimated* savings — never as measured.

## 5. Measure — turn estimates into numbers (`bench`)

A savings figure is real only after re-running the agent with the fixes applied and
measuring the delta at **constant task success**:

```bash
tracerazor bench --before before.json --after after.json --fixes fixes.json --format json
```

`bench` reports measured token and TAS deltas and, when `--fixes` is supplied, compares
the measured savings against each fix's `estimated_token_savings` so you can validate
(or falsify) the recommendation. Report the measured number; drop the estimate once you
have it. See the measured [case_study.md](case_study.md), which caught one fix that cost
tokens.

## 6. Compare and cost

```bash
# Regression gate between two trace versions (exit 1 if any metric regresses > threshold)
tracerazor compare baseline.json target.json --regression-threshold 10 --format json

# Project spend at a run volume (estimate)
tracerazor cost trace.json --runs 50000 --provider anthropic-claude-3-5-sonnet --format json
```

## 7. Verify — signed, tamper-evident reports and evidence bundles

```bash
# One-time keypair for signing
tracerazor keygen        # sets TRACERAZOR_SIGNING_KEY (secret) / TRACERAZOR_VERIFY_KEY (public)

# Re-verify a saved report against its trace
tracerazor verify report.json trace.json      # exit 0 = checks passed, exit 1 = TAMPERED/mismatch

# Package a portable evidence bundle (trace + signed report + weights + SHA256SUMS)
tracerazor export trace.json --bundle bundle.zip
tracerazor verify bundle.zip                   # trace arg optional; bundle is self-contained
```

`verify` checks the signature first, then the trace hash, then re-scores hermetic
bag-of-words runs metric-by-metric. The guarantee depends on signing:

- **Signed reports** (`TRACERAZOR_SIGNING_KEY` set at audit time): any edited field —
  TAS, savings, fixes, even the similarity-backend claim — exits 1 `TAMPERED`.
- **Unsigned reports** verify at `rescore-only (unsigned)` level at best: re-scoring
  catches tampering of scored/derived fields (exit 1 `mismatch`), but an edited
  non-scored field (e.g. `total_tokens`) can still pass with exit 0. Read the reported
  level, and sign reports for any hand-off where integrity matters.

## 8. Claude Code coach artifacts

Install the SessionEnd hook once; every session is then converted, audited hermetically,
and written to disk:

```bash
tracerazor claude install --scope local --mode coach
```

- `--scope` — `local` (per-project, default) / `project` / `user`.
- `--mode` — `coach` (writes advice) / `passive`. Coach mode **never** auto-edits
  prompts, settings, tools, or files.

Per session, under `.tracerazor/claude-code/<session-id>/`:

- `trace.json` — the converted trace.
- `report.json` — the full hermetic audit (same shape as §3).
- `fixes.json` — the raw `fixes[]` array (feed straight to `apply`).
- `coach.md` — human-readable coaching writeup.
- `summary.json` — compact index record: `{trace_id, agent_name, framework, tas_score,
  grade, total_tokens, estimated_tokens_saved, fix_count, trace, report, fixes, coach,
  validated: false, validation_status: "projected_only"}`. Note `validated: false` —
  the savings are projected until you run `bench`.

`.tracerazor/claude-code/index.json` — an array of the most recent session summaries
(newest first, deduped by `trace_id`, capped at 100). Read it to find the worst recent
sessions without opening each folder.

## 9. Failure modes and recovery (exact stderr)

- **Below minimum steps** (skip, not error): exits **0**, prints to stderr
  `Notice: Trace '<id>' has <n> steps (minimum 5 required). Use --min-steps to audit
  short traces.` and emits **no report** (in JSON mode, stdout is empty). Recovery: pass
  `--min-steps 2`, or accept that the run is too short to audit.
- **Unreadable / missing input**: exits **2**, `Error: Cannot read file: <path>: The
  system cannot find the file specified. (os error 2)`. Recovery: fix the path; check
  it is JSON (or JSONL for Claude Code).
- **Invalid trace** (bad schema, duplicate step ids, empty `trace_id`): exits **2** with
  a parse/validation error naming the field. Recovery: validate against
  [../schemas/trace.schema.json](../schemas/trace.schema.json).
- **Missing native binary**: `audit` fails (exit 2) with a message that the auditor
  binary is unavailable. Recovery: build it (`cargo build --release -p tracerazor`) or
  set `TRACERAZOR_BIN` (see §0).
- **Threshold / regression gate failed**: exits **1** — this is a *gate result*, not an
  error. Read the JSON; the score is valid.
- **Degraded ingest**: exit 0, but `manifest.ingest_quality.degraded_ingest: true` and a
  stderr warning. Recovery: re-export with real token counts and step content; caveat
  the token/content-derived metrics in your writeup.
