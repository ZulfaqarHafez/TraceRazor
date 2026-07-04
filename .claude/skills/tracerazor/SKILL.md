---
name: tracerazor
description: Audit AI-agent token waste and cost, fully offline with no API keys. Use this when someone says their agent burns too many tokens, asks to audit an agent run, a trace, or a Claude Code session, wonders why their context is so bloated, or wants to cut the token spend of an LLM app. TraceRazor scores a reasoning trace with a Token Alignment Score (0-100) and emits concrete fix patches for redundant steps, tool-call loops, verbose prompts, and duplicated context. It ingests Claude Code transcripts and exports from LangSmith, Langfuse, Arize Phoenix, OpenTelemetry GenAI, CrewAI, LangGraph, and OpenAI Agents, then reports per-metric waste plus estimated token and dollar savings you can validate with a before/after benchmark.
when_to_use: Trigger phrases include "my agent burns too many tokens", "audit this trace / agent run", "why is my context so bloated", "reduce my agent's token spend", "audit a Claude Code session", and "score my LangSmith, Langfuse, Phoenix, or OpenTelemetry trace". Also fires when the user wants fix patches to cut redundant steps, loops, or verbose prompts, or wants to gate token efficiency in CI.
license: MIT
---

# TraceRazor — audit AI-agent token waste (offline)

TraceRazor scores an AI-agent reasoning trace and returns concrete, low-risk fixes
for the token waste it finds. It runs fully offline and needs no API keys. Every
step below drives the `tracerazor` CLI; treat its JSON output as the source of truth.

## Honesty rules (obey when reporting results)

- The Token Alignment Score (TAS, 0-100) is an **ordinal heuristic**, not a physical
  quantity. Compare the same project/agent over time — never one agent against another
  or against an absolute bar.
- Every token, cost, and dollar figure (`savings.*`, each fix's
  `estimated_token_savings`) is a **heuristic ESTIMATE / projection**, not a
  measurement. `savings.monthly_runs_assumed: true` means the monthly/annual dollars
  assume a default run volume.
- Say "estimated" out loud whenever you present a number. A figure becomes real only
  after a before/after rerun measured with `tracerazor bench`, task success held constant.
- Never invent metrics, grades, or claims that are not in the JSON.

## Step 1 — Resolve the tool

1. Run `tracerazor --version`. If it prints a version, use it.
2. If missing or it exits 2: `pip install tracerazor` (the wheel may ship without the
   native binary).
3. Else point the `TRACERAZOR_BIN` env var at an existing binary.
4. Else build from source:
   ```sh
   git clone https://github.com/ZulfaqarHafez/tracerazor && cd tracerazor
   cargo build --release -p tracerazor    # binary at target/release/tracerazor[.exe]
   ```
The pure-Python `tracerazor-trice` CLI (runtime context compression) works without the
native binary if you only need TRICE.

## Step 2 — Identify the input

- **Claude Code session**: transcript JSONL at
  `~/.claude/projects/<munged-cwd>/<session-id>.jsonl`. Audit the `.jsonl` directly
  (auto-detected) or convert first: `tracerazor claude convert <file> --out trace.json`.
- **LangSmith / Langfuse / Arize Phoenix / OpenTelemetry GenAI exports**:
  `tracerazor import <file> --from auto --out trace.json --audit`.
- **Raw / native trace**: hand-authored JSON per `schemas/trace.schema.json` and
  `docs/trace-format.md`.

## Step 3 — Audit

```sh
tracerazor audit <input> --hermetic --format json
```

- `--hermetic` makes the score a pure function of (trace, config, version) — no local
  store reads/writes, so it is reproducible. Use it for anything you report.
- Add `--min-steps 2` for short runs (audits need >=5 steps by default; clamped >=2).
- Add `--threshold N` only to gate CI.
- **Exit codes: 0 = ok, 1 = `--threshold` gate failed, 2 = bad input.** A low score
  still exits 0 — read the JSON; do not infer quality from the exit code.

## Step 4 — Interpret the JSON

Real field names (verified against the binary):

- `score.score` — the TAS (0-100). `score.grade` — Excellent >=90 / Good >=70 /
  Fair >=50 / Poor. `score.raw_tas` is the pre-task-value score.
- `score.metric_normalised` — per-metric 0-1 scores (1.0 = clean, lower = more waste).
  Keys include `srr` (step redundancy), `ldi` (tool-call loops), `tca` (tool-call
  accuracy), `tur` (token utility), `cce` (context/cache efficiency), `rda` (reasoning
  depth), plus `isr`, `ccr`, `obs`, `dbo`, `vdi`, `shl`, `gar`, `csd`. Each metric also
  has a detail object at `score.<metric>` (e.g. `score.tca.misfires`,
  `score.cce.bloated_steps`).
- `fixes[]` — each has `fix_type`, `target`, `patch`, `risk`
  (`safe` / `needs_review` / `dangerous`), and `estimated_token_savings`.
- `savings` — projected `tokens_saved`, `reduction_pct`, `cost_saved_per_run_usd`,
  `monthly_savings_usd` (plus `monthly_runs` / `monthly_runs_assumed`). Estimates, per
  the honesty rules.
- `manifest` — provenance: `trace_sha256`, `tool_version`, `similarity_backend`,
  `hermetic`, `min_steps`, `ingest_quality`, and optional `signature`. Re-verify a
  saved report with `tracerazor verify report.json trace.json`.
- `total_steps`, `total_tokens`, `summary_oneliner` — headline context.

Watch stderr for a degraded-ingest warning, or a low `manifest.ingest_quality`
coverage: token- and content-derived metrics are unreliable when the exporter dropped
tokens or step content. Check the format flag (`--from` / `-F`) before trusting scores.

## Step 5 — Act

- Present the top waste signals (lowest `score.metric_normalised` entries) and the
  `fixes[]`, each with its `risk` label and *estimated* savings.
- Apply prompt-side fixes to a prompt file, previewing first:
  ```sh
  tracerazor apply <fixes.json> --to <prompt-file> --dry-run
  ```
  Safe fixes apply by default; `needs_review` requires `--all`; `dangerous` requires
  `--all --force`.
- Measure real deltas before claiming any savings:
  ```sh
  tracerazor bench --before before.json --after after.json --fixes fixes.json
  ```

## Claude Code integration

Audit every session automatically:

```sh
tracerazor claude install --scope local --mode coach
```

This writes `.tracerazor/claude-code/<session-id>/{trace,report,fixes,summary}.json`
plus `coach.md`, and an `index.json`, once per session. Coach mode never auto-edits
prompts, settings, tools, or files.

For deeper detail see `docs/trace-format.md` and `schemas/trace.schema.json`.
