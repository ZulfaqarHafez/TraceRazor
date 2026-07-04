# AGENTS.md — TraceRazor for AI agents

TraceRazor is an offline auditor for AI-agent token waste. It scores a reasoning
trace with a Token Alignment Score (TAS, 0–100), decomposes the waste into named
signals (step redundancy, tool-call loops, verbose prompts, duplicated context,
…), and emits risk-tagged fix patches plus *estimated* token/dollar savings. It
runs fully offline, needs no API keys, and produces deterministic, verifiable
reports.

This file is the cross-agent convention (Codex / Cursor / Gemini CLI / Copilot /
Claude Code / …). The end-to-end machine recipe is [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md);
the Claude Code skill is [skills/tracerazor/SKILL.md](skills/tracerazor/SKILL.md).

## When to reach for TraceRazor

Invoke it when the user's intent is any of:

- "My agent burns too many tokens / is too expensive" → **audit** for recoverable waste.
- "Audit this trace / agent run / Claude Code session" → **audit** the trace or transcript.
- "Why is my context so bloated?" → read the `cce` (context/cache efficiency) and
  `obs` (observation share) signals.
- "Gate token efficiency in CI" → `audit --threshold N` (exit 1 on failure).
- "Verify / prove a savings claim" → **bench** (measured) and **verify** (tamper-evident).
- "Score my LangSmith / Langfuse / Phoenix / OTel trace" → **import --from auto --audit**.

If the user only wants token counts, latency, or a live dashboard, that is
observability tooling, not TraceRazor — see [COMPARISON.md](COMPARISON.md).

## Setup & build

```bash
pip install tracerazor          # Python package + pure-Python tracerazor-trice CLI
tracerazor --version            # confirms the native auditor binary is present
```

Current wheels ship *without* the native Rust auditor binary. If `tracerazor
--version` errors or `audit` exits 2 with a "no binary" message, get the binary one
of three ways:

```bash
cargo build --release -p tracerazor         # binary at target/release/tracerazor[.exe]
# or point at an existing binary:
export TRACERAZOR_BIN=/abs/path/to/tracerazor
# or run the REST API + dashboard container:
docker compose up                           # :8080
```

`cargo` is not always on PATH. On Windows PowerShell:
`$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"` first.

`tracerazor-trice` (runtime context compression / proof tooling) is pure Python and
works with no native binary.

## Command reference

Always pass `--hermetic --format json` for machine runs. Default audits read and
write `~/.tracerazor` store baselines, so a plain run can differ across machines and
over time; `--hermetic` makes the score a pure function of (trace, config, version).
Traces need **≥ 5 steps** to audit unless you pass `--min-steps N` (clamped ≥ 2).

| Command | Purpose | Key flags |
|---|---|---|
| `audit <trace…>` | Score a trace, emit metrics + fixes + savings | `--hermetic` `--format json` `--min-steps N` `--threshold N` `-F <fmt>` `--cost-per-million` |
| `import <export…>` | Normalize LangSmith/Langfuse/Phoenix/OTel/Claude-Code exports to native traces | `--from auto` `--out <path>` `--audit` |
| `claude convert <jsonl>` | Convert one Claude Code transcript to a trace | `--out <file>` |
| `claude install` | Install the SessionEnd coach hook | `--scope local\|project\|user` `--mode coach\|passive` `--with-skill`¹ |
| `apply <fixes> --to <file>` | Append safe fix patches to a prompt file | `--dry-run` `--all` `--force` |
| `bench --before A --after B` | **Measured** token/TAS delta after re-running the agent | `--fixes <json>` `--format json` |
| `verify <report> [trace]` | Re-verify a saved/signed report or evidence bundle | (bundle: trace optional) |
| `compare <baseline> <target>` | Per-metric TAS delta + regression gate | `--regression-threshold N` `--format json` |
| `cost <trace…>` | Project monthly/annual cost at a run volume | `--runs N` `--provider <preset>` `--format json` |
| `export <trace> --bundle b.zip` | Verifiable evidence bundle (trace + signed report + weights + SHA256SUMS) | `--otel` `--webhook` `--print` |

¹ `--with-skill` is being added in this change-set — see notes.

### Exit-code contract (do not infer quality from the exit code)

- **0** — ran fine. A terrible score still exits 0. Read the JSON.
- **1** — *only* an explicit gate failed: `audit --threshold` below the bar, `compare`
  regression over threshold, or `verify` finding a **TAMPERED** report.
- **2** — bad input or usage: unreadable/invalid trace, unknown flag, missing binary.

A below-minimum-steps trace is a **skip, not a failure**: it prints a `Notice:` to
stderr and exits **0** with no report. In text mode this is easy to miss — always
check for a report body, or pass `--min-steps`.

## Programmatic checks (agents: run these)

```bash
# Rust workspace tests (add cargo to PATH first on Windows, see above)
cargo test --workspace

# Python tests
python -m pytest -q

# End-to-end sanity: audit the bundled sample and gate at 70 (scores ~83 → exit 0)
tracerazor audit traces/support-agent-run-2847.json --hermetic --threshold 70
```

## Honesty rules (verbatim policy — obey when reporting results)

- The **TAS is ordinal, not cardinal**. Compare the same project/agent over time —
  never one agent against another or against an absolute bar.
- Every token/cost/dollar figure (`savings.*`, each fix's `estimated_token_savings`)
  is a **heuristic ESTIMATE / projection**, not a measurement. `savings.monthly_runs_assumed:
  true` means the monthly/annual dollars assume a default run volume.
- Say "estimated" out loud whenever you present a number. A figure becomes real only
  after a before/after rerun measured with `tracerazor bench`, task success held constant.
- Never quote the README's illustrative $/month figures as measured, and never invent
  metrics, grades, or claims that are not in the JSON.

## Repo layout (short)

- `crates/tracerazor-{core,ingest,semantic,store,server,cli}` — Rust workspace (the auditor).
- `tracerazor/` — Python package (`Tracer`, client, sampling helpers, `tracerazor-trice`).
- `traces/` — sample traces (`support-agent-run-2847.json`).
- `schemas/` — `trace.schema.json`, `report.schema.json`¹, TRICE card schemas.
- `docs/` — [trace-format.md](docs/trace-format.md), [AGENT_GUIDE.md](docs/AGENT_GUIDE.md),
  [python_api.md](docs/python_api.md), [MCP.md](docs/MCP.md)¹, `trice_*` proof cards.
- `skills/tracerazor/SKILL.md` — the distributable Claude Code skill.
- `.tracerazor/claude-code/<session-id>/` — coach artifacts written per session.

¹ Added in this change-set — see notes.

## Gotchas

- **min-steps skip is silent in text mode**: below the floor it exits 0 with only a
  stderr `Notice:` — check for a report or pass `--min-steps`.
- **Store side effect**: non-`--hermetic` audits read/write `~/.tracerazor`; scores can
  drift across machines and runs. Use `--hermetic` for anything you report or compare.
- **Ingest quality**: check `manifest.ingest_quality` — token/content-derived metrics
  are unreliable when an exporter dropped tokens or step content (`degraded_ingest: true`
  or coverage < 1.0). Watch stderr for the degraded-ingest warning.
- **`.tracerazor/` is an artifacts dir**, not config — it holds per-session coach output
  and the store; safe to delete.
- **`docs/trice_*.{json,md,svg,tex}` are generated point-in-time proof cards** — do not
  hand-edit; they are regenerated by `tracerazor-trice`.
