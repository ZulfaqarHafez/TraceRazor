---
name: tracerazor
description: Audit and improve AI-agent token efficiency locally. Use when someone asks why an agent burns tokens, wants an agent-run or trace audit, needs same-workload efficiency regression checks, wants to diagnose tool loops or context bloat, or needs verified before/after savings.
license: MIT
---

# TraceRazor agent efficiency supervisor

TraceRazor captures and audits agent runs locally. It diagnoses structural waste,
proposes risk-labelled fixes, and treats savings as measured only after a
quality-preserving rerun.

## Non-negotiable reporting rules

- Token Alignment Score (TAS, 0-100) is an ordinal heuristic. Compare the same
  project or workload over time. Do not rank unrelated agents by TAS.
- Every value under `savings` and every `estimated_token_savings` value is an
  estimate until a before/after rerun is measured with `tracerazor bench`.
- Say "estimated" whenever presenting projected tokens or dollars.
- Check `manifest.ingest_quality` and token provenance. Degraded or estimated
  usage is advisory and must not drive enforcement.
- Preserve task success. Never claim an improvement from token reduction alone.

## Resolve and diagnose

1. Run `tracerazor --version`.
2. Run `tracerazor agent doctor --format json`.
3. If the binary is unavailable, install a supported platform wheel or use a
   signed standalone binary. Do not hide a missing native auditor.
4. Treat the JSON output as the source of truth.

## Trusted setup

Only install agent integration when the user or project explicitly requests it.

Preview first:

```sh
tracerazor agent install --host auto --scope project --mode coach --dry-run
```

After review, repeat without `--dry-run`. Coach mode captures and advises but
does not edit prompts, tools, settings, or working files. Do not enable
`--mode enforce` unless project policy supplies exact or approved usage data
and a task-quality verifier.

Useful lifecycle commands:

```sh
tracerazor agent status --format json
tracerazor agent uninstall --host auto --scope project
tracerazor agent run -- <agent-command>
```

## Find an input

Prefer the newest completed artifact under `.tracerazor/runs/<run-id>/trace.json`.
Otherwise use:

- Claude Code JSONL via `tracerazor claude convert <file> --out trace.json`.
- LangSmith, Langfuse, Phoenix, or OpenTelemetry export via
  `tracerazor import <file> --from auto --out trace.json --audit`.
- Native JSON conforming to `schemas/trace.schema.json`.

If MCP is available, use `latest_findings`, `audit_current_run`, or
`audit_trace` before asking the user to locate files manually.

## Audit

```sh
tracerazor audit <trace> --hermetic --format json
```

- Use `--min-steps 2` only for intentionally short traces.
- Use `--threshold N` only for an explicit same-workload gate.
- Exit 0 means the command ran or a gate passed; exit 1 means an explicit gate
  failed; exit 2 means invalid input or execution error. Read the JSON.

Report:

1. Ingest quality and whether token values are reported, estimated, or missing.
2. TAS with the ordinal caveat.
3. The lowest-scoring, highest-confidence waste signals.
4. Fixes with risk labels and estimated savings.
5. The next measurement needed to verify the recommendation.

## Preview and verify changes

Preview prompt-side fixes:

```sh
tracerazor apply <fixes.json> --to <prompt-file> --dry-run
```

Apply nothing automatically. Safe fixes still require user or project-policy
authorization. `needs_review` requires `--all`; `dangerous` requires
`--all --force`.

After rerunning the same task with its quality oracle:

```sh
tracerazor bench --before before.json --after after.json --fixes fixes.json
tracerazor verify report.json trace.json
```

Call savings measured only when token usage improves and task quality remains
non-inferior. Otherwise record the intervention as rejected.

## MCP workflow

Prefer these read-oriented tools when available:

- `doctor`
- `audit_trace` or `audit_current_run`
- `latest_findings`
- `compare_runs`
- `explain_signal`
- `preview_fix`
- `check_policy`
- `verify_evidence`

`record_validation` is an explicit write to an existing run artifact. Confirm
the run and verifier evidence before using it.
