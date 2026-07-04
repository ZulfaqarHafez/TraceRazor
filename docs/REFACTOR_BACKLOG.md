# TraceRazor — Cleanup & Refactor Backlog

> Working backlog for the next round. Written 2026-07-04, no changes applied yet.
> Each item is scoped, located (`file:line` where known), and tagged:
> **Effort** S(<1h) / M(half day) / L(multi-day) · **Risk** low/med/high ·
> **Status** `verified` (I confirmed it in source this round) or
> `reported` (surfaced by the code-map agents — re-confirm before acting).
>
> Suggested order: Section A (safe cleanups) → C (README split) → B (refactors,
> behind tests) → D (agent-native polish) → E (publish, needs your accounts).
> Do B refactors one at a time with `cargo test --workspace` + `pytest` green
> between each. cargo isn't on PATH: `$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"`.

## Status snapshot (green as of this round)

- Rust 302/302, Python 322 passed / 9 skipped, TRICE proof graph 17/17 verifiers pass.
- Agent-native surface shipped (uncommitted working tree): AGENTS.md, CLAUDE.md,
  llms.txt, docs/AGENT_GUIDE.md, docs/MCP.md, skills/tracerazor/, report.schema.json,
  session-start coach hook, `verify`/`list --format json`, `tracerazor-mcp`.
- **Nothing here is committed yet** — decide commit strategy before or alongside this backlog.

---

## Section A — Code cleaning (low risk, do first)

### A1. Fix `scoring.rs` metric-count comment rot — `verified`
`crates/tracerazor-core/src/scoring.rs`: lines 1 & 4 say "all fourteen metrics",
lines 183 & 270 say "all thirteen metrics". Neither states the real model (14 metrics
computed; 8 carry composite weight, 6 are diagnostics). Rewrite all four comments to the
current "8 composite + 6 diagnostic (14 computed)" language used in the README and
`docs/metric_effectiveness.md`. **S / low.** Comments only — no logic change.

### A2. Purge stale root `integrations/` residue — `verified`
Root `integrations/{anthropic-wrapper,crewai,langgraph,openai-agents,openai-wrapper,tracerazor}`
holds only `dist/*.whl|*.tar.gz` + `*.egg-info/` from an abandoned multi-package layout
(no live `.py` outside dist/egg-info/pycache). The **live** adapters are
`tracerazor/integrations/`. It's **untracked** (won't appear in clones) so this is local
hygiene, but it misleads anyone browsing the working copy. Either `rm -rf integrations/`
locally, or if any of those `PKG-INFO`/`SOURCES.txt` capture history worth keeping, move a
short note to `docs/` first. **S / low.**

### A3. Remove stray local store artifacts — `verified`
`tracerazor.db/` (a directory) sits at repo root from store-backed audit runs; it's
git-ignored via `*.db` so harmless, but delete it to keep the tree clean, and confirm
`.tracerazor/` (coach output dir) is ignored too. Add explicit `.gitignore` entries if not.
**S / low.**

### A4. Doc-rot sweep on narrative docs — `reported`
Re-confirm and fix, or stamp "point-in-time" headers on:
- `docs/external_agent_audits.md`, `docs/huggingface_agentinstruct_audit.md`: reportedly
  describe a "9-signal composite" (5 weight-0 diagnostics) pinned to tracerazor 0.5.0;
  current model is 8-composite/6-diagnostic at 1.0.3. Either regenerate the numbers or add a
  "generated under v0.5.0 scorer" banner so an agent doesn't read them as current.
- `docs/figures_manifest.json`: reportedly declares `tracerazor_version: 1.1.0` (repo is
  1.0.3). Fix the version or regenerate.
- Distinguish *generated point-in-time* proof cards (`docs/trice_*`) from *reference* docs so
  an agent doesn't quote a stale snapshot as fact — a one-line header on generated cards, or a
  `docs/README.md` index that labels each. **M / low.**

### A5. Add `.gitattributes eol=lf` for pinned text inputs — `verified` (root cause seen this round)
The TRICE install/contract/release cards hash `pyproject.toml`/`README.md` **raw bytes** with
no line-ending normalization. On Windows `core.autocrlf=true`, the worktree bytes (CRLF) can
never match an LF-generated pin, which is exactly what broke `verify-install` before I
regenerated the chain. Pin `pyproject.toml`, `README.md`, and other hashed text inputs to
`eol=lf` in `.gitattributes` (repo already has one — extend it). Prevents the recurring
CRLF/LF proof-card breakage (see commits 27da003, 328329a). **S / low.**

---

## Section B — Refactoring (structural, behind tests)

### B1. Rename/nest the top-level `benchmark` package — `verified` — **highest-value refactor**
`pyproject.toml` ships `packages = ["tracerazor", "benchmark"]`, so a generic `benchmark/`
lands in site-packages root and can be shadowed by any user project's own `benchmark/`.
`tracerazor.trice` is only a façade re-exporting ~110 names from `benchmark.trice`, so a
`tracerazor-trice` CLI can Imp­ortError on a name collision it doesn't own. Also drops
top-level `schemas/` and `examples/` into site-packages root (same pollution).
**Plan:** move the real implementation under the `tracerazor` namespace
(e.g. `tracerazor/_trice/…`), keep `tracerazor.trice` as the public façade, and repoint the
wheel to package only `tracerazor` + force-included data files. Keep `benchmark/` as a
dev-only dir excluded from the wheel (tests import from source). **L / high** (touches
imports across the trice CLI, tests, and `pyproject` force-includes; the TRICE proof cards
hash some of these paths — regenerate the card chain after, as done this round).

### B2. Split `crates/tracerazor-cli/src/main.rs` (4120 lines) into modules — `verified`
One file holds clap defs + every subcommand (audit, verify, claude, apply, bench, compare,
cost, simulate, optimize, export, list, keygen, serve). Extract into
`cli/{audit,verify,claude,apply,bench,compare,cost,simulate,optimize,export,list,keygen}.rs`
(or a `commands/` module), leaving `main.rs` as arg-parsing + dispatch. Pure code motion —
do it in one commit with no behavior change, `cargo test -p tracerazor` green before/after.
**M / med** (big diff, low logic risk).

### B3. Single source of truth for pass/fail threshold — `reported`
Python `tracerazor/_audit_client.py` recomputes `tas >= threshold` client-side, duplicating
the Rust CLI's gate logic — two places to drift. Have the client read the CLI's exit code /
`passed` field instead of recomputing. **S / med.**

### B4. Typed exception hierarchy for the Python API — `reported`
Programmatic callers currently catch a grab-bag: `RuntimeError` (with raw Rust stderr),
`FileNotFoundError`, `ImportError`, `TimeoutExpired`-wrapped `RuntimeError`, `AssertionError`.
Add `tracerazor.errors` with `TraceRazorError` base + `BinaryNotFoundError`,
`AuditError`, `BelowMinStepsError`, `VerificationError`. Note: `mcp_server.py` already defines
its own `BinaryNotFoundError` — unify on the shared one. **M / med** (public API surface —
keep old exceptions as subclasses for back-comat).

### B5. Don't construct-time explode when the binary is missing — `reported`
`Tracer(...)` calls `_find_binary()` in `__init__`, so instrumentation raises before a single
step is recorded on machines without the binary. Defer resolution to `analyse()` and raise a
typed `BinaryNotFoundError` (B4) with the launcher's recovery text. **S / med.**

### B6. Expose the audit flags the Python client hides — `reported`
`TraceRazorClient` hardcodes `audit <file> --format json --threshold N`; `--min-steps`,
`--hermetic`, `--weights`, `--enhanced`, `--store` are unreachable, so short ReAct traces
(<5 steps) dead-end in a `RuntimeError` with no way to pass `--min-steps 2`. Thread these
through, and default `analyse()` to `--hermetic` (today it silently reads/writes the local
store). **M / med** (behavior change — document the new hermetic default).

### B7. Collapse or differentiate `--mode passive|coach` — `reported`
The only code difference is the interpolated mode label in `coach.md`; both write identical
artifacts, yet help claims coach "emits richer guidance." Either make coach genuinely richer
(e.g. include the optimal-path recommendation and per-step annotations) or collapse to one
mode and drop the misleading help text. **S / low.**

### B8. Fix overloaded/misrouted CLI outputs — `reported`
- Legacy `optimize --format json` prints its summary to **stderr** while the prompt goes to
  stdout; an agent capturing stdout for JSON gets prompt text. Route the summary to stdout (or
  add `--summary-out`).
- Exit code `2` is triple-purposed (missing binary / clap usage / trice no-subcommand); can't
  distinguish "not installed" from "bad args". Consider a distinct code or a JSON error body.
  **S–M / med.**

---

## Section C — README & docs restructure

### C1. Break up the 1630-line / 85 KB README monolith — `verified` — **do early**
It's the only home for the CLI reference, REST API table, metric semantics, signing/verify,
and the Claude Code hook — so retrieval over `docs/` (71% generated proof cards) misses them.
**Plan:**
- Extract stable reference into `docs/`: `docs/cli-reference.md`, `docs/rest-api.md`,
  `docs/metrics.md` (the 8+6 table + grade scale + ordinal caveat), `docs/report-format.md`
  (pairs with the new `schemas/report.schema.json`).
- README keeps: value prop, 60-second start, the new **For AI agents** section, "how it
  compares", and links out to the extracted docs.
- Update `llms.txt` and `docs/AGENT_GUIDE.md` cross-links to the new paths.
**L / low** (mostly moving prose; verify every internal link after).

### C2. Reconcile README against this round's new surface — `verified`
The **For AI agents** section and AGENTS.md reference features added this round
(`claude install --with-skill`, `tracerazor-mcp`, `report.schema.json`, `verify`/`list
--format json`, skip-status JSON). Once committed these are real — remove any "being added"
hedging and confirm each command in README/AGENTS/AGENT_GUIDE/SKILL runs against the shipped
binary (the e2e agent truth-checked 25+; re-run after any README move). **S / low.**

### C3. Correct `docs/MCP.md` to the real tool contract — `verified` (e2e finding)
`audit_trace` returns the report **fields merged at the top level** of the payload (with
`passed`), **not** under a nested `report` key, and the param is `path` (not `trace_path`).
Make sure MCP.md's examples match the actual shape. **S / low.**

### C4. `docs/trace-format.md` completeness check — `verified` (e2e finding)
Confirm it states: each step **requires** an `id` (unique; duplicates rejected) and `type`
∈ {`reasoning`,`tool_call`,`handoff`,`unknown`} (a bad type → exit 2). The schema can't
express id-uniqueness or the ≥5-step audit floor, so these must live in prose. **S / low.**

### C5. Add a `docs/` index — `reported`
60 of ~84 docs are generated proof-card artifacts. A short `docs/README.md` (or a `## Docs`
map, complementing `llms.txt`) that separates **reference** from **generated snapshot** from
**research report** stops agents (and humans) treating a stale card as current truth. **S / low.**

---

## Section D — Finish the agent-native surface (small follow-ups from this round)

### D1. Package the skill for real distribution — `verified` (build follow-up)
`skills/tracerazor/` is in no distribution manifest. Decide: (a) commit `skills/` +
`.claude/skills/tracerazor/` (`.gitignore` already un-ignores the latter this round), and
(b) whether to force-include `skills/` in the wheel and/or submit to skills.sh so
`npx skills add ZulfaqarHafez/tracerazor` works. **S / low** (decision + manifest edit).

### D2. Make `--with-skill`'s `include_str!` a packaged asset — `verified` (build follow-up)
`main.rs` embeds the skill via `include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
"/../../skills/tracerazor/SKILL.md"))` (flagged in a code comment). That repo-relative path
won't resolve from a published crate tarball — **blocks any crates.io publish of the CLI
crate.** Copy `SKILL.md` under `crates/tracerazor-cli/` with a sync step, or embed at build
time. **S / med.**

### D3. Remaining machine-UX polish — `reported` (already did report.schema, verify/list JSON, skip-status)
Still prose-only or silently no-op: `apply` (add `--format json`: applied[]/skipped/target),
`keygen` (emit `{signing_key,verify_key}` JSON), and the silent-exit-0-empty-stdout paths in
`cost`/`simulate`/`export` when given no work (emit a `{"status":"skipped"}` object like audit
now does). Optional: accept `-` (stdin) for `audit`/`import`; a `mode`/`kind` discriminator so
single vs batch JSON is distinguishable; `--compact`/NDJSON for fleet batches. **M / low.**

### D4. HTTP-mode parity — `reported`
`TraceRazorClient` HTTP mode sends **no** `Authorization` header (can't hit a token-protected
server) and returns thinner metrics than CLI mode (only `avs`). Add bearer support and full
metric parsing so the three framework adapters work against a remote server. **M / med.**

### D5. Deterministic staleness test for the coach hook — `verified` (build follow-up)
`session-start` freshness currently falls back to file mtime; tests seed an explicit
`indexed_at` to avoid a `filetime` dev-dep. If you want mtime-path coverage too, add
`filetime` as a dev-dependency. **S / low.**

---

## Section E — Distribution / publish (needs your accounts — blocks real adoption)

These are the true blocker: **`pip install tracerazor` produces no working binary on any OS
today** (wheels are `py3-none-any`, `tracerazor/bin/` empty; the platform-wheel machinery
exists but has never run). Until one of these ships, agents must build from source.

### E1. Run the GitHub release workflow — `verified`
`release.yml` builds platform wheels (bundled binary) + standalone binaries for 4 targets +
PyPI trusted publishing, but has **never executed** (zero GitHub releases). Running it unblocks
the GitHub Action's default binary-download path and real `pip install` UX. `resolve-binary.sh`
Windows mapping and `docker-compose` tag were fixed this round. **M / — (your accounts).**

### E2. crates.io stage-1 publish — `reported`
Nothing on crates.io (`cargo install tracerazor` fails). Per `docs/trice_crates_card.md` the
publishable first stage is `tracerazor-core` + `tracerazor-semantic`. Do **D2** first (the CLI
crate can't publish with the repo-relative `include_str!`). **M / — (your accounts).**

### E3. Push the Docker image — `reported`
Multi-stage Dockerfile is sound but the image is never pushed to any registry, so
`docker run ghcr.io/...` — the easiest zero-toolchain path for an agent — doesn't exist.
Add a docker job to `release.yml`. **M / — (your accounts).**

---

## Appendix — how items were sourced

- `verified` items: I read the source/paths this round (scoring.rs comments, root
  `integrations/` contents, `main.rs` line count, `pyproject` packaging, the MCP/e2e findings,
  the CRLF/LF proof-card root cause).
- `reported` items: surfaced by the 6-agent code-map (ran without the Opus safety classifier —
  re-confirm the specific `file:line` before acting; the finding direction is reliable, exact
  line numbers may have shifted).
- Do **not** touch protected paths while cleaning: `target/`, `.git/`, `.claude/worktrees/`,
  `benchmark/trice/results/`, `dist/`, `paper/`, `.env`, version numbers (stay 1.0.3), and the
  generated `docs/trice_*` proof cards (regenerate via the `tracerazor-trice` commands, never
  hand-edit — the full regen chain is in the README's TRICE section).
