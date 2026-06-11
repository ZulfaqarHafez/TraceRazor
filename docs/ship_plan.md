# TraceRazor Ship Plan

> **Status (2026-06-11): executed.** Phases 0–4 have shipped — see
> `CHANGELOG.md` (releases v0.4.1 and v0.5.0). Highlights: Ed25519 signing
> and evidence bundles landed with the forgery suite as integration tests
> (`crates/tracerazor-cli/tests/signing.rs`), the measured case study is
> published in `docs/case_study.md`, the GitHub Action resolves prebuilt
> binaries, and 0.5.0 rationalised the composite to nine evidence-gated
> metrics. The tables below are kept as the historical plan of record; the
> "Now" scores reflect the pre-plan baseline, not the current state.

Drafted from the six-reviewer critique (academic researcher, platform engineer,
compliance officer, DX reviewer, eval engineer, technical buyer). Baseline
weighted product score: **4.6/10**. This plan targets **≥7.5/10** in 8 weeks.

The blunt diagnosis: the engine is honest, fast, and reproducible, but the
product's headline output (delete recommendations, fixes, AGF flags) is often
wrong, the advertised install path is broken, and the two ingestion adapters
buyers actually need fail on real exports. None of these are research
problems. They sequence into five phases below; every item carries an
acceptance criterion that can be turned into a test.

Effort key: **S** < 1 day · **M** 1–3 days · **L** 1–2 weeks.

---

## Phase 0 — Stop the bleeding (week 1) — trust hygiene — ✅ shipped

These cost days and currently make everything else look untrustworthy.

| # | Item | Effort | Acceptance criterion |
|---|------|--------|----------------------|
| 0.1 | **One version everywhere.** Workspace `version` is the single source; sync `pyproject.toml` (1.1.0), `Cargo.toml` (0.1.0), README banner, docker-compose (0.1.0), CHANGELOG (0.3.0). Decide the semver story once (recommend: everything → `0.4.0` truthfully, jump to 1.0 at Phase-4 release). | S | `grep -r` finds exactly one version string; `tracerazor --version` == pyproject == README. |
| 0.2 | **README dedup + repair pass.** Dual-version hero banner (L19–20); self-duplicated ASCII box (L31–36); contradictory problem paragraphs (L47–48); pasted-twice Pillar-1 blockquote (L70–71); mermaid says "14 signals", draws 13 (OBS missing) with shares that disagree with the table; "thirteen sub-metrics" vs 14; stale sample outputs (claims TAS 96.1, binary prints 85.0; 74 vs 75); CLI table missing `verify`/`list`; quickstarts referencing non-existent `traces/agent-run.json`, `trace.json`, `trace_v2.json`; internal ticket IDs "(E-05)" in user-facing help. | M | A README-snippet CI job extracts every fenced shell command and runs it against shipped files; job is green. |
| 0.3 | **Fix `_find_binary` path depth** (`tracerazor/_audit_client.py:275` walks four dirs up, missing the repo's own `target/release/`). | S | Fresh clone + `cargo build --release` + README quickstart works with no `TRACERAZOR_BIN` export. |
| 0.4 | **Exit-code contract.** `0` = pass, `1` = threshold gate failed, `2` = error (parse/IO). Gating becomes opt-in: no `--threshold` ⇒ never exit 1. | S | Integration tests for all three codes; nightly batch can distinguish broken trace from inefficient agent. |
| 0.5 | **Repo hygiene.** Move `LOOP_LOG.md` → `docs/research_log.md`; remove the `.docx` PRD from root; reconcile `benchmark/` vs `benchmarks/`; fix `examples/openai_agents` stale package name and the `../../../../.env` path; fix `publish.sh` (references non-existent `crates/tracerazor-proxy`). | S | Root contains only product files; `bash -n publish.sh` + dry-run passes. |
| 0.6 | **Regenerate stale doc tables from the current binary in CI** (`docs/external_agent_audits.md` says 55.9, `benchmarks/RESULTS.md` 55.5, binary prints 55.1; README "36–41% redundancy" is the airline subset — corpus-wide is 26.0%, report per-domain). | S | A CI step regenerates the tables and fails on drift. |

## Phase 1 — Make the verdicts right (weeks 1–3) — the product's actual job — ✅ shipped

Eval-engineer adjudication: **1 of 6 delete recommendations correct; AGF
ungrounded[] ≈ 0/10 correct.** This phase is the credibility core and the
largest single score lever (precision 3 → target ≥7).

| # | Item | Effort | Acceptance criterion |
|---|------|--------|----------------------|
| 1.1 | **Never delete the successful member of a fail→retry pair.** Delete the failure, keep the retry. (airline_task0 deleted the *successful* booking, step 14, as "85% sim" with the failed step 10; marshmallow deleted the post-fix verification run.) | M | Regression tests pinned on both adjudicated traces: step 14 KEEP / step 10 DELETE; marshmallow step 20 KEEP. |
| 1.2 | **Condition redundancy on `input_context`.** A step answering a *new* user/environment turn is never "redundant" with a pre-turn step (steps 6–7 were obligatory re-searches after the user rejected the first results). | M | airline_task0 steps 6, 7, 13 no longer flagged; existing SRR tests stay green. |
| 1.3 | **Mutating-tool guard + risk classes.** Heuristic classifier (name patterns `book/create/update/delete/exchange/send/post/write` + `tool_success`); successful mutating calls are never DELETE candidates; every fix tagged `safe` / `needs-review` / `dangerous`; `apply` refuses `dangerous` without `--force`. | M | Corpus-wide invariant test: zero successful-mutating-call deletions across all 37 shipped traces. |
| 1.4 | **AGF tokenizer rewrite.** Strip markdown (`**`, backticks) and shell/regex/awk/glob syntax classes before the literal check; same-step `input_context` counts as evidence (zip-code FP); split `kind` into `fabricated-path`/`derived-value` and drop pure syntax. Re-measure and reconcile the paper's corpus numbers (paper claims mean 0.854; measured 0.785 with a one-regex 0.0 outlier). | M | Scripted check: zero markdown/syntax artifacts in `ungrounded[]` across the 37-trace corpus; `os_2` ≠ 0.0; paper table regenerated. |
| 1.5 | **Fix generator: diagnose from the error, not boilerplate.** The `tool_schema` fix ("mark required parameters as required") was wrong on every adjudicated failure (the real cause was a $50 fee omission). Derive fix text from `tool_error` content; emit fixes as structured ops (no meta-prose — `apply` currently injects "Add to system prompt:" and debug entropy stats verbatim). | M | Adjudicated failures produce cause-specific fixes; `apply` output contains no meta-prose; `bench` runs as a post-apply validation gate. |
| 1.6 | **SRR "most similar prior" bug** — comment promises the most similar prior step, code breaks on the *first* ≥ threshold. | S | Unit test with two priors above threshold flags the more similar one. |

## Phase 2 — Make it installable and ingestible (weeks 2–5) — ✅ shipped

A buyer cannot currently install it as advertised (`pip` ships a shim that
finds no binary) nor feed it their traces (both adapters fail on real exports).

| # | Item | Effort | Acceptance criterion |
|---|------|--------|----------------------|
| 2.1 | **Wheels that work.** Wire `crates/tracerazor-py` (exists, `publish = false`) via maturin, or bundle the CLI binary into platform wheels with `[project.scripts]`. | L | Clean-room test: `docker run python:3.11 pip install tracerazor && tracerazor audit sample.json` exits 0 with no Rust toolchain. |
| 2.2 | **LangSmith adapter vs reality.** Flat `client.list_runs()` arrays must group by `trace_id`/`parent_run_id` (today `langsmith.rs:54` silently keeps only the first run); read tokens from `outputs.llm_output.token_usage` and top-level `total_tokens`, not just `extra.usage_metadata`. | M | Golden-file tests from real exports: ≥3-run export parses all runs, tokens > 0. |
| 2.3 | **OTel semconv coverage.** Accept protojson string `intValue`; `gen_ai.usage.prompt_tokens`/`completion_tokens`; message events (`gen_ai.input.messages`) and OpenLLMetry indexed attributes — content must not silently fall back to span names. | M | Golden OTLP-JSON fixture parses with tokens > 0 and real content. |
| 2.4 | **Degraded-ingest detection.** If > X% of steps carry 0 tokens or content == span name, print a loud warning and record `ingest_quality` in the report + manifest (a TAS computed on span names should never look authoritative). | S | Bad fixtures trigger the warning; manifest records it. |
| 2.5 | **Batch mode.** `audit --jsonl` / directory glob with an aggregate summary (mean/p50 TAS, worst-N list) + a documented LangSmith fetch script (`pull --langsmith` as M follow-up). | M | Fleet run over `traces/external/` produces one aggregate report. |

## Phase 3 — Make "verifiable" survive an adversary (weeks 4–6) — ✅ shipped

The compliance reviewer forged a "verified: full" report by flipping one
unsigned manifest field, and freely edited `agf`/`savings` on a verified
report. Provenance is the differentiator — it must be sound.

| # | Item | Effort | Acceptance criterion |
|---|------|--------|----------------------|
| 3.1 | **Sign the canonical report.** Digest over the full canonicalised report (signature field excluded, `analysis_duration_ms` zeroed), Ed25519 (`tracerazor keygen`, `TRACERAZOR_SIGNING_KEY`); signature stored in the manifest. `verify` checks the signature *first*; any changed field ⇒ TAMPERED. | M | The compliance reviewer's four forgery attacks (TAS edit, backend flip, agf edit, savings edit) all exit 1 — added as integration tests. |
| 3.2 | **`verify` compares the whole report** (agf, savings, fixes, summary), not just `score` + `metric_normalised`; a backend mismatch under a valid signature is TAMPERED, not "integrity-only". | M | Same forgery suite; unsigned legacy reports get an explicit "unsigned" verdict, never "full". |
| 3.3 | **Evidence bundle + store integrity.** `export --bundle` (zip: trace, report, weights, sha256 manifest, optional hash chain) for WORM handoff; append-only store mode and a retention window. | L | Bundle round-trips through `verify`; store append-only mode covered by tests. |

## Phase 4 — Prove it and launch (weeks 6–8) — ✅ shipped

| # | Item | Effort | Acceptance criterion |
|---|------|--------|----------------------|
| 4.1 | **The measured case study.** Audit → apply fixes → re-run the agent → `bench` before/after: tokens at constant pass rate, with CIs, on 3–5 tau-bench tasks. This is the only number marketing needs; until it exists, all savings remain self-admitted estimates. | L | A published table of *measured* (not projected) token deltas at unchanged pass rate. |
| 4.2 | **GitHub Action v2.** Download prebuilt release binary (works from any repo — today it `cargo build`s in the consumer's workspace); JSON-first parsing (no markdown grep, no silent score=0); wire `compare --regression-threshold`; PR comment + artifact. | M | The action gates a *different* demo repo end-to-end. |
| 4.3 | **Narrow the pitch.** README leads with Audit + Verify ("offline auditor that decomposes agent token waste, emits risk-tagged fix patches, and produces cryptographically verifiable reports"); sampling + substitutability demoted to `labs/` status. | S | One sentence, one product, above the fold. |
| 4.4 | **Server hardening for ops.** Bearer-token auth; document the `{"trace": ...}` envelope; `serve` alias. | S | Unauthenticated request → 401; docs show a working curl. |
| 4.5 | **Cut the release.** Tag, publish wheels + binaries, CHANGELOG promoted out of [Unreleased], announce with the case study. | S | `pip install tracerazor` on a clean machine reproduces the announcement demo. |

---

## Deferred (explicitly out of scope)

Hosted SaaS/dashboards, trace *capture* (stay complementary to
LangSmith/Langfuse), multi-language BoW, Sigstore keyless signing (after 3.1),
TVI/weight recalibration beyond documenting (needs more than n=37), human
annotation study for detector P/R (valuable, not ship-blocking — schedule with
4.1's API budget).

## Score projection against the review rubric

| Rubric metric (weight) | Now | After plan | Driven by |
|---|---:|---:|---|
| Finding precision (20%) | 3 | 7 | Phase 1 |
| Ingestion & integration (15%) | 2 | 7 | Phase 2 |
| Core correctness & honesty (15%) | 8 | 8 | (keep) |
| Time-to-first-value (13%) | 4 | 8 | 0.3, 2.1 |
| Provenance differentiation (12%) | 6 | 8 | Phase 3 |
| Packaging coherence (10%) | 3 | 8 | 0.1–0.5, 4.5 |
| CI/ops readiness (8%) | 4.5 | 7 | 0.4, 4.2, 4.4 |
| Scientific validity (7%) | 4 | 5 | 1.4, 4.1 |
| **Weighted total** | **4.6** | **≈7.5** | |

## Sequencing notes

- Phase 0 and Phase 1 start in parallel (hygiene is mechanical; precision is
  the long pole). Phases 2–3 overlap once 1.1–1.3 land.
- Every acceptance criterion above lands as a test in `cargo test --workspace`
  / `pytest` so "shippable" is enforced by CI, not by a checklist.
- The two hand-adjudicated traces (`gpt-4o_airline_task0`,
  `marshmallow_cursors`) become permanent regression fixtures with their
  documented step-level verdicts.
