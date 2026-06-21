# Changelog

All notable changes to TraceRazor are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [1.0.2] - 2026-06-21

### Added
- Publish TRICE live-suite library surfaces: command adapter profiles, run
  receipts, evidence bundles, schema validation, and regenerated LaTeX/PDF
  research artifacts.
- Add broad bundled live smoke evidence with six fresh-workspace tasks,
  verified bundles, and an explicit S-tier gate that refuses local-only claims.

### Changed
- Ship TRICE schemas and runnable examples in the Python package so PyPI and
  piwheels installs include the live evaluation contract.
- Align package metadata with the public PyPI line after the existing 1.0.1
  release.

### Scoring credibility (breaking: TAS values shift)
- **TUR demoted from the composite to diagnostics** (weight 0.10 → 0;
  non-zero weights now sum to 0.82). Unlike the five variance-based 0.5.0
  demotions, this one is on logical grounds: TUR's "useful tokens" are
  defined as tokens in steps *not already flagged* by SRR/LDI/TCA, so it
  re-aggregated signal the composite already carries (double-counting),
  and its 0.70 normalisation divisor was never calibrated. The detector,
  per-step utilisation breakdown and report row all still run. New
  post-normalisation shares: SRR 20.7%, LDI/TCA 15.9%, RDA/ISR/CCE 12.2%,
  OBS 7.3%, CCR 3.7%. `benchmark/RESULTS.md` regenerated (corpus mean
  74.1); README tables, mermaid and sample output refreshed from the
  current binary.
- **TCA retries are now matched by tool name** — previously *any* next
  tool call after a failure was attributed as "the retry", so a
  legitimate pivot to a different tool was mislabelled (wrong Retry flag,
  wrong annotation text, pivot tokens charged as waste). A retry is now
  the next call to the *same tool* within the next 3 tool calls; pivots
  are annotated as "no same-tool retry (agent pivoted or abandoned)".
  The score itself is unchanged (every failed call was and is a misfire);
  a dead, condition-shadowed second detection pass was removed. Two new
  unit tests pin the pivot and the gap-retry cases.
- **The 50,000 runs/month assumption is no longer silent** —
  `SavingsEstimate` now carries `monthly_runs` + `monthly_runs_assumed`
  (serialized only when set, so previously signed reports still verify),
  and every rendered surface (savings table, executive summary,
  one-liner) says "at an ASSUMED 50000 runs/month — illustration, not
  your bill" when the volume was defaulted rather than supplied.

### Adoption: get your own traces in, trust the server's numbers
- **`tools/convert_openai.py`** — converts plain OpenAI/Anthropic
  chat-completions logs (`{"messages": [...]}` or a bare messages array,
  including `tool_calls`/`tool_use`/`tool_result`) into native traces; the
  first user message becomes `metadata.task` so GAR/TPE anchor on the real
  goal. Token fallback is `len/4` when usage is absent.
- **Chat logs no longer dead-end** — feeding a `messages` payload to
  `audit` now explains what it is and points at the converter,
  `docs/trace-format.md` and `-F langsmith` / `-F otel`, instead of
  failing with a bare `missing field trace_id`.
- **`docs/trace-format.md` + `schemas/trace.schema.json`** — the native
  schema is now documented outside a Rust doc comment, with the
  field-level notes that change scores (`input_context` redundancy
  exemption, `metadata.task` goal anchoring, degraded-ingest triggers).
- **Server scoring is now explainable and CLI-reproducible** —
  `POST /api/audit` accepts `"hermetic": true` (no store reads/writes;
  verified to produce the exact CLI `--hermetic` TAS) and every response
  carries the provenance `manifest`, so server-vs-CLI deltas are always
  attributable to recorded store baselines instead of being mystery drift.
- **README: 60-second start** at the very top (pip / cargo install /
  docker, the bundled sample, the CI gate) plus the exit-code contract
  and trace-acquisition pointers; the Python quickstart's pinned sample
  numbers were replaced with a shape contract (they drifted every scorer
  release).
- **`publish.sh` ships the binary again** — the manual path now builds the
  sdist and the platform wheel (via `scripts/build_platform_wheel.sh`)
  instead of a binary-less wheel, and says plainly that multi-platform
  publishing is `release.yml`'s job.

### Hardening
- **Weights files are validated on load** — a negative, non-finite or
  zero-sum weight set is rejected with a clear error (exit 2) instead of
  producing a NaN TAS, which crashed fleet mode's sort. `Weights::validate()`
  also runs inside `analyse()` for library callers, and fleet sorting now
  uses `total_cmp` so it can never panic on NaN.
- **LangSmith ingest depth-limits the run tree** (128 levels) — a malformed
  `parent_run_id` chain in an export now fails with a clean parse error
  instead of overflowing the stack during tree rebuild/flatten.
- **Raw-JSON ingest rejects duplicate step ids** — metrics and fixes resolve
  steps by id and previously bound silently to the first match.
- **One-sided signature manifests now verify as TAMPERED** — a report
  carrying `signing_key_pub` without `signature` (or vice versa) fails
  verification (exit 1) instead of downgrading to the "unsigned" verdict;
  covered by two new strip-field forgery tests in `tests/signing.rs`.
- Ship plan (`docs/ship_plan.md`) marked executed — Phases 0–4 all shipped
  (releases v0.4.1, v0.5.0); the doc previously still read as future work.

## [0.5.0] - 2026-06-10

### Metric rationalisation: an evidence-gated composite (breaking: TAS values shift)
- **Five metrics demoted from the composite to detection-only diagnostics**
  — GAR, CSD (range-broken: max 0.62/0.68 ever observed on real traces),
  DBO (near-constant, r=0.76 with TCA), VDI (sd 0.038, zero correlation
  with the composite) and SHL (sd 0.033). A self-evaluation over **61 real
  traces** with pre-stated criteria (sd ≥ 0.05; max ≥ 0.80; |r| < 0.85 vs
  kept metrics) made the calls; evidence committed as
  `docs/metric_effectiveness.md`, regenerated by
  `python -m benchmark.metric_effectiveness`.
- **Nothing is deleted**: all five metrics still compute, render in a new
  "Diagnostics" report section, annotate steps, drive fixes (GAR still
  powers `goal_anchor`) and feed the AVS verbosity alert. A weights file
  re-enables any of them; the calibration tool still fits all fourteen.
- **TAS values shift upward** (corpus mean 73.5 → 78.4) because two
  range-broken metrics had subtracted a near-constant penalty from every
  trace. TAS remains ordinal: compare within one project over time.
  `benchmark/RESULTS.md` and the audit docs are regenerated; the live
  case-study tables in `docs/case_study.md` record TAS deltas measured
  under the 0.4 scorer (their token numbers are scorer-independent).
- Composite weights now sum to 0.92 pre-normalisation; new shares: SRR
  18.5%, LDI 14.1%, TCA 14.1%, RDA/ISR/TUR/CCE 10.9% each, OBS 6.5%,
  CCR 3.3%. README, paper (new §"Metric rationalisation") and all docs
  updated.

## [0.4.1] - 2026-06-10

### Release plumbing
- **PyPI publishing wired into the release workflow** — on tag push,
  `release.yml` now uploads the built wheels to PyPI via trusted publishing
  (OIDC, `pypa/gh-action-pypi-publish`; no long-lived token in the repo).
  One-time setup: register this repository as a trusted publisher for the
  `tracerazor` project on pypi.org (workflow `release.yml`, environment
  `pypi`). `publish.sh` remains the manual fallback.

### Documentation, paper and housekeeping
- **Paper reworked around the measured findings** (`paper/tracerazor.tex`) —
  new "measured intervention study" section reporting both rounds (the
  −5.6% negative result, the diagnosis, the rewritten patch, the +0.7%
  re-measurement) plus the cache-warmth accounting hazard; abstract,
  contributions, related work (cost-aware agent evaluation), fix-generation,
  verification (Ed25519 signing + bundles), limitations (study scope,
  coding-agent metric priors), reproducibility (test counts, committed
  dataset) and conclusion updated; five references added. Prior calibration,
  AgentInstruct and sampling content kept as-is — still accurate.
- **README visualized** — Mermaid product-loop diagram (audit → measure →
  verify) replaces the ASCII hero; new "Measure" section with the
  apply→re-run→bench sequence diagram and both case-study rounds; crate
  data-flow diagram in Architecture; layout tree now includes the
  measurement kit.
- **License standardized** — copyright unified to
  "(c) 2025-2026 Zulfaqar Hafez" across LICENSE, README and
  docs/python_api.md (was 2024 there); `dashboard/package.json` gains its
  missing `"license": "MIT"` field.
- **Dead-code audit: clean** — a full sweep (scripts, dashboard, calibration,
  benchmark tools, traces, Python modules, Rust crates, workflows, docs)
  found every file referenced by CI, tests or docs; nothing to delete.

### The live measured case study (docs/case_study.md)
- **Live case study executed** — 24 real agent runs (Claude Code headless,
  Haiku 4.5) over 6 pytest-verified Python tasks × 2 replicates, audit →
  `apply` → re-run per pair, measured with the bootstrap-CI harness at
  constant 12/12 pass rate. Total spend ≈ $1.30. Round 1 measured the
  shipped `goal_anchor` patch at **−5.6% tokens (a cost, not a saving)**;
  the harness's estimate-accuracy check surfaced it (−102%).
- **`goal_anchor` patch rewritten** from the measured evidence: the old
  wording told the agent to restate the objective before every reasoning
  step — a per-turn standing cost that exceeded recovered drift on
  on-track runs. The anchor now forbids restating ("do not restate the
  objective or summarise progress unless explicitly asked") while keeping
  the skip-non-advancing directive. Detection and the conservative
  estimate are unchanged.
- **Claude Code transcript converter** (`benchmark/convert_claude_code.py`)
  — turns any Claude Code session transcript into an auditable TraceRazor
  trace: per-message usage grouped by API message id, tool results joined
  to tool calls, marginal token accounting with cache reads and the
  (cache-warmth-dependent, ±22k observed) first-turn prefix encoding
  excluded and the convention stamped in trace metadata.
- **Live-run kit** (`benchmark/live/`) — task suite with objective pytest
  outcomes, headless runner with a tight tool envelope identical across
  conditions, per-pair audit-and-apply orchestration, and transcript
  reconversion so converter improvements never require re-running agents.

## [0.4.0] - 2026-06-10

### Ship-plan Phase 4 (prove it and launch)
- **Measured case-study harness** (`benchmark/case_study.py`) — turns
  before/after trace pairs into a published table of *measured* token deltas
  with seeded bootstrap 95% CIs, and refuses to call a delta a "saving" on
  any task whose pass flag flipped (constant-pass-rate requirement, exit 1).
  Methodology + status in `docs/case_study.md`. (The live run landed
  post-0.4.0 — see Unreleased above.)
- **GitHub Action v2** — downloads a prebuilt release binary (works from any
  repo, no Rust toolchain; build-from-source is explicit opt-in), parses the
  report JSON (a malformed/empty report fails the step instead of silently
  scoring 0), wires `compare --regression-threshold` as a second gate
  (`baseline-trace` input), posts a sticky PR comment, uploads the JSON
  report as an artifact. Logic lives in shell scripts under
  `.github/actions/tracerazor/`, exercised locally against the binary.
- **Release binaries** — `release.yml` now builds standalone CLI tarballs
  (linux x86_64, macOS arm64/x86_64) next to the wheels and attaches both to
  the GitHub release on tag push; the action's download path consumes them.
- **Server hardening for ops** — `TRACERAZOR_API_TOKEN` enables bearer-token
  auth on every `/api` route and `/ws` (constant-time compare; missing/wrong
  token → 401; health probes stay open); the server warns when binding
  non-loopback unauthenticated. New `tracerazor serve` CLI alias (the server
  crate is now also a library). The `{"trace": ...}` envelope and a working
  curl are documented in the README.
- **README narrowed to one product** — hero leads with Audit + Verify
  ("offline auditor … cryptographically verifiable reports"); sampling and
  substitutability demoted to explicit `Labs (experimental)` status; new
  Verify section documents `keygen`/signing/`verify`/`--bundle`; CLI table
  gains `verify`-bundle, `keygen`, `serve`.

### Ship-plan Phase 3 (adversary-proof verification)
- **Ed25519 report signing** — `tracerazor keygen` generates a keypair;
  with `TRACERAZOR_SIGNING_KEY` set, every audit signs the canonical report
  (`analysis_duration_ms` zeroed, signature fields excluded) and embeds the
  signature + public key in the manifest. `verify` checks the signature
  *first*: any edited field — TAS, AGF, savings, fixes, summary, even the
  `similarity_backend` claim — exits 1 TAMPERED. The compliance reviewer's
  four forgery attacks are pinned as integration tests
  (`crates/tracerazor-cli/tests/signing.rs`).
- **Whole-report verification** — re-score compares AGF score, savings,
  fix count, and summary in addition to TAS + every normalised metric.
  Unsigned reports get an explicit `rescore-only (unsigned)` verdict, never
  "full"; only signed + reproduced reports earn
  `full (Ed25519-authenticated + reproduced …)`.
- **Evidence bundles** — `tracerazor export <trace> --bundle evidence.zip`
  packs trace + signed report + weights + SHA256SUMS for WORM hand-off;
  `tracerazor verify evidence.zip` checks bundle integrity then runs the
  full signature/hash/re-score chain (no separate trace argument).
- **Deterministic canonical bytes** — LDI loop output is sorted (HashMap
  iteration order leaked into the report), and signing normalises floats
  through a JSON round-trip so sign-time and verify-time serialisations
  agree byte-for-byte.

### Ship-plan Phase 2 (installable + ingestible)
- **Platform wheels with the bundled CLI** — `scripts/build_platform_wheel.sh`
  builds a wheel carrying the Rust binary at `tracerazor/bin/`; a new
  `tracerazor` console script (`tracerazor/_launcher.py`) and the Python
  client prefer the bundled binary, so `pip install <wheel>` delivers a
  working auditor with no Rust toolchain (clean-room smoke test in the new
  `release.yml` workflow, linux+macos matrix). Dev-status classifier
  corrected to Beta.
- **LangSmith adapter vs real exports** — flat `client.list_runs()` arrays now
  rebuild the run tree from `parent_run_id` and keep **every** run (previously
  only the first survived, silently); tokens are read from run-level
  `total_tokens`/`prompt+completion`, `outputs.llm_output.token_usage`, and
  `outputs.usage_metadata`, not just `extra`. Golden-file tests.
- **OTel semconv coverage** — spec-compliant protojson string `intValue`
  parsing; `gen_ai.usage.prompt_tokens`/`completion_tokens`; content from
  message events (`gen_ai.user.message`/`gen_ai.choice`), structured
  `gen_ai.input/output.messages`, and OpenLLMetry indexed attributes —
  content no longer silently falls back to span names. Golden-file tests.
- **Degraded-ingest detection** — `IngestQuality` (zero-token share,
  placeholder-content share) computed on every audit, recorded in the run
  manifest, with a loud stderr warning when either exceeds 50%: a TAS
  computed over span names never looks authoritative again.
- **Batch/fleet mode** — `tracerazor audit <DIR>` (or multiple files) audits
  hermetically per file and emits one aggregate report (mean/median TAS,
  worst-5, recoverable-token sum; JSON or markdown); `--threshold` gates the
  mean. Plus `tools/fetch_langsmith.py` for one-command project export.

### Ship-plan Phase 1 (verdict precision)
- **Responsiveness rules in SRR** — a similar pair is exempt when (1) new
  external input arrived at or between the pair (a step answering a new user
  turn is never redundant with a pre-turn step), (2) it is a fail→retry of the
  same tool (the retry is the productive member; TCA already penalises the
  failure), or (3) both are successful tool calls with an intervening
  state-changing step (re-running a check after an edit is verification).
  On the reviewer-adjudicated airline trace this took delete-recommendation
  precision from 1/6 correct to 6/6; corpus airline SRR 35.9%→15.5%.
- **Verification-aware LDI** — a state-hash repeat after an intervening
  mutation restarts the chain instead of counting as a loop iteration, and
  parametric-loop occurrence chains split at mutations (test→edit→test cycles
  are verification, not looping). The marshmallow post-fix verification run is
  no longer deleted.
- **Mutating-call protection** — `TraceStep::is_mutating()` (name vocabulary +
  command-text scan); a successful state-changing call (booking, edit, write)
  is never a delete candidate in the optimal-path diff. Corpus-wide invariant
  test across all real traces.
- **Fix risk classes** — every fix carries `safe` / `needs_review` /
  `dangerous`; `apply` auto-applies safe only, `--all` adds needs_review,
  `dangerous` (e.g. termination guards, which can suppress verification
  re-runs) additionally requires `--force`. `apply` now appends only the
  quoted prompt directive, never the report's analysis meta-prose.
- **Error-derived tool fixes** — the `tool_schema` fix diagnoses from the
  recorded `tool_error` text (value errors get a pre-call validation
  recommendation) instead of the one-size "mark parameters required"
  boilerplate that was wrong on every adjudicated failure.
- **AGF tokenizer rewrite** — markdown emphasis, shell variables, regex/awk
  classes, and glob patterns are no longer extracted as "claims"; quoted
  syntax spans are rejected; content-creation params (edit/write/insert) are
  treated as the agent's new artifact, not assertions; a step's own
  `input_context` counts as evidence; apostrophes in prose no longer open
  phantom quote spans. Corpus acceptance test: zero syntax artifacts in
  `ungrounded[]`. AgentInstruct AGF 0.854→0.951 with the failure trace lowest.
- **SRR most-similar fix** — the flagged pair now points at the *most* similar
  prior step, not the first/oldest one above threshold.
- All published tables/claims regenerated under the new scorer (24-trace
  corpus mean TAS 71.3→73.5; README sample report; paper Table tab:taubench;
  docs/external_agent_audits.md narrative).

### Ship-plan Phase 0 (trust hygiene)
- **One version everywhere: 0.4.0** — workspace Cargo.toml, inter-crate dep
  declarations, pyproject, `__version__`, docker-compose, README banner all
  agree; enforced by a pytest (`tests/test_readme_claims.py`).
- **Exit-code contract** — `0` success/pass, `1` explicit gate failed
  (`--threshold`, regression, tamper), `2` error (bad input/IO/parse).
  Gating is now opt-in: without `--threshold`, a low TAS never exits non-zero.
- **`_find_binary` fixed** — the Python client searched four directories up
  and missed a source checkout's own `target/release/`; now one level up.
- **README repaired** — dual-version banner, duplicated ASCII/blockquote
  lines, contradictory problem paragraphs, stale sample outputs (75/0.70/0.833
  as the binary actually prints), quickstart output corrected to the measured
  `TAS 80.4`, phantom file paths replaced with shipped traces, mermaid now
  shows all 14 signals with table-consistent shares, "thirteen"→fourteen,
  CLI table gains `verify`/`list`, redundancy claim restated corpus-wide
  (26% mean; 36–41% airline / 15% retail / 22% SWE). Internal ticket IDs
  stripped from `--help`.
- **Doc tables regenerated + CI drift check** — `benchmark/RESULTS.md` and
  `docs/external_agent_audits.md` regenerated against 0.4.0 with hermetic,
  order-independent audits (`run_benchmarks.py` now uses `--hermetic` with a
  fresh state dir per audit); CI fails if RESULTS.md drifts from the scorer.
- **Repo hygiene** — `LOOP_LOG.md` → `docs/research_log.md`; PRD `.docx`
  removed; `benchmarks/` merged into `benchmark/`; `publish.sh` no longer
  references a non-existent crate; example fixes (package name, `.env` path).

### Added
- **Run manifest + `tracerazor verify`** — every audit report embeds a
  provenance manifest (trace SHA-256, tool version, timestamp, actual
  similarity backend incl. recorded embedding→BoW fallbacks, exact weights +
  weights SHA-256, threshold, step floor, store-derived baselines). New
  `--hermetic` flag makes scoring a pure function of (trace, config, version);
  `tracerazor verify <report> <trace>` re-checks the hash and exactly
  re-scores hermetic BoW runs, exiting non-zero on tamper or divergence.
  Non-reproducible conditions (embeddings, store-influenced baselines) are
  detected and reported as hash-only verification.
- **AGF (Action/Claim Grounding Fidelity) diagnostic** — deterministic,
  model-free provenance metric: share of tool-call argument literals grounded
  in prior context, and of final-answer literals grounded in
  environment-provided text; every ungrounded literal itemised per step.
  Reported alongside TAS, not folded into the composite pending calibration.
- **Metric-validity audit on 37 real traces** — per-metric fire rates,
  realised-vs-nominal weight influence, and correlation structure documented
  in the paper: TVI dominates final TAS (r=0.89), TUR carries 28% of raw-TAS
  variance, GAR/CSD never exceed 0.62/0.68 on real data, DBO≈TCA (r=0.81).
  Recorded as the baseline for quantile recalibration in `calibration/`.

### Fixed
- **`--store false` now works** — the flag previously rejected an explicit
  value, making store write-back impossible to disable.

### Performance
- **Memoised TF vectors in the default similarity closure** — the BoW backend
  re-tokenised both strings on every call (9,534 calls over 191 distinct
  texts on a 100-step trace); each distinct text is now tokenised once.
  Output is identical (equivalence-tested).
- **Incremental CCE prior-n-gram set** — replaces the O(n²·len) whole-prefix
  re-join per step; boundary-spanning n-grams preserved via a tail carry,
  equivalence-tested against the original implementation.

### Added (real-data evaluation cycle)
- **Hugging Face real-data audit harness** — sourced real ReAct agent
  trajectories from the Hugging Face dataset `zai-org/AgentInstruct` (bash + SQL
  splits), a converter (`tools/convert_agentinstruct.py`), a bundled/disk/live
  loader (`benchmark/hf_loader.py`), a statistics harness
  (`benchmark/hf_audit_stats.py` → `docs/huggingface_agentinstruct_audit.md`),
  and a `cargo test` statistics gate
  (`crates/tracerazor-cli/tests/huggingface_real_data.rs`) that audits the
  corpus end-to-end. Establishes measured behaviour on tool-using agents
  (de-contaminated corpus: mean TAS 78.0 at the default floor, 82.9 over the
  full corpus with `--min-steps 2`). Every audit runs in a fresh state
  directory so measurements are independent of audit order.
- **`tracerazor audit --min-steps N`** — opt-in floor override (default
  unchanged at 5; clamped ≥2). With few-shot scaffolding excluded, ~69% of real
  AgentInstruct trajectories finish in 3–4 steps; the flag makes that real
  trace class auditable by explicit choice, and the skip notice now points at
  it.

### Fixed (real-data evaluation cycle)
- **AgentInstruct converter no longer audits few-shot scaffolding** — upstream
  rows embed the dataset's fixed one-shot demo (and the db split's "Ok." ack)
  before the real task, marked `loss=false` on gpt turns. The converter now
  audits only real-task turns (loss-flag rule, text-marker fallback).
  Previously the identical demo steps were pseudo-replicated into every trace
  and mis-anchored goal metrics; their removal moved mean TAS 82.8→78.0 (the
  demo was padding every score). Corpus widened with 4 more real rows (os_7,
  os_11, os_16, os_18) fetched live from the Hub.
- **DBO cold-start no longer structurally penalises single-tool agents** — the
  retry/thrash signals keyed on the bare tool name, capping a bash/SQL operator
  near the 0.5 floor by construction (n calls = n−1 "retries"). They now key on
  the invocation (tool + params): re-running a tool with new arguments is
  progress; re-issuing an identical call is a retry. Corpus mean DBOₙₒᵣₘ
  0.59→0.88 with genuine-failure traces still discriminated.

### Changed
- **GAR/CSD reduce fenced code to its argument literals** — scored ReAct turns
  previously fed prose+code into BoW similarity. A wholesale fence-strip
  ablation made scores *worse* (the code's paths/quoted strings are the goal
  anchors); the shipped reduction keeps argument literals and drops command
  names/flags/operators — the inverse of the LDI skeleton. Corpus mean GARₙₒᵣₘ
  0.202→0.348 across the exercise.
- **LDI now detects parametric loops** — loop detection previously keyed on an
  exact tool+params state hash and missed the dominant real loop shape for
  tool-using agents: the same command template run once per argument (e.g. a
  shell command repeated per file). A parametric detector abstracts argument
  literals into a command skeleton and flags ≥3 repeats, scoped to command-style
  tools so structured tools are unaffected. Surfaced by the AgentInstruct corpus.
- **GAR/CSD are reasoning-aware on ReAct agents** — goal-advancement and
  cross-step-drift scored only `Reasoning`-typed steps, so they collapsed on
  ReAct agents whose reasoning is fused into the tool-call turn ("Think: … Act:
  …"). They now also score tool-call steps that carry substantive reasoning
  prose (≥12 words) while still ignoring bare invocations.

### Added (earlier in this cycle)
- **OBS metric (Observation Token Share)** — the fraction of tokens spent on
  tool I/O, promoted into the composite (weight 0.06, ~4.8% share) after it was
  the one candidate feature that predicted real recoverable token waste and
  replicated across two real datasets. Default raw weights now sum to 1.26.
- **Weight calibration toolkit (`calibration/`)** — `calibrate.py` fits
  non-negative composite weights to measured recoverable waste (target
  `1 - recoverable_fraction`) with k-fold CV and a `--features` mode; `adapt.py`
  builds manifests from CSV/JSONL/paired-dir exports; `sources/from_messages.py`
  and `sources/from_taubench.py` convert OpenAI/ShareGPT trajectory datasets into
  before/after pairs. `report.features` exposes experimental context-accumulation
  signals (observation share/compressibility, stale-observation retention,
  context growth, redundant/repeated tool calls) for calibration research.
- **Real-data calibration results** — on real tau-bench/tau2-bench before/after
  pairs the original metrics did not predict recoverable waste (negative CV R²);
  adding the observation features raised it positive (+0.08 / +0.12), which is
  what justified promoting OBS. Documented in the paper (`paper/tracerazor.tex`).
- **Feature/ceiling exploration** — path/length features (`step_count_norm`,
  `mean_step_tokens_norm`, `reasoning_run_max`, `revisit_rate`, `tool_diversity`)
  added to `report.features` and a `--feature-keys` flag added to the calibrator.
  Tested with non-linear models (ridge, gradient boosting): none beat the
  regularised convex fit, so the structural-feature ceiling for predicting
  recoverable waste is ~0.1; richer semantic signal is needed to exceed it
  (recorded in the paper).
- **Real-framework integration tests** (`tests/test_integrations_real.py`) —
  drive the LangGraph callback with real `langchain_core` events and the
  OpenAI-Agents hooks bound to the real `RunHooks` base, auditing against the
  binary. CI installs the framework extras so they run there.
- **HTTP server health checks** — `GET /healthz` (liveness) and `/readyz`
  (readiness, checks the store) plus a `--health-check` binary probe used by the
  container HEALTHCHECK (no curl needed).
- **Python CI job** — builds the CLI, installs the package, runs `pytest`, an
  end-to-end audit, and a calibration smoke on real in-repo pairs.
- **`paper/tracerazor.tex`** — an honest technical report (metrics, calibration,
  real-data results, threats to validity), built to a PDF artifact in CI.

### Security
- Server hardening: SSRF guard on the export endpoints, dashboard XSS escaping,
  loopback-by-default bind, restricted CORS, 16 MiB request-body limit, and a
  50k-step audit cap. Docker image runs as a non-root user and pins its tag.

### Changed
- **Benchmarks are now real** — `benchmarks/` audits the real public traces in
  `traces/external/` (tau-bench + SWE-agent); the synthetic scenario traces and
  the synthetic calibration worked-example were removed in favour of real data.
- **HTTP mode parity** — the Python client now maps the full server response
  (`total_steps`/`total_tokens`/`fixes`/`savings`) instead of dropping fields.
- **Framework integrations fixed** — the LangGraph/CrewAI/OpenAI-Agents callbacks
  passed an unsupported `threshold=` to `analyse()`; they now set it on the
  client, so the integrations work at runtime (regression-tested).

### Removed
- **Dead `tracerazor-proxy` crate** — was declared as a server dependency but
  never used; dropped from the workspace, Dockerfile, and docs.

- **Trajectory Path Entropy (TPE)** — a genuine information-theoretic
  "staying on the path" diagnostic (`metrics::tpe`). Classifies step-to-step
  goal-progress increments as advance/stall/regress and reports the normalised
  Shannon entropy of that distribution plus a directed `focus_score`. Reported
  alongside TAS (and in the `PATH ENTROPY` report section) but **not** folded
  into the composite, so published per-metric shares are unchanged.
- **`goal_anchor` fix type** — GAR/TPE drift was previously detection-only;
  the audit now emits a goal-re-anchoring prompt patch when a trajectory
  drifts off its objective.
- **`Trace::task_goal()`** — resolves the real task objective from trace
  `metadata` (`task`/`goal`/`objective`/…) so goal-oriented metrics can score
  progress toward the actual goal rather than the agent's own final step.

### Changed
- **GAR** now anchors on the real task goal via `gar::compute_with_goal` when
  the trace provides one (falling back to the final-step proxy otherwise).
  Scoring against the agent's own last step rewarded confident convergence on
  the *wrong* answer.
- **ISR** novelty scan is bounded to a 64-step recent-context window instead of
  the full prefix, removing the dominant quadratic term in `analyse()`
  (200-step traces ~2.5× faster, 1000-step ~3.3× faster). Results are identical
  for traces up to the window length.
- **DBO** `normalised()` now clamps to [0, 1] defensively.
- **Honesty pass on claims**: README version banner corrected `v1.0.0 → v0.1.0`
  (matches `Cargo.toml`); the unsourced and mutually-contradictory "30–60%"
  (README) / "40–70%" (Overview) redundancy figures replaced with the
  actually-measured 36–41% step-redundancy number from
  `docs/external_agent_audits.md`; the unsupported RDA "80–85% agreement on a
  500-trace benchmark" claim removed (no such dataset exists in the repo); the
  "sub-5 ms per trace" headline replaced with measured, size-qualified
  benchmark numbers; savings output and the synthetic benchmark table
  relabelled as heuristic projections rather than measured re-runs.

### Changed (earlier)
- **License**: project relicensed under MIT (was Apache-2.0); LICENSE, all
  `Cargo.toml` files and CONTRIBUTING aligned with the README/PyPI metadata
  that already advertised MIT.
- **Storage backend**: replaced SurrealDB / SurrealKV with SQLite via
  `tokio-rusqlite`. Two-table schema (`traces`, `kb_entries`), JSON
  payloads, statically-linked SQLite (no system library required). The
  public `TraceStore` API is unchanged. The CLI's persistent path moved
  from `~/.tracerazor/store/` (a SurrealKV directory) to
  `~/.tracerazor/store.db` (a SQLite file) — existing local data does not
  migrate.
- **Python source layout**: `src/redundancy/` moved to `python/redundancy/`
  so the top-level `src/` directory is no longer mistaken for the Rust
  workspace root. `sys.path.insert(0, 'src')` is now `... 'python'`.
- **bow.rs**: removed the "TF-IDF" label — implementation has always been
  plain term-frequency cosine; rustdoc and crate description updated.

### Added
- Term-frequency BoW similarity is now correctly labelled.
- SRR pair scan bounded to a 256-step lookback window (`LOOKBACK_WINDOW`)
  so very long traces stay O(n · window) instead of O(n²).
- `tracerazor-cli` integration test suite (assert_cmd + tempfile) covering
  audit, threshold gating, missing-file, cost, simulate, compare.
- Criterion benchmark `analyse` for 10/50/200/1000-step traces, runnable
  via `cargo bench -p tracerazor-core`, backing the sub-5 ms claim.
- File-backed persistence round-trip test in `tracerazor-store`.

### Known TODO
- Substitutability AUC numbers in `docs/findings_v5.md` and the README
  remain on synthetic data. Re-running against real tau-bench transcripts
  is queued; the evaluator is ready to consume them once collected.

### Added (prior)
- **Phase 0: Production readiness**
  - Replaced all `unwrap()` calls in production code with safe alternatives
  - Configurable CORS via `TRACERAZOR_CORS_ORIGINS` env var (comma-separated origins; defaults to permissive)
  - Integration test suite: full lifecycle (audit/retrieve/delete), compare, agents, KB, malformed input
  - CONTRIBUTING.md and CHANGELOG.md
  - Python package preparation for PyPI publishing (classifiers, keywords, project URLs)
  - `avs` and `fixes` fields in `/api/audit` JSON response

### Fixed
- `partial_cmp().unwrap()` in 5 production sites (cost, report, store, API) — could panic on NaN
- `min_by_key().unwrap()` in DBO metric — replaced with `let-else` for clarity
- `as_deref().unwrap()` in DBO metric — replaced with `unwrap_or("unknown")`

## [0.3.0] - 2026-04-12

### Added
- **Verbosity metrics (P2)**: reformulation detection via bigram Jaccard overlap (threshold 0.70), Shannon entropy pre-filter (<3.8 bits/char), Aggregate Verbosity Score (AVS), VERBOSITY ALERT in report when AVS > 0.40
- **New fix types**: `VerbosityReduction`, `HedgeReduction`, `CavemanPromptInsert`, `ReformulationGuard`
- **Proxy Layer 4**: verbosity directive injection when rolling CCR >= 0.35 (standard) or > 0.50 (ultra)
- `StepFlag::Reformulation` — flags steps that paraphrase their input context
- `TasScore::avs` field in JSON output
- `ProxyRequest::rolling_ccr` field for Layer 4 integration
- 7 new proxy tests (verbosity directive standard/ultra/boundary/no-op)
- 6 new core tests (reformulation detection + annotation, entropy flagging)

## [0.2.0] - 2026-04-12

### Added
- **Verbosity metrics (P1)**: VDI (Verbosity Density Index), SHL (Sycophancy/Hedging Level), CCR (Caveman Compression Ratio)
- Shared `verbosity_data` module with HEDGE_PHRASES, PREAMBLE_PATTERNS, FILLER_WORDS
- TAS weight redistribution: 8 metrics -> 11 metrics (SRR 20->17%, LDI 15->13%, TCA 15->13%, DBO 10->9%)
- 14 new tests across VDI (4), SHL (5), CCR (5)

### Changed
- `TasScore` now carries `vdi`, `shl`, `ccr` result fields
- `scoring::compute()` accepts 11 metric results
- Report markdown includes verbosity metrics separator and three new rows

## [0.1.0] - 2026-04-11

### Added
- Initial release: 8 TAS metrics (SRR, LDI, TCA, RDA, ISR, TUR, CCE, DBO)
- CLI with audit, compare, simulate, cost, export commands
- Axum REST server with embedded Alpine.js dashboard
- SurrealDB persistence (file-backed + in-memory)
- Per-metric anomaly detection (9 rolling baselines at 2σ)
- Known-Good-Paths KB (auto-capture at TAS >= 85)
- Multi-agent scoring with per-agent TAS breakdown
- Executive summaries (paragraph + one-liner)
- Auto-generated fixes: tool_schema, prompt_insert, termination_guard, context_compression
- Four-layer guardrail proxy (semantic, scope, budget)
- Python adapters: CrewAI, OpenAI Agents SDK, LangGraph
- GitHub Action for CI/CD efficiency gating
- Docker deployment with health checks and volume persistence
