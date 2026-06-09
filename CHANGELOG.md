# Changelog

All notable changes to TraceRazor are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
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