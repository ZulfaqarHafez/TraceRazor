# Changelog

All notable changes to TraceRazor are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
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