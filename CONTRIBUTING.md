# Contributing to TraceRazor

Thank you for considering contributing. This guide covers everything you need to get started.

## Prerequisites

- **Rust 1.82+** (`rustup update stable`)
- **Node 20+** (optional — only for the React dashboard build)
- **Python 3.10+** (optional — only for integration adapters)

## Setup

```bash
git clone https://github.com/ZulfaqarHafez/tracerazor
cd tracerazor
cargo build --workspace
cargo test --workspace
```

All 102+ tests should pass with zero clippy warnings:

```bash
cargo clippy --workspace -- -D warnings
```

## Project Structure

```
crates/
  tracerazor-core/      # Metrics, scoring, simulation — zero network deps
  tracerazor-ingest/    # Trace parsers (raw JSON, LangSmith, OTEL)
  tracerazor-semantic/  # BoW + optional OpenAI embeddings
  tracerazor-store/     # SurrealDB persistence layer
  tracerazor-server/    # Axum REST + WebSocket server
  tracerazor-proxy/     # Four-layer guardrail interceptor
  tracerazor-cli/       # CLI entry point
integrations/           # Python adapters (CrewAI, OpenAI Agents, LangGraph)
```

## Making Changes

1. **Fork and branch.** Create a feature branch from `main`.
2. **Write tests.** Every new metric, fix type, or API endpoint needs tests.
3. **Run the full suite** before submitting:
   ```bash
   cargo test --workspace
   cargo clippy --workspace -- -D warnings
   ```
4. **Keep PRs focused.** One feature or fix per PR. Bundled PRs are fine for tightly coupled changes.

## Code Style

- Follow existing patterns in the crate you're modifying.
- No `unwrap()` in production code — use `?`, `unwrap_or()`, or `expect()` with a message explaining the invariant.
- No `unsafe` blocks.
- Add doc comments (`///`) for public functions and types.
- Prefer descriptive error messages over generic ones.

## Adding a New Metric

1. Create `crates/tracerazor-core/src/metrics/<name>.rs`.
2. Add the result struct with `Serialize`/`Deserialize` derives.
3. Implement `normalised()` returning 0.0-1.0 (higher = better for TAS).
4. Register in `metrics/mod.rs`, add to `TasScore` in `scoring.rs`.
5. Update `scoring::compute()` with the new weight.
6. Add the metric to `report.rs` (markdown output) and `simulate.rs` (placeholder).
7. Write at least 3 tests covering clean input, pathological input, and empty trace.

## Adding a New Fix Type

1. Add the variant to `FixType` in `fixes.rs`.
2. Add the `Display` impl and generation logic in `generate_fixes()`.
3. Update the fix table in `README.md`.

## Reporting Issues

Use [GitHub Issues](https://github.com/ZulfaqarHafez/tracerazor/issues). Include:
- TraceRazor version (`tracerazor --version`)
- Minimal reproduction trace (anonymise if needed)
- Expected vs. actual output

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.