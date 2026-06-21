# TraceRazor Public Trust Matrix

Last checked: 2026-06-21.

This matrix tracks public proof signals that a new user can verify without
private context. A signal is green only when the public registry, workflow, or
artifact is already visible.

| Signal | Status | Evidence | Owner action |
|---|---|---|---|
| PyPI package | Green for 1.0.2, pending 1.0.3 | `https://pypi.org/project/tracerazor/` shows the latest published Python package. | Publish 1.0.3 after this change set passes. |
| piwheels file | Green for 1.0.2, pending 1.0.3 | `https://www.piwheels.org/project/tracerazor/` has a visible 1.0.2 wheel. | Wait for piwheels builder after PyPI 1.0.3 publish. |
| crates.io CLI | Red | `https://crates.io/crates/tracerazor` is not published yet. | Publish crates in dependency order or keep README source-build guidance. |
| GitHub tag alignment | Green for v1.0.2, pending v1.0.3 | `v1.0.2` points at the released main commit. | Tag v1.0.3 only after local gates and package checks pass. |
| GitHub Actions: Agent Efficiency Gate | Green at last public check | Latest checked run succeeded. | Keep gate required. |
| GitHub Actions: TraceRazor CI | Red at last public check, fixed locally for 1.0.3 | Known public failures were Rust clippy and supply-chain audit. Local `cargo clippy --workspace -- -D warnings`, `cargo audit`, `cargo deny check`, and `pip-audit` pass after this change set. | Re-run Actions after pushing 1.0.3 fixes. |
| GitHub Actions: Release | Pending at last public check | Release workflow was queued during the 1.0.2 check window. | If 1.0.2 is immutable on PyPI, cut 1.0.3 instead. |
| Security policy | Green after this file set | `SECURITY.md` exists in the repository root. | Keep disclosure and support windows current. |
| Citation metadata | Green after this file set | `CITATION.cff` exists in the repository root. | Update version and date each release. |
| TRICE Suite Readiness | Green for smoke preflight | `docs/trice_suite_readiness.json` reports the bundled suite is `smoke_ready`, not pilot or claim ready. | Regenerate before every pilot or claim run. |
| TRICE Claim Card | Green for non-claim smoke | `docs/trice_claim_card.json` binds the broad smoke suite result, evidence manifest, requirements, and non-claims. | Regenerate after every suite run. |
| TRICE S-tier evidence | Red, not claimed | Local smoke and suite machinery exist, but the held-out 50 x 3 remote-git gate has not passed. Claim card level is `smoke`, not `s_tier`. | Run pilot, then held-out claim suite, then publish evidence bundle. |

## Automated Check

Run:

```bash
tracerazor-trice doctor --format text
tracerazor-trice doctor --format json
```

Use `--offline` for deterministic local tests. Offline mode verifies local
package, CLI availability, schemas, and git alignment, while skipping public
HTTP checks. In a source checkout, CLI availability can be `source-build`; in a
platform wheel it should be `bundled`.

## Cargo Install Honesty

The README must not claim `cargo install tracerazor` until crates.io has a
published `tracerazor` crate and `cargo install tracerazor` succeeds on a clean
machine. Until then, the public Rust path is source build:

```bash
cargo build --release -p tracerazor
```

## S-Tier Claim Rule

README may say "S-tier evidence passed" only when the generated suite result
contains:

```json
{
  "s_tier_gate": {
    "passed": true
  }
}
```

Anything less is a pilot, smoke, or readiness result.

The public claim card is the user-facing guardrail for that rule:

```bash
tracerazor-trice claim \
  --suite-result benchmark/trice/results/v2-broad-smoke/trice_suite_results.json \
  --manifest benchmark/trice/results/v2-broad-smoke/trice_suite_evidence_manifest.json \
  --out docs/trice_claim_card.json
tracerazor-trice suite readiness examples/trice_suite_bundled_live.json \
  --out docs/trice_suite_readiness.json
tracerazor-trice suite verify-readiness docs/trice_suite_readiness.json \
  --manifest examples/trice_suite_bundled_live.json
tracerazor-trice verify-claim docs/trice_claim_card.json
```
