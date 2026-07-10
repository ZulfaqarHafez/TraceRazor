# TraceRazor Public Trust Matrix

Last checked: 2026-06-21.

This matrix tracks public proof signals that a new user can verify without
private context. A signal is green only when the public registry, workflow, or
artifact is already visible.

| Signal | Status | Evidence | Owner action |
|---|---|---|---|
| PyPI package | Green for 1.0.3, pending 1.1.0 | `https://pypi.org/project/tracerazor/` shows the latest published Python package. | Publish 1.1.0 only after clean platform-wheel gates pass. |
| piwheels file | Informational for 1.1 | piwheels builds from source distributions; TraceRazor 1.1 deliberately ships platform wheels only so every supported install contains the native auditor. | Do not add an unsafe source-only distribution solely for piwheels. |
| crates.io CLI | Informational / Labs | No stable public Rust API is declared for 1.1, so crates.io is not a GA distribution target. | Keep source-build guidance and do not add a cargo-install claim. |
| GitHub tag alignment | Green for v1.0.2, pending v1.1.0 | `v1.0.2` is the latest public release tag. | Tag v1.1.0 only after local gates and package checks pass. |
| GitHub Actions: Agent Efficiency Gate | Green at last public check | Latest checked run succeeded. | Keep gate required. |
| GitHub Actions: TraceRazor CI | Pending for 1.1.0 | Python, Rust, packaging, security, and agent-host contract gates must all pass at the release commit. | Re-run Actions after pushing the reviewed 1.1.0 change set. |
| GitHub Actions: Release | Pending for 1.1.0 | The workflow builds five native wheels and standalone binaries; Linux floors are explicitly glibc 2.35 x64 and 2.39 ARM64. Its OCI path pins both multi-arch base indexes, uses exact dependency locks and a fixed source epoch, smoke-tests the exact amd64/arm64 digest, verifies signed registry provenance and anonymous public pull, uploads the image receipt, then promotes tags as the final registry mutation. Bit-for-bit OCI reproducibility still requires a repeat-build proof because BuildKit attestations may contain builder metadata and platform package artifacts differ. | Set the GHCR package to Public, then publish only after clean-install, imported-GLIBC-symbol, artifact-origin, both OCI architecture smokes, anonymous pull, and digest-attestation gates pass. |
| OpenSSF Scorecard | Red until first published result | `.github/workflows/scorecard.yml` runs Scorecard and uploads SARIF, but `https://api.scorecard.dev/projects/github.com/ZulfaqarHafez/TraceRazor` has no published result until the workflow runs on GitHub. | Run the workflow and keep public score >= 7.0 before marking release ready. |
| GitHub artifact attestations | Added, pending release run | `.github/workflows/release.yml` now generates `trice_release_evidence.*`, `trice_crates_card.*`, `trice_install_card.*`, and `trice_research_card.*` assets and runs `actions/attest` over platform wheels, binaries, checksums, and proof/evidence files. | Verify attestations after the next GitHub release run. |
| Security policy | Green after this file set | `SECURITY.md` exists in the repository root. | Keep disclosure and support windows current. |
| Citation metadata | Green after this file set | `CITATION.cff` exists in the repository root. | Update version and date each release. |
| TRICE Suite Readiness | Green for smoke preflight | `docs/trice_suite_readiness.json` reports the bundled suite is `smoke_ready`, not pilot or claim ready. | Regenerate before every pilot or claim run. |
| TRICE Protocol Lock | Green for smoke protocol | `docs/trice_protocol_lock.json` pre-registers metric, guardrails, evidence-recall floor, clustered CI, source, adapter, receipt, claim-card, and artifact-card requirements. Current level is `smoke_protocol_locked`. | Regenerate before each pilot or claim run and verify before execution. |
| TRICE Design Card | Green for smoke design | `docs/trice_design_card.json` projects statistical signal from task-cluster means while refusing claim design because the protocol is not held-out/replicated enough. Current level is `smoke_design_observed`. | Regenerate after protocol or suite-result changes. |
| TRICE Claim Card | Green for non-claim smoke | `docs/trice_claim_card.json` binds the broad smoke suite result, evidence recall, evidence manifest, requirements, and non-claims. | Regenerate after every suite run. |
| TRICE Remote Smoke Claim Card | Green for remote-git smoke | `docs/trice_remote_smoke_claim_card.json` binds one locked PyPA `sampleproject` clone, remote suite result, evidence recall, evidence manifest, and non-claims. | Keep as a small public smoke. Do not use it as S-tier evidence. |
| TRICE Reproduction Card | Green for reviewer replay smoke | `docs/trice_reproduction_card.json` binds exact verifier commands plus readiness, protocol, design, claim, bundle, paper-manifest, and paper-result input hashes. | Regenerate after paper, evidence, protocol, design, claim, or bundle changes. |
| TRICE Contract Card | Green for library contract | `docs/trice_contract_card.json` binds SemVer, public imports, `tracerazor-trice` commands, shipped schemas, examples, docs, and package metadata. Current level is `library_contract_locked`. | Regenerate before every release and after any public API, CLI, schema, example, or doc-bound package change. |
| TRICE Research Card | Green for paper-basis proof | `docs/trice_research_card.json` binds 165 ledgered sources, 165 unique URLs, category coverage, row hashes, and the research ledger hash. Current level is `research_basis_locked`. | Regenerate after research ledger, paper, README, or proof graph changes. |
| TRICE Artifact Card | Green for review-ready smoke | `docs/trice_artifact_card.json` binds and `tracerazor-trice verify-artifact` verifies the README, paper, readiness card, protocol lock, design card, reproduction card, contract card, installability card, research card, broad/remote claim cards, broad/remote evidence bundles, library doc, and schemas as one packet. | Regenerate after README, paper, evidence, protocol, design, reproduction, contract, installability, research, or schema changes. |
| TRICE Installability Card | Green for the local Windows x64 candidate; release matrix pending | `docs/trice_install_card.json` proves a clean, non-repository install of the 1.1.0 Windows x64 wheel, including the bundled auditor, MCP self-test, packaged skill/assets, agent doctor, and sample audit. Current level is `full_cli_install_ready`; this is one platform cell, not proof of the full release matrix. | Run the same clean-machine card for macOS x64/ARM64 and Linux x64/ARM64 in release CI before publication. |
| TRICE Release Evidence | Green for the local Windows x64 candidate; public release pending | `tracerazor-trice release-evidence` binds the local 1.1.0 platform wheel, Rust CLI binary, proof cards, paper artifacts, evidence bundles, SHA-256 checksums, CycloneDX-style SBOMs, and an in-toto/SLSA-shaped provenance sidecar. Source distributions remain deliberately excluded because they cannot guarantee a bundled auditor. | Regenerate from every platform artifact in release CI, then verify the published GitHub assets and attestations. |
| TRICE Release Card | Yellow for local release candidate | `docs/trice_release_card.json` snapshots `tracerazor-trice doctor`, binds proof cards including the contract and installability cards, and keeps `public_release_ready=false` while the platform install, PyPI, tag, Actions, or Scorecard gates are red. piwheels/crates.io are informational. | Regenerate after every release-publication attempt. |
| TRICE Crates Publish Card | Labs preflight only | `docs/trice_crates_card.json` binds workspace Cargo manifests and README cargo-install honesty. `cargo_install_claim_allowed=false`. | Keep crates outside GA until a stable Rust API is intentionally declared. |
| TRICE Integrity Card | Green for local proof graph | `docs/trice_integrity_card.json` binds offline doctor output, proof-card verifiers, release evidence, crates publish card, installability card, research card, paper manifest, shipped schemas, and CI/release/Scorecard workflow hooks. Current level is `proof_graph_integrity_locked`. | Regenerate after changing schemas, proof cards, workflows, paper, release evidence, installability, research, crates status, or public docs. |
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
tracerazor-trice protocol --manifest examples/trice_suite_bundled_live.json \
  --out docs/trice_protocol_lock.json
tracerazor-trice verify-protocol docs/trice_protocol_lock.json \
  --manifest examples/trice_suite_bundled_live.json
tracerazor-trice design --protocol docs/trice_protocol_lock.json \
  --suite-result benchmark/trice/results/v2-broad-smoke/trice_suite_results.json \
  --out docs/trice_design_card.json
tracerazor-trice verify-design docs/trice_design_card.json \
  --protocol docs/trice_protocol_lock.json \
  --suite-result benchmark/trice/results/v2-broad-smoke/trice_suite_results.json
tracerazor-trice verify-claim docs/trice_claim_card.json
tracerazor-trice reproduction --out docs/trice_reproduction_card.json
tracerazor-trice verify-reproduction docs/trice_reproduction_card.json
tracerazor-trice install --out docs/trice_install_card.json --dist-dir dist
tracerazor-trice verify-install docs/trice_install_card.json
tracerazor-trice research --out docs/trice_research_card.json
tracerazor-trice verify-research docs/trice_research_card.json
tracerazor-trice artifact --out docs/trice_artifact_card.json
tracerazor-trice verify-artifact docs/trice_artifact_card.json
tracerazor-trice release-evidence --out docs/trice_release_evidence.json --dist-dir dist
tracerazor-trice verify-release-evidence docs/trice_release_evidence.json
tracerazor-trice release --out docs/trice_release_card.json --timeout-s 10
tracerazor-trice verify-release docs/trice_release_card.json
tracerazor-trice crates --out docs/trice_crates_card.json --timeout-s 10
tracerazor-trice verify-crates docs/trice_crates_card.json
tracerazor-trice integrity --out docs/trice_integrity_card.json
tracerazor-trice verify-integrity docs/trice_integrity_card.json
```
