# TraceRazor Public Trust Matrix

Last checked: 2026-07-12.

This matrix tracks public proof signals that a new user can verify without
private context. A signal is green only when the public registry, workflow, or
artifact is already visible.

| Signal | Status | Evidence | Owner action |
|---|---|---|---|
| PyPI package | Green for 1.1.0 | `https://pypi.org/project/tracerazor/` publishes five native platform wheels and no 1.1 source distribution. | Keep the platform-wheel-only contract until a source install can guarantee the native auditor. |
| piwheels file | Informational for 1.1 | piwheels builds from source distributions; TraceRazor 1.1 deliberately ships platform wheels only so every supported install contains the native auditor. | Do not add an unsafe source-only distribution solely for piwheels. |
| crates.io CLI | Informational / Labs | No stable public Rust API is declared for 1.1, so crates.io is not a GA distribution target. | Keep source-build guidance and do not add a cargo-install claim. |
| GitHub tag alignment | Green for v1.1.0 | `v1.1.0` is the latest public release and matches the Python and Cargo versions. | Never mutate or reuse the published tag. |
| GitHub Actions: Agent Efficiency Gate | Green at v1.1.0 commit | The release commit's gate completed successfully. | Keep the gate required. |
| GitHub Actions: TraceRazor CI | Green at v1.1.0 commit | Python, Rust, packaging, security, and agent-host contract gates passed at the release commit. | Keep the full matrix green before the next release. |
| GitHub Actions: Release | Green for v1.1.0 | The successful run published five native wheels, five standalone archives, checksums, proof assets, and a tested linux/amd64 + linux/arm64 agent image. Linux floors are glibc 2.35 x64 and 2.39 ARM64. | Preserve immutable release assets and repeat all clean-install and image gates for the next version. |
| OpenSSF Scorecard | Red against the project target | A public result now exists, but the 2026-07-10 score is 3.8, below the repository target of 7.0. | Address branch protection, review, dependency pinning, and other reported gaps before marking this signal green. |
| GitHub artifact attestations | Green for the v1.1.0 release run | The release workflow generated release-evidence assets and hosted attestations for published subjects. | Keep verifying the exact release subjects and immutable digest on every release. |
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
| TRICE Installability Card | Green for the CI Linux x64 proof of the unpublished 1.2.0 wheel | `docs/trice_install_card.json` is the card from the `release-evidence` artifact of release run [35892178950](https://github.com/ZulfaqarHafez/TraceRazor/actions/runs/35892178950) (tag `v1.2.0`, commit `3a99c6a1`). It proves a clean, non-repository install of the CI-built `tracerazor-1.2.0-py3-none-manylinux_2_35_x86_64.whl` (SHA-256 `ea5db152c4c3804d019c0bc518e786675c19e2a6844d99a778f365d5a460d182`), including the bundled auditor, MCP self-test, packaged skill/assets, agent doctor, and sample audit. The same run's five platform-wheel jobs passed. As of 2026-10-29 the wheel is not on PyPI. | Commit the release workflow's card for every new version, not a local build. After 1.2.0 reaches PyPI, confirm the published Linux x64 wheel has this SHA-256. |
| TRICE Release Evidence | Red: CI packet for unpublished subjects; does not verify in a checkout | `docs/trice_release_evidence.*` are the unmodified `release-evidence` artifact of release run 35892178950 (artifact digest `sha256:1596265122ffe977cdaf4290ce1abce8ab1f7da7c26d89721d87fbb735118d30`). Its `release_evidence_ready` level and 100/100 score mean every bound file was present and hashed in that job; they do not mean 1.2.0 is published. As of 2026-10-29 PyPI's latest version is 1.1.0 and no v1.2.0 GitHub release exists: the run's `Publish to PyPI with Trusted Publishing` step failed with `invalid-publisher` (no PyPI trusted publisher is registered and `PYPI_API_TOKEN` is unset), so `Publish GitHub release` was skipped. The same run did publish the agent image: `ghcr.io/zulfaqarhafez/tracerazor-agent:v1.2.0` and `:latest` resolve to `sha256:dfd6eb4ea3fd7c5450d16936952bf1cda0f0eac53f542cd1360e0e58b5ec406c`. Since that build, `cargo deny check` reports 12 RustSec advisories, published in October 2026, against crates in its Cargo.lock: `rustls`, `rustls-webpki`, `webpki-roots`, `tokio`, `zip`, `zstd-safe` and `zstd-sys`. Those artifacts and that image bundle them; the `Cargo.lock` on `main` now pins patched versions and `zip` 9 without zstd. The packet binds the five 1.2.0 wheels under `dist/` and the Linux x64 CLI at `release-cli/tracerazor`, which exist only as that run's Actions artifacts (they expire on 2026-12-22), so `tracerazor-trice verify-release-evidence` fails in a checkout. It also pins the artifact card from before that job regenerated the install card, so the artifact card had to be regenerated here; the contract card changed with this matrix, and the crates card with the CLI's `Cargo.toml`. `release.yml` now regenerates the artifact card before `release-evidence` and ships it with the packet, so later packets pin only cards they carry. | Configure PyPI publishing (register the trusted publisher or set `PYPI_API_TOKEN`), then ship a patched release from `main` instead of this run's artifacts; that release replaces this packet. Never regenerate this packet locally, and mark this row green only when a published packet verifies. |
| TRICE Release Card | Yellow for local release candidate | `docs/trice_release_card.json` snapshots `tracerazor-trice doctor`, binds proof cards including the contract and installability cards, and keeps `public_release_ready=false` while the platform install, PyPI, tag, Actions, or Scorecard gates are red. piwheels/crates.io are informational. | Regenerate after every release-publication attempt. |
| TRICE Crates Publish Card | Labs preflight only | `docs/trice_crates_card.json` binds workspace Cargo manifests and README cargo-install honesty. `cargo_install_claim_allowed=false`. It is generated with `--offline`, as the release workflow does, so its registry status reads `skipped`; it is not a registry check. On the same tree `--offline` reproduces the release workflow's card byte for byte, so it is regenerated locally whenever a bound `Cargo.toml` or the README changes. | Keep crates outside GA until a stable Rust API is intentionally declared. |
| TRICE Integrity Card | Red while release evidence does not verify | `docs/trice_integrity_card.json` binds offline doctor output, proof-card verifiers, release evidence, crates publish card, installability card, research card, paper manifest, shipped schemas, and CI/release/Scorecard workflow hooks. It is `partial_integrity` (90/100): every bound verifier passes except release evidence, so it cannot reach `proof_graph_integrity_locked` (see above). CI's `trice-integrity` job checks the graph against freshly built release subjects. | Regenerate after changing schemas, proof cards, workflows, paper, release evidence, installability, research, crates status, or public docs. |
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
# docs/trice_install_card.* comes from the same release-evidence artifact as the
# release evidence (it installs the workflow's own wheel); verify it, never
# regenerate it locally.
tracerazor-trice verify-install docs/trice_install_card.json
tracerazor-trice research --out docs/trice_research_card.json
tracerazor-trice verify-research docs/trice_research_card.json
tracerazor-trice artifact --out docs/trice_artifact_card.json
tracerazor-trice verify-artifact docs/trice_artifact_card.json
# docs/trice_release_evidence.* comes from the release workflow's release-evidence
# artifact; never regenerate it locally. In a checkout this verifier fails (see above).
tracerazor-trice verify-release-evidence docs/trice_release_evidence.json
tracerazor-trice release --out docs/trice_release_card.json --timeout-s 10
tracerazor-trice verify-release docs/trice_release_card.json
tracerazor-trice crates --offline --out docs/trice_crates_card.json --format tex
tracerazor-trice verify-crates docs/trice_crates_card.json
tracerazor-trice integrity --out docs/trice_integrity_card.json
tracerazor-trice verify-integrity docs/trice_integrity_card.json
```
