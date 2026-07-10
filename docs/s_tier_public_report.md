# TraceRazor S-Tier Public Report

Status: not passed yet.

TraceRazor has useful A-tier mechanics today: offline audit, signed evidence,
safe fix patches, TRICE live rollouts, command adapter profiles, receipts,
evidence bundles, and suite-level gates. The S-tier public claim is withheld
until a held-out remote-git suite passes the configured gate.

## Public Scope

The claim scope is narrow:

Python software-repair and context-control tasks on held-out remote Git
repositories, using fixed adapter profiles and objective verifier commands.

The claim does not cover all programming languages, all coding agents, or all
software engineering tasks.

## Current Proof State

| Layer | State | Notes |
|---|---|---|
| TRICE live smoke | Present | Fresh workspaces, real edits, verifier commands, receipts, and evidence manifests exist. |
| TRICE suite gate | Present | Suite-level clustered savings, evidence recall, and S-tier missing-requirement reporting exist. |
| Remote-git scaffold | Added | `tracerazor-trice suite scaffold` generates locked manifests from curated task lists. |
| Remote-git smoke | Present | One locked public PyPA `sampleproject` clone verifies the remote source, patch, receipt, evidence recall, suite, claim-card, and bundle path. It is smoke evidence only. |
| Public doctor | Added | `tracerazor-trice doctor` reports package, schema, registry, tag, and CI state. |
| Suite Readiness | Added | `tracerazor-trice suite readiness` preflights manifests before live pilot/claim spend. Current bundled level is smoke-ready only. |
| Protocol Lock | Added | `tracerazor-trice protocol` pre-registers the metric, pass guardrail, clustered CI, locked-source, adapter-profile, receipt, claim-card, and artifact-card requirements before a live outcome claim. Current bundled level is smoke-protocol only. |
| Design Card | Added | `tracerazor-trice design` projects whether observed task-cluster variance would clear the claim target, while refusing claim readiness when protocol/replicate/holdout requirements are missing. Current bundled level is smoke-design only. |
| Claim Card | Added | `tracerazor-trice claim` generates JSON, Markdown, LaTeX, and SVG claim boundaries. Current level is smoke, not S-tier. |
| Contract Card | Added | `tracerazor-trice contract` binds SemVer, public imports, CLI commands, shipped schemas, examples, docs, and package metadata so release promises have a declared API boundary. |
| Research Card | Added | `tracerazor-trice research` binds the research ledger, 165 source rows, category coverage, row hashes, Markdown/SVG/LaTeX outputs, and non-claim boundary. |
| Artifact Card | Added | `tracerazor-trice artifact` binds the public README, paper, broad and remote evidence bundles, readiness card, protocol lock, design card, reproduction card, contract card, installability card, research card, claim cards, library doc, and schemas; `tracerazor-trice verify-artifact` checks the card and bound hashes. |
| Installability Card | Added | `tracerazor-trice install` proves the built wheel in a clean virtual environment, imports packaged schemas/APIs, runs `tracerazor-trice`, and separately checks bundled Rust CLI availability. Current generic-wheel target is Python/TRICE install ready, not full no-Rust-toolchain CLI ready. |
| Release Evidence | Added | `tracerazor-trice release-evidence` binds platform wheels, Rust CLI binaries, proof cards, paper artifacts, evidence bundles, checksums, CycloneDX-style SBOMs, and provenance sidecars. This is release-asset evidence, not an S-tier outcome claim. |
| Release Card | Added | `tracerazor-trice release` snapshots doctor output, binds proof cards including the contract and installability cards, and keeps public-release readiness false until PyPI, piwheels, crates.io, tag, Actions, and OpenSSF Scorecard are green. |
| Crates Publish Card | Added | `tracerazor-trice crates` binds workspace Cargo manifests, topological publish order, crates.io status, and README install honesty. Current level is publish-plan locked, not cargo-install live. |
| GitHub release attestations | Added to workflow | The release workflow now generates deterministic release-evidence assets and runs GitHub artifact attestation over platform wheels, binaries, checksums, and release-evidence sidecars. |
| OpenSSF Scorecard | Added to workflow and doctor | `.github/workflows/scorecard.yml` publishes SARIF/Scorecard results; `tracerazor-trice doctor` reports the public Scorecard API and requires score >= 7.0 for release readiness. |
| Integrity Card | Added | `tracerazor-trice integrity` binds offline doctor output, proof-card verifiers, release evidence, crates publish card, installability card, research card, paper manifest, schemas, and CI/release/Scorecard workflow hooks so the proof graph cannot silently drift. |
| Held-out 10 x 2 pilot | Pending | Must run before any 50 x 3 claim run. |
| Held-out 50 x 3 claim run | Pending | Required for S-tier evidence. |

## Method

Run:

```bash
tracerazor-trice suite scaffold --source remote-git-list.json --out suite.json
tracerazor-trice suite readiness suite.json --out docs/trice_suite_readiness.json
tracerazor-trice suite verify-readiness docs/trice_suite_readiness.json --manifest suite.json
tracerazor-trice protocol --manifest suite.json --out docs/trice_protocol_lock.json
tracerazor-trice verify-protocol docs/trice_protocol_lock.json --manifest suite.json
tracerazor-trice suite suite.json --out-dir benchmark/trice/results/heldout-pilot --rounds 1
tracerazor-trice verify-suite benchmark/trice/results/heldout-pilot/trice_suite_evidence_manifest.json
tracerazor-trice design \
  --protocol docs/trice_protocol_lock.json \
  --suite-result benchmark/trice/results/heldout-pilot/trice_suite_results.json \
  --out docs/trice_design_card.json
tracerazor-trice verify-design docs/trice_design_card.json
tracerazor-trice claim \
  --suite-result benchmark/trice/results/heldout-pilot/trice_suite_results.json \
  --manifest benchmark/trice/results/heldout-pilot/trice_suite_evidence_manifest.json \
  --out docs/trice_claim_card.json
tracerazor-trice verify-claim docs/trice_claim_card.json
tracerazor-trice contract --out docs/trice_contract_card.json
tracerazor-trice verify-contract docs/trice_contract_card.json
tracerazor-trice install --out docs/trice_install_card.json --dist-dir dist
tracerazor-trice verify-install docs/trice_install_card.json
tracerazor-trice research --out docs/trice_research_card.json
tracerazor-trice verify-research docs/trice_research_card.json
tracerazor-trice artifact --out docs/trice_artifact_card.json
tracerazor-trice verify-artifact docs/trice_artifact_card.json
tracerazor-trice release-evidence --out docs/trice_release_evidence.json --dist-dir dist
tracerazor-trice verify-release-evidence docs/trice_release_evidence.json
tracerazor-trice crates --out docs/trice_crates_card.json --timeout-s 10
tracerazor-trice verify-crates docs/trice_crates_card.json
tracerazor-trice integrity --out docs/trice_integrity_card.json
tracerazor-trice verify-integrity docs/trice_integrity_card.json
scorecard --repo=github.com/ZulfaqarHafez/TraceRazor
```

The suite manifest must include:

- `git.url`
- `git.rev`
- `task_id`
- `prompt`
- `verify_cmd`
- `adapter_profile`
- `replicates`

The readiness report must say `pilot_execution_ready = true` before the 10 x 2
pilot, and `claim_execution_ready = true` before the 50 x 3 claim run.

## Remote Smoke Result

The current checked-in remote smoke is:

- Source: `https://github.com/pypa/sampleproject.git`
- Revision: `621e4974ca25ce531773def586ba3ed8e736b3fc`
- Task: change `sample.simple.add_one` so `add_one(1) == 3`
- Adapter: declarative source-only JSON patch
- Verifier: `python -c "import sys; sys.path.insert(0, 'src'); from sample.simple import add_one; assert add_one(1) == 3"`
- Result: 83.2% measured input-token savings, zero pass regressions, 100% evidence recall, zero recall failures
- Bundle: 17 hashed entries, verified by `tracerazor-trice verify-bundle`
- Claim card: `docs/trice_remote_smoke_claim_card.json`, `claim_allowed = false`

This proves the remote-git evidence path is operational. It does not change the
S-tier status because one public repo and one replicate do not satisfy the
held-out 10 x 2 pilot or 50 x 3 claim requirements.

## S-Tier Gate

The claim run must satisfy:

| Metric | Required |
|---|---:|
| Remote task clusters | 50 |
| Replicates per task | 3 |
| Mean input-token savings | >= 60% |
| Clustered savings CI lower bound | >= 60% |
| Pass regressions | 0 |
| Evidence recall on solved traces | >= 95% |
| Receipt validation | 100% |
| Evidence manifests | 100% valid |

## Public Trust Work Still Required

- Publish Rust crates or keep the cargo install claim out of public docs.
- Re-run GitHub CI after local clippy and supply-chain fixes.
- Publish 1.1.0 to PyPI only after local gates pass.
- Wait for the piwheels 1.1.0 build and verify visibility.
- Attach SBOMs, checksums, and the verified evidence bundle to the GitHub
  release.

## Decision

Do not call the product S-tier yet. The product may say:

"TraceRazor includes an S-tier evidence gate and live TRICE suite runner."

It may not say:

"TraceRazor has passed S-tier evidence."

That sentence becomes true only when the held-out suite result contains
`s_tier_gate.passed = true`.

The committed claim card should then also say:

```json
{
  "claim_level": "s_tier",
  "claim_allowed": true
}
```
