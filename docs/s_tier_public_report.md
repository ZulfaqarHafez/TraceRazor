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
| TRICE suite gate | Present | Suite-level clustered savings and S-tier missing-requirement reporting exist. |
| Remote-git scaffold | Added | `tracerazor-trice suite scaffold` generates locked manifests from curated task lists. |
| Public doctor | Added | `tracerazor-trice doctor` reports package, schema, registry, tag, and CI state. |
| Suite Readiness | Added | `tracerazor-trice suite readiness` preflights manifests before live pilot/claim spend. Current bundled level is smoke-ready only. |
| Claim Card | Added | `tracerazor-trice claim` generates JSON, Markdown, LaTeX, and SVG claim boundaries. Current level is smoke, not S-tier. |
| Held-out 10 x 2 pilot | Pending | Must run before any 50 x 3 claim run. |
| Held-out 50 x 3 claim run | Pending | Required for S-tier evidence. |

## Method

Run:

```bash
tracerazor-trice suite scaffold --source remote-git-list.json --out suite.json
tracerazor-trice suite readiness suite.json --out docs/trice_suite_readiness.json
tracerazor-trice suite verify-readiness docs/trice_suite_readiness.json --manifest suite.json
tracerazor-trice suite suite.json --out-dir benchmark/trice/results/heldout-pilot --rounds 1
tracerazor-trice verify-suite benchmark/trice/results/heldout-pilot/trice_suite_evidence_manifest.json
tracerazor-trice claim \
  --suite-result benchmark/trice/results/heldout-pilot/trice_suite_results.json \
  --manifest benchmark/trice/results/heldout-pilot/trice_suite_evidence_manifest.json \
  --out docs/trice_claim_card.json
tracerazor-trice verify-claim docs/trice_claim_card.json
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
- Publish 1.0.3 to PyPI only after local gates pass.
- Wait for the piwheels 1.0.3 build and verify visibility.
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
