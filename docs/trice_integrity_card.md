# TRICE Integrity Card

- Scope: `TRICE proof graph integrity`
- Integrity level: `proof_graph_integrity_locked`
- Integrity score: **100/100**

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| offline_doctor_core | yes | local_package=installed; bundled_cli=on-path; schemas=available | local package, CLI, and schemas pass offline doctor |
| contract_card_verifies | yes | library_contract_locked | public API/CLI/schema contract verifies |
| artifact_card_verifies | yes | review_ready_smoke | artifact-review packet verifies |
| reproduction_card_verifies | yes | reviewer_replay_ready_smoke | reviewer reproduction packet verifies |
| release_card_verifies | yes | local_release_candidate | release trust card verifies |
| release_evidence_verifies | yes | release_evidence_ready | release evidence packet verifies |
| crates_card_verifies | yes | publish_plan_locked | crates staged-publish card verifies |
| install_card_verifies | yes | full_cli_install_ready | clean-wheel installability card verifies |
| research_card_verifies | yes | research_basis_locked | research-basis card verifies |
| paper_manifest_verifies | yes | 43 | paper manifest and bound result verify |
| schemas_available | yes | 19/19 | all shipped TRICE schemas are present |
| workflows_bound | yes | release_workflow=ok; ci_workflow=ok; scorecard_workflow=ok | CI, release, and Scorecard workflows contain integrity/provenance hooks |
| claim_honesty_bound | yes | claim_allowed=False public_release_ready=False | smoke evidence remains a non-S-tier claim |

## Workflows

| Workflow | Present | Markers | Path |
|---|---:|---|---|
| release_workflow | yes | yes | `.github/workflows/release.yml` |
| ci_workflow | yes | yes | `.github/workflows/tracerazor.yml` |
| scorecard_workflow | yes | yes | `.github/workflows/scorecard.yml` |

## Next Actions

- Run this card in CI after building release artifacts.
- Regenerate the integrity card after changing proof cards, schemas, workflows, README, paper, or release evidence.
- Keep S-tier wording blocked until the held-out remote claim card passes.

## Hash

- integrity card: `1c271b25769d773e70a9d1c60ac50e589c7588113afea0fad24a8b74e687c1e1`
