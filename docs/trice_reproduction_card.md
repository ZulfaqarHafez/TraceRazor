# TRICE Reproduction Card

- Reproduction level: `reviewer_replay_ready_smoke`
- Reproduction score: **100/100**
- Claim allowed: `false`

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| inputs_available | yes | 10/10 | 10/10 inputs present |
| readiness_reproduces | yes | smoke_ready | readiness verifier ok |
| protocol_reproduces | yes | smoke_protocol_locked | protocol verifier ok |
| design_reproduces | yes | smoke_design_observed | design verifier ok |
| claim_reproduces | yes | smoke | claim-card verifier ok |
| bundle_reproduces | yes | 77 | bundle verifier ok |
| paper_reproduces | yes | 43 | paper manifest verifier ok |

## Commands

- `verify-readiness`: `python -m tracerazor.trice suite verify-readiness docs/trice_suite_readiness.json --manifest examples/trice_suite_bundled_live.json`
- `verify-protocol`: `python -m tracerazor.trice verify-protocol docs/trice_protocol_lock.json --manifest examples/trice_suite_bundled_live.json`
- `verify-design`: `python -m tracerazor.trice verify-design docs/trice_design_card.json --protocol docs/trice_protocol_lock.json --suite-result benchmark/trice/results/v2-broad-smoke/trice_suite_results.json`
- `verify-claim`: `python -m tracerazor.trice verify-claim docs/trice_claim_card.json --suite-result benchmark/trice/results/v2-broad-smoke/trice_suite_results.json --manifest benchmark/trice/results/v2-broad-smoke/trice_suite_evidence_manifest.json`
- `verify-bundle`: `python -m tracerazor.trice verify-bundle benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip`
- `verify-paper`: `python -m tracerazor.trice verify paper/trice_v3_research_manifest.json --result benchmark/trice/results/v2-smoke/trice_v2_live_results.json`
- `verify-artifact`: `python -m tracerazor.trice verify-artifact docs/trice_artifact_card.json`

## Hash

- reproduction card: `620e78496b2071b9139fa0b55df908c0cbd7d1b13798907e285753da664d90cd`
