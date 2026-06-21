# TRICE Live Suite Report

Suite: `fix-offby-one-suite`
Algorithm: `trice-v2-suite-live-user-conditioned-rollout`
Evidence manifest: `trice_suite_evidence_manifest.json`

## Tasks

| Task | Replicate | Rounds | Mean savings | Accepted | Pass regressions | Child manifest |
|---|---:|---:|---:|---:|---:|---|
| fix-offby-one-suite | 1 | 1 | 76.6% | 1 | 0 | `tasks/fix-offby-one-suite/replicate-1/trice_v2_evidence_manifest.json` |
| fix-offby-one-suite | 2 | 1 | 76.6% | 1 | 0 | `tasks/fix-offby-one-suite/replicate-2/trice_v2_evidence_manifest.json` |
| fix-offby-one-suite | 3 | 1 | 76.6% | 1 | 0 | `tasks/fix-offby-one-suite/replicate-3/trice_v2_evidence_manifest.json` |

## Aggregate Gate

- Mean savings: 76.6%
- Savings 95% bootstrap CI: [76.6%, 76.6%]
- Clustered-by-task savings 95% CI: [76.6%, 76.6%]
- Task clusters: 1
- Replicates: 3
- Pass regressions: 0
- Local smoke gate passed: yes
- Broad claim allowed: no
- Rationale: local deterministic smoke passed; broad claim still requires held-out provider runs with repeated trials and clustered confidence intervals

## Interpretation

A suite report is aggregate evidence. Each child task manifest must also
verify, because the suite manifest intentionally hashes child manifests
rather than duplicating every trace and context artifact.
