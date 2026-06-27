# TRICE Live Suite Report

Suite: `bundled-live-six-task-suite`
Algorithm: `trice-v2-suite-live-user-conditioned-rollout`
Evidence manifest: `trice_suite_evidence_manifest.json`

## Tasks

| Task | Replicate | Rounds | Mean savings | Accepted | Pass regressions | Child manifest |
|---|---:|---:|---:|---:|---:|---|
| csv-filter | 1 | 1 | 80.1% | 1 | 0 | `tasks/csv-filter/replicate-1/trice_v2_evidence_manifest.json` |
| dedupe-helpers | 1 | 1 | 84.2% | 1 | 0 | `tasks/dedupe-helpers/replicate-1/trice_v2_evidence_manifest.json` |
| fix-imports | 1 | 1 | 84.2% | 1 | 0 | `tasks/fix-imports/replicate-1/trice_v2_evidence_manifest.json` |
| fix-offby-one | 1 | 1 | 76.6% | 1 | 0 | `tasks/fix-offby-one/replicate-1/trice_v2_evidence_manifest.json` |
| implement-median | 1 | 1 | 79.7% | 1 | 0 | `tasks/implement-median/replicate-1/trice_v2_evidence_manifest.json` |
| rename-api | 1 | 1 | 84.1% | 1 | 0 | `tasks/rename-api/replicate-1/trice_v2_evidence_manifest.json` |

## Aggregate Gate

- Mean savings: 81.5%
- Savings 95% bootstrap CI: [79.0%, 83.5%]
- Clustered-by-task savings 95% CI: [79.0%, 83.5%]
- Task clusters: 6
- Replicates: 6
- Pass regressions: 0
- Evidence recall minimum: 100.0%
- Evidence recall failures: 0
- Local smoke gate passed: yes
- Broad claim allowed: no
- Rationale: local deterministic smoke passed; broad claim still requires held-out provider runs with repeated trials and clustered confidence intervals
- S-tier gate passed: no

## Adapter Breakdown

| Adapter | Runs | Mean savings | Pass regressions |
|---|---:|---:|---:|
| command_profile | 6 | 81.5% | 0 |

## Failure Modes

- Pass regression runs: 0
- Unaccepted runs: 0
- Failed smoke-gate runs: 0

## S-Tier Gate

- Claim level: `not_s_tier`
- Passed: no
- Missing requirements: task_clusters, replicates_per_task, locked_git_sources, remote_git_sources
- Rationale: suite evidence is useful but not broad enough for an S-tier claim

## Interpretation

A suite report is aggregate evidence. Each child task manifest must also
verify, because the suite manifest intentionally hashes child manifests
rather than duplicating every trace and context artifact.
Repo tree fingerprints and intervention provenance are recorded in
`trice_suite_sources.json` before live execution. JSON patch tasks
record patch-spec SHA-256 hashes; command tasks record argv, timeout,
and test-edit policy; adapter-profile tasks record profile SHA-256.
