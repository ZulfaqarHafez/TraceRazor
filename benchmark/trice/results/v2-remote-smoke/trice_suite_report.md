# TRICE Live Suite Report

Suite: `trice-remote-smoke`
Algorithm: `trice-v2-suite-live-user-conditioned-rollout`
Evidence manifest: `trice_suite_evidence_manifest.json`

## Tasks

| Task | Replicate | Rounds | Mean savings | Accepted | Pass regressions | Child manifest |
|---|---:|---:|---:|---:|---:|---|
| pypa-sampleproject-add-two | 1 | 1 | 83.2% | 1 | 0 | `tasks/pypa-sampleproject-add-two/replicate-1/trice_v2_evidence_manifest.json` |

## Aggregate Gate

- Mean savings: 83.2%
- Savings 95% bootstrap CI: [83.2%, 83.2%]
- Clustered-by-task savings 95% CI: [83.2%, 83.2%]
- Task clusters: 1
- Replicates: 1
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
| json_patch | 1 | 83.2% | 0 |

## Failure Modes

- Pass regression runs: 0
- Unaccepted runs: 0
- Failed smoke-gate runs: 0

## S-Tier Gate

- Claim level: `not_s_tier`
- Passed: no
- Missing requirements: task_clusters, replicates_per_task
- Rationale: suite evidence is useful but not broad enough for an S-tier claim

## Interpretation

A suite report is aggregate evidence. Each child task manifest must also
verify, because the suite manifest intentionally hashes child manifests
rather than duplicating every trace and context artifact.
Repo tree fingerprints and intervention provenance are recorded in
`trice_suite_sources.json` before live execution. JSON patch tasks
record patch-spec SHA-256 hashes; command tasks record argv, timeout,
and test-edit policy; adapter-profile tasks record profile SHA-256.
