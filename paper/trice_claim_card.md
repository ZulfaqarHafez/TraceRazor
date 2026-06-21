# TRICE Claim Card

- Scope: `python software-repair/context-control tasks on held-out Git repositories`
- Claim level: `smoke`
- Claim allowed: `false`
- Determinism contract score: **84/100**
- Mean input-token savings: **81.5%**
- Clustered CI lower bound: **79.0%**
- Pass regressions: **0**
- Accepted runs: **6/6**
- Evidence verification: **ok**

## Requirements

| Requirement | Passed | Observed | Required |
|---|---:|---|---|
| accepted_runs | yes | 0 | 0 unaccepted runs |
| adapter_profiles | yes | ["trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json"] | all tasks use adapter_profile |
| clustered_savings_ci_low | yes | 0.789866 | >= 0.600 |
| locked_git_sources | no | ["local", "local", "local", "local", "local", "local"] | all tasks use locked git sources |
| mean_savings | yes | 0.81489 | >= 0.600 |
| pass_regressions | yes | 0 | <= 0 |
| receipt_validation | yes | enabled | enabled |
| remote_git_sources | no | ["local", "local", "local", "local", "local", "local"] | all tasks use remote Git URLs |
| replicates_per_task | no | {"csv-filter": 1, "dedupe-helpers": 1, "fix-imports": 1, "fix-offby-one": 1, "implement-median": 1, "rename-api": 1} | each task >= 3 |
| task_clusters | no | 6 | >= 50 |

## Non-Claims

- Not an S-tier claim; missing task_clusters, replicates_per_task, locked_git_sources, remote_git_sources.
- Does not claim universal all-language or all-agent performance.
- Does not certify replay-only savings as live savings.

## Hashes

- suite result: `db2b55255c76f4cdf8216a1eb7cb996fd7b569e3ae17940e2105e924080b6eec`
- suite manifest: `b43e7b0e11912909928ced531a3429b778b05674935e97d227fb5ac25766d2a8`
- claim card: `4b8ac48b6d982a500bf0012cf31cc212194f0174c6193215c5534c6a5b754d86`
