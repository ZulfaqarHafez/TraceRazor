# TRICE Claim Card

- Scope: `python software-repair/context-control tasks on held-out Git repositories`
- Claim level: `failed`
- Claim allowed: `false`
- Determinism contract score: **88/100**
- Mean input-token savings: **81.5%**
- Clustered CI lower bound: **79.0%**
- Pass regressions: **0**
- Evidence recall minimum: **100.0%**
- Evidence recall failures: **0**
- Accepted runs: **6/6**
- Evidence verification: **failed**

## Requirements

| Requirement | Passed | Observed | Required |
|---|---:|---|---|
| accepted_runs | yes | 0 | 0 unaccepted runs |
| adapter_profiles | yes | ["trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json"] | all tasks use adapter_profile |
| clustered_savings_ci_low | yes | 0.789866 | >= 0.600 |
| evidence_recall | yes | {"failures": 0, "minimum": 1.0} | >= 0.950 on every accepted optimized run |
| locked_git_sources | no | ["local", "local", "local", "local", "local", "local"] | all tasks use locked git sources |
| mean_savings | yes | 0.81489 | >= 0.600 |
| pass_regressions | yes | 0 | <= 0 |
| receipt_validation | yes | enabled | enabled |
| remote_git_sources | no | ["local", "local", "local", "local", "local", "local"] | all tasks use remote Git URLs |
| replicates_per_task | no | {"csv-filter": 1, "dedupe-helpers": 1, "fix-imports": 1, "fix-offby-one": 1, "implement-median": 1, "rename-api": 1} | each task >= 3 |
| task_clusters | no | 6 | >= 50 |

## Non-Claims

- Not an S-tier claim; missing task_clusters, replicates_per_task, locked_git_sources, remote_git_sources.
- Evidence verification did not pass.
- Does not claim universal all-language or all-agent performance.
- Does not certify replay-only savings as live savings.

## Hashes

- suite result: `1622ffb36464e6e41cbadecd1c5ed6cceb678af89a2a5a5a3c999258559c3f2d`
- suite manifest: `1a564b4756d0d2432a0e40ba279fa2e23577b040168fa40961f547a0ded4941a`
- claim card: `5c03bc9f6da19d7cabc89e654a93eacfc6d25021ddec5ac9922877be6d771aaa`
