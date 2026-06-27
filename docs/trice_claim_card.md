# TRICE Claim Card

- Scope: `python software-repair/context-control tasks on held-out Git repositories`
- Claim level: `smoke`
- Claim allowed: `false`
- Determinism contract score: **100/100**
- Mean input-token savings: **81.5%**
- Clustered CI lower bound: **79.0%**
- Pass regressions: **0**
- Evidence recall minimum: **100.0%**
- Evidence recall failures: **0**
- Accepted runs: **6/6**
- Evidence verification: **ok**

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
- Does not claim universal all-language or all-agent performance.
- Does not certify replay-only savings as live savings.

## Hashes

- suite result: `1622ffb36464e6e41cbadecd1c5ed6cceb678af89a2a5a5a3c999258559c3f2d`
- suite manifest: `1d05fcbdb173da8629910545191ec55e3144743a8ded98f687d66d4df8f1d46b`
- claim card: `6f20ecbd0a990960b4d6295f820dba886836661e51c7cf9ebd0541637783fa72`
