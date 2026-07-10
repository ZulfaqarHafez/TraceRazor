# TRICE Claim Card

- Scope: `remote-git smoke path on one locked public Python repository`
- Claim level: `smoke`
- Claim allowed: `false`
- Determinism contract score: **100/100**
- Mean input-token savings: **83.2%**
- Clustered CI lower bound: **83.2%**
- Pass regressions: **0**
- Evidence recall minimum: **100.0%**
- Evidence recall failures: **0**
- Accepted runs: **1/1**
- Evidence verification: **ok**

## Requirements

| Requirement | Passed | Observed | Required |
|---|---:|---|---|
| accepted_runs | yes | 0 | 0 unaccepted runs |
| adapter_profiles | yes | [null] | not required |
| clustered_savings_ci_low | yes | 0.831914 | >= 0.600 |
| evidence_recall | yes | {"failures": 0, "minimum": 1.0} | >= 0.950 on every accepted optimized run |
| locked_git_sources | yes | ["git"] | all tasks use locked git sources |
| mean_savings | yes | 0.831914 | >= 0.600 |
| pass_regressions | yes | 0 | <= 0 |
| receipt_validation | yes | enabled | enabled |
| remote_git_sources | yes | ["https://github.com/pypa/sampleproject.git"] | all tasks use remote Git URLs |
| replicates_per_task | no | {"pypa-sampleproject-add-two": 1} | each task >= 3 |
| task_clusters | no | 1 | >= 50 |

## Non-Claims

- Not an S-tier claim; missing task_clusters, replicates_per_task.
- Does not claim universal all-language or all-agent performance.
- Does not certify replay-only savings as live savings.

## Hashes

- suite result: `a3d099d31dca99057da805dedf01a1a0a3dc70e0c74bc8b33d6bec4862272ff5`
- suite manifest: `f8128784fee5d095855fa5cff8db4c83b7fea9f7218f1327ad030adc48f7c220`
- claim card: `a00a66f27f85f397e8c25bef28d651cad3f128cd039e0110b4659dc7a2cfbe98`
