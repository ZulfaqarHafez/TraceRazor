# TRICE Protocol Lock

- Protocol id: `bundled-live-six-task-suite:fd4335bea5164e4e`
- Scope: `python software-repair/context-control tasks on held-out Git repositories`
- Protocol level: `smoke_protocol_locked`
- Protocol score: **81/100**
- Claim allowed by protocol: `false`
- Task clusters: **6**
- Planned runs: **6**
- Primary metric: `input_token_savings`
- Target savings: **60.0%**

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| manifest_valid | yes | trice-suite/v1 | trice-suite/v1 |
| unique_task_ids | yes | ["csv-filter", "dedupe-helpers", "fix-imports", "fix-offby-one", "implement-median", "rename-api"] | all task_id values unique |
| deterministic_interventions | yes | [1, 1, 1, 1, 1, 1] | exactly one intervention per task |
| primary_metric | yes | input_token_savings | input_token_savings |
| cost_quality_joint_gate | yes | ["input_token_savings", "pass_regressions"] | savings measured with pass preservation |
| target_mean_savings | yes | 0.6 | >= 0.600 |
| clustered_ci_target | yes | 0.6 | >= 0.600 |
| evidence_recall_gate | yes | 0.95 | >= 0.950 |
| pass_regression_gate | yes | 0 | 0 |
| receipt_validation_gate | yes | True | true |
| remote_git_sources | no | [null, null, null, null, null, null] | all tasks use remote Git URLs |
| commit_sha_revisions | no | [null, null, null, null, null, null] | all git.rev values are 40-hex commit SHA |
| adapter_profiles | yes | ["trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json"] | all tasks use adapter_profile |
| pilot_task_clusters | no | 6 | >= 10 |
| pilot_replicates_per_task | no | {"csv-filter": 1, "dedupe-helpers": 1, "fix-imports": 1, "fix-offby-one": 1, "implement-median": 1, "rename-api": 1} | each task >= 2 |
| claim_task_clusters | no | 6 | >= 50 |
| claim_replicates_per_task | no | {"csv-filter": 1, "dedupe-helpers": 1, "fix-imports": 1, "fix-offby-one": 1, "implement-median": 1, "rename-api": 1} | each task >= 3 |
| all_runs_accepted_gate | yes | True | all optimized runs accepted |
| evidence_bundle_required | yes | .trice.zip | portable evidence bundle verifies |
| claim_card_required | yes | trice-claim-card/v1 | claim card verifies before README S-tier wording |
| artifact_card_required | yes | trice-artifact-card/v1 | artifact card verifies before release claim |

## Non-Claims

- Protocol lock is not outcome evidence; it does not claim measured savings or task success.
- Not a claim-ready protocol until the suite has 50 task clusters and 3 replicates per task.
- Not a held-out remote-repo protocol until every task uses a locked remote Git commit.
- Does not permit README S-tier wording without a passing claim card and verified artifact card.

## Next Actions

- Replace local suite tasks with remote Git URLs pinned to immutable 40-hex commits.
- Build the 10-task x 2-replicate pilot protocol before the claim run.
- Scale to 50 task clusters and 3 replicates per task for the S-tier protocol.

## Hashes

- suite manifest: `fd4335bea5164e4e0dd479ff5154947c54941d9c9704258b018b51c172a7859b`
- readiness preflight: `6f4baca234454b98a138be96f9d4792e1e1d79c59fd8977b86288da9d8ba2ea4`
- protocol lock: `d3cebfda67594e0bd5e69d6cf96ff2126728c95fa5c665de2a5447c340761a70`
