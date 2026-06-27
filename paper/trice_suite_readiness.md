# TRICE Suite Readiness

- Scope: `python software-repair/context-control tasks on held-out Git repositories`
- Suite: `bundled-live-six-task-suite`
- Readiness level: `smoke_ready`
- Pilot execution ready: `false`
- Claim execution ready: `false`
- Readiness score: **60/100**
- Task clusters: **6**
- Planned runs: **6**
- Minimum verifier invocations: **12**

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| manifest_valid | yes | trice-suite/v1 | trice-suite/v1 |
| unique_task_ids | yes | ["csv-filter", "dedupe-helpers", "fix-imports", "fix-offby-one", "implement-median", "rename-api"] | all task_id values unique |
| prompts | yes | [true, true, true, true, true, true] | every task has prompt |
| verify_commands | yes | [true, true, true, true, true, true] | every task has verify_cmd |
| interventions | yes | [1, 1, 1, 1, 1, 1] | exactly one intervention per task |
| pilot_task_clusters | no | 6 | >= 10 |
| pilot_replicates_per_task | no | {"csv-filter": 1, "dedupe-helpers": 1, "fix-imports": 1, "fix-offby-one": 1, "implement-median": 1, "rename-api": 1} | each task >= 2 |
| claim_task_clusters | no | 6 | >= 50 |
| claim_replicates_per_task | no | {"csv-filter": 1, "dedupe-helpers": 1, "fix-imports": 1, "fix-offby-one": 1, "implement-median": 1, "rename-api": 1} | each task >= 3 |
| remote_git_sources | no | [null, null, null, null, null, null] | all tasks use remote Git URLs |
| commit_sha_revisions | no | [null, null, null, null, null, null] | all git.rev values are 40-hex commit SHA |
| adapter_profiles | yes | ["trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json", "trice_adapter_profile_bundled_tasks.json"] | all tasks use adapter_profile |
| target_savings | yes | 0.6 | >= 0.600 |
| evidence_recall_gate | yes | 0.95 | >= 0.950 |
| pass_regression_gate | yes | 0 | 0 |
| receipt_validation_gate | yes | True | true |

## Recommendations

- Use locked remote Git sources instead of local paths for pilot and claim suites.
- Pin every git.rev to an immutable 40-hex commit SHA before running held-out evidence.
- Add held-out task clusters until the pilot has at least 10 distinct task_id values.
- Add held-out task clusters until the claim suite has at least 50 distinct task_id values.
- Set replicates to at least 3 for every claim task.

## Contract

- Preflight only: no savings, pass-rate, or S-tier result is claimed.
- Claim execution requires held-out remote Git tasks, immutable revisions, fixed adapter profiles, and repeated live runs.
- Outcome evidence must come from suite results, evidence manifests, claim cards, and bundle verification.

## Hashes

- suite manifest: `fd4335bea5164e4e0dd479ff5154947c54941d9c9704258b018b51c172a7859b`
- readiness report: `6f4baca234454b98a138be96f9d4792e1e1d79c59fd8977b86288da9d8ba2ea4`
