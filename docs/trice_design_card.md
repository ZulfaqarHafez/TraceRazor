# TRICE Design Card

- Design level: `smoke_design_observed`
- Design score: **65/100**
- Claim design ready: `false`
- Observed mean savings: **81.5%**
- Observed clustered CI low: **79.0%**
- Projected claim lower bound: **80.6%**
- Projected clusters required by variance: **1**

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| protocol_verifies | yes | smoke_protocol_locked | protocol lock verifies |
| suite_result_schema | yes | trice-suite-result/v1 | trice-suite-result/v1 |
| primary_metric_locked | yes | input_token_savings | input_token_savings |
| observed_mean_above_target | yes | 0.81489 | >= 0.600 |
| observed_clustered_ci_above_target | yes | 0.789866 | >= 0.600 |
| zero_pass_regressions | yes | 0 | 0 |
| all_runs_accepted | yes | 6/6 | all runs accepted |
| pilot_task_clusters | no | 6 | >= 10 |
| claim_task_clusters | no | 6 | >= 50 |
| claim_replicates | no | {"csv-filter": 1, "dedupe-helpers": 1, "fix-imports": 1, "fix-offby-one": 1, "implement-median": 1, "rename-api": 1} | each task >= 3 |
| projected_claim_ci_above_target | yes | 0.806114 | >= 0.600 |
| protocol_claim_ready | no | smoke_protocol_locked | claim_protocol_ready |

## Next Actions

- Run or curate at least 10 task clusters before interpreting pilot design.
- Scale the claim suite to at least 50 held-out task clusters.
- Run at least 3 replicates per task cluster for the claim suite.
- Regenerate a claim-ready protocol lock with held-out remote Git commits and adapter profiles.

## Hashes

- protocol lock: `3e71413180ad94f43be0b6a35717ced48359b75ea382035c524a9b91e5e8296e`
- suite result: `73717655c94009153c47254c548f28650144df0aa3649d6e791eef6868992c5b`
- design card: `3b35c89b335f0f6deadaf7bf3d83c8296e945cde7e582cf3d411ae1abd422817`
