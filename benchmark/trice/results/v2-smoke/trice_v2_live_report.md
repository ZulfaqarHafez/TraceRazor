# TRICE V2 Live Rollout Report

Algorithm: `trice-v2-live-user-conditioned-rollout`
Target savings: 60%
Final budget ratio: 40%

## Evidence

| Task | Round | Baseline tokens | TRICE tokens | Savings | Baseline pass | TRICE pass | Accepted |
|---|---:|---:|---:|---:|---|---|---|
| csv-filter | 1 | 1895 | 430 | 77.3% | yes | yes | yes |
| dedupe-helpers | 1 | 1826 | 361 | 80.2% | yes | yes | yes |
| fix-imports | 1 | 1806 | 341 | 81.1% | yes | yes | yes |
| fix-offby-one | 1 | 2527 | 634 | 74.9% | yes | yes | yes |
| implement-median | 1 | 1889 | 424 | 77.6% | yes | yes | yes |
| rename-api | 1 | 1814 | 349 | 80.8% | yes | yes | yes |

## User-Learned Policy

- user target set to 60% input-token savings
- user requires live rollout evidence, not replay-only acceptance
- replay is allowed only as a preflight, not as final proof
- adapt budget and safety from user feedback before acting
- prefer aggressive compression when live pass preservation holds
- never edit tests during managed rollouts
- round 1: accepted 77% savings with pass preservation
- round 2: accepted 80% savings with pass preservation
- round 3: accepted 81% savings with pass preservation
- round 4: accepted 75% savings with pass preservation
- round 5: accepted 78% savings with pass preservation
- round 6: accepted 81% savings with pass preservation

## Interpretation

This is live-rollout evidence: each condition used a fresh copied workspace,
made real source edits, and passed or failed on the verifier command. It is
not replay evidence. The managed adapter is deterministic for CI; provider
adapters can reuse the same gate as long as they report assembled input
tokens and objective verifier results.
