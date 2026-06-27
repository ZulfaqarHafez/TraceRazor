# TRICE V2 Live Rollout Report

Algorithm: `trice-v2-live-user-conditioned-rollout`
Target savings: 60%
Final budget ratio: 40%
Evidence manifest: `trice_v2_evidence_manifest.json`

## Evidence

| Task | Round | Baseline tokens | TRICE tokens | Savings | Baseline pass | TRICE pass | Accepted |
|---|---:|---:|---:|---:|---|---|---|
| rename-api | 1 | 1742 | 277 | 84.1% | yes | yes | yes |

## Deterministic Claim Gate

- Scope: `local_deterministic_smoke`
- Mean savings: 84.1%
- Savings 95% bootstrap CI: [84.1%, 84.1%]
- TRICE pass rate: 100.0% (Wilson 95% CI [20.7%, 100.0%])
- Pass regressions: 0
- Local smoke gate passed: yes
- Broad claim allowed: no
- Rationale: local deterministic smoke passed; broad claim still requires held-out provider runs with repeated trials and clustered confidence intervals

## User-Learned Policy

- user target set to 60% input-token savings
- user requires live rollout evidence, not replay-only acceptance
- replay is allowed only as a preflight, not as final proof
- prefer aggressive compression when live pass preservation holds
- round 1: accepted 84% savings with pass preservation

## Interpretation

This is live-rollout evidence: each condition used a fresh copied workspace,
made real source edits, and passed or failed on the verifier command. It is
not replay evidence. The managed adapter is deterministic for CI; provider
adapters can reuse the same gate as long as they report assembled input
tokens and objective verifier results.
Verifier duration text is normalized and wall-clock metadata is excluded
from evidence hashes because timing noise is not decision evidence.
