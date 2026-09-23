# TRICE Release Card

- Release level: `local_release_candidate`
- Release score: **90/100**
- Public release ready: `false`
- Local version: `1.1.0`

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| local_package | yes | installed: version 1.1.0 | local package imports with version |
| bundled_cli | yes | bundled: /home/user/TraceRazor/tracerazor/bin/tracerazor; binary=1.1.0 | CLI binary is bundled or source-build reachable |
| schemas | yes | available: 19 schemas | all public contract schemas are shipped |
| artifact_card_verifies | yes | review_ready_smoke | artifact card verifies |
| reproduction_card_verifies | yes | reviewer_replay_ready_smoke | reproduction card verifies |
| contract_card_verifies | yes | library_contract_locked | public API/CLI/schema contract card verifies |
| install_card_verifies | yes | full_cli_install_ready | clean-wheel installability card verifies |
| release_docs_present | yes | 8/8 | README, trust matrix, release checklist, pyproject, contract card, and install card present |
| pypi | yes | matched: latest=1.1.0 local=1.1.0 | PyPI latest version matches local version |
| piwheels | no | unknown: <urlopen error Tunnel connection failed: 403 Forbidden> | piwheels exposes the local version file |
| crates_io | no | missing: crate tracerazor is not published | crates.io package is published |
| github_tag | no | pending: head=9ca945772fe9 local_tag=False remote_tag=True | local version tag points at current commit locally and remotely |
| github_actions | yes | green: Agent Efficiency Gate=completed/success; Release=completed/success; TraceRazor CI=completed/success | required public workflows are green |
| openssf_scorecard | no | unknown: <urlopen error Tunnel connection failed: 403 Forbidden> | OpenSSF Scorecard is published with score >= 7.0 |
| provenance_plan_documented | yes | trusted publishing/OIDC | trusted publishing and OIDC documented |
| attestation_plan_documented | yes | GitHub artifact attestations | GitHub release artifact attestation documented |
| sbom_plan_documented | yes | CycloneDX/SHA-256 | SBOM and checksum release assets documented |

## Next Actions

- Informational only for 1.1: do not add an sdist solely for piwheels.
- Optional: publish Rust crates only after declaring a stable public Rust API; keep cargo-install claims out of the README meanwhile.
- The remote v1.1.0 tag already exists and must not be reused; fetch it for verification, or bump the version and create a new immutable tag for the next release.
- Run and publish OpenSSF Scorecard until the public score is at least 7.0.

## Hash

- release card: `ac412e95b5edbc148e90a38e42f8ae74600ef8688c07a94a630fa79d62fed1aa`
