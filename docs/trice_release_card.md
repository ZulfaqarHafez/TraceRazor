# TRICE Release Card

- Release level: `local_release_candidate`
- Release score: **74/100**
- Public release ready: `false`
- Local version: `1.1.0`

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| local_package | yes | installed: version 1.1.0 | local package imports with version |
| bundled_cli | yes | on-path: C:\Users\zulfa\AppData\Local\Programs\Python\Python311\Scripts\tracerazor.exe; binary=1.1.0 | CLI binary is bundled or source-build reachable |
| schemas | yes | available: 19 schemas | all public contract schemas are shipped |
| artifact_card_verifies | yes | review_ready_smoke | artifact card verifies |
| reproduction_card_verifies | yes | reviewer_replay_ready_smoke | reproduction card verifies |
| contract_card_verifies | yes | library_contract_locked | public API/CLI/schema contract card verifies |
| install_card_verifies | yes | full_cli_install_ready | clean-wheel installability card verifies |
| release_docs_present | yes | 8/8 | README, trust matrix, release checklist, pyproject, contract card, and install card present |
| pypi | no | skipped: offline mode | PyPI latest version matches local version |
| piwheels | no | skipped: offline mode | piwheels exposes the local version file |
| crates_io | no | skipped: offline mode | crates.io package is published |
| github_tag | no | pending: head=6be5ac5b8677 local_tag=False remote_tag=False | local version tag points at current commit locally and remotely |
| github_actions | no | skipped: offline mode | required public workflows are green |
| openssf_scorecard | no | skipped: offline mode | OpenSSF Scorecard is published with score >= 7.0 |
| provenance_plan_documented | yes | trusted publishing/OIDC | trusted publishing and OIDC documented |
| attestation_plan_documented | yes | GitHub artifact attestations | GitHub release artifact attestation documented |
| sbom_plan_documented | yes | CycloneDX/SHA-256 | SBOM and checksum release assets documented |

## Next Actions

- Publish 1.1.0 to PyPI only after local release gates pass.
- Informational only for 1.1: do not add an sdist solely for piwheels.
- Optional: publish Rust crates only after declaring a stable public Rust API; keep cargo-install claims out of the README meanwhile.
- Create and push the v1.1.0 tag only after local gates pass.
- Re-run and fix GitHub Actions until CI, Agent Efficiency Gate, and Release are green.
- Run and publish OpenSSF Scorecard until the public score is at least 7.0.

## Hash

- release card: `0ee102e519af5d215fb1722d34dcfcdc4ebd9abcedccee081c3f31afe0c80a68`
