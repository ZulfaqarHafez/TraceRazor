# TRICE Release Card

- Release level: `local_release_candidate`
- Release score: **74/100**
- Public release ready: `false`
- Local version: `1.2.0`

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| local_package | yes | installed: version 1.2.0 | local package imports with version |
| bundled_cli | yes | source-build: /home/user/TraceRazor/target/release/tracerazor; binary=1.2.0 | CLI binary is bundled or source-build reachable |
| schemas | yes | available: 19 schemas | all public contract schemas are shipped |
| artifact_card_verifies | yes | review_ready_smoke | artifact card verifies |
| reproduction_card_verifies | yes | reviewer_replay_ready_smoke | reproduction card verifies |
| contract_card_verifies | yes | library_contract_locked | public API/CLI/schema contract card verifies |
| install_card_verifies | yes | full_cli_install_ready | clean-wheel installability card verifies |
| release_docs_present | yes | 8/8 | README, trust matrix, release checklist, pyproject, contract card, and install card present |
| pypi | no | mismatch: latest=1.1.0 local=1.2.0 | PyPI latest version matches local version |
| piwheels | no | missing: files=0 local=1.2.0 | piwheels exposes the local version file |
| crates_io | no | missing: crate tracerazor is not published | crates.io package is published |
| github_tag | no | pending: head=a8e130f79e8a local_tag=False remote_tag=True | local version tag points at current commit locally and remotely |
| github_actions | no | not-green: Agent Efficiency Gate=completed/success; Release=completed/failure; TraceRazor CI=completed/success | required public workflows are green |
| openssf_scorecard | no | below-threshold: score=3.8 minimum=7.0 date=2026-09-14T09:07:41Z commit=9ca945772fe9 | OpenSSF Scorecard is published with score >= 7.0 |
| provenance_plan_documented | yes | trusted publishing/OIDC | trusted publishing and OIDC documented |
| attestation_plan_documented | yes | GitHub artifact attestations | GitHub release artifact attestation documented |
| sbom_plan_documented | yes | CycloneDX/SHA-256 | SBOM and checksum release assets documented |

## Next Actions

- Publish 1.2.0 to PyPI only after local release gates pass.
- Informational only for 1.1: do not add an sdist solely for piwheels.
- Optional: publish Rust crates only after declaring a stable public Rust API; keep cargo-install claims out of the README meanwhile.
- The remote v1.2.0 tag already exists and must not be reused; fetch it for verification, or bump the version and create a new immutable tag for the next release.
- Re-run and fix GitHub Actions until CI, Agent Efficiency Gate, and Release are green.
- Run and publish OpenSSF Scorecard until the public score is at least 7.0.

## Hash

- release card: `5fb356e90a38bddd7a62460efbd512b36c67a6c54e6a22b4ba4d00bae2cd6739`
