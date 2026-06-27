# TRICE Release Card

- Release level: `local_release_candidate`
- Release score: **65/100**
- Public release ready: `false`
- Local version: `1.0.3`

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| local_package | yes | installed: version 1.0.3 | local package imports with version |
| bundled_cli | yes | source-build: C:\Users\zulfa\TraceRazor\target\release\tracerazor.exe; binary=1.0.3 | CLI binary is bundled or source-build reachable |
| schemas | yes | available: 19 schemas | all public contract schemas are shipped |
| artifact_card_verifies | yes | review_ready_smoke | artifact card verifies |
| reproduction_card_verifies | yes | reviewer_replay_ready_smoke | reproduction card verifies |
| contract_card_verifies | yes | library_contract_locked | public API/CLI/schema contract card verifies |
| install_card_verifies | yes | python_trice_install_ready | clean-wheel installability card verifies |
| release_docs_present | yes | 8/8 | README, trust matrix, release checklist, pyproject, contract card, and install card present |
| pypi | no | mismatch: latest=1.0.2 local=1.0.3 | PyPI latest version matches local version |
| piwheels | no | missing: files=0 local=1.0.3 | piwheels exposes the local version file |
| crates_io | no | missing: crate tracerazor is not published | crates.io package is published |
| github_tag | no | pending: head=ada742be30bb local_tag=False remote_tag=False | local version tag points at current commit locally and remotely |
| github_actions | no | not-green: Agent Efficiency Gate=completed/success; Release=queued/None; TraceRazor CI=completed/failure | required public workflows are green |
| openssf_scorecard | no | missing: OpenSSF Scorecard result is not published yet | OpenSSF Scorecard is published with score >= 7.0 |
| provenance_plan_documented | yes | trusted publishing/OIDC | trusted publishing and OIDC documented |
| attestation_plan_documented | yes | GitHub artifact attestations | GitHub release artifact attestation documented |
| sbom_plan_documented | yes | CycloneDX/SHA-256 | SBOM and checksum release assets documented |

## Next Actions

- Publish 1.0.3 to PyPI or lower the local version until it matches the public registry.
- Wait for piwheels to expose the 1.0.3 file after PyPI publish.
- Publish the Rust crates or keep cargo-install claims out of the README.
- Create and push the v1.0.3 tag only after local gates pass.
- Re-run and fix GitHub Actions until CI, Agent Efficiency Gate, and Release are green.
- Run and publish OpenSSF Scorecard until the public score is at least 7.0.

## Hash

- release card: `a840896aecebd928cdd5b73499c811879a32e2dac91a656928b8f1760a370006`
