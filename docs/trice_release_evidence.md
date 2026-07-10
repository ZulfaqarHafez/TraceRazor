# TRICE Release Evidence

- Package: `tracerazor`
- Version: `1.1.0`
- Evidence level: `release_evidence_ready`
- Evidence score: **100/100**

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| wheel_present | yes | 1 | at least one wheel artifact |
| sdist_absent | yes | 0 | no source distribution until it can satisfy the bundled-auditor contract |
| cli_binary_present | yes | 1 | one built CLI binary |
| proof_cards_present | yes | 6/6 | contract, artifact, reproduction, crates, installability, and research cards |
| evidence_bundles_present | yes | 2/2 | broad and remote smoke evidence bundles |
| paper_artifacts_present | yes | 2/2 | paper PDF and paper manifest |
| artifact_hashes_present | yes | 12/12 | every present artifact has a SHA-256 digest |
| python_sbom_generated | yes | 19 | CycloneDX-style Python SBOM |
| cargo_sbom_generated | yes | 383 | CycloneDX-style Cargo SBOM |
| provenance_statement_generated | yes | 12 | in-toto/SLSA-shaped provenance statement |
| sidecars_hashed | yes | 4/4 | checksums, SBOMs, and provenance sidecars have hashes |

## Release Artifacts

| Artifact | Kind | Present | Path | SHA-256 |
|---|---|---:|---|---|
| rust_cli | binary | yes | `target/release/tracerazor.exe` | `7548683efc5aa9b7a20f5b6a2dabd3ca5d3326ac57b9bc53a1af1d7126c21ae1` |
| paper_pdf | paper | yes | `paper/trice_v3_research_paper.pdf` | `a63a553d01c2ff0e2fe4be7658334d05f74bf4adf64f7eb373ae01fb29a02a45` |
| paper_manifest | paper-manifest | yes | `paper/trice_v3_research_manifest.json` | `b7bd76063e5b53b98d934562f642e4778a59e1c0b03b229de03d3aa9701ef19e` |
| artifact_card | proof-card | yes | `docs/trice_artifact_card.json` | `2ca622a1999b25d27ecf92af85d4549d0d0bac5f9d663fef84cb8ae4851af10a` |
| contract_card | proof-card | yes | `docs/trice_contract_card.json` | `1837de7eda82dc02abf5d9568d307431aac5749ba3eefdbf5a0d2c20a85c4461` |
| crates_card | proof-card | yes | `docs/trice_crates_card.json` | `92f9bb02700bca505e64a665431e9b0e9c46458c87519d9d8cfd87538e412420` |
| install_card | proof-card | yes | `docs/trice_install_card.json` | `7855087064b41c51e9f41e160d8aa977c3a5baddbcaa1cea0ed3c7b2e8a1b618` |
| reproduction_card | proof-card | yes | `docs/trice_reproduction_card.json` | `34762f6ff0511476a038dd1d4bf52e6923aa1ca899280a717ecab15afd66df54` |
| research_card | proof-card | yes | `docs/trice_research_card.json` | `2095432965bb86bb3a559aa2bd2ef3375d688910ea4f00315fb75ac4c6019e18` |
| broad_evidence_bundle | trice-bundle | yes | `benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip` | `a43eeac3ba5239031376aa7b3be97e01ccada6c36648faa15e1c2b3e2d2d2564` |
| remote_smoke_bundle | trice-bundle | yes | `benchmark/trice/results/v2-remote-smoke/trice_remote_smoke_evidence.trice.zip` | `93f31fc51b6c75279b1dc8a637a46e51c7fd9f2aa3ea0ef45fe9d581ffef493d` |
| tracerazor-1.1.0-py3-none-win_amd64.whl | wheel | yes | `build/final-release-1.1.0-agent-native/tracerazor-1.1.0-py3-none-win_amd64.whl` | `2a87bc3c2d6c2bd2a029724203f22591d1eeb6e12f1c8aea360b3808afb99f1f` |

## Sidecars

| Sidecar | File | SHA-256 |
|---|---|---|
| checksums | `trice_release_evidence.checksums.txt` | `23002e6662f03294c0283ace8d46b453f52c7a06b110a1f7f91a0bacd3c50b45` |
| python_sbom | `trice_release_evidence.python.cdx.json` | `d32c8d276cfea0d9aeb3e3589abddbf003f4b9d14ec4b7b16bc8e2c5dc2c79cb` |
| cargo_sbom | `trice_release_evidence.cargo.cdx.json` | `796ad45d98665e70e1f0487374893cc87787cf5f25a7324c51377c081da304b2` |
| provenance | `trice_release_evidence.intoto.json` | `ac8f7f96009fb08b24a712303fcf59c478e1bd7b29aadc91c03ff66850d5ad62` |

## Next Actions

- Attach the release evidence card, checksums, SBOMs, provenance statement, platform wheels, binaries, paper, and evidence bundles to the GitHub release.
- Regenerate this packet after every package rebuild, proof-card change, or evidence-bundle change.
- Publish registry attestations through trusted publishing where supported.

## Hash

- release evidence: `6d45938f447d4a699cbbed6faa7fe4549db5246b22e0b36c4c4ee276cb743004`
