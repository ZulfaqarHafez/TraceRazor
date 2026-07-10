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
| paper_pdf | paper | yes | `paper/trice_v3_research_paper.pdf` | `2c2e8a764fec74e1a9a9b5b3146b375eb9aec39054fdfa4e86468913a64b3467` |
| paper_manifest | paper-manifest | yes | `paper/trice_v3_research_manifest.json` | `564eb14772d42ef613a5def3a7f18ec6a5d50f378f8d8789fde1d46f0897d8cc` |
| artifact_card | proof-card | yes | `docs/trice_artifact_card.json` | `9f28fe165889580da35c2aff05ce217d3efba1bb74289c090a447efc8a63724e` |
| contract_card | proof-card | yes | `docs/trice_contract_card.json` | `9ad2dafb9665067dc652a713c2614829b1051729f607ae48d88597c9f363ae11` |
| crates_card | proof-card | yes | `docs/trice_crates_card.json` | `c91d94c89917648a342f296684973e638cc820863025800e316d3bb3c39d6ef0` |
| install_card | proof-card | yes | `docs/trice_install_card.json` | `77f76c7ddbb3970f659c63c9c6c2cfa81cbe214e43470b4fcff8af6acaeb3180` |
| reproduction_card | proof-card | yes | `docs/trice_reproduction_card.json` | `ed4a21787b9120ec50dfb223f98bffa20b1a97c4ccad78001f83f8b59e209c44` |
| research_card | proof-card | yes | `docs/trice_research_card.json` | `2095432965bb86bb3a559aa2bd2ef3375d688910ea4f00315fb75ac4c6019e18` |
| broad_evidence_bundle | trice-bundle | yes | `benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip` | `08ee5538891932f64e5c4e80ac63833257802b533d5b4a5a50adbe860b65a37c` |
| remote_smoke_bundle | trice-bundle | yes | `benchmark/trice/results/v2-remote-smoke/trice_remote_smoke_evidence.trice.zip` | `93f31fc51b6c75279b1dc8a637a46e51c7fd9f2aa3ea0ef45fe9d581ffef493d` |
| tracerazor-1.1.0-py3-none-win_amd64.whl | wheel | yes | `dist/tracerazor-1.1.0-py3-none-win_amd64.whl` | `2a87bc3c2d6c2bd2a029724203f22591d1eeb6e12f1c8aea360b3808afb99f1f` |

## Sidecars

| Sidecar | File | SHA-256 |
|---|---|---|
| checksums | `trice_release_evidence.checksums.txt` | `5dc2a2a58f34221bc37043a597910c028c23a5feeb816c31b28bdb677c09022a` |
| python_sbom | `trice_release_evidence.python.cdx.json` | `d32c8d276cfea0d9aeb3e3589abddbf003f4b9d14ec4b7b16bc8e2c5dc2c79cb` |
| cargo_sbom | `trice_release_evidence.cargo.cdx.json` | `796ad45d98665e70e1f0487374893cc87787cf5f25a7324c51377c081da304b2` |
| provenance | `trice_release_evidence.intoto.json` | `acc5632f61a5d6f9297ad0589dd9801f292003e2a50ad3a1e5336ce862a7b26b` |

## Next Actions

- Attach the release evidence card, checksums, SBOMs, provenance statement, platform wheels, binaries, paper, and evidence bundles to the GitHub release.
- Regenerate this packet after every package rebuild, proof-card change, or evidence-bundle change.
- Publish registry attestations through trusted publishing where supported.

## Hash

- release evidence: `eb8d29014eb7082c50f324769454d6c78ce8c9d0a10ed8b33b3ec63e79c06625`
