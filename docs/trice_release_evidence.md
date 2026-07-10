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
| paper_pdf | paper | yes | `paper/trice_v3_research_paper.pdf` | `c5d1982b11f4f5746d9147c89085fb1bc378ef7c64bd1b2221ba830a1cc96a3d` |
| paper_manifest | paper-manifest | yes | `paper/trice_v3_research_manifest.json` | `47cb26cfe76abb0a70ea159d610abd9460cb1b7d6ee5fdefee03d710b9cf890b` |
| artifact_card | proof-card | yes | `docs/trice_artifact_card.json` | `1a1a3255ae72f9f4dde8f285e37a7e1130e5839a5b7817716e6077df9afa6f33` |
| contract_card | proof-card | yes | `docs/trice_contract_card.json` | `bb25751668e879c2580ce9ea3d818f29850daca679ecdab1f56d617d3d3fdba1` |
| crates_card | proof-card | yes | `docs/trice_crates_card.json` | `c91d94c89917648a342f296684973e638cc820863025800e316d3bb3c39d6ef0` |
| install_card | proof-card | yes | `docs/trice_install_card.json` | `350c93766564c5bf47ceb91d086474d275565395d6dbc17b2961038f79e90969` |
| reproduction_card | proof-card | yes | `docs/trice_reproduction_card.json` | `e5f083ab4c57b95acbbdb10a1a83dfd3bca87b73d9c1904d4bdda2a264975fce` |
| research_card | proof-card | yes | `docs/trice_research_card.json` | `2095432965bb86bb3a559aa2bd2ef3375d688910ea4f00315fb75ac4c6019e18` |
| broad_evidence_bundle | trice-bundle | yes | `benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip` | `a43eeac3ba5239031376aa7b3be97e01ccada6c36648faa15e1c2b3e2d2d2564` |
| remote_smoke_bundle | trice-bundle | yes | `benchmark/trice/results/v2-remote-smoke/trice_remote_smoke_evidence.trice.zip` | `93f31fc51b6c75279b1dc8a637a46e51c7fd9f2aa3ea0ef45fe9d581ffef493d` |
| tracerazor-1.1.0-py3-none-win_amd64.whl | wheel | yes | `dist/tracerazor-1.1.0-py3-none-win_amd64.whl` | `2a87bc3c2d6c2bd2a029724203f22591d1eeb6e12f1c8aea360b3808afb99f1f` |

## Sidecars

| Sidecar | File | SHA-256 |
|---|---|---|
| checksums | `trice_release_evidence.checksums.txt` | `79688b9be66798f2bb02c0333038e79f0a3a701ba56ec03a690baabd41b36759` |
| python_sbom | `trice_release_evidence.python.cdx.json` | `d32c8d276cfea0d9aeb3e3589abddbf003f4b9d14ec4b7b16bc8e2c5dc2c79cb` |
| cargo_sbom | `trice_release_evidence.cargo.cdx.json` | `796ad45d98665e70e1f0487374893cc87787cf5f25a7324c51377c081da304b2` |
| provenance | `trice_release_evidence.intoto.json` | `2b5dc4403a08bdc3f5c35db7a92ae8a016e6017501d59e15c4af993053d3296c` |

## Next Actions

- Attach the release evidence card, checksums, SBOMs, provenance statement, platform wheels, binaries, paper, and evidence bundles to the GitHub release.
- Regenerate this packet after every package rebuild, proof-card change, or evidence-bundle change.
- Publish registry attestations through trusted publishing where supported.

## Hash

- release evidence: `be303111e02dcaaa62373dd790df029427fdcff22f852c3b44f89048d16dc0f3`
