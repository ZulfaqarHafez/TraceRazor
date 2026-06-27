# TRICE Release Evidence

- Package: `tracerazor`
- Version: `1.0.3`
- Evidence level: `release_evidence_ready`
- Evidence score: **100/100**

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| wheel_present | yes | 1 | at least one wheel artifact |
| sdist_present | yes | 1 | one source distribution artifact |
| cli_binary_present | yes | 1 | one built CLI binary |
| proof_cards_present | yes | 6/6 | contract, artifact, reproduction, crates, installability, and research cards |
| evidence_bundles_present | yes | 2/2 | broad and remote smoke evidence bundles |
| paper_artifacts_present | yes | 2/2 | paper PDF and paper manifest |
| artifact_hashes_present | yes | 13/13 | every present artifact has a SHA-256 digest |
| python_sbom_generated | yes | 17 | CycloneDX-style Python SBOM |
| cargo_sbom_generated | yes | 367 | CycloneDX-style Cargo SBOM |
| provenance_statement_generated | yes | 13 | in-toto/SLSA-shaped provenance statement |
| sidecars_hashed | yes | 4/4 | checksums, SBOMs, and provenance sidecars have hashes |

## Release Artifacts

| Artifact | Kind | Present | Path | SHA-256 |
|---|---|---:|---|---|
| rust_cli | binary | yes | `target/release/tracerazor.exe` | `623f468108d970eaf918362d50c44db4670906f00341733a9336d0d4fdc90855` |
| paper_pdf | paper | yes | `paper/trice_v3_research_paper.pdf` | `e989dc19be087a9991ff05424457fd9aeef541056a30472add2e2a1db24b597d` |
| paper_manifest | paper-manifest | yes | `paper/trice_v3_research_manifest.json` | `c381bf08088c8c4d1227ee3b3d0f92439860f00195b362a01c7b2956ffee94d6` |
| artifact_card | proof-card | yes | `docs/trice_artifact_card.json` | `8e6f523699b23b264876f48d41b042c5e74b3b30ba8381ef605bdc1f710f0b8f` |
| contract_card | proof-card | yes | `docs/trice_contract_card.json` | `9d2cb75affbee6676d7e9a00949f41ca5995202b14f9fc0b7a18240f461e0b5c` |
| crates_card | proof-card | yes | `docs/trice_crates_card.json` | `f16c2a70c6b97b0bb311b5dc1cc4398eeb2d7a974f0e00255bda33a122fb16cf` |
| install_card | proof-card | yes | `docs/trice_install_card.json` | `b019465c0cc37cf4af4ef6a2e8d9f5574cdf1ecc539a802ec7bf4ff59b9935a7` |
| reproduction_card | proof-card | yes | `docs/trice_reproduction_card.json` | `579d3b8898f6bdb7ab8b20aa1e58dc6f5118923f3d12a09b8d512fc9fe26eb84` |
| research_card | proof-card | yes | `docs/trice_research_card.json` | `2613a20f18eeecc0e1430474c5ea1138a87b917182608275c41f714426df08e1` |
| tracerazor-1.0.3.tar.gz | sdist | yes | `dist/tracerazor-1.0.3.tar.gz` | `c2a3187435d2e3d6e360273ae488dc6ed1a708a1e9194fd737a19f83d6749adf` |
| broad_evidence_bundle | trice-bundle | yes | `benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip` | `a43eeac3ba5239031376aa7b3be97e01ccada6c36648faa15e1c2b3e2d2d2564` |
| remote_smoke_bundle | trice-bundle | yes | `benchmark/trice/results/v2-remote-smoke/trice_remote_smoke_evidence.trice.zip` | `93f31fc51b6c75279b1dc8a637a46e51c7fd9f2aa3ea0ef45fe9d581ffef493d` |
| tracerazor-1.0.3-py3-none-any.whl | wheel | yes | `dist/tracerazor-1.0.3-py3-none-any.whl` | `2698409bb78759f6e661ad5dd1097815bb5ea1d27ce8445fbe8692acc13f3712` |

## Sidecars

| Sidecar | File | SHA-256 |
|---|---|---|
| checksums | `trice_release_evidence.checksums.txt` | `78f51f677538fa5a0423d532281ae857e9d91806e35ced93859f7e8afa7514fb` |
| python_sbom | `trice_release_evidence.python.cdx.json` | `d91ea8a2b868d5e3be5b983e3a2215ab1076181318cffb5e0d83fa2d3c30a26f` |
| cargo_sbom | `trice_release_evidence.cargo.cdx.json` | `636f4f9876ae0d80fd9e0951ececa3fc31e827f36d0a8452b8f16cf32bc3de6d` |
| provenance | `trice_release_evidence.intoto.json` | `4493e797691d3421d9e368f6824c273f90ad262cd925f24858f87fc6c5b047d0` |

## Next Actions

- Attach the release evidence card, checksums, SBOMs, provenance statement, wheel, sdist, binary, paper, and evidence bundles to the GitHub release.
- Regenerate this packet after every package rebuild, proof-card change, or evidence-bundle change.
- Publish registry attestations through trusted publishing where supported.

## Hash

- release evidence: `0324545e8a1134bec36a96f5fe898b22f9a7f93f265ae0594a2e85c4372a38bb`
