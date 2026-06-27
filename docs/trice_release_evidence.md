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
| paper_pdf | paper | yes | `paper/trice_v3_research_paper.pdf` | `a63a553d01c2ff0e2fe4be7658334d05f74bf4adf64f7eb373ae01fb29a02a45` |
| paper_manifest | paper-manifest | yes | `paper/trice_v3_research_manifest.json` | `b7bd76063e5b53b98d934562f642e4778a59e1c0b03b229de03d3aa9701ef19e` |
| artifact_card | proof-card | yes | `docs/trice_artifact_card.json` | `273a1df9a8542526243ed59fbc78032886e66a100c62d5bb33cdab432c1d2f36` |
| contract_card | proof-card | yes | `docs/trice_contract_card.json` | `2a6e22b58c2ab3af68bf77cea900c758d1e0e687a70a10eefd74743d6d303126` |
| crates_card | proof-card | yes | `docs/trice_crates_card.json` | `f16c2a70c6b97b0bb311b5dc1cc4398eeb2d7a974f0e00255bda33a122fb16cf` |
| install_card | proof-card | yes | `docs/trice_install_card.json` | `5ce2df710d1d8763ed25bdf6ab79464ada6e22b45982a0b37efbd7eb7a5244f6` |
| reproduction_card | proof-card | yes | `docs/trice_reproduction_card.json` | `34762f6ff0511476a038dd1d4bf52e6923aa1ca899280a717ecab15afd66df54` |
| research_card | proof-card | yes | `docs/trice_research_card.json` | `2613a20f18eeecc0e1430474c5ea1138a87b917182608275c41f714426df08e1` |
| tracerazor-1.0.3.tar.gz | sdist | yes | `dist/tracerazor-1.0.3.tar.gz` | `2be33fcbc0dad91032d6f5c15df1fc5f0e40816f6e88cc3b7bee7c9e0e347095` |
| broad_evidence_bundle | trice-bundle | yes | `benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip` | `a43eeac3ba5239031376aa7b3be97e01ccada6c36648faa15e1c2b3e2d2d2564` |
| remote_smoke_bundle | trice-bundle | yes | `benchmark/trice/results/v2-remote-smoke/trice_remote_smoke_evidence.trice.zip` | `93f31fc51b6c75279b1dc8a637a46e51c7fd9f2aa3ea0ef45fe9d581ffef493d` |
| tracerazor-1.0.3-py3-none-any.whl | wheel | yes | `dist/tracerazor-1.0.3-py3-none-any.whl` | `bda47590b13ab949078d45f866ff2ae61ae2f736f2211945a66cba098facad36` |

## Sidecars

| Sidecar | File | SHA-256 |
|---|---|---|
| checksums | `trice_release_evidence.checksums.txt` | `f3883c618534f9487c65cf56563715ad7f4120b488eb49eea15260b2f54ae32b` |
| python_sbom | `trice_release_evidence.python.cdx.json` | `d91ea8a2b868d5e3be5b983e3a2215ab1076181318cffb5e0d83fa2d3c30a26f` |
| cargo_sbom | `trice_release_evidence.cargo.cdx.json` | `636f4f9876ae0d80fd9e0951ececa3fc31e827f36d0a8452b8f16cf32bc3de6d` |
| provenance | `trice_release_evidence.intoto.json` | `72450124d448ac4ee3cbfd7426d994aae43c6deb96aa9f6d0ada711766502be8` |

## Next Actions

- Attach the release evidence card, checksums, SBOMs, provenance statement, wheel, sdist, binary, paper, and evidence bundles to the GitHub release.
- Regenerate this packet after every package rebuild, proof-card change, or evidence-bundle change.
- Publish registry attestations through trusted publishing where supported.

## Hash

- release evidence: `794f08e1a203bce0f420f7616d2445b1205f6e25d44daaa0c1531f757819363c`
