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
| python_sbom_generated | yes | 18 | CycloneDX-style Python SBOM |
| cargo_sbom_generated | yes | 367 | CycloneDX-style Cargo SBOM |
| provenance_statement_generated | yes | 13 | in-toto/SLSA-shaped provenance statement |
| sidecars_hashed | yes | 4/4 | checksums, SBOMs, and provenance sidecars have hashes |

## Release Artifacts

| Artifact | Kind | Present | Path | SHA-256 |
|---|---|---:|---|---|
| rust_cli | binary | yes | `target/release/tracerazor.exe` | `b9e1e29b9a9ce49c937eb69e1614573939d5a7d41efa7c1d11d73d9e1dc128ee` |
| paper_pdf | paper | yes | `paper/trice_v3_research_paper.pdf` | `a63a553d01c2ff0e2fe4be7658334d05f74bf4adf64f7eb373ae01fb29a02a45` |
| paper_manifest | paper-manifest | yes | `paper/trice_v3_research_manifest.json` | `b7bd76063e5b53b98d934562f642e4778a59e1c0b03b229de03d3aa9701ef19e` |
| artifact_card | proof-card | yes | `docs/trice_artifact_card.json` | `e624a39cb688bd22bbd568374d334bc8bfbd75fbf1f8fe96e4a3452669febc40` |
| contract_card | proof-card | yes | `docs/trice_contract_card.json` | `547ec54d1fe6a0e8f10ffe27bcf49fc30b3ac6aa44c5b97c97ad7d8a96fcad7c` |
| crates_card | proof-card | yes | `docs/trice_crates_card.json` | `3ae2ba5b02bfa9348bdd7666e43d4e2e925c82fb00c27f45a4e94cec7f828c4e` |
| install_card | proof-card | yes | `docs/trice_install_card.json` | `8fec158d6bf7c73d084419e2b5737c53228c77a7746d532000d1db2998ba5191` |
| reproduction_card | proof-card | yes | `docs/trice_reproduction_card.json` | `34762f6ff0511476a038dd1d4bf52e6923aa1ca899280a717ecab15afd66df54` |
| research_card | proof-card | yes | `docs/trice_research_card.json` | `2613a20f18eeecc0e1430474c5ea1138a87b917182608275c41f714426df08e1` |
| tracerazor-1.0.3.tar.gz | sdist | yes | `dist/tracerazor-1.0.3.tar.gz` | `f86cae681c21f4af9c759e9a2799f1e69500859185a04d066bf55808e9c0304c` |
| broad_evidence_bundle | trice-bundle | yes | `benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip` | `a43eeac3ba5239031376aa7b3be97e01ccada6c36648faa15e1c2b3e2d2d2564` |
| remote_smoke_bundle | trice-bundle | yes | `benchmark/trice/results/v2-remote-smoke/trice_remote_smoke_evidence.trice.zip` | `93f31fc51b6c75279b1dc8a637a46e51c7fd9f2aa3ea0ef45fe9d581ffef493d` |
| tracerazor-1.0.3-py3-none-any.whl | wheel | yes | `dist/tracerazor-1.0.3-py3-none-any.whl` | `14d9b19c3abc3ac132edbfe3709d8bc673a397dd25fa671742da9b2dd3e58368` |

## Sidecars

| Sidecar | File | SHA-256 |
|---|---|---|
| checksums | `trice_release_evidence.checksums.txt` | `3bfee25411afbe7e196ab83fa0b62cec45e068c4b4eec5f58309cabe93d2bf10` |
| python_sbom | `trice_release_evidence.python.cdx.json` | `c3be794b414e11db70b4d272cdaf4c0d3425a8d448b288c19f4bdb4731f57689` |
| cargo_sbom | `trice_release_evidence.cargo.cdx.json` | `636f4f9876ae0d80fd9e0951ececa3fc31e827f36d0a8452b8f16cf32bc3de6d` |
| provenance | `trice_release_evidence.intoto.json` | `8dd03d0b22c954883086edb2d01196771df1a3dafd7b7030cdd24458ba354c86` |

## Next Actions

- Attach the release evidence card, checksums, SBOMs, provenance statement, wheel, sdist, binary, paper, and evidence bundles to the GitHub release.
- Regenerate this packet after every package rebuild, proof-card change, or evidence-bundle change.
- Publish registry attestations through trusted publishing where supported.

## Hash

- release evidence: `c47d3b9c4f33acbcd088823cc06d3b09ebaaeff01738fcc63dffad5a15519181`
