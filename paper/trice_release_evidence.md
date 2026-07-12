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
| license_notices_present | yes | 2/2 | project license and third-party notices |
| evidence_bundles_present | yes | 2/2 | broad and remote smoke evidence bundles |
| paper_artifacts_present | yes | 2/2 | paper PDF and paper manifest |
| artifact_hashes_present | yes | 14/14 | every present artifact has a SHA-256 digest |
| python_sbom_generated | yes | 19 | CycloneDX-style Python SBOM |
| cargo_sbom_generated | yes | 383 | CycloneDX-style Cargo SBOM |
| python_project_license_in_sbom | yes | 1 | the TraceRazor Python component carries its SPDX license expression |
| cargo_sbom_license_coverage | yes | 383 | every Cargo component carries a resolved license expression |
| provenance_statement_generated | yes | 14 | in-toto/SLSA-shaped provenance statement |
| sidecars_hashed | yes | 4/4 | checksums, SBOMs, and provenance sidecars have hashes |

## Release Artifacts

| Artifact | Kind | Present | Path | SHA-256 |
|---|---|---:|---|---|
| rust_cli | binary | yes | `target/release/tracerazor.exe` | `7548683efc5aa9b7a20f5b6a2dabd3ca5d3326ac57b9bc53a1af1d7126c21ae1` |
| project_license | legal | yes | `LICENSE` | `3dcfbb340c8cf8be36a907935370813acfca0adfc29be5049798140931b7591d` |
| third_party_notices | legal | yes | `THIRD_PARTY_NOTICES.md` | `6e1ad2d2451c1e4e854e34d83fccde037a2a523d5e846646b33ae91bf4b3c0dc` |
| paper_pdf | paper | yes | `paper/trice_v3_research_paper.pdf` | `238219dcdb923f0e35ef1f24772b4bda13ae4f40ac453bf347eca24196c831ca` |
| paper_manifest | paper-manifest | yes | `paper/trice_v3_research_manifest.json` | `4ca9de56f71c5498f84fdda4fdbc955bd5b0adcc45f7bb64869aa032010fb60a` |
| artifact_card | proof-card | yes | `docs/trice_artifact_card.json` | `dbf86de28860a8a23d59e049ad43122a65dfc338cea423e715502461fb7b42df` |
| contract_card | proof-card | yes | `docs/trice_contract_card.json` | `f4635c58f7f396076b119fbe1e1bae5e14385ec7493c4142fd4e66840ca733b3` |
| crates_card | proof-card | yes | `docs/trice_crates_card.json` | `b18bbabb2e7d04a380ba68cdd8d99b96c0208fadff9942371d05d294c4a77d75` |
| install_card | proof-card | yes | `docs/trice_install_card.json` | `d5528d855bd86026b9f6279e4f4eaa23605e92209b56ad6f3f3c6775ce5f0892` |
| reproduction_card | proof-card | yes | `docs/trice_reproduction_card.json` | `74ec65592fa039648a1b78fd73f133b6abf8a528ceb848fab35416ca02436cfa` |
| research_card | proof-card | yes | `docs/trice_research_card.json` | `c1753dfa8b6c68b9bb2a1ebda01a3e37bd022a6801d2b5e136b2e94e8d77b804` |
| broad_evidence_bundle | trice-bundle | yes | `benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip` | `08ee5538891932f64e5c4e80ac63833257802b533d5b4a5a50adbe860b65a37c` |
| remote_smoke_bundle | trice-bundle | yes | `benchmark/trice/results/v2-remote-smoke/trice_remote_smoke_evidence.trice.zip` | `93f31fc51b6c75279b1dc8a637a46e51c7fd9f2aa3ea0ef45fe9d581ffef493d` |
| tracerazor-1.1.0-py3-none-win_amd64.whl | wheel | yes | `C:/Users/zulfa/AppData/Local/Temp/tracerazor-readme-final-4f9708d1c6b84b009d6254bde70f7218/tracerazor-1.1.0-py3-none-win_amd64.whl` | `e0bca1a49cf564e3373cfa2dbd6ef761a9cfefeb115b24c129d67ff833b9998e` |

## Sidecars

| Sidecar | File | SHA-256 |
|---|---|---|
| checksums | `trice_release_evidence.checksums.txt` | `8cf00b947726776516a6967fb44b838ddb6b06d2a33f5d6ec46af05516ee1e4b` |
| python_sbom | `trice_release_evidence.python.cdx.json` | `e851b27bb14731ab38770552f2d6e7fb60ca69f25ee277ca78edcad5b6ad057c` |
| cargo_sbom | `trice_release_evidence.cargo.cdx.json` | `fe84717da1da274f1d57478e0610ba9a1c4fc270f613948b5ac2a36d232c997f` |
| provenance | `trice_release_evidence.intoto.json` | `3ced0b616c8c15f366b216fae00153fd501809c9612bfe7c345e97ec0e860b71` |

## Next Actions

- Attach the release evidence card, checksums, SBOMs, provenance statement, platform wheels, binaries, paper, and evidence bundles to the GitHub release.
- Regenerate this packet after every package rebuild, proof-card change, or evidence-bundle change.
- Publish registry attestations through trusted publishing where supported.

## Hash

- release evidence: `a6f9686410ec592e7ccd222cf87c8232fd3c860d2394a1ae08d504ab9f7ce2ab`
