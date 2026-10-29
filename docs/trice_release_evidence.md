# TRICE Release Evidence

- Package: `tracerazor`
- Version: `1.2.0`
- Evidence level: `release_evidence_ready`
- Evidence score: **100/100**

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| wheel_present | yes | 5 | at least one wheel artifact |
| sdist_absent | yes | 0 | no source distribution until it can satisfy the bundled-auditor contract |
| cli_binary_present | yes | 1 | one built CLI binary |
| proof_cards_present | yes | 6/6 | contract, artifact, reproduction, crates, installability, and research cards |
| license_notices_present | yes | 2/2 | project license and third-party notices |
| evidence_bundles_present | yes | 2/2 | broad and remote smoke evidence bundles |
| paper_artifacts_present | yes | 2/2 | paper PDF and paper manifest |
| artifact_hashes_present | yes | 18/18 | every present artifact has a SHA-256 digest |
| python_sbom_generated | yes | 19 | CycloneDX-style Python SBOM |
| cargo_sbom_generated | yes | 379 | CycloneDX-style Cargo SBOM |
| python_project_license_in_sbom | yes | 1 | the TraceRazor Python component carries its SPDX license expression |
| cargo_sbom_license_coverage | yes | 379 | every Cargo component carries a resolved license expression |
| provenance_statement_generated | yes | 18 | in-toto/SLSA-shaped provenance statement |
| sidecars_hashed | yes | 4/4 | checksums, SBOMs, and provenance sidecars have hashes |

## Release Artifacts

| Artifact | Kind | Present | Path | SHA-256 |
|---|---|---:|---|---|
| rust_cli | binary | yes | `release-cli/tracerazor` | `a1643254fd784e8a28e7bb82c068a0f4ddd6f1887aaaaedd0d8e3463d0562402` |
| project_license | legal | yes | `LICENSE` | `c9a9384e1f51f9a463b2d7eda6153a16000cffb7973e6ccfefdd6032e9cbe7f4` |
| third_party_notices | legal | yes | `THIRD_PARTY_NOTICES.md` | `d08d1eb48c5de4e890044a42ed865a25d18d14626565b92adb9db27be2d18e1b` |
| paper_pdf | paper | yes | `paper/trice_v3_research_paper.pdf` | `238219dcdb923f0e35ef1f24772b4bda13ae4f40ac453bf347eca24196c831ca` |
| paper_manifest | paper-manifest | yes | `paper/trice_v3_research_manifest.json` | `4ca9de56f71c5498f84fdda4fdbc955bd5b0adcc45f7bb64869aa032010fb60a` |
| artifact_card | proof-card | yes | `docs/trice_artifact_card.json` | `b35e4ad0fb2afc7d57f28f4b19e1b953acc65c4d1f5ef8fe66e9ba4ca494914a` |
| contract_card | proof-card | yes | `docs/trice_contract_card.json` | `ced41a779dff9e8c3c17752a78fb505b09115d09a745e2c2db0293a12753d233` |
| crates_card | proof-card | yes | `docs/trice_crates_card.json` | `561d2e228c5b95425e02685be39d5aca985bd55d939f2186c0a1672e7bcd4102` |
| install_card | proof-card | yes | `docs/trice_install_card.json` | `85c6a5b2bc0bf7dd446517f023b063116ce9b7e4065d18fc4f395798f3379d7f` |
| reproduction_card | proof-card | yes | `docs/trice_reproduction_card.json` | `74ec65592fa039648a1b78fd73f133b6abf8a528ceb848fab35416ca02436cfa` |
| research_card | proof-card | yes | `docs/trice_research_card.json` | `c1753dfa8b6c68b9bb2a1ebda01a3e37bd022a6801d2b5e136b2e94e8d77b804` |
| broad_evidence_bundle | trice-bundle | yes | `benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip` | `08ee5538891932f64e5c4e80ac63833257802b533d5b4a5a50adbe860b65a37c` |
| remote_smoke_bundle | trice-bundle | yes | `benchmark/trice/results/v2-remote-smoke/trice_remote_smoke_evidence.trice.zip` | `93f31fc51b6c75279b1dc8a637a46e51c7fd9f2aa3ea0ef45fe9d581ffef493d` |
| tracerazor-1.2.0-py3-none-macosx_11_0_arm64.whl | wheel | yes | `dist/tracerazor-1.2.0-py3-none-macosx_11_0_arm64.whl` | `30067b835aa61eacd786e43883d051c7b8bb0cf1b7537f9e0f3c8f42a99c2b67` |
| tracerazor-1.2.0-py3-none-macosx_11_0_x86_64.whl | wheel | yes | `dist/tracerazor-1.2.0-py3-none-macosx_11_0_x86_64.whl` | `98747baa9d531fc3c1c75e939d9bff7b4ea68ddc4fbd9ac7ff38e345f61e5005` |
| tracerazor-1.2.0-py3-none-manylinux_2_35_x86_64.whl | wheel | yes | `dist/tracerazor-1.2.0-py3-none-manylinux_2_35_x86_64.whl` | `ea5db152c4c3804d019c0bc518e786675c19e2a6844d99a778f365d5a460d182` |
| tracerazor-1.2.0-py3-none-manylinux_2_39_aarch64.whl | wheel | yes | `dist/tracerazor-1.2.0-py3-none-manylinux_2_39_aarch64.whl` | `8cfc50cf31ef6ea1b3c856c187a8a9a6fb7030f77869a85cd3bf17dc742d7666` |
| tracerazor-1.2.0-py3-none-win_amd64.whl | wheel | yes | `dist/tracerazor-1.2.0-py3-none-win_amd64.whl` | `14f94efd55a41b13c15b5e646a09307147f76917034451b0190cbd6fdf488a69` |

## Sidecars

| Sidecar | File | SHA-256 |
|---|---|---|
| checksums | `trice_release_evidence.checksums.txt` | `035c8cfa90e0105b4fdc43b4e4d7f04538f582906ab675109fad031821b1f5a7` |
| python_sbom | `trice_release_evidence.python.cdx.json` | `df97aadaa498028e6016dbb0bee301e0ad92a6310cdd5398f584c95631be51b9` |
| cargo_sbom | `trice_release_evidence.cargo.cdx.json` | `e46c044626aaaecda34e34e27b9e8077302ad5fca088effe53a5b3de82678679` |
| provenance | `trice_release_evidence.intoto.json` | `625a5a3e6c22c12324ae579657a64a5ea300cd73445d0a856a6b4e382eac7445` |

## Next Actions

- Attach the release evidence card, checksums, SBOMs, provenance statement, platform wheels, binaries, paper, and evidence bundles to the GitHub release.
- Regenerate this packet after every package rebuild, proof-card change, or evidence-bundle change.
- Publish registry attestations through trusted publishing where supported.

## Hash

- release evidence: `63707ad993754dd77dd2991f39397f61530834981ad97c87b1f49459e55ba4be`
