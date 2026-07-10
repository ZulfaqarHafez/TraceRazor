# TRICE Artifact Card

- Scope: `TRICE deterministic context-control evidence package`
- Artifact level: `review_ready_smoke`
- Artifact review score: **100/100**
- Readiness level: `smoke_ready`
- Claim allowed: `false`

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| artifacts_available | yes | 17/17 | 17/17 present |
| readiness_verifies | yes | smoke_ready | readiness hash and suite manifest hash verify |
| protocol_lock_verifies | yes | smoke_protocol_locked | protocol hash and deterministic suite rebuild verify |
| design_card_verifies | yes | smoke_design_observed | design-card hash and deterministic protocol/result rebuild verify |
| reproduction_card_verifies | yes | reviewer_replay_ready_smoke | reproduction-card hash, input hashes, and deterministic rebuild verify |
| contract_card_verifies | yes | library_contract_locked | public API/CLI/schema contract card verifies |
| install_card_verifies | yes | full_cli_install_ready | clean-wheel installability card verifies |
| research_card_verifies | yes | research_basis_locked | research-basis card verifies |
| claim_card_verifies | yes | smoke | claim-card hash and bound suite hashes verify |
| remote_smoke_claim_verifies | yes | smoke | remote-git smoke claim card hash and bound suite hashes verify |
| evidence_bundle_verifies | yes | 77 | bundle hashes and child manifests verify |
| remote_smoke_bundle_verifies | yes | 17 | remote-git smoke bundle hashes and child manifest verify |
| paper_manifest_verifies | yes | 43 | paper artifacts and result hash verify |
| schemas_available | yes | 19/19 | 19/19 schemas present |
| claim_honesty | yes | smoke | non-S-tier evidence must not allow S-tier claim |
| remote_smoke_honesty | yes | smoke | remote smoke evidence must not allow S-tier claim |
| readiness_honesty | yes | smoke_ready | smoke package must not be claim-ready |

## Availability

| Artifact | Present | Path | SHA-256 |
|---|---:|---|---|
| readiness | yes | `docs/trice_suite_readiness.json` | `f98e2481ce47a262e618f0adee5481ee7227347e900613bfc68115cc420be1d7` |
| protocol_lock | yes | `docs/trice_protocol_lock.json` | `3e71413180ad94f43be0b6a35717ced48359b75ea382035c524a9b91e5e8296e` |
| design_card | yes | `docs/trice_design_card.json` | `dffa666fb9ba1deff7ba467da24b68b7dd4c19d5f0b0fa2e8c0f3cb45be74d39` |
| reproduction_card | yes | `docs/trice_reproduction_card.json` | `ed4a21787b9120ec50dfb223f98bffa20b1a97c4ccad78001f83f8b59e209c44` |
| contract_card | yes | `docs/trice_contract_card.json` | `9ad2dafb9665067dc652a713c2614829b1051729f607ae48d88597c9f363ae11` |
| install_card | yes | `docs/trice_install_card.json` | `77f76c7ddbb3970f659c63c9c6c2cfa81cbe214e43470b4fcff8af6acaeb3180` |
| research_card | yes | `docs/trice_research_card.json` | `2095432965bb86bb3a559aa2bd2ef3375d688910ea4f00315fb75ac4c6019e18` |
| claim | yes | `docs/trice_claim_card.json` | `5bf4de6c6ae6c74b7bfcef12a924890ae633183ad8120d707f6531a9a1b8f9cf` |
| remote_smoke_claim | yes | `docs/trice_remote_smoke_claim_card.json` | `e296651963ef02ccfde115ab59f0cc87e5d938548878ed63ef8f8660113da641` |
| evidence_bundle | yes | `benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip` | `08ee5538891932f64e5c4e80ac63833257802b533d5b4a5a50adbe860b65a37c` |
| remote_smoke_bundle | yes | `benchmark/trice/results/v2-remote-smoke/trice_remote_smoke_evidence.trice.zip` | `93f31fc51b6c75279b1dc8a637a46e51c7fd9f2aa3ea0ef45fe9d581ffef493d` |
| paper_manifest | yes | `paper/trice_v3_research_manifest.json` | `564eb14772d42ef613a5def3a7f18ec6a5d50f378f8d8789fde1d46f0897d8cc` |
| paper_result | yes | `benchmark/trice/results/v2-smoke/trice_v2_live_results.json` | `edf72f856f37e2c773ffefc31c2ae0b06cab67cc6e456044df3ff03911b58d8a` |
| paper_tex | yes | `paper/trice_v3_research_paper.tex` | `d2b054232679cedf748ccf2a3b95c283e3a148a57e7c0cbaafbde0a3ef527709` |
| paper_pdf | yes | `paper/trice_v3_research_paper.pdf` | `2c2e8a764fec74e1a9a9b5b3146b375eb9aec39054fdfa4e86468913a64b3467` |
| readme | yes | `README.md` | `9e4785baab70f8818940e90ee01050074048d7bd73c9dbca2416699cd59f9638` |
| library_doc | yes | `docs/trice_library.md` | `a256eb788c0dc9c04a976e19e1ca0f8a788c2c5b03e2b8433f3114cff55524bf` |

## Next Actions

- Publish the artifact card with the release assets.
- Run the held-out remote pilot, then regenerate readiness, protocol, design, reproduction, contract, claim, bundle, paper, and artifact cards.
- Do not upgrade the README to S-tier passed until claim_allowed is true on held-out evidence.

## Hash

- artifact card: `fc4dfe2194384110ad32a4c892eb93f8a424b08a87a235c4a878212016ce53ec`
