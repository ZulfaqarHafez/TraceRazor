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
| install_card_verifies | yes | python_trice_install_ready | clean-wheel installability card verifies |
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
| design_card | yes | `docs/trice_design_card.json` | `cbdbb49b901d8c51e6e76504cc7f664b86177dc84f1ee7879345bf8d98261a0b` |
| reproduction_card | yes | `docs/trice_reproduction_card.json` | `89820ed6ab230c9114a86b1aa7b4a987ec5d0d33938208b799f4d2ef78673ecd` |
| contract_card | yes | `docs/trice_contract_card.json` | `013b9902508491740ab317f15de37a5d8b82dab2bf16d27e798d9fc506d40491` |
| install_card | yes | `docs/trice_install_card.json` | `e8bc7be728b9f61af4b1988428c6b82c7b2055fe6916284895f4ea20a8850dc8` |
| research_card | yes | `docs/trice_research_card.json` | `2095432965bb86bb3a559aa2bd2ef3375d688910ea4f00315fb75ac4c6019e18` |
| claim | yes | `docs/trice_claim_card.json` | `43fbd6d1329da52fb78239ace0037f0cfb1a460ab3a281ebaff1f4dc99f7d4c6` |
| remote_smoke_claim | yes | `docs/trice_remote_smoke_claim_card.json` | `e296651963ef02ccfde115ab59f0cc87e5d938548878ed63ef8f8660113da641` |
| evidence_bundle | yes | `benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip` | `a43eeac3ba5239031376aa7b3be97e01ccada6c36648faa15e1c2b3e2d2d2564` |
| remote_smoke_bundle | yes | `benchmark/trice/results/v2-remote-smoke/trice_remote_smoke_evidence.trice.zip` | `93f31fc51b6c75279b1dc8a637a46e51c7fd9f2aa3ea0ef45fe9d581ffef493d` |
| paper_manifest | yes | `paper/trice_v3_research_manifest.json` | `cc0ed7cdf452a6fe10d378b21a41992ef32c6ff33cd568ad78b80d424e73ee81` |
| paper_result | yes | `benchmark/trice/results/v2-smoke/trice_v2_live_results.json` | `edf72f856f37e2c773ffefc31c2ae0b06cab67cc6e456044df3ff03911b58d8a` |
| paper_tex | yes | `paper/trice_v3_research_paper.tex` | `8f5db5311aca8061917e8dd2c972aa2696611f1404eb4d5c63623ed137b233d8` |
| paper_pdf | yes | `paper/trice_v3_research_paper.pdf` | `e989dc19be087a9991ff05424457fd9aeef541056a30472add2e2a1db24b597d` |
| readme | yes | `README.md` | `bca276ad2145f0861fd0c444bda5258f83e3096a518df9a762739d19b099242e` |
| library_doc | yes | `docs/trice_library.md` | `a256eb788c0dc9c04a976e19e1ca0f8a788c2c5b03e2b8433f3114cff55524bf` |

## Next Actions

- Publish the artifact card with the release assets.
- Run the held-out remote pilot, then regenerate readiness, protocol, design, reproduction, contract, claim, bundle, paper, and artifact cards.
- Do not upgrade the README to S-tier passed until claim_allowed is true on held-out evidence.

## Hash

- artifact card: `5420f1a644c47dca33624ce0d0382213c4f7ae8dc6bedbe1a827efe09cc18f02`
