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
| readiness | yes | `docs/trice_suite_readiness.json` | `30a32533b71a878abb81be23e241617550a94ebfd1df4f23dcec5a68d59bf9a4` |
| protocol_lock | yes | `docs/trice_protocol_lock.json` | `877f06f9d5c312f5d9396fcc2bb3001122d27c91bdeb575236361c5d4c841d05` |
| design_card | yes | `docs/trice_design_card.json` | `a9568aa5b32a54c99ab36bec232132c1a7f7ff7a25427353f0df1ef6f10140f2` |
| reproduction_card | yes | `docs/trice_reproduction_card.json` | `34762f6ff0511476a038dd1d4bf52e6923aa1ca899280a717ecab15afd66df54` |
| contract_card | yes | `docs/trice_contract_card.json` | `0153cf143e17eaec75f5ff0def49476fcf7a548cc5247cf5ed2e6e6a9ec0e3ae` |
| install_card | yes | `docs/trice_install_card.json` | `7e0ab240e9a92ecb079bd7720f81f7cc560b0981a63d647a1318443dd9a42a37` |
| research_card | yes | `docs/trice_research_card.json` | `2613a20f18eeecc0e1430474c5ea1138a87b917182608275c41f714426df08e1` |
| claim | yes | `docs/trice_claim_card.json` | `c2c8758ed555ccbc9328c894b63b20da289c56f754fd597c3b411e4bbad5009e` |
| remote_smoke_claim | yes | `docs/trice_remote_smoke_claim_card.json` | `528f1f9e24d0497aff6b1302cb9fc66e05f2c63495d7bfb708f5748af4de9fc7` |
| evidence_bundle | yes | `benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip` | `a43eeac3ba5239031376aa7b3be97e01ccada6c36648faa15e1c2b3e2d2d2564` |
| remote_smoke_bundle | yes | `benchmark/trice/results/v2-remote-smoke/trice_remote_smoke_evidence.trice.zip` | `93f31fc51b6c75279b1dc8a637a46e51c7fd9f2aa3ea0ef45fe9d581ffef493d` |
| paper_manifest | yes | `paper/trice_v3_research_manifest.json` | `b7bd76063e5b53b98d934562f642e4778a59e1c0b03b229de03d3aa9701ef19e` |
| paper_result | yes | `benchmark/trice/results/v2-smoke/trice_v2_live_results.json` | `f2e7956fa5ffe1cae48311188dfe8f2afc5fbe41ab7d3a6381b745b382fe95b8` |
| paper_tex | yes | `paper/trice_v3_research_paper.tex` | `046bbd935e7bddf6d8340e8da150caaed256b6954f585c24eb1310e6d1c2a508` |
| paper_pdf | yes | `paper/trice_v3_research_paper.pdf` | `a63a553d01c2ff0e2fe4be7658334d05f74bf4adf64f7eb373ae01fb29a02a45` |
| readme | yes | `README.md` | `d40843c15ff2528f3478436b5e3c86da899f6c1651021e0ac4889c27d663a856` |
| library_doc | yes | `docs/trice_library.md` | `a256eb788c0dc9c04a976e19e1ca0f8a788c2c5b03e2b8433f3114cff55524bf` |

## Next Actions

- Publish the artifact card with the release assets.
- Run the held-out remote pilot, then regenerate readiness, protocol, design, reproduction, contract, claim, bundle, paper, and artifact cards.
- Do not upgrade the README to S-tier passed until claim_allowed is true on held-out evidence.

## Hash

- artifact card: `beca7656c30f16a143ea1e55b7153f0dc1f9885a46ff4d6cc651cbac4c2a0dc5`
