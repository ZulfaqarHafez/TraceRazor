# TRICE Research Card

- Scope: `TRICE deterministic research basis`
- Research level: `research_basis_locked`
- Research score: **100/100**
- Sources: **165** total, **165** unique URLs
- Ledger: `docs/trice_research_ledger.md`

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| ledger_present | yes | `docs/trice_research_ledger.md` | `ledger file exists` |
| minimum_sources | yes | `165` | `>= 150 ledger rows` |
| unique_source_urls | yes | `165` | `>= 127 unique source URLs` |
| category_coverage | yes | `{"agent_evaluation": 43, "context_efficiency": 75, "other": 8, "public_trust": 25, "statistical_quality": 9, "trice_internal": 5}` | `{"agent_evaluation": 30, "context_efficiency": 30, "public_trust": 20, "statistical_quality": 5}` |
| valid_source_links | yes | `0.9818` | `>= 0.95 valid http/local links` |
| takeaways_present | yes | `{"missing": 0, "present": 165}` | `every row has a takeaway` |
| index_sequence | yes | `{"count": 165, "first": 1, "last": 165, "unique": 165}` | `rows are uniquely indexed in ascending order` |
| product_decision_section | yes | `Product Decision From The Ledger` | `ledger includes product decision synthesis` |

## Category Coverage

| Category | Count |
|---|---:|
| agent_evaluation | 43 |
| context_efficiency | 75 |
| other | 8 |
| public_trust | 25 |
| statistical_quality | 9 |
| trice_internal | 5 |

## Source Domains

| Domain | Count |
|---|---:|
| hf.co | 120 |
| arxiv.org | 16 |
| local | 4 |
| docs.pypi.org | 2 |
| github.com | 2 |
| packaging.python.org | 2 |
| peps.python.org | 2 |
| slsa.dev | 2 |
| www.acm.org | 2 |
| csrc.nist.gov | 1 |
| cyclonedx.org | 1 |
| doc.rust-lang.org | 1 |

## Research Basis

- Agent evaluation research motivates fixed task environments, repeated trials, and cost-aware metrics.
- Prompt-compression and long-context research motivate decision-preserving context control rather than shortest-prompt contests.
- RAG and evidence-recall research motivate explicit recall floors for hidden or compressed evidence.
- Reproducible-build and supply-chain practice motivates hash-bound research, package, release, and provenance artifacts.
- TRICE treats its literature base as a versioned product input: if the ledger drifts, the paper, README, and integrity proof must be regenerated.

## Non-Claims

- A green research card does not prove the 60% S-tier outcome gate.
- A green research card does not prove that every cited paper was replicated.
- A green research card proves that the current ledger is broad, hashed, categorized, and bound to generated artifacts.

## Next Actions

- Regenerate this card whenever the ledger, paper, README, or proof graph changes.
- Keep S-tier outcome claims gated by live held-out suite results, not by research-card status.
- Scale the ledger toward 300+ primary sources before submitting the paper outside the repo.

## Hash

- research card: `8fc4acd4bbfc30a06263602dcd44cdb2eb58a59cfcf03d337a9e38894e91b07b`
