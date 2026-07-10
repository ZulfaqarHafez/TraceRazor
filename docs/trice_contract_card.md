# TRICE Contract Card

- Package: `tracerazor`
- Version: `1.1.0`
- Contract level: `library_contract_locked`
- Contract score: **100/100**

## Checks

| Check | Pass | Observed | Required |
|---|---:|---|---|
| semver_version | yes | 1.1.0 | MAJOR.MINOR.PATCH |
| top_level_api | yes | {"count": 25, "missing": []} | tracerazor.__all__ resolves |
| trice_api | yes | {"count": 139, "missing": []} | tracerazor.trice.__all__ resolves |
| cli_contract | yes | ["artifact", "bundle", "claim", "contract", "crates", "design", "doctor", "install", "integrity", "protocol", "release", "release-evidence", "reproduction", "research", "run", "schema", "suite", "validate-adapter", "validate-patch", "validate-receipt", "validate-suite", "verify", "verify-artifact", "verify-bundle", "verify-claim", "verify-contract", "verify-crates", "verify-design", "verify-install", "verify-integrity", "verify-protocol", "verify-release", "verify-release-evidence", "verify-reproduction", "verify-research", "verify-suite"] | all documented tracerazor-trice commands exist |
| schemas_shipped | yes | 19/19 | all TRICE JSON Schemas are present |
| contract_schema_shipped | yes | trice_contract_card.schema.json | contract-card schema ships |
| examples_shipped | yes | 9/9 | public examples are present |
| docs_shipped | yes | 5/5 | README/library/research/trust docs are present |

## Public API

- `tracerazor`: 25 exported names
- `tracerazor.trice`: 139 exported names
- `tracerazor-trice`: 36 subcommands
- Schemas: 19/19
- Examples: 9/9

## Research Basis

- Semantic Versioning requires the public API to be declared before compatibility claims are meaningful.
- Python packaging makes version identifiers public registry facts, so library contracts must bind version and import surface together.
- JSON Schema gives users and downstream agents a machine-checkable boundary for TRICE receipts, suites, cards, and bundles.
- SLSA, in-toto, and CycloneDX motivate release evidence that binds checksums, SBOMs, provenance statements, and public artifacts.
- Reproducible-build practice motivates checking that public examples, schemas, CLI commands, release evidence, and the integrity proof graph match the source being packaged.
- Cargo publication trust requires staged registry facts because downstream crates cannot honestly claim cargo-install readiness before upstream workspace crates are indexed.
- Clean-wheel installability must be verified after build because packaged data, console scripts, and bundled binaries can diverge from checkout behavior.
- Research-ledger integrity must be machine-checkable because product claims and papers can drift from the sources that supposedly justify them.

## Next Actions

- Treat this card as the public API boundary for SemVer compatibility.
- Regenerate the contract card before every release and after any public API, CLI, schema, or example change.
- Promote only documented imports and schemas into long-term compatibility guarantees.

## Hash

- contract card: `6357f7dcc80f341212dff89d6209ac84f97063b479fe84cb2fb44b16629a07df`
