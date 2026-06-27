# TraceRazor Release Checklist

Use this checklist for 1.0.3 and later. Do not mutate an already published PyPI
release; cut a follow-up version instead.

## 1. Local Gates

```bash
cargo check --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo audit
cargo deny check
python -m pip_audit --progress-spinner off .
python -m pytest
python -m build --sdist --wheel
python -m twine check dist/*
tracerazor-trice crates --out docs/trice_crates_card.json --timeout-s 10
tracerazor-trice verify-crates docs/trice_crates_card.json
tracerazor-trice install --out docs/trice_install_card.json --dist-dir dist
tracerazor-trice verify-install docs/trice_install_card.json
tracerazor-trice research --out docs/trice_research_card.json
tracerazor-trice verify-research docs/trice_research_card.json
tracerazor-trice release-evidence --out docs/trice_release_evidence.json --dist-dir dist
tracerazor-trice verify-release-evidence docs/trice_release_evidence.json
tracerazor-trice integrity --out docs/trice_integrity_card.json
tracerazor-trice verify-integrity docs/trice_integrity_card.json
scorecard --repo=github.com/ZulfaqarHafez/TraceRazor
```

Then test a clean wheel install:

```bash
python -m venv .venv-release-check
.venv-release-check\Scripts\python -m pip install --upgrade pip
.venv-release-check\Scripts\python -m pip install dist\tracerazor-1.0.3-py3-none-any.whl
.venv-release-check\Scripts\tracerazor-trice doctor --format json --offline
```

The generated installability card is the canonical clean-wheel proof. The
manual virtualenv smoke above is a human sanity check, while
`tracerazor-trice install` records the wheel hash, packaged schema/API import
surface, console-script result, and bundled Rust CLI status.

The generated research card is the canonical paper-basis proof. It records the
research ledger hash, row hashes, source counts, category coverage, and
non-claim boundary so paper and README claims cannot drift from their cited
source base.

## 2. Rust Crate Publish Order

Generate and verify the staged publish card before any upload:

```bash
tracerazor-trice crates --out docs/trice_crates_card.json --timeout-s 10
tracerazor-trice verify-crates docs/trice_crates_card.json
```

The card must report `local_publish_plan_locked = true`. It may still report
`public_crates_live = false` before publication; that is expected and keeps the
README cargo-install claim blocked.

Publish only after local Rust gates pass. `cargo package` for a dependent
workspace crate requires its internal dependencies to already exist on
crates.io, so package verification is staged and cannot all happen before the
first internal crate publish:

```bash
cargo package -p tracerazor-core --allow-dirty
cargo package -p tracerazor-semantic --allow-dirty
```

Actual publish order is dependency order. After each publish is visible in the
index, rerun `cargo package` without `--no-verify` for the next dependent crate
before publishing it:

```bash
cargo publish -p tracerazor-core
cargo package -p tracerazor-ingest --allow-dirty
cargo package -p tracerazor-store --allow-dirty
cargo publish -p tracerazor-semantic
cargo publish -p tracerazor-ingest
cargo publish -p tracerazor-store
cargo package -p tracerazor-server --allow-dirty
cargo publish -p tracerazor-server
cargo package -p tracerazor --allow-dirty
cargo publish -p tracerazor
```

If `CARGO_REGISTRY_TOKEN` is unavailable, do not publish and do not add
`cargo install tracerazor` back to README.

## 3. Python Publish

Build and check:

```bash
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
python -m build --sdist --wheel
python -m twine check dist/*
```

Preferred release path is PyPI trusted publishing from GitHub Actions using
OIDC, so releases do not depend on long-lived upload tokens. Manual Twine
upload is a fallback:

```bash
python -m twine upload dist/*
```

After PyPI publish, wait for piwheels to build and verify:

```bash
tracerazor-trice doctor --format json
```

PyPI Trusted Publishing should emit registry-side digital attestations for the
uploaded distributions. Do not fall back to manual Twine upload unless the
trusted publishing path is unavailable and the reason is documented in the
release notes.

## 4. Public Health And Attestations

The OpenSSF Scorecard workflow must complete and publish a score before the
release card can be public-ready. The release gate treats a missing Scorecard
or a score below 7.0 as a public trust blocker.

The release workflow must attach GitHub artifact attestations for release
assets generated in Actions. The expected release asset set is:

## 5. GitHub Release Assets

Attach:

- Wheels and sdist.
- Platform CLI binaries.
- SHA-256 checksums.
- CycloneDX SBOM for Python dependencies.
- CycloneDX SBOM for Cargo dependencies.
- TRICE evidence bundle for any published proof claim.
- Machine-verifiable suite manifest and result JSON.
- TRICE suite readiness JSON, Markdown, LaTeX, and SVG.
- TRICE protocol lock JSON, Markdown, LaTeX, and SVG.
- TRICE design card JSON, Markdown, LaTeX, and SVG.
- TRICE claim card JSON, Markdown, LaTeX, and SVG.
- TRICE reproduction card JSON, Markdown, LaTeX, and SVG.
- TRICE artifact card JSON, Markdown, LaTeX, and SVG.
- TRICE installability card JSON, Markdown, LaTeX, and SVG.
- TRICE release evidence JSON, Markdown, LaTeX, SVG, checksums, Python SBOM,
  Cargo SBOM, and in-toto/SLSA-shaped provenance statement.
- TRICE release card JSON, Markdown, LaTeX, and SVG.
- TRICE crates publish card JSON, Markdown, LaTeX, and SVG.
- TRICE integrity card JSON, Markdown, LaTeX, and SVG.

## 6. S-Tier Evidence Gate

Pilot:

- 10 remote Git task clusters.
- 2 replicates per cluster.
- Locked commits.
- Adapter profiles, not ad hoc commands.
- Zero pass regressions.
- Valid receipts.

Preflight:

```bash
tracerazor-trice suite readiness suite.json --out docs/trice_suite_readiness.json
tracerazor-trice suite verify-readiness docs/trice_suite_readiness.json --manifest suite.json
tracerazor-trice protocol --manifest suite.json --out docs/trice_protocol_lock.json
tracerazor-trice verify-protocol docs/trice_protocol_lock.json --manifest suite.json
```

The pilot suite should report `pilot_execution_ready = true` and
`pilot_protocol_ready`; the claim suite should report
`claim_execution_ready = true` and `claim_protocol_ready` before live execution.
After execution, the generated design card should report
`claim_design_ready = true`.

Claim run:

- 50 remote Git task clusters.
- 3 replicates per cluster.
- Mean input-token savings >= 60%.
- Clustered CI lower bound >= 60%.
- Zero pass regressions.
- Evidence recall >= 95% on solved traces.
- Every child evidence manifest verifies.

README can claim S-tier only if the generated result says:

```json
{
  "s_tier_gate": {
    "passed": true
  }
}
```

Also regenerate and inspect the claim card:

```bash
tracerazor-trice design \
  --protocol docs/trice_protocol_lock.json \
  --suite-result benchmark/trice/results/heldout-claim/trice_suite_results.json \
  --out docs/trice_design_card.json
tracerazor-trice verify-design docs/trice_design_card.json
tracerazor-trice claim \
  --suite-result benchmark/trice/results/heldout-claim/trice_suite_results.json \
  --manifest benchmark/trice/results/heldout-claim/trice_suite_evidence_manifest.json \
  --out docs/trice_claim_card.json
tracerazor-trice verify-claim docs/trice_claim_card.json
tracerazor-trice reproduction --out docs/trice_reproduction_card.json
tracerazor-trice verify-reproduction docs/trice_reproduction_card.json
tracerazor-trice install --out docs/trice_install_card.json --dist-dir dist
tracerazor-trice verify-install docs/trice_install_card.json
tracerazor-trice research --out docs/trice_research_card.json
tracerazor-trice verify-research docs/trice_research_card.json
tracerazor-trice artifact --out docs/trice_artifact_card.json
tracerazor-trice verify-artifact docs/trice_artifact_card.json
tracerazor-trice release-evidence --out docs/trice_release_evidence.json --dist-dir dist
tracerazor-trice verify-release-evidence docs/trice_release_evidence.json
tracerazor-trice release --out docs/trice_release_card.json --timeout-s 10
tracerazor-trice verify-release docs/trice_release_card.json
tracerazor-trice crates --out docs/trice_crates_card.json --timeout-s 10
tracerazor-trice verify-crates docs/trice_crates_card.json
tracerazor-trice integrity --out docs/trice_integrity_card.json
tracerazor-trice verify-integrity docs/trice_integrity_card.json
```

## References

- Python Packaging User Guide: https://packaging.python.org/tutorials/packaging-projects/
- Twine documentation: https://twine.readthedocs.io/
- piwheels FAQ: https://www.piwheels.org/faq.html
- SLSA specification: https://slsa.dev/spec/
- in-toto attestations: https://in-toto.io/
- CycloneDX specification: https://cyclonedx.org/specification/overview/
- PyPI trusted publishing: https://docs.pypi.org/trusted-publishers/
- PyPI digital attestations: https://docs.pypi.org/attestations/
- Cargo publishing: https://doc.rust-lang.org/cargo/reference/publishing.html
