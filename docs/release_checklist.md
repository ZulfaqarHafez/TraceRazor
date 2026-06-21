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
```

Then test a clean wheel install:

```bash
python -m venv .venv-release-check
.venv-release-check\Scripts\python -m pip install --upgrade pip
.venv-release-check\Scripts\python -m pip install dist\tracerazor-1.0.3-py3-none-any.whl
.venv-release-check\Scripts\tracerazor-trice doctor --format json --offline
```

## 2. Rust Crate Publish Order

Publish only after all package dry-runs pass:

```bash
cargo package -p tracerazor-core --allow-dirty
cargo package -p tracerazor-semantic --allow-dirty
cargo package -p tracerazor-ingest --allow-dirty
cargo package -p tracerazor-store --allow-dirty
cargo package -p tracerazor-server --allow-dirty
cargo package -p tracerazor --allow-dirty
```

Actual publish order:

```bash
cargo publish -p tracerazor-core
cargo publish -p tracerazor-semantic
cargo publish -p tracerazor-ingest
cargo publish -p tracerazor-store
cargo publish -p tracerazor-server
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

Preferred release path is PyPI trusted publishing from GitHub Actions. Manual
Twine upload is a fallback:

```bash
python -m twine upload dist/*
```

After PyPI publish, wait for piwheels to build and verify:

```bash
tracerazor-trice doctor --format json
```

## 4. GitHub Release Assets

Attach:

- Wheels and sdist.
- Platform CLI binaries.
- SHA-256 checksums.
- CycloneDX SBOM for Python dependencies.
- CycloneDX SBOM for Cargo dependencies.
- TRICE evidence bundle for any published proof claim.
- Machine-verifiable suite manifest and result JSON.
- TRICE suite readiness JSON, Markdown, LaTeX, and SVG.
- TRICE claim card JSON, Markdown, LaTeX, and SVG.

## 5. S-Tier Evidence Gate

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
```

The pilot suite should report `pilot_execution_ready = true`; the claim suite
should report `claim_execution_ready = true` before live execution.

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
tracerazor-trice claim \
  --suite-result benchmark/trice/results/heldout-claim/trice_suite_results.json \
  --manifest benchmark/trice/results/heldout-claim/trice_suite_evidence_manifest.json \
  --out docs/trice_claim_card.json
tracerazor-trice verify-claim docs/trice_claim_card.json
```

## References

- Python Packaging User Guide: https://packaging.python.org/tutorials/packaging-projects/
- Twine documentation: https://twine.readthedocs.io/
- piwheels FAQ: https://www.piwheels.org/faq.html
