# Security Policy

## Supported Versions

Security fixes are accepted for the latest published minor line. At the time of
this policy, the active line is 1.0.x.

## Reporting a Vulnerability

Please report security issues privately by email to:

`zulfaqarhafez@gmail.com`

Include:

- Affected version or commit.
- Operating system and install method.
- Minimal reproduction steps.
- Impact assessment if known.
- Whether the issue affects trace contents, signatures, bundles, release
  artifacts, or command execution.

Please do not open a public GitHub issue for an undisclosed vulnerability.

## Scope

In scope:

- Trace parsing or report verification bugs that can hide tampering.
- Evidence bundle verification bypasses.
- Unsafe path handling in patch, adapter, or suite execution.
- Supply-chain issues in release, packaging, or CI workflows.
- Secrets exposure through logs, reports, receipts, or bundles.

Out of scope:

- Vulnerabilities in third-party agent tools run by a user-provided adapter.
- Benchmarks that intentionally execute untrusted repositories without an
  isolation boundary.
- Token-saving claims that are inaccurate but do not create a security impact.

## Release Security Gates

Before a public release, maintainers should run:

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
tracerazor-trice doctor --format json
```

Any dependency exception must be documented with the advisory id, affected
package, reason, and expiry date.

## Evidence Bundle Handling

TRICE bundles may include prompts, trace excerpts, file paths, receipts, and
hashes. Treat bundles as potentially sensitive. Publish only curated held-out
evidence bundles and avoid secrets in adapter receipts.
