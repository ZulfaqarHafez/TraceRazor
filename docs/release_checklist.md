# TraceRazor 1.1 release checklist

TraceRazor 1.1 is a platform-wheel, standalone-binary, and OCI-image release.
Do not mutate an existing tag or reuse a published version.

## 1. Local quality gates

Run from a clean checkout with the Rust toolchain on `PATH`:

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo audit
cargo deny check licenses
python -m pip_audit --progress-spinner off .
python -m pytest -q
git diff --check
```

Validate the agent-native contracts separately:

```bash
cargo test -p tracerazor --test agent_bootstrap
python -m pytest -q \
  tests/test_skill_pack.py \
  tests/test_codex_plugin.py \
  tests/test_runtime_api.py \
  tests/test_runtime_events.py \
  tests/test_runtime_persistence.py \
  tests/test_runtime_guardrails.py \
  tests/test_mcp_server.py \
  tests/test_mcp_guardrails.py
python -m pytest -q benchmark/agent_native/tests
```

The synthetic evaluation tests validate the evaluator, not TraceRazor's
efficacy. Never quote them as a measured product result.

## 2. Distribution contract

Build one native wheel on each supported runner:

- Linux x86-64 at glibc 2.35 (`manylinux_2_35_x86_64`);
- Linux ARM64 at glibc 2.39 (`manylinux_2_39_aarch64`);
- macOS x86-64 and ARM64;
- Windows x86-64.

These are the actual 1.1 Linux compatibility floors, not aliases for older
manylinux releases. The build fails if the runner's glibc changes or the ELF
binary imports a GLIBC symbol above its declared floor, and the clean-machine
smoke runs on that same oldest-supported native runner. Supporting older Linux
distributions requires a future dedicated manylinux builder and smoke matrix.
Alpine/musl is unsupported in 1.1. Do not publish a source distribution: a
source-only install does not satisfy the bundled-auditor contract. crates.io is
also outside the 1.1 GA contract until TraceRazor intentionally exposes a
stable public Rust API.

Each clean-machine wheel job must run outside the checkout with
`TRACERAZOR_BIN`, `PYTHONPATH`, `PYTHONHOME`, and ambient launcher paths
removed. It must prove:

```text
tracerazor --version
tracerazor agent doctor --format json
python -m tracerazor.mcp_server --selftest
tracerazor audit <sample> --hermetic --format json
```

The installed distribution must resolve its own native binary and contain the
event schema, canonical Agent Skill, Codex plugin, Claude plugin, Gemini
extension, policy template, MCP catalog, project license, and third-party
notices.

Generate the install receipt against the Linux x86-64 wheel explicitly; never
let an alphabetical wheel selection choose a foreign platform wheel:

```bash
tracerazor-trice install \
  --wheel dist/<linux-x86_64-wheel>.whl \
  --dist-dir dist \
  --out docs/trice_install_card.json
tracerazor-trice verify-install docs/trice_install_card.json
```

## 3. Release workflow

The tag must be exactly `v<pyproject version>`, and that version must match the
Cargo workspace and `tracerazor.__version__`. The release workflow runs Python
3.10 and 3.12 tests, Rust format/test/clippy, `pip-audit`, and `cargo audit`
before any public upload.

Publish only after the release evidence packet passes. Required public
artifacts are:

- five platform wheels;
- five matching standalone archives;
- the project license and third-party notices inside every wheel, standalone
  archive, and runtime image;
- `SHA256SUMS` covering every wheel, archive, and evidence file;
- CycloneDX SBOMs plus SHA-256 checksums for the staged release artifacts;
- GitHub artifact attestations;
- the agent OCI image for linux/amd64 and linux/arm64;
- install, run/comparison where applicable, research, and release receipts.

Existing release assets are immutable. A rerun may skip a byte-identical asset,
but must fail if an asset with the same name has different bytes.

PyPI trusted publishing uses GitHub OIDC and fails if the version already exists; it never
silently accepts an existing file without proving byte identity. Release
evidence extracts its CLI subject from the downloaded standalone archive built
by the binary matrix, rather than compiling a replacement in the evidence job.
The composite GitHub Action defaults to the
immutable `v1.1.0` archive and verifies it against the release checksum. The
mutable `latest` alias is an explicit caller opt-in.

### Agent OCI image gate

The release workflow first pushes the multi-architecture image under the
unpromoted `build-<commit>` tag. It then addresses that image by digest and
runs `scripts/smoke_agent_image.sh` for both `linux/amd64` and `linux/arm64`.
Each smoke runs as the image's numeric `10001:10001` user with networking and
Linux capabilities removed, `no-new-privileges` enabled, and a read-only root
filesystem. It proves:

- the default command is the JSON agent doctor;
- the image policy is local-redacted coach mode with enforcement disabled;
- a mounted `/workspace/tracerazor.toml` overrides the image fallback policy;
- the image-scope installation and ownership ledger are healthy;
- the provisioning receipts are present;
- the packaged MCP catalog self-test succeeds; and
- a hermetic sample audit succeeds without a checkout or writable store.

The image's Python build and runtime dependency closures are exact-version
locks installed with `--no-deps` and checked with `pip check`; MCP is fixed to
one version. `SOURCE_DATE_EPOCH` comes from the tagged commit and normalizes
wheel timestamps plus the image provisioning ledger and receipts. This removes
known wall-clock and dependency-resolution drift. It does **not** establish a
bit-for-bit reproducible OCI digest by itself: the Rust and Python multi-arch
base indexes are digest-pinned, but platform-specific package artifacts may
differ and BuildKit SBOM/provenance attestations can carry builder metadata.
CI must prove repeat builds before making a reproducibility claim. The
immutable version-tag check fails closed if a rebuild produces a different
digest.

The manifest must contain exactly the supported real platforms
`linux/amd64` and `linux/arm64` (BuildKit attestation descriptors may appear as
`unknown/unknown`). BuildKit emits per-platform SBOM and maximum-mode
provenance attestations. GitHub then signs the tested index digest with
`actions/attest`, pushes that attestation to GHCR, and verifies it with
`gh attestation verify`. The deterministic `agent-image-release.json` receipt
is generated and uploaded before promotion; it records the tested digest,
platforms, fixed source epoch, source revision, and trust event and is later
included in `SHA256SUMS` and the GitHub release asset attestation.

GHCR package visibility is a one-time manual prerequisite: the
`zulfaqarhafez/tracerazor-agent` package must already exist and be set to
**Public** in GitHub package settings. Before promotion, the workflow logs out
and anonymously inspects and pulls the exact staging digest; a private package
fails the release. It then restores publisher authentication. Promotion of the
immutable `v<version>` and mutable `latest` tags is the job's final registry
mutation. The preflight tag lookup treats only HTTP 404 as absent; auth,
rate-limit, and network failures stop the release rather than overwriting.

Consumers requiring a durable trust decision should use the version tag plus
attestation verification, or pin the reported `sha256:` digest. The local
Docker daemon is not a release gate; these two architecture smokes must pass
on GitHub-hosted release infrastructure.

## 4. Claims and efficacy gate

The preregistered study machinery lives in `benchmark/agent_native/`:

```bash
python -m benchmark.agent_native.evaluate --print-protocol-sha256
python -m benchmark.agent_native.evaluate \
  --input <held-out-results.jsonl> \
  --output <evaluation-report.json>
```

Exit `0` is required before making efficacy claims. Exit `1` means a complete
study failed one or more gates. Exit `2` means malformed or incomplete
evidence. Estimated or missing token counts are excluded from efficacy.

The locked product gates include 100% clean-machine installation, activation
precision/recall, provider-token agreement, parent/child linkage, runtime
overhead, measured token reduction with a positive confidence-interval lower
bound, task-success non-inferiority, recommendation precision, sandbox
containment, and secret-redaction safety.

Until real held-out results pass, describe TraceRazor as an efficiency
supervisor and regression diagnostic. Keep autonomous optimization and TRICE
as Labs surfaces; do not market measured savings from synthetic fixtures or
heuristic report estimates.
