# Third-party notices

TraceRazor includes or can be distributed with software developed by third
parties. TraceRazor's MIT license applies to the original TraceRazor source and
project artwork. Third-party components remain subject to their own licenses.

## Distribution boundary

The TraceRazor 1.1 platform wheels bundle the native Rust auditor. The
standalone archives and container images also contain compiled Rust
dependencies, and the dashboard image contains compiled JavaScript
dependencies. This notice accompanies those distributions. It records the
distribution boundary and license policy; it is not yet a dependency-specific
attribution bundle containing every required copyright and license text.
Generate and CI-check that merged attribution bundle before the next public
release.

The current resolved Rust dependency graph is governed by the permissive
license allowlist in the repository's `.cargo/deny.toml`. Exact package
versions are locked in `Cargo.lock`. Run the following from a source checkout
to inspect every resolved crate and its detected license:

```bash
cargo deny check licenses
cargo deny list --format json --layout crate
```

The dashboard dependency graph and declared licenses are locked in
`dashboard/package-lock.json`. Optional Python extras are installed separately
by users and retain the licenses published by their respective distributions.

Release assets include CycloneDX Python and Cargo component inventories. The
release evidence generator adds resolved Cargo license expressions to the
Cargo inventory when `cargo metadata --locked` is available. See the
`trice_release_evidence.*.cdx.json` assets attached to each GitHub release.

The resolved dependency set includes permissive licenses such as MIT,
Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Unicode-3.0,
CDLA-Permissive-2.0, CC0-1.0, Zlib, and approved dual-license expressions.
The exact expression attached to each component is authoritative over this
summary.

## Research data

External research traces and the vendored AgentInstruct sample are source-tree
research inputs, not TraceRazor product assets. The 1.1 wheels, standalone
archives, and runtime container images exclude them. The public source tree and
legacy 1.0.3 source distribution still contain the derived fixtures, and the
AgentInstruct redistribution license remains unconfirmed. The project MIT
license does not relicense them.

## No endorsement

Third-party names are used only to identify dependencies or trace formats. No
affiliation or endorsement is implied.
