#!/usr/bin/env bash
# TraceRazor release script.
#
# Publishes the Python wheel to PyPI and the Rust crates to crates.io in
# dependency order. Run from the repo root:
#
#     ./publish.sh                   # build + publish everything
#     ./publish.sh --dry-run         # cargo publish --dry-run + python -m build (no upload)
#     ./publish.sh --python-only     # PyPI only
#     ./publish.sh --rust-only       # crates.io only
#
# Requirements:
#   pip install build twine
#   cargo login    (or CARGO_REGISTRY_TOKEN exported)
set -euo pipefail

DRY_RUN=0
PYTHON=1
RUST=1

for arg in "$@"; do
    case "$arg" in
        --dry-run)     DRY_RUN=1 ;;
        --python-only) RUST=0 ;;
        --rust-only)   PYTHON=0 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

# Rust crates must publish in topological dependency order. The CLI binary
# depends on server + store + semantic + ingest + core; store depends on core; etc.
CRATE_ORDER=(
    crates/tracerazor-core
    crates/tracerazor-ingest
    crates/tracerazor-semantic
    crates/tracerazor-store
    crates/tracerazor-server
    crates/tracerazor-cli
)

if [[ "$RUST" == "1" ]]; then
    echo "==> Cargo publish (crates.io)"
    for crate_dir in "${CRATE_ORDER[@]}"; do
        echo "--> $crate_dir"
        if [[ "$DRY_RUN" == "1" ]]; then
            (cd "$crate_dir" && cargo publish --dry-run --allow-dirty)
        else
            (cd "$crate_dir" && cargo publish)
            # crates.io needs a moment for an upstream crate to become
            # resolvable before the next one in the chain can publish.
            sleep 15
        fi
    done
fi

if [[ "$PYTHON" == "1" ]]; then
    echo "==> Building Python artifacts (PyPI)"
    rm -rf dist build *.egg-info
    # sdist first, then the platform wheel with the bundled CLI binary —
    # `python -m build` alone produces a binary-less wheel whose `tracerazor`
    # command cannot work after install.
    python -m build --sdist
    bash scripts/build_platform_wheel.sh
    echo "NOTE: this builds only THIS machine's platform wheel; the"
    echo "      release.yml matrix (tag push / release-cut branch) is the"
    echo "      real multi-platform publish path."
    if [[ "$DRY_RUN" == "0" ]]; then
        echo "==> Uploading to PyPI"
        twine upload dist/*
    fi
fi

echo "Done."
