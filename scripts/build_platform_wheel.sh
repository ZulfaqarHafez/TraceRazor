#!/usr/bin/env bash
# Build a platform wheel with the Rust CLI bundled, so
# `pip install tracerazor` delivers a working `tracerazor` command with no
# Rust toolchain (ship-plan 2.1). Run from the repo root.
set -euo pipefail

cargo build --release -p tracerazor
mkdir -p tracerazor/bin
cp target/release/tracerazor tracerazor/bin/tracerazor
chmod +x tracerazor/bin/tracerazor

python -m pip install --quiet build wheel
python -m build --wheel

# Tag the wheel with the real platform (it contains a native binary).
PLAT=$(python -c "import sysconfig; print(sysconfig.get_platform().replace('-', '_').replace('.', '_'))")
WHEEL=$(ls -t dist/tracerazor-*-py3-none-any.whl | head -1)
python -m wheel tags --platform-tag "${PLAT}" --remove "${WHEEL}"
ls -l dist/
