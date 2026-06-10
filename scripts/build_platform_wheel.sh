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
# On Linux, PyPI rejects the bare `linux_*` tag (PEP 600): tag with the
# builder's actual glibc as `manylinux_<maj>_<min>_<arch>` instead — honest
# (the binary needs exactly that glibc or newer) and PyPI-acceptable.
PLAT=$(python - <<'PY'
import platform, sysconfig
plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
if plat.startswith("linux_"):
    arch = plat.split("_", 1)[1]
    libc, ver = platform.libc_ver()
    if libc == "glibc" and ver:
        maj, minor = ver.split(".")[:2]
        plat = f"manylinux_{maj}_{minor}_{arch}"
print(plat)
PY
)
WHEEL=$(ls -t dist/tracerazor-*-py3-none-any.whl | head -1)
python -m wheel tags --platform-tag "${PLAT}" --remove "${WHEEL}"
ls -l dist/
