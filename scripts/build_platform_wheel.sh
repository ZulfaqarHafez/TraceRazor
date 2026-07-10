#!/usr/bin/env bash
# Build a platform wheel with the Rust CLI bundled, so
# `pip install tracerazor` delivers a working `tracerazor` command with no
# Rust toolchain (ship-plan 2.1). Run from the repo root.
set -euo pipefail

cargo build --release -p tracerazor
mkdir -p tracerazor/bin
rm -f tracerazor/bin/tracerazor tracerazor/bin/tracerazor.exe
cp target/release/tracerazor tracerazor/bin/tracerazor
chmod +x tracerazor/bin/tracerazor

python -m pip install --quiet build wheel
python -m build --wheel

# Tag the wheel with the real platform (it contains a native binary).
# On Linux, PyPI rejects the bare `linux_*` tag (PEP 600): tag with the
# builder's actual glibc as `manylinux_<maj>_<min>_<arch>` instead — honest
# (the binary needs exactly that glibc or newer) and PyPI-acceptable.
PLAT=$(python - <<'PY'
import os
import platform
import sysconfig
plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
if plat.startswith("linux_"):
    arch = plat.split("_", 1)[1]
    libc, ver = platform.libc_ver()
    if libc == "glibc" and ver:
        maj, minor = ver.split(".")[:2]
        expected = os.environ.get("TRACERAZOR_EXPECTED_GLIBC", "")
        actual = f"{maj}.{minor}"
        if expected and actual != expected:
            raise SystemExit(f"glibc baseline drift: expected {expected}, found {actual}")
        plat = f"manylinux_{maj}_{minor}_{arch}"
print(plat)
PY
)
if [[ -n "${TRACERAZOR_EXPECTED_WHEEL_PLATFORM:-}" \
      && "$PLAT" != "$TRACERAZOR_EXPECTED_WHEEL_PLATFORM" ]]; then
  echo "wheel platform drift: expected $TRACERAZOR_EXPECTED_WHEEL_PLATFORM, found $PLAT" >&2
  exit 1
fi

if [[ "$PLAT" == manylinux_* ]]; then
  # Check the imported symbol ceiling as well as the builder's libc version.
  GLIBC_BASELINE="${TRACERAZOR_EXPECTED_GLIBC:-}"
  if [[ -z "$GLIBC_BASELINE" ]]; then
    if [[ "$PLAT" =~ ^manylinux_([0-9]+)_([0-9]+)_ ]]; then
      GLIBC_BASELINE="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
    else
      echo "unable to derive GLIBC baseline from $PLAT" >&2
      exit 1
    fi
  fi
  python scripts/verify_glibc_baseline.py \
    tracerazor/bin/tracerazor "$GLIBC_BASELINE"
fi

WHEEL=$(ls -t dist/tracerazor-*-py3-none-any.whl | head -1)
python -m wheel tags --platform-tag "${PLAT}" --remove "${WHEEL}"
ls -l dist/
