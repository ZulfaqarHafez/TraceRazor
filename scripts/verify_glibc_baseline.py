"""Fail when an ELF binary imports GLIBC symbols above its release floor."""

from __future__ import annotations

import re
import subprocess
import sys


def imported_versions(binary: str) -> list[tuple[int, int]]:
    output = subprocess.check_output(
        ["readelf", "--version-info", binary], text=True
    )
    return [
        tuple(map(int, match))
        for match in re.findall(r"GLIBC_(\d+)\.(\d+)", output)
    ]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_glibc_baseline.py <binary> <major.minor>")
    versions = imported_versions(sys.argv[1])
    if not versions:
        raise SystemExit("could not determine imported GLIBC symbol versions")
    ceiling = tuple(map(int, sys.argv[2].split(".")))
    if max(versions) > ceiling:
        raise SystemExit(
            f"binary imports GLIBC_{'.'.join(map(str, max(versions)))} "
            f"above declared {sys.argv[2]}"
        )
    print(
        f"GLIBC symbol ceiling {'.'.join(map(str, max(versions)))} "
        f"is within declared {sys.argv[2]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
