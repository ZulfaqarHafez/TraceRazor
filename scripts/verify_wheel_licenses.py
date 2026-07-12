"""Fail a release when a TraceRazor wheel omits its legal notices."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


REQUIRED_SUFFIXES = (
    ".dist-info/METADATA",
    ".dist-info/licenses/LICENSE",
    ".dist-info/licenses/THIRD_PARTY_NOTICES.md",
    "tracerazor/agent_assets/plugins/tracerazor/LICENSE",
    "tracerazor/agent_assets/extensions/claude-code/tracerazor/LICENSE",
    "tracerazor/agent_assets/extensions/gemini-cli/tracerazor/LICENSE",
)


def verify_wheel(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = [
            suffix for suffix in REQUIRED_SUFFIXES if not any(name.endswith(suffix) for name in names)
        ]
        if missing:
            return [f"missing {suffix}" for suffix in missing]

        metadata = _read_suffix(archive, names, ".dist-info/METADATA")
        if "License-Expression: MIT" not in metadata:
            missing.append("METADATA License-Expression: MIT")
        for license_file in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            if f"License-File: {license_file}" not in metadata:
                missing.append(f"METADATA License-File: {license_file}")

        for suffix in (
            ".dist-info/licenses/LICENSE",
            "tracerazor/agent_assets/plugins/tracerazor/LICENSE",
            "tracerazor/agent_assets/extensions/claude-code/tracerazor/LICENSE",
            "tracerazor/agent_assets/extensions/gemini-cli/tracerazor/LICENSE",
        ):
            if not _read_suffix(archive, names, suffix).startswith("MIT License"):
                missing.append(f"substantive MIT text at {suffix}")

        notices = _read_suffix(archive, names, ".dist-info/licenses/THIRD_PARTY_NOTICES.md")
        if not notices.startswith("# Third-party notices"):
            missing.append("substantive third-party notice boundary")
        if "not yet a dependency-specific attribution bundle" not in " ".join(notices.split()):
            missing.append("explicit attribution-completeness status")
    return missing


def _read_suffix(archive: zipfile.ZipFile, names: set[str], suffix: str) -> str:
    name = next(name for name in names if name.endswith(suffix))
    return archive.read(name).decode("utf-8")


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    root = Path(args[0]) if args else Path("dist")
    wheels = [root] if root.is_file() else sorted(root.glob("tracerazor-*.whl"))
    if not wheels:
        print(f"no TraceRazor wheel found under {root}", file=sys.stderr)
        return 2
    failed = False
    for wheel in wheels:
        missing = verify_wheel(wheel)
        if missing:
            failed = True
            print(f"{wheel}: legal-boundary check failed: {', '.join(missing)}", file=sys.stderr)
        else:
            print(f"{wheel}: PEP 639 metadata and substantive legal-boundary files present")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
