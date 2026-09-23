#!/usr/bin/env python3
"""Propagate the canonical TraceRazor Agent Skill to its mirrors.

``skills/tracerazor/SKILL.md`` is the single source. Each mirror has a real
consumer (host discovery paths, the wheel's agent bundles, and the CLI's
``include_str!``), so they stay real byte-identical files, not symlinks.

    python scripts/sync_skill.py          # copy the source over every mirror
    python scripts/sync_skill.py --check  # exit 1 if any mirror has drifted
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "skills" / "tracerazor" / "SKILL.md"
MIRRORS = [
    ROOT / ".claude" / "skills" / "tracerazor" / "SKILL.md",
    ROOT / ".agents" / "skills" / "tracerazor" / "SKILL.md",
    ROOT / "plugins" / "tracerazor" / "skills" / "tracerazor" / "SKILL.md",
    ROOT / "extensions" / "claude-code" / "tracerazor" / "skills" / "tracerazor" / "SKILL.md",
    ROOT / "extensions" / "gemini-cli" / "tracerazor" / "skills" / "tracerazor" / "SKILL.md",
    ROOT / "crates" / "tracerazor-cli" / "assets" / "tracerazor-skill" / "SKILL.md",
]


def main(argv: list[str]) -> int:
    source = SOURCE.read_bytes()
    stale = [m for m in MIRRORS if not m.is_file() or m.read_bytes() != source]
    if "--check" in argv:
        for mirror in stale:
            print(f"drifted: {mirror.relative_to(ROOT)}", file=sys.stderr)
        return 1 if stale else 0
    for mirror in stale:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_bytes(source)
        print(f"updated {mirror.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
