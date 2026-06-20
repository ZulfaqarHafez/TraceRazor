"""Locate the TraceRazor CLI for benchmark and report scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _target_binaries(repo: Path) -> list[Path]:
    candidates: list[Path] = []
    for profile in ("release", "debug"):
        for name in ("tracerazor.exe", "tracerazor"):
            path = repo / "target" / profile / name
            if path.is_file():
                candidates.append(path)
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def _supports_required_audit_flags(binary: Path, flags: tuple[str, ...]) -> bool:
    if not flags:
        return True
    try:
        out = subprocess.run(
            [str(binary), "audit", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and all(flag in out.stdout for flag in flags)


def find_tracerazor_binary(
    repo: Path = REPO_ROOT,
    required_audit_flags: tuple[str, ...] = ("--hermetic",),
) -> str:
    """Return a usable TraceRazor binary, preferring fresh source builds.

    Source checkouts often have both target/release and target/debug artifacts.
    During local iteration the debug binary may be newer than a stale release
    binary, so target artifacts are sorted by mtime instead of hard-coding the
    release profile first.
    """

    paths: list[Path] = []
    env = os.environ.get("TRACERAZOR_BIN")
    if env:
        paths.append(Path(env))

    which = shutil.which("tracerazor")
    if which:
        paths.append(Path(which))

    paths.extend(_target_binaries(repo))

    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        if _supports_required_audit_flags(resolved, required_audit_flags):
            return str(resolved)

    need = ", ".join(required_audit_flags) if required_audit_flags else "benchmark support"
    raise RuntimeError(
        "could not find a usable tracerazor binary "
        f"(required audit flag support: {need}). "
        "Run `cargo build --release -p tracerazor` or set TRACERAZOR_BIN."
    )
