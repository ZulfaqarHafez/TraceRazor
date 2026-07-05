"""Deterministic source provenance helpers for TRICE suites."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

IGNORED_DIRS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


@dataclass(frozen=True)
class TreeFingerprint:
    algorithm: str
    digest: str
    file_count: int
    bytes: int
    ignored_dirs: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["ignored_dirs"] = list(self.ignored_dirs)
        return data


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_contained_file(root: Path, path: Path) -> None:
    base = root.resolve()
    resolved = path.resolve()
    if resolved != base and base not in resolved.parents:
        rel = path.relative_to(root).as_posix()
        raise ValueError(f"refusing to hash file outside tree: {rel}")


def fingerprint_tree(root: str | Path, *, ignored_dirs: Iterable[str] = IGNORED_DIRS) -> TreeFingerprint:
    """Hash a source tree using relative paths plus per-file SHA-256 hashes."""

    base = Path(root).resolve()
    ignored = tuple(sorted(set(ignored_dirs)))
    h = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in _iter_files(base, ignored):
        _assert_contained_file(base, path)
        rel = path.relative_to(base).as_posix()
        digest = hash_file(path)
        size = path.stat().st_size
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(str(size).encode("ascii"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
        count += 1
        total_bytes += size
    return TreeFingerprint(
        algorithm="trice-tree-sha256/v1",
        digest=h.hexdigest(),
        file_count=count,
        bytes=total_bytes,
        ignored_dirs=ignored,
    )


def _iter_files(root: Path, ignored_dirs: tuple[str, ...]):
    ignored = set(ignored_dirs)
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        rel_parts = path.relative_to(root).parts
        if any(part in ignored for part in rel_parts[:-1]):
            continue
        if path.is_file():
            yield path
