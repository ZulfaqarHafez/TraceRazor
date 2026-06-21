"""Deterministic evidence manifests for TRICE live runs.

TRICE is only ship-worthy if a result can be audited later without trusting the
machine that produced it. This module provides canonical JSON, stable hashing,
and manifest verification for live rollout artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_RESULT_FILENAMES = ("trice_v2_live_results.json", "trice_suite_results.json")


@dataclass(frozen=True)
class ArtifactHash:
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class EvidenceManifest:
    schema_version: str
    algorithm: str
    created_by: str
    python_version: str
    platform: str
    artifacts: tuple[ArtifactHash, ...]
    result_sha256: str
    canonical_result_sha256: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifacts"] = [asdict(a) for a in self.artifacts]
        data["notes"] = list(self.notes)
        return data


def canonical_json(data: Any) -> str:
    """Return a deterministic JSON representation for hashing and papers."""

    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    result: dict[str, Any],
    result_path: str | Path,
    artifact_paths: list[str | Path],
    algorithm: str,
    notes: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> EvidenceManifest:
    base = Path(base_dir) if base_dir else Path(result_path).parent
    artifacts = tuple(_artifact_hash(p, base) for p in artifact_paths)
    return EvidenceManifest(
        schema_version="trice-evidence-manifest/v1",
        algorithm=algorithm,
        created_by="TraceRazor TRICE",
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        artifacts=artifacts,
        result_sha256=sha256_file(result_path),
        canonical_result_sha256=sha256_text(canonical_json(result)),
        notes=tuple(notes or ()),
    )


def write_manifest(manifest: EvidenceManifest, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> EvidenceManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    artifacts = tuple(ArtifactHash(**a) for a in data.get("artifacts", []))
    return EvidenceManifest(
        schema_version=data["schema_version"],
        algorithm=data["algorithm"],
        created_by=data["created_by"],
        python_version=data["python_version"],
        platform=data["platform"],
        artifacts=artifacts,
        result_sha256=data["result_sha256"],
        canonical_result_sha256=data["canonical_result_sha256"],
        notes=tuple(data.get("notes") or ()),
    )


def verify_manifest(manifest_path: str | Path, result_path: str | Path | None = None) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    base = Path(manifest_path).parent
    errors: list[str] = []

    resolved_result = Path(result_path) if result_path else _default_result_path(base)
    if not resolved_result.is_file():
        errors.append(f"missing result file: {resolved_result}")
    else:
        file_hash = sha256_file(resolved_result)
        if file_hash != manifest.result_sha256:
            errors.append("result_sha256 mismatch")
        data = json.loads(resolved_result.read_text(encoding="utf-8"))
        canonical_hash = sha256_text(canonical_json(data))
        if canonical_hash != manifest.canonical_result_sha256:
            errors.append("canonical_result_sha256 mismatch")

    for artifact in manifest.artifacts:
        p = base / artifact.path
        if not p.is_file():
            errors.append(f"missing artifact: {artifact.path}")
            continue
        if p.stat().st_size != artifact.bytes:
            errors.append(f"byte-size mismatch: {artifact.path}")
        if sha256_file(p) != artifact.sha256:
            errors.append(f"sha256 mismatch: {artifact.path}")

    return {
        "ok": not errors,
        "errors": errors,
        "manifest": manifest.to_dict(),
    }


def _artifact_hash(path: str | Path, base: Path) -> ArtifactHash:
    p = Path(path)
    rel = Path(os.path.relpath(p, base)).as_posix()
    return ArtifactHash(path=rel, sha256=sha256_file(p), bytes=p.stat().st_size)


def _default_result_path(base: Path) -> Path:
    for filename in DEFAULT_RESULT_FILENAMES:
        candidate = base / filename
        if candidate.is_file():
            return candidate
    return base / DEFAULT_RESULT_FILENAMES[0]
