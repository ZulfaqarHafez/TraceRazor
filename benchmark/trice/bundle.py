"""Portable TRICE evidence bundles."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .evidence import load_manifest, sha256_file, verify_manifest
from .suite import verify_suite_evidence

BUNDLE_SCHEMA_VERSION = "trice-evidence-bundle/v1"
FIXED_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class BundleEntry:
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class BundleManifest:
    schema_version: str
    root_manifest: str
    root_result: str
    verifier: str
    entries: tuple[BundleEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entries"] = [asdict(entry) for entry in self.entries]
        return data


def export_evidence_bundle(
    manifest_path: str | Path,
    out_path: str | Path | None = None,
    *,
    result_path: str | Path | None = None,
) -> Path:
    manifest = Path(manifest_path)
    base = manifest.parent.resolve()
    result = Path(result_path) if result_path else _default_result_for_manifest(manifest)
    bundle_path = Path(out_path) if out_path else manifest.with_suffix(".trice.zip")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    files = _collect_manifest_files(manifest, result)
    entries = tuple(
        BundleEntry(path=rel, sha256=sha256_file(path), bytes=path.stat().st_size)
        for rel, path in sorted(files.items())
    )
    bundle_manifest = BundleManifest(
        schema_version=BUNDLE_SCHEMA_VERSION,
        root_manifest=_rel(manifest, base),
        root_result=_rel(result, base),
        verifier="tracerazor-trice verify-bundle",
        entries=entries,
    )
    ro_crate = _ro_crate_metadata(bundle_manifest)

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _writestr_deterministic(zf, "trice_bundle_manifest.json", json.dumps(bundle_manifest.to_dict(), indent=2, sort_keys=True) + "\n")
        _writestr_deterministic(zf, "ro-crate-metadata.json", json.dumps(ro_crate, indent=2, sort_keys=True) + "\n")
        for rel, path in sorted(files.items()):
            _write_file_deterministic(zf, path, rel)
    return bundle_path


def verify_evidence_bundle(bundle_path: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_path)
    errors: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="trice-bundle-") as td:
            root = Path(td)
            with zipfile.ZipFile(bundle, "r") as zf:
                names = zf.namelist()
                for name in names:
                    try:
                        _validate_bundle_member(name)
                    except ValueError as exc:
                        errors.append(str(exc))
                if errors:
                    return {"ok": False, "errors": errors, "bundle": str(bundle)}
                zf.extractall(root)

            manifest_path = root / "trice_bundle_manifest.json"
            if not manifest_path.is_file():
                return {"ok": False, "errors": ["missing trice_bundle_manifest.json"], "bundle": str(bundle)}
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if data.get("schema_version") != BUNDLE_SCHEMA_VERSION:
                errors.append(f"schema_version must be {BUNDLE_SCHEMA_VERSION}")
            for entry in data.get("entries", []):
                rel = str(entry.get("path", ""))
                try:
                    _validate_bundle_member(rel)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                p = root / rel
                if not p.is_file():
                    errors.append(f"missing bundled file: {rel}")
                    continue
                if p.stat().st_size != int(entry.get("bytes", -1)):
                    errors.append(f"byte-size mismatch: {rel}")
                if sha256_file(p) != entry.get("sha256"):
                    errors.append(f"sha256 mismatch: {rel}")

            root_manifest = root / str(data.get("root_manifest", ""))
            root_result = root / str(data.get("root_result", ""))
            if not root_manifest.is_file():
                errors.append("missing root manifest")
                root_verdict = {"ok": False, "errors": ["missing root manifest"]}
            elif root_result.name == "trice_suite_results.json":
                root_verdict = verify_suite_evidence(root_manifest, root_result)
            else:
                root_verdict = verify_manifest(root_manifest, root_result)
            errors.extend(root_verdict.get("errors", []))
            return {
                "ok": not errors and root_verdict.get("ok", False),
                "errors": errors,
                "bundle": str(bundle),
                "root_manifest": data.get("root_manifest"),
                "root_result": data.get("root_result"),
                "entry_count": len(data.get("entries", [])),
                "root_verdict": root_verdict,
            }
    except (zipfile.BadZipFile, OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "errors": [f"invalid evidence bundle: {exc}"], "bundle": str(bundle)}


def _collect_manifest_files(manifest_path: Path, result_path: Path, bundle_base: Path | None = None) -> dict[str, Path]:
    base = manifest_path.parent.resolve()
    root = bundle_base.resolve() if bundle_base else base
    files: dict[str, Path] = {}
    _add_file(files, root, manifest_path)
    _add_file(files, root, result_path)
    manifest = load_manifest(manifest_path)
    for artifact in manifest.artifacts:
        artifact_path = _resolve_bundle_path(base, artifact.path)
        _add_file(files, root, artifact_path)
        if artifact_path.name.endswith("_evidence_manifest.json"):
            child_result = _default_result_for_manifest(artifact_path)
            if child_result.is_file():
                files.update(_collect_manifest_files(artifact_path, child_result, root))
    return files


def _add_file(files: dict[str, Path], base: Path, path: Path) -> None:
    resolved = path.resolve()
    rel = _rel(resolved, base)
    files[rel] = resolved


def _default_result_for_manifest(manifest_path: Path) -> Path:
    base = manifest_path.parent
    for filename in ("trice_suite_results.json", "trice_v2_live_results.json"):
        candidate = base / filename
        if candidate.is_file():
            return candidate
    return base / "trice_v2_live_results.json"


def _resolve_bundle_path(base: Path, rel: str) -> Path:
    _validate_bundle_member(rel)
    resolved = (base / rel).resolve()
    if base != resolved and base not in resolved.parents:
        raise ValueError(f"bundle artifact escapes root: {rel}")
    return resolved


def _rel(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def _validate_bundle_member(path: str) -> None:
    p = Path(path)
    if not path or p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe bundle path: {path}")


def _write_file_deterministic(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    info = zipfile.ZipInfo(arcname, FIXED_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, path.read_bytes())


def _writestr_deterministic(zf: zipfile.ZipFile, arcname: str, content: str) -> None:
    info = zipfile.ZipInfo(arcname, FIXED_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, content.encode("utf-8"))


def _ro_crate_metadata(bundle_manifest: BundleManifest) -> dict[str, Any]:
    return {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": [
            {
                "@id": "./",
                "@type": "Dataset",
                "name": "TRICE evidence bundle",
                "hasPart": [{"@id": entry.path} for entry in bundle_manifest.entries],
            },
            {
                "@id": "trice_bundle_manifest.json",
                "@type": "File",
                "encodingFormat": "application/json",
                "name": "TRICE bundle manifest",
            },
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export or verify portable TRICE evidence bundles.")
    sub = ap.add_subparsers(dest="command")

    export = sub.add_parser("export", help="Export a TRICE evidence bundle zip.")
    export.add_argument("manifest", type=Path)
    export.add_argument("--result", type=Path, default=None)
    export.add_argument("--out", type=Path, default=None)

    verify = sub.add_parser("verify", help="Verify a TRICE evidence bundle zip.")
    verify.add_argument("bundle", type=Path)

    args = ap.parse_args(argv)
    if args.command == "export":
        path = export_evidence_bundle(args.manifest, args.out, result_path=args.result)
        print(path)
        return 0
    if args.command == "verify":
        verdict = verify_evidence_bundle(args.bundle)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
