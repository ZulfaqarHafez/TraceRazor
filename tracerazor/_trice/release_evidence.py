"""Deterministic release evidence packets for TraceRazor/TRICE."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib
from pathlib import Path
from typing import Any

from .evidence import canonical_json, sha256_file

RELEASE_EVIDENCE_SCHEMA_VERSION = "trice-release-evidence/v1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "docs" / "trice_release_evidence.json"
DEFAULT_DIST = REPO / "dist"
DEFAULT_ARTIFACT_PATHS = {
    "contract_card": ("proof-card", REPO / "docs" / "trice_contract_card.json"),
    "artifact_card": ("proof-card", REPO / "docs" / "trice_artifact_card.json"),
    "reproduction_card": ("proof-card", REPO / "docs" / "trice_reproduction_card.json"),
    "crates_card": ("proof-card", REPO / "docs" / "trice_crates_card.json"),
    "install_card": ("proof-card", REPO / "docs" / "trice_install_card.json"),
    "research_card": ("proof-card", REPO / "docs" / "trice_research_card.json"),
    "broad_evidence_bundle": (
        "trice-bundle",
        REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_broad_smoke_evidence.trice.zip",
    ),
    "remote_smoke_bundle": (
        "trice-bundle",
        REPO / "benchmark" / "trice" / "results" / "v2-remote-smoke" / "trice_remote_smoke_evidence.trice.zip",
    ),
    "paper_pdf": ("paper", REPO / "paper" / "trice_v3_research_paper.pdf"),
    "paper_manifest": ("paper-manifest", REPO / "paper" / "trice_v3_research_manifest.json"),
}


def build_release_evidence_card(
    *,
    dist_dir: str | Path = DEFAULT_DIST,
    cli_binary_path: str | Path | None = None,
    sidecar_stem: str = "trice_release_evidence",
    package_name: str = "tracerazor",
) -> dict[str, Any]:
    """Build a deterministic release evidence packet card."""

    dist_path = Path(dist_dir)
    pyproject_path = REPO / "pyproject.toml"
    cargo_lock_path = REPO / "Cargo.lock"
    pyproject = _load_toml(pyproject_path)
    version = str((pyproject.get("project") or {}).get("version") or "")
    artifacts = _release_artifacts(dist_path, Path(cli_binary_path) if cli_binary_path is not None else None)
    artifacts.extend(_static_artifacts())
    artifacts.sort(key=lambda row: (row["kind"], row["path"], row["name"]))
    python_sbom = _python_sbom(pyproject, package_name, version)
    cargo_sbom = _cargo_sbom(cargo_lock_path)
    provenance_statement = _provenance_statement(package_name, version, artifacts, pyproject_path, cargo_lock_path)
    sidecars = _sidecar_rows(sidecar_stem, artifacts, python_sbom, cargo_sbom, provenance_statement)
    checks = [
        _check("wheel_present", _kind_count(artifacts, "wheel") >= 1, _kind_count(artifacts, "wheel"), "at least one wheel artifact"),
        _check("sdist_absent", _kind_count(artifacts, "sdist") == 0, _kind_count(artifacts, "sdist"), "no source distribution until it can satisfy the bundled-auditor contract"),
        _check("cli_binary_present", _kind_count(artifacts, "binary") >= 1, _kind_count(artifacts, "binary"), "one built CLI binary"),
        _check("proof_cards_present", _all_named_present(artifacts, ["contract_card", "artifact_card", "reproduction_card", "crates_card", "install_card", "research_card"]), _present_named(artifacts, ["contract_card", "artifact_card", "reproduction_card", "crates_card", "install_card", "research_card"]), "contract, artifact, reproduction, crates, installability, and research cards"),
        _check("evidence_bundles_present", _all_named_present(artifacts, ["broad_evidence_bundle", "remote_smoke_bundle"]), _present_named(artifacts, ["broad_evidence_bundle", "remote_smoke_bundle"]), "broad and remote smoke evidence bundles"),
        _check("paper_artifacts_present", _all_named_present(artifacts, ["paper_pdf", "paper_manifest"]), _present_named(artifacts, ["paper_pdf", "paper_manifest"]), "paper PDF and paper manifest"),
        _check("artifact_hashes_present", all(row.get("sha256") for row in artifacts if row.get("present")), _hash_count(artifacts), "every present artifact has a SHA-256 digest"),
        _check("python_sbom_generated", python_sbom["component_count"] >= 1, python_sbom["component_count"], "CycloneDX-style Python SBOM"),
        _check("cargo_sbom_generated", cargo_sbom["component_count"] >= 1, cargo_sbom["component_count"], "CycloneDX-style Cargo SBOM"),
        _check("provenance_statement_generated", len(provenance_statement["subject"]) >= 1, len(provenance_statement["subject"]), "in-toto/SLSA-shaped provenance statement"),
        _check("sidecars_hashed", all(row.get("sha256") for row in sidecars), _hash_count(sidecars), "checksums, SBOMs, and provenance sidecars have hashes"),
    ]
    card = {
        "schema_version": RELEASE_EVIDENCE_SCHEMA_VERSION,
        "package": package_name,
        "version": version,
        "release_evidence_level": _release_evidence_level(checks),
        "release_evidence_score": _release_evidence_score(checks),
        "inputs": {
            "dist_dir": _dir_row(dist_path),
            "cli_binary": _artifact_row("rust_cli", "binary", Path(cli_binary_path) if cli_binary_path is not None else _default_cli_binary()),
            "pyproject": _artifact_row("pyproject", "source-metadata", pyproject_path),
            "cargo_lock": _artifact_row("cargo_lock", "source-metadata", cargo_lock_path),
        },
        "artifacts": artifacts,
        "checks": checks,
        "sidecars": sidecars,
        "python_sbom": python_sbom,
        "cargo_sbom": cargo_sbom,
        "provenance_statement": provenance_statement,
        "research_basis": [
            "SLSA provenance separates release trust from local tests by naming subjects, builders, materials, and build parameters.",
            "in-toto statements provide a portable envelope with subjects and predicateType for supply-chain attestations.",
            "CycloneDX SBOMs make Python and Cargo dependency inventories inspectable by downstream users.",
            "PyPI trusted publishing and attestations make registry-side provenance stronger when paired with deterministic local release evidence.",
            "TRICE release evidence keeps checksums, SBOMs, provenance, benchmark bundles, and paper artifacts under one verifier.",
            "Research cards keep the paper's cited basis under the same release-evidence packet as the wheel and benchmark artifacts.",
        ],
        "next_actions": _next_actions(checks),
    }
    card["release_evidence_sha256"] = hashlib.sha256(canonical_json(_without_release_evidence_hash(card)).encode("utf-8")).hexdigest()
    return card


def verify_release_evidence_file(path: str | Path) -> dict[str, Any]:
    """Verify a release evidence card, bound artifacts, and sidecars."""

    card_path = Path(path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if card.get("schema_version") != RELEASE_EVIDENCE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {RELEASE_EVIDENCE_SCHEMA_VERSION}")
    expected_hash = str(card.get("release_evidence_sha256") or "")
    actual_hash = hashlib.sha256(canonical_json(_without_release_evidence_hash(card)).encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        errors.append("release_evidence_sha256 mismatch")

    checked_artifacts: list[str] = []
    for row in card.get("artifacts") or []:
        if isinstance(row, dict):
            _verify_artifact_row(card_path, row, errors, checked_artifacts)
        else:
            errors.append("artifact row must be an object")

    sidecar_content = _expected_sidecars(card)
    checked_sidecars: list[str] = []
    for row in card.get("sidecars") or []:
        if not isinstance(row, dict):
            errors.append("sidecar row must be an object")
            continue
        filename = str(row.get("filename") or "")
        expected = sidecar_content.get(row.get("name"))
        sidecar_path = card_path.parent / filename
        if expected is None:
            errors.append(f"unknown sidecar {row.get('name')}")
            continue
        if not sidecar_path.is_file():
            errors.append(f"sidecar missing: {filename}")
            continue
        actual_text = sidecar_path.read_text(encoding="utf-8")
        if actual_text != expected:
            errors.append(f"sidecar content mismatch: {filename}")
        if sidecar_path.stat().st_size != int(row.get("bytes") or 0):
            errors.append(f"sidecar byte count mismatch: {filename}")
        if hashlib.sha256(actual_text.encode("utf-8")).hexdigest() != row.get("sha256"):
            errors.append(f"sidecar sha256 mismatch: {filename}")
        else:
            checked_sidecars.append(filename)

    try:
        rebuilt = build_release_evidence_card(
            dist_dir=_resolve_path(card_path, str(((card.get("inputs") or {}).get("dist_dir") or {}).get("path") or "dist")),
            cli_binary_path=_resolve_path(card_path, str(((card.get("inputs") or {}).get("cli_binary") or {}).get("path") or _display_path(_default_cli_binary()))),
            sidecar_stem=_sidecar_stem_from_rows(card.get("sidecars") or []),
            package_name=str(card.get("package") or "tracerazor"),
        )
        if canonical_json(_without_release_evidence_hash(rebuilt)) != canonical_json(_without_release_evidence_hash(card)):
            errors.append("release evidence card does not match deterministic rebuild from bound inputs")
    except Exception as exc:
        errors.append(f"release evidence rebuild failed: {exc}")

    return {
        "ok": not errors,
        "schema_version": card.get("schema_version"),
        "release_evidence_level": card.get("release_evidence_level"),
        "release_evidence_score": card.get("release_evidence_score"),
        "release_evidence_sha256": expected_hash,
        "computed_release_evidence_sha256": actual_hash,
        "checked_artifacts": checked_artifacts,
        "checked_sidecars": checked_sidecars,
        "errors": errors,
    }


def render_release_evidence_markdown(card: dict[str, Any]) -> str:
    lines = [
        "# TRICE Release Evidence",
        "",
        f"- Package: `{card['package']}`",
        f"- Version: `{card['version']}`",
        f"- Evidence level: `{card['release_evidence_level']}`",
        f"- Evidence score: **{card['release_evidence_score']}/100**",
        "",
        "## Checks",
        "",
        "| Check | Passed | Observed | Required |",
        "|---|---:|---|---|",
    ]
    for row in card["checks"]:
        lines.append(f"| {row['name']} | {'yes' if row['passed'] else 'no'} | {_md(row['observed'])} | {_md(row['required'])} |")
    lines.extend(["", "## Release Artifacts", "", "| Artifact | Kind | Present | Path | SHA-256 |", "|---|---|---:|---|---|"])
    for row in card["artifacts"]:
        lines.append(f"| {row['name']} | {row['kind']} | {'yes' if row['present'] else 'no'} | `{row['path']}` | `{row['sha256']}` |")
    lines.extend(["", "## Sidecars", "", "| Sidecar | File | SHA-256 |", "|---|---|---|"])
    for row in card["sidecars"]:
        lines.append(f"| {row['name']} | `{row['filename']}` | `{row['sha256']}` |")
    lines.extend(["", "## Next Actions", ""])
    for item in card["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Hash", "", f"- release evidence: `{card['release_evidence_sha256']}`", ""])
    return "\n".join(lines)


def render_release_evidence_tex(card: dict[str, Any]) -> str:
    rows = "\n".join(
        f"{_tex(row['name'])} & {'yes' if row['passed'] else 'no'} & {_tex(str(row['required']))} \\\\"
        for row in card["checks"]
    )
    return (
        "\\section{Release Evidence}\n"
        f"Release evidence level: \\texttt{{{_tex(card['release_evidence_level'])}}}; "
        f"score: {card['release_evidence_score']}/100.\n\n"
        "\\begin{tabular}{lll}\n"
        "Check & Pass & Required \\\\\n"
        "\\hline\n"
        f"{rows}\n"
        "\\end{tabular}\n"
    )


def render_release_evidence_svg(card: dict[str, Any]) -> str:
    stages = [
        ("packages", _check_passed(card, "wheel_present") and _check_passed(card, "sdist_absent")),
        ("binary", _check_passed(card, "cli_binary_present")),
        ("proof", _check_passed(card, "proof_cards_present") and _check_passed(card, "evidence_bundles_present")),
        ("sbom", _check_passed(card, "python_sbom_generated") and _check_passed(card, "cargo_sbom_generated")),
        ("slsa", _check_passed(card, "provenance_statement_generated")),
        ("hashes", _check_passed(card, "artifact_hashes_present") and _check_passed(card, "sidecars_hashed")),
    ]
    width, height = 1040, 280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE release evidence</text>',
        f'<text x="28" y="58" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Score {card["release_evidence_score"]}/100 | level {card["release_evidence_level"]} | hash {card["release_evidence_sha256"][:16]}...</text>',
    ]
    x0, y = 28, 96
    for idx, (label, passed) in enumerate(stages):
        x = x0 + idx * 164
        fill = "#2563eb" if passed else "#e5e7eb"
        text = "#ffffff" if passed else "#374151"
        parts.append(f'<rect x="{x}" y="{y}" width="126" height="58" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{x + 63}" y="{y + 36}" text-anchor="middle" font-family="Inter,Segoe UI,Arial" font-size="15" font-weight="700" fill="{text}">{label}</text>')
        if idx < len(stages) - 1:
            parts.append(f'<line x1="{x + 136}" y1="{y + 29}" x2="{x + 156}" y2="{y + 29}" stroke="#9ca3af" stroke-width="3"/>')
    parts.append(f'<text x="28" y="202" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#111827">Artifacts {sum(1 for row in card["artifacts"] if row["present"])}/{len(card["artifacts"])} | Python SBOM components {card["python_sbom"]["component_count"]} | Cargo SBOM components {card["cargo_sbom"]["component_count"]}</text>')
    parts.append('<text x="28" y="228" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">Release evidence is a local verifier packet; the 1.1 contract still requires platform-wheel, image, tag, CI, and PyPI publication checks.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_release_evidence_outputs(card: dict[str, Any], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_text_lf(out, json.dumps(card, indent=2, sort_keys=True) + "\n")
    md = out.with_suffix(".md")
    tex = out.with_suffix(".tex")
    svg = out.with_suffix(".svg")
    _write_text_lf(md, render_release_evidence_markdown(card))
    _write_text_lf(tex, render_release_evidence_tex(card))
    _write_text_lf(svg, render_release_evidence_svg(card))
    sidecars = _expected_sidecars(card)
    sidecar_outputs: dict[str, str] = {}
    for row in card["sidecars"]:
        path = out.parent / row["filename"]
        _write_text_lf(path, sidecars[row["name"]])
        sidecar_outputs[row["name"]] = str(path)
    return {"json": str(out), "markdown": str(md), "tex": str(tex), "svg": str(svg), **sidecar_outputs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate deterministic TRICE release evidence with checksums, SBOMs, and provenance.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dist-dir", type=Path, default=DEFAULT_DIST)
    ap.add_argument("--cli-binary", type=Path, default=None)
    ap.add_argument("--format", choices=["json", "markdown", "tex"], default="json")
    args = ap.parse_args(argv)
    card = build_release_evidence_card(dist_dir=args.dist_dir, cli_binary_path=args.cli_binary, sidecar_stem=args.out.stem)
    outputs = write_release_evidence_outputs(card, args.out)
    if args.format == "markdown":
        print(render_release_evidence_markdown(card))
    elif args.format == "tex":
        print(render_release_evidence_tex(card))
    else:
        print(json.dumps({"release_evidence": card, "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if card["release_evidence_level"] != "release_evidence_missing" else 1


def verify_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify deterministic TRICE release evidence.")
    ap.add_argument("release_evidence", type=Path)
    args = ap.parse_args(argv)
    verdict = verify_release_evidence_file(args.release_evidence)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def _release_artifacts(dist_dir: Path, cli_binary_path: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if dist_dir.is_dir():
        for path in sorted(dist_dir.iterdir(), key=lambda p: p.name):
            if path.is_file() and path.name.startswith("tracerazor-") and path.suffix == ".whl":
                rows.append(_artifact_row(path.name, "wheel", path))
            elif path.is_file() and path.name.startswith("tracerazor-") and path.name.endswith(".tar.gz"):
                rows.append(_artifact_row(path.name, "sdist", path))
    rows.append(_artifact_row("rust_cli", "binary", cli_binary_path or _default_cli_binary()))
    return rows


def _static_artifacts() -> list[dict[str, Any]]:
    return [_artifact_row(name, kind, path) for name, (kind, path) in DEFAULT_ARTIFACT_PATHS.items()]


def _artifact_row(name: str, kind: str, path: Path) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "path": _display_path(path),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _dir_row(path: Path) -> dict[str, Any]:
    return {"path": _display_path(path), "present": path.is_dir()}


def _python_sbom(pyproject: dict[str, Any], package_name: str, version: str) -> dict[str, Any]:
    project = pyproject.get("project") if isinstance(pyproject.get("project"), dict) else {}
    components = [
        {
            "type": "application",
            "name": package_name,
            "version": version,
            "purl": f"pkg:pypi/{package_name}@{version}" if version else f"pkg:pypi/{package_name}",
        }
    ]
    for dep in project.get("dependencies") or []:
        name = _requirement_name(str(dep))
        if name:
            components.append({"type": "library", "name": name, "version": _requirement_spec(str(dep)), "purl": f"pkg:pypi/{name}"})
    optional = project.get("optional-dependencies") if isinstance(project.get("optional-dependencies"), dict) else {}
    for group, deps in sorted(optional.items()):
        for dep in deps or []:
            name = _requirement_name(str(dep))
            if name:
                components.append({
                    "type": "library",
                    "name": name,
                    "version": _requirement_spec(str(dep)),
                    "purl": f"pkg:pypi/{name}",
                    "properties": [{"name": "trice:optional_group", "value": str(group)}],
                })
    components = _dedupe_components(components)
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": _urn_uuid(canonical_json({"python": components})),
        "version": 1,
        "metadata": {
            "component": {"type": "application", "name": package_name, "version": version},
            "tools": [{"vendor": "TraceRazor", "name": "tracerazor-trice release-evidence"}],
        },
        "components": components,
    }
    return {"format": "CycloneDX", "spec_version": "1.6", "component_count": len(components), "bom": bom}


def _cargo_sbom(cargo_lock: Path) -> dict[str, Any]:
    packages = []
    if cargo_lock.is_file():
        data = tomllib.loads(cargo_lock.read_text(encoding="utf-8"))
        packages = data.get("package") if isinstance(data.get("package"), list) else []
    components = []
    for pkg in packages:
        if not isinstance(pkg, dict):
            continue
        name = str(pkg.get("name") or "")
        version = str(pkg.get("version") or "")
        if not name:
            continue
        row = {"type": "library", "name": name, "version": version, "purl": f"pkg:cargo/{name}@{version}"}
        checksum = pkg.get("checksum")
        if checksum:
            row["hashes"] = [{"alg": "SHA-256", "content": str(checksum)}]
        components.append(row)
    components = _dedupe_components(components)
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": _urn_uuid(canonical_json({"cargo": components})),
        "version": 1,
        "metadata": {
            "component": {"type": "application", "name": "tracerazor", "version": _cargo_workspace_version()},
            "tools": [{"vendor": "TraceRazor", "name": "tracerazor-trice release-evidence"}],
        },
        "components": components,
    }
    return {"format": "CycloneDX", "spec_version": "1.6", "component_count": len(components), "bom": bom}


def _provenance_statement(package_name: str, version: str, artifacts: list[dict[str, Any]], pyproject: Path, cargo_lock: Path) -> dict[str, Any]:
    subjects = [
        {"name": row["path"], "digest": {"sha256": row["sha256"]}}
        for row in artifacts
        if row.get("present") and row.get("sha256")
    ]
    materials = [
        {"uri": _display_path(pyproject), "digest": {"sha256": sha256_file(pyproject)}} if pyproject.is_file() else None,
        {"uri": _display_path(cargo_lock), "digest": {"sha256": sha256_file(cargo_lock)}} if cargo_lock.is_file() else None,
    ]
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": subjects,
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://tracerazor.dev/trice/release-evidence/v1",
                "externalParameters": {"package": package_name, "version": version},
                "internalParameters": {"generator": "tracerazor-trice release-evidence"},
                "resolvedDependencies": [item for item in materials if item is not None],
            },
            "runDetails": {
                "builder": {"id": "https://github.com/ZulfaqarHafez/tracerazor"},
                "metadata": {"reproducible": True},
            },
        },
    }


def _sidecar_rows(
    sidecar_stem: str,
    artifacts: list[dict[str, Any]],
    python_sbom: dict[str, Any],
    cargo_sbom: dict[str, Any],
    provenance_statement: dict[str, Any],
) -> list[dict[str, Any]]:
    sidecars = {
        "checksums": _checksums_text(artifacts),
        "python_sbom": json.dumps(python_sbom["bom"], indent=2, sort_keys=True) + "\n",
        "cargo_sbom": json.dumps(cargo_sbom["bom"], indent=2, sort_keys=True) + "\n",
        "provenance": json.dumps(provenance_statement, indent=2, sort_keys=True) + "\n",
    }
    suffixes = {
        "checksums": ".checksums.txt",
        "python_sbom": ".python.cdx.json",
        "cargo_sbom": ".cargo.cdx.json",
        "provenance": ".intoto.json",
    }
    rows = []
    for name, text in sidecars.items():
        rows.append({
            "name": name,
            "filename": f"{sidecar_stem}{suffixes[name]}",
            "bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    return rows


def _expected_sidecars(card: dict[str, Any]) -> dict[str, str]:
    artifacts = card.get("artifacts") if isinstance(card.get("artifacts"), list) else []
    return {
        "checksums": _checksums_text([row for row in artifacts if isinstance(row, dict)]),
        "python_sbom": json.dumps((card.get("python_sbom") or {}).get("bom") or {}, indent=2, sort_keys=True) + "\n",
        "cargo_sbom": json.dumps((card.get("cargo_sbom") or {}).get("bom") or {}, indent=2, sort_keys=True) + "\n",
        "provenance": json.dumps(card.get("provenance_statement") or {}, indent=2, sort_keys=True) + "\n",
    }


def _checksums_text(artifacts: list[dict[str, Any]]) -> str:
    lines = [
        f"{row['sha256']}  {row['path']}"
        for row in sorted(artifacts, key=lambda item: str(item.get("path") or ""))
        if row.get("present") and row.get("sha256")
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _verify_artifact_row(card_path: Path, row: dict[str, Any], errors: list[str], checked: list[str]) -> None:
    raw_path = str(row.get("path") or "")
    if not raw_path:
        errors.append("artifact path missing")
        return
    path = _resolve_path(card_path, raw_path)
    should_exist = bool(row.get("present"))
    if not path.is_file():
        if should_exist:
            errors.append(f"artifact missing: {raw_path}")
        return
    if not should_exist:
        errors.append(f"artifact unexpectedly exists: {raw_path}")
        return
    if path.stat().st_size != int(row.get("bytes") or 0):
        errors.append(f"artifact byte count mismatch: {raw_path}")
    if sha256_file(path) != row.get("sha256"):
        errors.append(f"artifact sha256 mismatch: {raw_path}")
    else:
        checked.append(str(row.get("name") or raw_path))


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _default_cli_binary() -> Path:
    exe = "tracerazor.exe" if sys.platform.startswith("win") else "tracerazor"
    return REPO / "target" / "release" / exe


def _cargo_workspace_version() -> str:
    cargo_toml = REPO / "Cargo.toml"
    if not cargo_toml.is_file():
        return ""
    data = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
    workspace = data.get("workspace") if isinstance(data.get("workspace"), dict) else {}
    package = workspace.get("package") if isinstance(workspace.get("package"), dict) else {}
    return str(package.get("version") or "")


def _requirement_name(req: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", req)
    return match.group(1).replace("_", "-").lower() if match else ""


def _requirement_spec(req: str) -> str:
    name = _requirement_name(req)
    return req[len(name):].strip() if name and len(req) > len(name) else ""


def _dedupe_components(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out = []
    for row in sorted(components, key=lambda item: (item.get("type", ""), item.get("name", ""), item.get("version", ""))):
        key = (str(row.get("type") or ""), str(row.get("name") or ""), str(row.get("version") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _urn_uuid(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"urn:uuid:{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _release_evidence_level(checks: list[dict[str, Any]]) -> str:
    passed = {row["name"] for row in checks if row["passed"]}
    required = {
        "wheel_present",
        "sdist_absent",
        "cli_binary_present",
        "proof_cards_present",
        "evidence_bundles_present",
        "paper_artifacts_present",
        "artifact_hashes_present",
        "python_sbom_generated",
        "cargo_sbom_generated",
        "provenance_statement_generated",
        "sidecars_hashed",
    }
    if required.issubset(passed):
        return "release_evidence_ready"
    if {"proof_cards_present", "python_sbom_generated", "cargo_sbom_generated", "provenance_statement_generated"}.issubset(passed):
        return "partial_release_evidence"
    return "release_evidence_missing"


def _release_evidence_score(checks: list[dict[str, Any]]) -> int:
    weights = {
        "wheel_present": 10,
        "sdist_absent": 10,
        "cli_binary_present": 8,
        "proof_cards_present": 10,
        "evidence_bundles_present": 10,
        "paper_artifacts_present": 8,
        "artifact_hashes_present": 10,
        "python_sbom_generated": 10,
        "cargo_sbom_generated": 10,
        "provenance_statement_generated": 10,
        "sidecars_hashed": 4,
    }
    return min(100, sum(weights.get(row["name"], 0) for row in checks if row["passed"]))


def _next_actions(checks: list[dict[str, Any]]) -> list[str]:
    missing = [row["name"] for row in checks if not row["passed"]]
    if not missing:
        return [
            "Attach the release evidence card, checksums, SBOMs, provenance statement, platform wheels, binaries, paper, and evidence bundles to the GitHub release.",
            "Regenerate this packet after every package rebuild, proof-card change, or evidence-bundle change.",
            "Publish registry attestations through trusted publishing where supported.",
        ]
    actions = []
    mapping = {
        "wheel_present": "Build the Python wheel before generating release evidence.",
        "sdist_absent": "Remove the source distribution until it can satisfy the bundled-auditor contract.",
        "cli_binary_present": "Build the Rust CLI binary before generating release evidence.",
        "proof_cards_present": "Regenerate contract, artifact, reproduction, and crates cards before generating release evidence.",
        "evidence_bundles_present": "Regenerate broad and remote smoke .trice.zip bundles.",
        "paper_artifacts_present": "Regenerate the paper PDF and paper manifest.",
    }
    for name in missing:
        if name in mapping:
            actions.append(mapping[name])
    return actions or ["Regenerate release evidence after all release artifacts exist."]


def _kind_count(rows: list[dict[str, Any]], kind: str) -> int:
    return sum(1 for row in rows if row.get("kind") == kind and row.get("present"))


def _hash_count(rows: list[dict[str, Any]]) -> str:
    return f"{sum(1 for row in rows if row.get('sha256'))}/{len(rows)}"


def _all_named_present(rows: list[dict[str, Any]], names: list[str]) -> bool:
    present = {row.get("name") for row in rows if row.get("present")}
    return set(names).issubset(present)


def _present_named(rows: list[dict[str, Any]], names: list[str]) -> str:
    present = {row.get("name") for row in rows if row.get("present")}
    return f"{sum(1 for name in names if name in present)}/{len(names)}"


def _sidecar_stem_from_rows(rows: list[Any]) -> str:
    for row in rows:
        if isinstance(row, dict) and str(row.get("filename") or "").endswith(".checksums.txt"):
            return str(row["filename"])[: -len(".checksums.txt")]
    return "trice_release_evidence"


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def _check_passed(card: dict[str, Any], name: str) -> bool:
    return any(row["name"] == name and row["passed"] for row in card.get("checks", []))


def _without_release_evidence_hash(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out.pop("release_evidence_sha256", None)
    return out


def _resolve_path(card_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    repo_candidate = REPO / path
    if repo_candidate.exists():
        return repo_candidate
    card_relative = card_path.parent / path
    return card_relative if card_relative.exists() else repo_candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _md(value: Any) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return text.replace("|", "\\|")


def _tex(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


if __name__ == "__main__":
    raise SystemExit(main())
