"""Deterministic crates.io publish-plan cards for TraceRazor."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .evidence import canonical_json, sha256_file

CRATES_CARD_SCHEMA_VERSION = "trice-crates-card/v1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "docs" / "trice_crates_card.json"
CRATES_API = "https://crates.io/api/v1/crates/{name}"
PUBLISH_ORDER = [
    ("tracerazor-core", "crates/tracerazor-core/Cargo.toml"),
    ("tracerazor-semantic", "crates/tracerazor-semantic/Cargo.toml"),
    ("tracerazor-ingest", "crates/tracerazor-ingest/Cargo.toml"),
    ("tracerazor-store", "crates/tracerazor-store/Cargo.toml"),
    ("tracerazor-server", "crates/tracerazor-server/Cargo.toml"),
    ("tracerazor", "crates/tracerazor-cli/Cargo.toml"),
]


def build_crates_card(
    *,
    cargo_toml_path: str | Path = REPO / "Cargo.toml",
    readme_path: str | Path = REPO / "README.md",
    offline: bool = False,
    timeout_s: float = 5.0,
    status_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic crates.io publish-plan card.

    The card separates local publication readiness from public registry truth.
    It intentionally does not require tokens and does not upload anything.
    """

    cargo_toml = Path(cargo_toml_path)
    readme = Path(readme_path)
    workspace = _load_toml(cargo_toml)
    workspace_package = workspace.get("workspace", {}).get("package", {})
    version = str(workspace_package.get("version") or "")
    status = status_snapshot or _registry_snapshot(offline=offline, timeout_s=timeout_s)
    packages = [
        _package_row(name, REPO / manifest, workspace_package, status)
        for name, manifest in PUBLISH_ORDER
    ]
    checks = [
        _check("workspace_manifest_present", cargo_toml.is_file(), _display_path(cargo_toml), "workspace Cargo.toml is present"),
        _check("crate_manifests_present", all(row["manifest"]["present"] for row in packages), _present_count([row["manifest"] for row in packages]), "all publish crate manifests are present"),
        _check("version_alignment", all(row["version"] == version for row in packages), _version_summary(version, packages), "workspace and crate package versions match"),
        _check("publish_order_topological", _publish_order_topological(packages), _dependency_summary(packages), "each local dependency appears earlier in the publish order"),
        _check("dependency_versions_pinned", _dependency_versions_pinned(packages, version), _dependency_version_summary(packages), "all local crate dependencies pin the workspace version"),
        _check("metadata_complete", _metadata_complete(packages), _metadata_summary(packages), "description, license, repository, readme, keywords, and categories are present"),
        _check("stage_one_publishable", _stage_one_publishable(packages), _stage_one_summary(packages), "first-stage crates have no unpublished local dependencies"),
        _check("readme_install_honesty", _readme_install_honesty(readme, status, version), _readme_install_summary(readme, status, version), "README does not claim cargo install until tracerazor is live on crates.io"),
        _check("public_crates_live", all(row["registry"]["version_published"] for row in packages), _registry_summary(packages), "all six crates are published at the local version"),
        _check("cargo_install_truth", _cli_published(status, version), _cli_summary(status, version), "cargo install tracerazor is true for the local version"),
    ]
    inputs = {
        "workspace": _input_row(cargo_toml),
        "readme": _input_row(readme),
        **{row["name"]: row["manifest"] for row in packages},
    }
    card = {
        "schema_version": CRATES_CARD_SCHEMA_VERSION,
        "scope": "TraceRazor crates.io staged publication",
        "workspace_version": version,
        "crates_card_level": _crates_level(checks),
        "crates_publish_score": _crates_score(checks),
        "local_publish_plan_locked": _local_publish_plan_locked(checks),
        "public_crates_live": _check_passed(checks, "public_crates_live"),
        "cargo_install_claim_allowed": _check_passed(checks, "cargo_install_truth"),
        "packages": packages,
        "checks": checks,
        "inputs": inputs,
        "status_snapshot": status,
        "status_snapshot_sha256": hashlib.sha256(canonical_json(status).encode("utf-8")).hexdigest(),
        "publish_commands": _publish_commands(packages),
        "research_basis": [
            "Cargo packaging verifies the tarball that will be uploaded; downstream workspace crates still require upstream crates to exist in the registry.",
            "Registry publication is an external public fact, so cargo-install documentation must be gated on crates.io visibility rather than repository intent.",
            "Staged release plans need a topological dependency order, pinned local dependency versions, and a clear distinction between package-ready and published.",
            "TRICE proof cards turn release checklists into machine-verifiable evidence that can be bound by the release and integrity cards.",
        ],
        "next_actions": _next_actions(checks, packages, version),
    }
    card["crates_card_sha256"] = hashlib.sha256(canonical_json(_without_card_hash(card)).encode("utf-8")).hexdigest()
    return card


def verify_crates_card_file(path: str | Path) -> dict[str, Any]:
    """Verify a crates card self hash, bound manifests, and deterministic rebuild."""

    card_path = Path(path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if card.get("schema_version") != CRATES_CARD_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CRATES_CARD_SCHEMA_VERSION}")
    expected_hash = str(card.get("crates_card_sha256") or "")
    actual_hash = hashlib.sha256(canonical_json(_without_card_hash(card)).encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        errors.append("crates_card_sha256 mismatch")
    status = card.get("status_snapshot") if isinstance(card.get("status_snapshot"), dict) else {}
    if hashlib.sha256(canonical_json(status).encode("utf-8")).hexdigest() != card.get("status_snapshot_sha256"):
        errors.append("status_snapshot_sha256 mismatch")

    checked_inputs: list[str] = []
    resolved = _verify_inputs(card_path, card.get("inputs"), errors, checked_inputs)
    if {"workspace", "readme"}.issubset(resolved) and status:
        rebuilt = build_crates_card(
            cargo_toml_path=resolved["workspace"],
            readme_path=resolved["readme"],
            status_snapshot=status,
        )
        if canonical_json(_without_card_hash(rebuilt)) != canonical_json(_without_card_hash(card)):
            errors.append("crates card does not match deterministic rebuild from bound inputs")

    return {
        "ok": not errors,
        "schema_version": card.get("schema_version"),
        "crates_card_level": card.get("crates_card_level"),
        "crates_publish_score": card.get("crates_publish_score"),
        "local_publish_plan_locked": bool(card.get("local_publish_plan_locked")),
        "public_crates_live": bool(card.get("public_crates_live")),
        "cargo_install_claim_allowed": bool(card.get("cargo_install_claim_allowed")),
        "crates_card_sha256": expected_hash,
        "computed_crates_card_sha256": actual_hash,
        "checked_inputs": checked_inputs,
        "errors": errors,
    }


def render_crates_markdown(card: dict[str, Any]) -> str:
    lines = [
        "# TRICE Crates Publish Card",
        "",
        f"- Scope: `{card['scope']}`",
        f"- Workspace version: `{card['workspace_version']}`",
        f"- Crates level: `{card['crates_card_level']}`",
        f"- Publish score: **{card['crates_publish_score']}/100**",
        f"- Local publish plan locked: `{str(card['local_publish_plan_locked']).lower()}`",
        f"- Cargo install claim allowed: `{str(card['cargo_install_claim_allowed']).lower()}`",
        "",
        "## Checks",
        "",
        "| Check | Passed | Observed | Required |",
        "|---|---:|---|---|",
    ]
    for row in card["checks"]:
        lines.append(f"| {row['name']} | {'yes' if row['passed'] else 'no'} | {_md(row['observed'])} | {_md(row['required'])} |")
    lines.extend(["", "## Publish Order", "", "| Stage | Crate | Local dependencies | Registry | Currently publishable |", "|---:|---|---|---|---:|"])
    for row in card["packages"]:
        deps = ", ".join(row["local_dependencies"]) or "none"
        reg = row["registry"]["status"]
        lines.append(f"| {row['stage']} | `{row['name']}` | {deps} | {reg} | {'yes' if row['currently_publishable'] else 'no'} |")
    lines.extend(["", "## Commands", ""])
    for row in card["publish_commands"]:
        lines.append(f"- `{row['command']}`")
    lines.extend(["", "## Next Actions", ""])
    for item in card["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Hash", "", f"- crates card: `{card['crates_card_sha256']}`", ""])
    return "\n".join(lines)


def render_crates_tex(card: dict[str, Any]) -> str:
    rows = "\n".join(
        f"{_tex(row['name'])} & {'yes' if row['passed'] else 'no'} & {_tex(row['required'])} \\\\"
        for row in card["checks"]
    )
    order_rows = "\n".join(
        f"{row['stage']} & {_tex(row['name'])} & {_tex(', '.join(row['local_dependencies']) or 'none')} & {'yes' if row['currently_publishable'] else 'no'} \\\\"
        for row in card["packages"]
    )
    return (
        "\\section{Crates Publish Card}\n"
        f"Crates level: \\texttt{{{_tex(card['crates_card_level'])}}}; "
        f"score: {card['crates_publish_score']}/100; "
        f"cargo-install claim allowed: {'yes' if card['cargo_install_claim_allowed'] else 'no'}.\n\n"
        "\\begin{tabular}{lrl}\n"
        "Check & Passed & Required \\\\\n"
        "\\hline\n"
        f"{rows}\n"
        "\\end{tabular}\n\n"
        "\\begin{tabular}{rllr}\n"
        "Stage & Crate & Local dependencies & Publishable now \\\\\n"
        "\\hline\n"
        f"{order_rows}\n"
        "\\end{tabular}\n"
    )


def render_crates_svg(card: dict[str, Any]) -> str:
    stages = [
        ("metadata", _check_passed(card["checks"], "metadata_complete")),
        ("dag", _check_passed(card["checks"], "publish_order_topological") and _check_passed(card["checks"], "dependency_versions_pinned")),
        ("stage 1", _check_passed(card["checks"], "stage_one_publishable")),
        ("crates.io", _check_passed(card["checks"], "public_crates_live")),
        ("install", _check_passed(card["checks"], "cargo_install_truth")),
    ]
    width, height = 980, 300
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE crates publish card</text>',
        f'<text x="28" y="58" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Score {card["crates_publish_score"]}/100 | level {card["crates_card_level"]} | version {card["workspace_version"]}</text>',
    ]
    x0, y = 36, 96
    for idx, (label, passed) in enumerate(stages):
        x = x0 + idx * 184
        fill = "#047857" if passed else "#e5e7eb"
        text = "#ffffff" if passed else "#374151"
        parts.append(f'<rect x="{x}" y="{y}" width="138" height="58" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{x + 69}" y="{y + 36}" text-anchor="middle" font-family="Inter,Segoe UI,Arial" font-size="15" font-weight="700" fill="{text}">{label}</text>')
        if idx < len(stages) - 1:
            parts.append(f'<line x1="{x + 148}" y1="{y + 29}" x2="{x + 176}" y2="{y + 29}" stroke="#9ca3af" stroke-width="3"/>')
    currently = [row["name"] for row in card["packages"] if row["currently_publishable"] and not row["registry"]["version_published"]]
    parts.append(f'<text x="28" y="202" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#111827">Currently publishable: {_svg_text(", ".join(currently) or "none")}</text>')
    parts.append(f'<text x="28" y="228" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">Cargo install claim allowed: {str(card["cargo_install_claim_allowed"]).lower()} | hash {card["crates_card_sha256"][:16]}...</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_crates_outputs(card: dict[str, Any], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = out.with_suffix(".md")
    tex = out.with_suffix(".tex")
    svg = out.with_suffix(".svg")
    md.write_text(render_crates_markdown(card), encoding="utf-8")
    tex.write_text(render_crates_tex(card), encoding="utf-8")
    svg.write_text(render_crates_svg(card), encoding="utf-8")
    return {"json": str(out), "markdown": str(md), "tex": str(tex), "svg": str(svg)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a deterministic crates.io staged publish card.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=5.0)
    ap.add_argument("--format", choices=["json", "markdown", "tex"], default="json")
    args = ap.parse_args(argv)
    card = build_crates_card(offline=args.offline, timeout_s=args.timeout_s)
    outputs = write_crates_outputs(card, args.out)
    if args.format == "markdown":
        print(render_crates_markdown(card))
    elif args.format == "tex":
        print(render_crates_tex(card))
    else:
        print(json.dumps({"crates_card": card, "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if card["crates_card_level"] != "not_publishable" else 1


def verify_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a deterministic crates.io staged publish card.")
    ap.add_argument("crates_card", type=Path)
    args = ap.parse_args(argv)
    verdict = verify_crates_card_file(args.crates_card)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _package_row(name: str, manifest_path: Path, workspace_package: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    manifest = _load_toml(manifest_path) if manifest_path.is_file() else {}
    package = manifest.get("package", {}) if isinstance(manifest.get("package"), dict) else {}
    dependencies = manifest.get("dependencies", {}) if isinstance(manifest.get("dependencies"), dict) else {}
    local_deps = _local_dependencies(dependencies)
    stage = _stage_for(local_deps)
    registry = _registry_row(name, status, _workspace_value(package, workspace_package, "version"))
    metadata = {
        "description": str(package.get("description") or ""),
        "license": _workspace_value(package, workspace_package, "license"),
        "repository": _workspace_value(package, workspace_package, "repository"),
        "homepage": _workspace_value(package, workspace_package, "homepage"),
        "readme": _workspace_value(package, workspace_package, "readme"),
        "keywords": _workspace_value(package, workspace_package, "keywords"),
        "categories": _workspace_value(package, workspace_package, "categories"),
    }
    return {
        "name": name,
        "manifest": _input_row(manifest_path),
        "version": _workspace_value(package, workspace_package, "version"),
        "stage": stage,
        "metadata": metadata,
        "local_dependencies": [dep["name"] for dep in local_deps],
        "local_dependency_specs": local_deps,
        "registry": registry,
        "currently_publishable": all(_registry_row(dep["name"], status, dep["version"]).get("version_published") for dep in local_deps),
    }


def _workspace_value(package: dict[str, Any], workspace_package: dict[str, Any], key: str) -> Any:
    value = package.get(key)
    if isinstance(value, dict) and value.get("workspace") is True:
        return workspace_package.get(key)
    return value if value not in (None, "") else workspace_package.get(key)


def _local_dependencies(dependencies: dict[str, Any]) -> list[dict[str, Any]]:
    deps = []
    for dep_name, spec in sorted(dependencies.items()):
        if not isinstance(spec, dict):
            continue
        path = str(spec.get("path") or "")
        if "tracerazor-" not in path:
            continue
        deps.append({
            "name": dep_name,
            "path": path.replace("\\", "/"),
            "version": str(spec.get("version") or ""),
        })
    return deps


def _stage_for(local_deps: list[dict[str, Any]]) -> int:
    if not local_deps:
        return 1
    order = {name: idx for idx, (name, _) in enumerate(PUBLISH_ORDER)}
    max_idx = max(order.get(dep["name"], 0) for dep in local_deps)
    if max_idx <= 1:
        return 2
    if max_idx <= 3:
        return 3
    return 4


def _registry_snapshot(*, offline: bool, timeout_s: float) -> dict[str, Any]:
    rows = {
        name: _fetch_crate(name, timeout_s) if not offline else {"status": "skipped", "published": None, "version_published": None, "detail": "offline mode"}
        for name, _ in PUBLISH_ORDER
    }
    return {
        "offline": offline,
        "source": "crates.io",
        "url_template": CRATES_API,
        "crates": rows,
    }


def _fetch_crate(name: str, timeout_s: float) -> dict[str, Any]:
    url = CRATES_API.format(name=name)
    req = urllib.request.Request(url, headers={"User-Agent": "TraceRazor TRICE crates card"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"status": "missing", "published": False, "version_published": False, "latest_version": None, "detail": "crate is not published", "url": url}
        return {"status": "unknown", "published": None, "version_published": None, "detail": f"HTTP {exc.code}", "url": url}
    except Exception as exc:
        return {"status": "unknown", "published": None, "version_published": None, "detail": str(exc), "url": url}
    crate = data.get("crate") if isinstance(data.get("crate"), dict) else {}
    newest = str(crate.get("newest_version") or "")
    versions = data.get("versions") if isinstance(data.get("versions"), list) else []
    return {
        "status": "published" if newest else "missing",
        "published": bool(newest),
        "version_published": None,
        "latest_version": newest or None,
        "versions_seen": [str(row.get("num") or "") for row in versions[:25] if isinstance(row, dict)],
        "detail": f"newest={newest or 'unknown'}",
        "url": url,
    }


def _registry_row(name: str, status: dict[str, Any], version: str) -> dict[str, Any]:
    row = {}
    crates = status.get("crates") if isinstance(status.get("crates"), dict) else {}
    if isinstance(crates.get(name), dict):
        row = dict(crates[name])
    versions = set(row.get("versions_seen") or [])
    latest = row.get("latest_version")
    version_published = bool(version and (version in versions or latest == version))
    if row.get("published") is False:
        version_published = False
    if row.get("published") is None:
        version_published = False
    row["target_version"] = version
    row["version_published"] = version_published
    row.setdefault("status", "unknown")
    row.setdefault("published", bool(row.get("latest_version")))
    return row


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def _present_count(rows: list[dict[str, Any]]) -> str:
    return f"{sum(1 for row in rows if row.get('present'))}/{len(rows)}"


def _version_summary(version: str, packages: list[dict[str, Any]]) -> dict[str, Any]:
    return {"workspace": version, "packages": {row["name"]: row["version"] for row in packages}}


def _dependency_summary(packages: list[dict[str, Any]]) -> dict[str, Any]:
    return {row["name"]: row["local_dependencies"] for row in packages}


def _dependency_version_summary(packages: list[dict[str, Any]]) -> dict[str, Any]:
    return {row["name"]: {dep["name"]: dep["version"] for dep in row["local_dependency_specs"]} for row in packages}


def _metadata_summary(packages: list[dict[str, Any]]) -> dict[str, Any]:
    missing = {}
    for row in packages:
        missing[row["name"]] = [key for key, value in row["metadata"].items() if not value]
    return missing


def _stage_one_summary(packages: list[dict[str, Any]]) -> dict[str, Any]:
    return {row["name"]: row["currently_publishable"] for row in packages if row["stage"] == 1}


def _registry_summary(packages: list[dict[str, Any]]) -> dict[str, Any]:
    return {row["name"]: row["registry"].get("status") for row in packages}


def _cli_summary(status: dict[str, Any], version: str) -> str:
    row = _registry_row("tracerazor", status, version)
    return f"{row.get('status')}: target={version} latest={row.get('latest_version') or 'none'}"


def _readme_install_summary(readme: Path, status: dict[str, Any], version: str) -> str:
    contains = "cargo install tracerazor" in (readme.read_text(encoding="utf-8").lower() if readme.is_file() else "")
    return f"contains_cargo_install={contains}; {_cli_summary(status, version)}"


def _publish_order_topological(packages: list[dict[str, Any]]) -> bool:
    order = {row["name"]: idx for idx, row in enumerate(packages)}
    return all(order.get(dep, 999) < idx for idx, row in enumerate(packages) for dep in row["local_dependencies"])


def _dependency_versions_pinned(packages: list[dict[str, Any]], version: str) -> bool:
    return all(dep["version"] == version for row in packages for dep in row["local_dependency_specs"])


def _metadata_complete(packages: list[dict[str, Any]]) -> bool:
    return all(all(bool(value) for value in row["metadata"].values()) for row in packages)


def _stage_one_publishable(packages: list[dict[str, Any]]) -> bool:
    roots = [row for row in packages if row["stage"] == 1]
    return bool(roots) and all(row["currently_publishable"] for row in roots)


def _readme_install_honesty(readme: Path, status: dict[str, Any], version: str) -> bool:
    text = readme.read_text(encoding="utf-8").lower() if readme.is_file() else ""
    return "cargo install tracerazor" not in text or _cli_published(status, version)


def _cli_published(status: dict[str, Any], version: str) -> bool:
    return bool(_registry_row("tracerazor", status, version).get("version_published"))


def _check_passed(checks: list[dict[str, Any]], name: str) -> bool:
    return any(row["name"] == name and row["passed"] for row in checks)


def _local_publish_plan_locked(checks: list[dict[str, Any]]) -> bool:
    required = {
        "workspace_manifest_present",
        "crate_manifests_present",
        "version_alignment",
        "publish_order_topological",
        "dependency_versions_pinned",
        "metadata_complete",
        "stage_one_publishable",
        "readme_install_honesty",
    }
    return required.issubset({row["name"] for row in checks if row["passed"]})


def _crates_level(checks: list[dict[str, Any]]) -> str:
    if _check_passed(checks, "public_crates_live") and _check_passed(checks, "cargo_install_truth"):
        return "crates_live"
    if _local_publish_plan_locked(checks):
        return "publish_plan_locked"
    return "not_publishable"


def _crates_score(checks: list[dict[str, Any]]) -> int:
    weights = {
        "workspace_manifest_present": 8,
        "crate_manifests_present": 8,
        "version_alignment": 10,
        "publish_order_topological": 12,
        "dependency_versions_pinned": 10,
        "metadata_complete": 12,
        "stage_one_publishable": 10,
        "readme_install_honesty": 10,
        "public_crates_live": 12,
        "cargo_install_truth": 8,
    }
    return min(100, sum(weights.get(row["name"], 0) for row in checks if row["passed"]))


def _publish_commands(packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    commands = []
    for row in packages:
        package = row["name"]
        commands.append({
            "name": f"package-{package}",
            "command": f"cargo package -p {package} --allow-dirty",
            "stage": row["stage"],
            "sha256": hashlib.sha256(f"cargo package -p {package} --allow-dirty".encode("utf-8")).hexdigest(),
        })
        commands.append({
            "name": f"publish-{package}",
            "command": f"cargo publish -p {package}",
            "stage": row["stage"],
            "sha256": hashlib.sha256(f"cargo publish -p {package}".encode("utf-8")).hexdigest(),
        })
    commands.append({
        "name": "install-verify",
        "command": "cargo install tracerazor --locked",
        "stage": 5,
        "sha256": hashlib.sha256(b"cargo install tracerazor --locked").hexdigest(),
    })
    return commands


def _next_actions(checks: list[dict[str, Any]], packages: list[dict[str, Any]], version: str) -> list[str]:
    if _check_passed(checks, "public_crates_live") and _check_passed(checks, "cargo_install_truth"):
        return ["Verify `cargo install tracerazor --locked` from a clean Cargo home and then allow the README cargo-install claim."]
    if not _local_publish_plan_locked(checks):
        return ["Fix local Cargo metadata, version pins, publish order, or README install claims before publishing."]
    publishable = [row["name"] for row in packages if row["currently_publishable"] and not row["registry"]["version_published"]]
    if publishable:
        return [
            f"Publish stage-ready crates for {version}: {', '.join(publishable)}.",
            "Regenerate this crates card after crates.io indexes each published crate.",
            "Continue stage by stage until the final `tracerazor` crate is live, then verify `cargo install tracerazor --locked`.",
        ]
    return [
        "Wait for upstream crates.io index propagation, regenerate the card, and publish the next dependent stage.",
        "Keep README cargo-install claims disabled until the final CLI crate is visible.",
    ]


def _input_row(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "path": _display_path(path),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _verify_inputs(card_path: Path, raw_inputs: Any, errors: list[str], checked_inputs: list[str]) -> dict[str, Path]:
    if not isinstance(raw_inputs, dict):
        errors.append("inputs must be an object")
        return {}
    resolved: dict[str, Path] = {}
    for name, row in sorted(raw_inputs.items()):
        if not isinstance(row, dict):
            errors.append(f"input {name} must be an object")
            continue
        raw_path = str(row.get("path") or "")
        if not raw_path:
            errors.append(f"input {name} path missing")
            continue
        path = _resolve_bound_path(card_path, raw_path)
        resolved[name] = path
        if not path.is_file():
            errors.append(f"input {name} missing: {raw_path}")
            continue
        if row.get("bytes") != path.stat().st_size:
            errors.append(f"input {name} byte count mismatch")
        if row.get("sha256") != sha256_file(path):
            errors.append(f"input {name} sha256 mismatch")
        else:
            checked_inputs.append(name)
    return resolved


def _resolve_bound_path(card_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    repo_candidate = REPO / candidate
    if repo_candidate.exists():
        return repo_candidate
    return card_path.parent / candidate


def _without_card_hash(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out.pop("crates_card_sha256", None)
    return out


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _md(value: Any) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return text.replace("|", "\\|")


def _tex(value: Any) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _svg_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    raise SystemExit(main())
