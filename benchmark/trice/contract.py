"""Deterministic public contract cards for the TRICE library surface."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any

from .evidence import canonical_json, sha256_file

CONTRACT_CARD_SCHEMA_VERSION = "trice-contract-card/v1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "docs" / "trice_contract_card.json"

REQUIRED_TRICE_COMMANDS = [
    "run",
    "suite",
    "verify",
    "verify-suite",
    "bundle",
    "verify-bundle",
    "doctor",
    "claim",
    "verify-claim",
    "artifact",
    "verify-artifact",
    "protocol",
    "verify-protocol",
    "design",
    "verify-design",
    "reproduction",
    "verify-reproduction",
    "release",
    "verify-release",
    "release-evidence",
    "verify-release-evidence",
    "integrity",
    "verify-integrity",
    "research",
    "verify-research",
    "crates",
    "verify-crates",
    "install",
    "verify-install",
    "contract",
    "verify-contract",
    "schema",
    "validate-patch",
    "validate-adapter",
    "validate-receipt",
    "validate-suite",
]


def build_contract_card(*, scope: str = "TraceRazor/TRICE public library contract") -> dict[str, Any]:
    """Build a machine-verifiable contract for public library surfaces."""

    import tracerazor

    version = str(getattr(tracerazor, "__version__", ""))
    top_exports = _module_exports("tracerazor")
    trice_exports = _module_exports("tracerazor.trice")
    commands = _cli_commands()
    schema_rows = _schema_rows()
    example_rows = _example_rows()
    doc_paths = {
        "readme": REPO / "README.md",
        "library_doc": REPO / "docs" / "trice_library.md",
        "research_ledger": REPO / "docs" / "trice_research_ledger.md",
        "public_trust_matrix": REPO / "docs" / "public_trust_matrix.md",
        "pyproject": REPO / "pyproject.toml",
    }
    checks = [
        _check("semver_version", _is_semver(version), version, "MAJOR.MINOR.PATCH"),
        _check("top_level_api", not top_exports["missing"] and top_exports["count"] >= 10, _export_summary(top_exports), "tracerazor.__all__ resolves"),
        _check("trice_api", not trice_exports["missing"] and trice_exports["count"] >= 50, _export_summary(trice_exports), "tracerazor.trice.__all__ resolves"),
        _check("cli_contract", all(cmd in commands["commands"] for cmd in REQUIRED_TRICE_COMMANDS), commands["commands"], "all documented tracerazor-trice commands exist"),
        _check("schemas_shipped", all(row["present"] for row in schema_rows), _present_count(schema_rows), "all TRICE JSON Schemas are present"),
        _check("contract_schema_shipped", any(row["name"] == "trice_contract_card.schema.json" and row["present"] for row in schema_rows), "trice_contract_card.schema.json", "contract-card schema ships"),
        _check("examples_shipped", all(row["present"] for row in example_rows) and len(example_rows) >= 8, _present_count(example_rows), "public examples are present"),
        _check("docs_shipped", all(path.is_file() for path in doc_paths.values()), _present_count([_availability_row(name, path) for name, path in doc_paths.items()]), "README/library/research/trust docs are present"),
    ]
    card = {
        "schema_version": CONTRACT_CARD_SCHEMA_VERSION,
        "scope": scope,
        "package": "tracerazor",
        "version": version,
        "semver": _semver_parts(version),
        "contract_level": _contract_level(checks),
        "contract_score": _contract_score(checks),
        "checks": checks,
        "public_api": {
            "tracerazor": top_exports,
            "tracerazor.trice": trice_exports,
        },
        "cli": commands,
        "schemas": schema_rows,
        "examples": example_rows,
        "inputs": {
            name: _input_row(path)
            for name, path in doc_paths.items()
        },
        "research_basis": [
            "Semantic Versioning requires the public API to be declared before compatibility claims are meaningful.",
            "Python packaging makes version identifiers public registry facts, so library contracts must bind version and import surface together.",
            "JSON Schema gives users and downstream agents a machine-checkable boundary for TRICE receipts, suites, cards, and bundles.",
            "SLSA, in-toto, and CycloneDX motivate release evidence that binds checksums, SBOMs, provenance statements, and public artifacts.",
            "Reproducible-build practice motivates checking that public examples, schemas, CLI commands, release evidence, and the integrity proof graph match the source being packaged.",
            "Cargo publication trust requires staged registry facts because downstream crates cannot honestly claim cargo-install readiness before upstream workspace crates are indexed.",
            "Clean-wheel installability must be verified after build because packaged data, console scripts, and bundled binaries can diverge from checkout behavior.",
            "Research-ledger integrity must be machine-checkable because product claims and papers can drift from the sources that supposedly justify them.",
        ],
        "next_actions": _next_actions(checks),
    }
    card["contract_card_sha256"] = hashlib.sha256(canonical_json(_without_contract_hash(card)).encode("utf-8")).hexdigest()
    return card


def verify_contract_card_file(path: str | Path) -> dict[str, Any]:
    """Verify a contract card self hash, bound inputs, and deterministic rebuild."""

    card_path = Path(path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if card.get("schema_version") != CONTRACT_CARD_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CONTRACT_CARD_SCHEMA_VERSION}")
    expected_hash = str(card.get("contract_card_sha256") or "")
    actual_hash = hashlib.sha256(canonical_json(_without_contract_hash(card)).encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        errors.append("contract_card_sha256 mismatch")

    checked_inputs: list[str] = []
    for group_name in ("schemas", "examples"):
        rows = card.get(group_name) if isinstance(card.get(group_name), list) else []
        for row in rows:
            if not isinstance(row, dict):
                errors.append(f"{group_name} row must be an object")
                continue
            _check_bound_file(row, errors, checked_inputs)

    inputs = card.get("inputs") if isinstance(card.get("inputs"), dict) else {}
    for name, row in sorted(inputs.items()):
        if not isinstance(row, dict):
            errors.append(f"input {name} must be an object")
            continue
        _check_bound_file(row, errors, checked_inputs, fallback_name=name)

    try:
        rebuilt = build_contract_card(scope=str(card.get("scope") or "TraceRazor/TRICE public library contract"))
        if canonical_json(_without_contract_hash(rebuilt)) != canonical_json(_without_contract_hash(card)):
            errors.append("contract card does not match deterministic rebuild from current public surface")
    except Exception as exc:
        errors.append(f"contract card rebuild failed: {exc}")

    return {
        "schema_version": CONTRACT_CARD_SCHEMA_VERSION,
        "ok": not errors,
        "errors": errors,
        "contract_level": card.get("contract_level"),
        "contract_score": card.get("contract_score"),
        "contract_card_sha256": expected_hash,
        "computed_contract_card_sha256": actual_hash,
        "checked_inputs": sorted(set(checked_inputs)),
    }


def render_contract_markdown(card: dict[str, Any]) -> str:
    lines = [
        "# TRICE Contract Card",
        "",
        f"- Package: `{card['package']}`",
        f"- Version: `{card['version']}`",
        f"- Contract level: `{card['contract_level']}`",
        f"- Contract score: **{card['contract_score']}/100**",
        "",
        "## Checks",
        "",
        "| Check | Pass | Observed | Required |",
        "|---|---:|---|---|",
    ]
    for row in card["checks"]:
        observed = json.dumps(row["observed"], sort_keys=True) if isinstance(row["observed"], (dict, list)) else str(row["observed"])
        lines.append(f"| {row['name']} | {'yes' if row['passed'] else 'no'} | {observed} | {row['required']} |")
    lines.extend(
        [
            "",
            "## Public API",
            "",
            f"- `tracerazor`: {card['public_api']['tracerazor']['count']} exported names",
            f"- `tracerazor.trice`: {card['public_api']['tracerazor.trice']['count']} exported names",
            f"- `tracerazor-trice`: {len(card['cli']['commands'])} subcommands",
            f"- Schemas: {sum(1 for row in card['schemas'] if row['present'])}/{len(card['schemas'])}",
            f"- Examples: {sum(1 for row in card['examples'] if row['present'])}/{len(card['examples'])}",
            "",
            "## Research Basis",
            "",
        ]
    )
    for item in card["research_basis"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Actions", ""])
    for item in card["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Hash", "", f"- contract card: `{card['contract_card_sha256']}`", ""])
    return "\n".join(lines)


def render_contract_tex(card: dict[str, Any]) -> str:
    rows = "\n".join(
        f"{_tex(row['name'])} & {'yes' if row['passed'] else 'no'} & {_tex(str(row['required']))} \\\\"
        for row in card["checks"]
    )
    basis = "\n".join(f"\\item {_tex(item)}" for item in card["research_basis"])
    actions = "\n".join(f"\\item {_tex(item)}" for item in card["next_actions"])
    return (
        "\\section{Contract Card}\n"
        f"Package \\texttt{{{_tex(card['package'])}}} version \\texttt{{{_tex(card['version'])}}} "
        f"has contract level \\texttt{{{_tex(card['contract_level'])}}} and score {card['contract_score']}/100.\n\n"
        "\\begin{tabular}{lll}\n"
        "Check & Pass & Required \\\\\n"
        "\\hline\n"
        f"{rows}\n"
        "\\end{tabular}\n\n"
        "\\noindent Research basis:\n\\begin{itemize}\n"
        f"{basis}\n"
        "\\end{itemize}\n\n"
        "\\noindent Next actions:\n\\begin{itemize}\n"
        f"{actions}\n"
        "\\end{itemize}\n"
    )


def render_contract_svg(card: dict[str, Any]) -> str:
    checks = [
        ("semver", _check_passed(card, "semver_version")),
        ("api", _check_passed(card, "top_level_api") and _check_passed(card, "trice_api")),
        ("cli", _check_passed(card, "cli_contract")),
        ("schemas", _check_passed(card, "schemas_shipped") and _check_passed(card, "contract_schema_shipped")),
        ("examples", _check_passed(card, "examples_shipped")),
        ("docs", _check_passed(card, "docs_shipped")),
    ]
    width = 900
    height = 260
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="900" height="260" fill="#f8fafc"/>',
        '<text x="28" y="34" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE public contract card</text>',
        f'<text x="28" y="58" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Score {card["contract_score"]}/100 | level {card["contract_level"]} | version {card["version"]}</text>',
    ]
    x = 28
    y = 92
    for label, ok in checks:
        color = "#16a34a" if ok else "#dc2626"
        parts.append(f'<rect x="{x}" y="{y}" width="120" height="72" rx="8" fill="#ffffff" stroke="#d1d5db"/>')
        parts.append(f'<circle cx="{x + 24}" cy="{y + 28}" r="9" fill="{color}"/>')
        parts.append(f'<text x="{x + 42}" y="{y + 32}" font-family="Inter,Segoe UI,Arial" font-size="14" fill="#111827">{label}</text>')
        parts.append(f'<text x="{x + 18}" y="{y + 56}" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">{"locked" if ok else "check"}</text>')
        x += 140
    parts.append(f'<text x="28" y="202" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#111827">API exports {card["public_api"]["tracerazor"]["count"]}+{card["public_api"]["tracerazor.trice"]["count"]} | CLI commands {len(card["cli"]["commands"])} | schemas {len(card["schemas"])}</text>')
    parts.append('<text x="28" y="228" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">Contract card only: SemVer promises are credible only for this declared public surface.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_contract_outputs(card: dict[str, Any], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = out.with_suffix(".md")
    tex = out.with_suffix(".tex")
    svg = out.with_suffix(".svg")
    md.write_text(render_contract_markdown(card), encoding="utf-8")
    tex.write_text(render_contract_tex(card), encoding="utf-8")
    svg.write_text(render_contract_svg(card), encoding="utf-8")
    return {"json": str(out), "markdown": str(md), "tex": str(tex), "svg": str(svg)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a deterministic TRICE public contract card.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--format", choices=["json", "markdown", "tex"], default="json")
    ap.add_argument("--scope", default="TraceRazor/TRICE public library contract")
    args = ap.parse_args(argv)
    card = build_contract_card(scope=args.scope)
    outputs = write_contract_outputs(card, args.out)
    if args.format == "markdown":
        print(render_contract_markdown(card))
    elif args.format == "tex":
        print(render_contract_tex(card))
    else:
        print(json.dumps({"contract_card": card, "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if card["contract_level"] != "contract_unusable" else 1


def verify_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a deterministic TRICE public contract card.")
    ap.add_argument("contract_card", type=Path)
    args = ap.parse_args(argv)
    verdict = verify_contract_card_file(args.contract_card)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def _module_exports(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    exported = list(getattr(module, "__all__", []))
    missing = [name for name in exported if not hasattr(module, name)]
    return {
        "module": module_name,
        "count": len(exported),
        "exports": exported,
        "missing": missing,
    }


def _cli_commands() -> dict[str, Any]:
    cli_path = REPO / "tracerazor" / "trice" / "cli.py"
    text = cli_path.read_text(encoding="utf-8") if cli_path.is_file() else ""
    commands = sorted(set(re.findall(r'sub\.add_parser\("([^"]+)"', text)))
    missing = [cmd for cmd in REQUIRED_TRICE_COMMANDS if cmd not in commands]
    return {
        "entrypoint": "tracerazor-trice",
        "source": _display_path(cli_path),
        "commands": commands,
        "required": REQUIRED_TRICE_COMMANDS,
        "missing": missing,
        "sha256": sha256_file(cli_path) if cli_path.is_file() else None,
    }


def _schema_rows() -> list[dict[str, Any]]:
    schema_dir = REPO / "schemas"
    names = sorted(path.name for path in schema_dir.glob("trice_*.schema.json"))
    return [_availability_row(name, schema_dir / name) for name in names]


def _example_rows() -> list[dict[str, Any]]:
    examples_dir = REPO / "examples"
    names = sorted(path.name for path in examples_dir.glob("trice_*") if path.is_file())
    return [_availability_row(name, examples_dir / name) for name in names]


def _availability_row(name: str, path: Path) -> dict[str, Any]:
    return {
        "name": name,
        "path": _display_path(path),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _input_row(path: Path) -> dict[str, Any]:
    row = _availability_row(path.name, path)
    row["path"] = _display_path(path)
    return row


def _check_bound_file(row: dict[str, Any], errors: list[str], checked_inputs: list[str], *, fallback_name: str | None = None) -> None:
    raw_path = row.get("path")
    name = str(row.get("name") or fallback_name or raw_path or "unknown")
    if not raw_path:
        errors.append(f"{name} has no path")
        return
    path = _resolve_path(str(raw_path))
    if not path.is_file():
        errors.append(f"{name} missing: {raw_path}")
        return
    checked_inputs.append(name)
    expected = row.get("sha256")
    actual = sha256_file(path)
    if expected and actual != expected:
        errors.append(f"{name} sha256 mismatch")
    expected_bytes = row.get("bytes")
    if isinstance(expected_bytes, int) and path.stat().st_size != expected_bytes:
        errors.append(f"{name} byte count mismatch")


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _is_semver(version: str) -> bool:
    return bool(re.fullmatch(r"\d+\.\d+\.\d+", version))


def _semver_parts(version: str) -> dict[str, int | None]:
    if not _is_semver(version):
        return {"major": None, "minor": None, "patch": None}
    major, minor, patch = version.split(".")
    return {"major": int(major), "minor": int(minor), "patch": int(patch)}


def _export_summary(exports: dict[str, Any]) -> dict[str, Any]:
    return {"count": exports["count"], "missing": exports["missing"]}


def _contract_level(checks: list[dict[str, Any]]) -> str:
    passed = {row["name"] for row in checks if row["passed"]}
    required = {"semver_version", "top_level_api", "trice_api", "cli_contract", "schemas_shipped", "contract_schema_shipped", "examples_shipped"}
    if required.issubset(passed):
        return "library_contract_locked"
    if {"top_level_api", "trice_api", "schemas_shipped"}.issubset(passed):
        return "partial_contract"
    return "contract_unusable"


def _contract_score(checks: list[dict[str, Any]]) -> int:
    weights = {
        "semver_version": 10,
        "top_level_api": 15,
        "trice_api": 20,
        "cli_contract": 15,
        "schemas_shipped": 15,
        "contract_schema_shipped": 10,
        "examples_shipped": 10,
        "docs_shipped": 5,
    }
    return min(100, sum(weights.get(row["name"], 0) for row in checks if row["passed"]))


def _next_actions(checks: list[dict[str, Any]]) -> list[str]:
    missing = [row["name"] for row in checks if not row["passed"]]
    if not missing:
        return [
            "Treat this card as the public API boundary for SemVer compatibility.",
            "Regenerate the contract card before every release and after any public API, CLI, schema, or example change.",
            "Promote only documented imports and schemas into long-term compatibility guarantees.",
        ]
    actions = []
    if "semver_version" in missing:
        actions.append("Normalize package version to MAJOR.MINOR.PATCH before publishing.")
    if "top_level_api" in missing or "trice_api" in missing:
        actions.append("Fix missing public __all__ exports before shipping the library.")
    if "cli_contract" in missing:
        actions.append("Wire missing tracerazor-trice commands or remove them from the public contract.")
    if "schemas_shipped" in missing or "contract_schema_shipped" in missing:
        actions.append("Ship every public JSON Schema in the wheel.")
    if "examples_shipped" in missing:
        actions.append("Ship deterministic examples that exercise the public contracts.")
    return actions or ["Fix public contract drift before release."]


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def _check_passed(card: dict[str, Any], name: str) -> bool:
    return any(row["name"] == name and row["passed"] for row in card.get("checks", []))


def _present_count(rows: list[dict[str, Any]]) -> str:
    return f"{sum(1 for row in rows if row.get('present'))}/{len(rows)}"


def _without_contract_hash(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out.pop("contract_card_sha256", None)
    return out


def _tex(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )
