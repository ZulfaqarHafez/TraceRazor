"""Release trust cards for TRICE/TraceRazor public distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .artifact import verify_artifact_card_file
from .contract import verify_contract_card_file
from .doctor import doctor_report
from .evidence import canonical_json, sha256_file, write_text_lf
from .install import verify_install_card_file
from .reproduction import verify_reproduction_card_file

RELEASE_CARD_SCHEMA_VERSION = "trice-release-card/v1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = REPO / "docs" / "trice_artifact_card.json"
DEFAULT_REPRODUCTION = REPO / "docs" / "trice_reproduction_card.json"
DEFAULT_CONTRACT = REPO / "docs" / "trice_contract_card.json"
DEFAULT_INSTALL = REPO / "docs" / "trice_install_card.json"
DEFAULT_README = REPO / "README.md"
DEFAULT_PUBLIC_TRUST = REPO / "docs" / "public_trust_matrix.md"
DEFAULT_RELEASE_CHECKLIST = REPO / "docs" / "release_checklist.md"
DEFAULT_PYPROJECT = REPO / "pyproject.toml"


def build_release_card(
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    reproduction_path: str | Path = DEFAULT_REPRODUCTION,
    contract_path: str | Path = DEFAULT_CONTRACT,
    install_path: str | Path = DEFAULT_INSTALL,
    readme_path: str | Path = DEFAULT_README,
    public_trust_path: str | Path = DEFAULT_PUBLIC_TRUST,
    release_checklist_path: str | Path = DEFAULT_RELEASE_CHECKLIST,
    pyproject_path: str | Path = DEFAULT_PYPROJECT,
    offline: bool = False,
    timeout_s: float = 10.0,
    doctor_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic release-readiness card from a doctor snapshot."""

    paths = {
        "artifact_card": Path(artifact_path),
        "reproduction_card": Path(reproduction_path),
        "contract_card": Path(contract_path),
        "install_card": Path(install_path),
        "readme": Path(readme_path),
        "public_trust_matrix": Path(public_trust_path),
        "release_checklist": Path(release_checklist_path),
        "pyproject": Path(pyproject_path),
    }
    doctor = doctor_snapshot or doctor_report(offline=offline, timeout_s=timeout_s)
    artifact_verdict = _safe_verify(lambda: verify_artifact_card_file(paths["artifact_card"]))
    reproduction_verdict = _safe_verify(lambda: verify_reproduction_card_file(paths["reproduction_card"]))
    contract_verdict = _safe_verify(lambda: verify_contract_card_file(paths["contract_card"]))
    install_verdict = _safe_verify(lambda: verify_install_card_file(paths["install_card"]))
    checklist_text = paths["release_checklist"].read_text(encoding="utf-8") if paths["release_checklist"].is_file() else ""
    checks = [
        _doctor_check("local_package", doctor, "local package imports with version"),
        _doctor_check("bundled_cli", doctor, "CLI binary is bundled or source-build reachable"),
        _doctor_check("schemas", doctor, "all public contract schemas are shipped"),
        _check("artifact_card_verifies", bool(artifact_verdict.get("ok")), artifact_verdict.get("artifact_level"), "artifact card verifies"),
        _check("reproduction_card_verifies", bool(reproduction_verdict.get("ok")), reproduction_verdict.get("reproduction_level"), "reproduction card verifies"),
        _check("contract_card_verifies", bool(contract_verdict.get("ok")), contract_verdict.get("contract_level"), "public API/CLI/schema contract card verifies"),
        _check("install_card_verifies", bool(install_verdict.get("ok")), install_verdict.get("install_level"), "clean-wheel installability card verifies"),
        _check("release_docs_present", _docs_present(paths), _present_count(paths), "README, trust matrix, release checklist, pyproject, contract card, and install card present"),
        _doctor_check("pypi", doctor, "PyPI latest version matches local version"),
        _doctor_check("piwheels", doctor, "piwheels exposes the local version file"),
        _doctor_check("crates_io", doctor, "crates.io package is published"),
        _doctor_check("github_tag", doctor, "local version tag points at current commit locally and remotely"),
        _doctor_check("github_actions", doctor, "required public workflows are green"),
        _doctor_check("openssf_scorecard", doctor, "OpenSSF Scorecard is published with score >= 7.0"),
        _check("provenance_plan_documented", "trusted publishing" in checklist_text.lower() and "oidc" in checklist_text.lower(), "trusted publishing/OIDC" if checklist_text else None, "trusted publishing and OIDC documented"),
        _check("attestation_plan_documented", "artifact attestation" in checklist_text.lower() and "github release" in checklist_text.lower(), "GitHub artifact attestations" if checklist_text else None, "GitHub release artifact attestation documented"),
        _check("sbom_plan_documented", "cyclonedx" in checklist_text.lower() and "sha-256" in checklist_text.lower(), "CycloneDX/SHA-256" if checklist_text else None, "SBOM and checksum release assets documented"),
    ]
    card = {
        "schema_version": RELEASE_CARD_SCHEMA_VERSION,
        "release_level": _release_level(checks),
        "release_score": _release_score(checks),
        "public_release_ready": _public_release_ready(checks),
        "s_tier_claim_allowed": False,
        "package": {
            "name": str(doctor.get("package") or "tracerazor"),
            "local_version": str(doctor.get("local_version") or ""),
            "python": doctor.get("python"),
            "platform": doctor.get("platform"),
            "offline": bool(doctor.get("offline")),
        },
        "doctor_snapshot": doctor,
        "doctor_snapshot_sha256": hashlib.sha256(canonical_json(doctor).encode("utf-8")).hexdigest(),
        "inputs": {
            name: _input_row(path)
            for name, path in paths.items()
        },
        "checks": checks,
        "verdicts": {
            "artifact_card": _compact_verdict(artifact_verdict),
            "reproduction_card": _compact_verdict(reproduction_verdict),
            "contract_card": _compact_verdict(contract_verdict),
            "install_card": _compact_verdict(install_verdict),
        },
        "research_basis": [
            "SLSA frames provenance as a release property: consumers need to know what built an artifact, how, and from which inputs.",
            "OpenSSF Scorecard treats public project health as a set of automated checks rather than maintainer promises.",
            "PyPI trusted publishing reduces long-lived credential exposure and enables index-hosted attestations.",
            "GitHub artifact attestations provide hosted provenance records for release assets generated in Actions.",
            "CycloneDX-style SBOMs and SHA-256 checksums make release assets inspectable by downstream users.",
            "SemVer trust depends on a declared public API; the TRICE contract card binds imports, CLI commands, schemas, examples, and docs before publication.",
            "Clean-wheel installability verifies packaged data and console scripts after build, separating Python/TRICE readiness from platform-bundled Rust CLI readiness.",
            "TRICE release cards separate local proof readiness from public distribution readiness and refuse S-tier wording while public gates are red.",
        ],
        "next_actions": _next_actions(checks, str(doctor.get("local_version") or "unknown")),
        "commands": _commands(),
    }
    card["release_card_sha256"] = hashlib.sha256(canonical_json(_without_release_hash(card)).encode("utf-8")).hexdigest()
    return card


def verify_release_card_file(path: str | Path) -> dict[str, Any]:
    """Verify a release card self hash, bound input hashes, and rebuild."""

    card_path = Path(path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if card.get("schema_version") != RELEASE_CARD_SCHEMA_VERSION:
        errors.append(f"schema_version must be {RELEASE_CARD_SCHEMA_VERSION}")
    expected_hash = str(card.get("release_card_sha256") or "")
    actual_hash = hashlib.sha256(canonical_json(_without_release_hash(card)).encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        errors.append("release_card_sha256 mismatch")
    doctor = card.get("doctor_snapshot") if isinstance(card.get("doctor_snapshot"), dict) else {}
    doctor_hash = hashlib.sha256(canonical_json(doctor).encode("utf-8")).hexdigest()
    if doctor_hash != card.get("doctor_snapshot_sha256"):
        errors.append("doctor_snapshot_sha256 mismatch")

    inputs = card.get("inputs") if isinstance(card.get("inputs"), dict) else {}
    resolved: dict[str, Path] = {}
    checked_inputs: list[str] = []
    for name, row in sorted(inputs.items()):
        if not isinstance(row, dict):
            errors.append(f"input {name} must be an object")
            continue
        raw_path = row.get("path")
        if not raw_path:
            errors.append(f"input {name} path is missing")
            continue
        p = _resolve_bound_path(card_path, str(raw_path))
        resolved[name] = p
        if not p.is_file():
            errors.append(f"input {name} file not found: {_display_path(p)}")
            continue
        if p.stat().st_size != int(row.get("bytes") or 0):
            errors.append(f"input {name} byte count mismatch")
        if sha256_file(p) != row.get("sha256"):
            errors.append(f"input {name} sha256 mismatch")
        else:
            checked_inputs.append(name)

    if _required_inputs_present(resolved) and doctor:
        rebuilt = build_release_card(
            artifact_path=resolved["artifact_card"],
            reproduction_path=resolved["reproduction_card"],
            contract_path=resolved["contract_card"],
            install_path=resolved["install_card"],
            readme_path=resolved["readme"],
            public_trust_path=resolved["public_trust_matrix"],
            release_checklist_path=resolved["release_checklist"],
            pyproject_path=resolved["pyproject"],
            doctor_snapshot=doctor,
        )
        if canonical_json(_without_release_hash(rebuilt)) != canonical_json(_without_release_hash(card)):
            errors.append("release card does not match deterministic rebuild from bound inputs")

    return {
        "ok": not errors,
        "schema_version": card.get("schema_version"),
        "release_level": card.get("release_level"),
        "release_score": card.get("release_score"),
        "public_release_ready": bool(card.get("public_release_ready")),
        "release_card_sha256": expected_hash,
        "computed_release_card_sha256": actual_hash,
        "checked_inputs": checked_inputs,
        "errors": errors,
    }


def render_release_markdown(card: dict[str, Any]) -> str:
    lines = [
        "# TRICE Release Card",
        "",
        f"- Release level: `{card['release_level']}`",
        f"- Release score: **{card['release_score']}/100**",
        f"- Public release ready: `{str(card['public_release_ready']).lower()}`",
        f"- Local version: `{card['package']['local_version']}`",
        "",
        "## Checks",
        "",
        "| Check | Passed | Observed | Required |",
        "|---|---:|---|---|",
    ]
    for row in card["checks"]:
        lines.append(f"| {row['name']} | {'yes' if row['passed'] else 'no'} | {_md(row['observed'])} | {_md(row['required'])} |")
    lines.extend(["", "## Next Actions", ""])
    for item in card["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Hash", "", f"- release card: `{card['release_card_sha256']}`", ""])
    return "\n".join(lines)


def render_release_tex(card: dict[str, Any]) -> str:
    rows = "\n".join(
        f"{_tex(row['name'])} & {'yes' if row['passed'] else 'no'} & {_tex(row['required'])} \\\\"
        for row in card["checks"]
    )
    next_actions = "\n".join(f"\\item {_tex(item)}" for item in card["next_actions"])
    return (
        "\\section{Release Card}\n"
        f"Release level: \\texttt{{{_tex(card['release_level'])}}}; "
        f"score: {card['release_score']}/100; "
        f"public release ready: {'yes' if card['public_release_ready'] else 'no'}.\n\n"
        "\\begin{table}[h]\n\\centering\n"
        "\\begin{tabular}{lrl}\n\\toprule\nCheck & Passed & Required \\\\\n\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{Distribution trust checks for TraceRazor/TRICE release readiness.}\n"
        "\\end{table}\n\n"
        "\\noindent Next actions:\n\\begin{itemize}\n"
        f"{next_actions}\n"
        "\\end{itemize}\n"
    )


def render_release_svg(card: dict[str, Any]) -> str:
    stages = [
        ("local", _check_passed(card, "local_package") and _check_passed(card, "bundled_cli") and _check_passed(card, "schemas")),
        ("proof", _check_passed(card, "artifact_card_verifies") and _check_passed(card, "reproduction_card_verifies") and _check_passed(card, "contract_card_verifies") and _check_passed(card, "install_card_verifies")),
        ("package", _check_passed(card, "pypi")),
        ("ci", _check_passed(card, "github_tag") and _check_passed(card, "github_actions") and _check_passed(card, "openssf_scorecard")),
        ("install", _check_passed(card, "install_card_verifies")),
    ]
    width, height = 980, 280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE release card</text>',
        f'<text x="28" y="58" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Score {card["release_score"]}/100 | level {card["release_level"]} | public ready {str(card["public_release_ready"]).lower()}</text>',
    ]
    x0, y = 36, 96
    for idx, (label, passed) in enumerate(stages):
        x = x0 + idx * 184
        fill = "#0f766e" if passed else "#e5e7eb"
        text = "#ffffff" if passed else "#374151"
        parts.append(f'<rect x="{x}" y="{y}" width="138" height="58" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{x + 69}" y="{y + 36}" text-anchor="middle" font-family="Inter,Segoe UI,Arial" font-size="16" font-weight="700" fill="{text}">{label}</text>')
        if idx < len(stages) - 1:
            parts.append(f'<line x1="{x + 148}" y1="{y + 29}" x2="{x + 176}" y2="{y + 29}" stroke="#9ca3af" stroke-width="3"/>')
    failed = [row["name"] for row in card["checks"] if not row["passed"]]
    parts.append(f'<text x="28" y="202" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#111827">Local version {card["package"]["local_version"]} | blockers {len(failed)} | hash {card["release_card_sha256"][:16]}...</text>')
    parts.append('<text x="28" y="228" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">Release card only: 1.1 public readiness requires platform artifacts, PyPI, tag, CI, and Scorecard to be green.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_release_outputs(card: dict[str, Any], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(out, json.dumps(card, indent=2, sort_keys=True) + "\n")
    md = out.with_suffix(".md")
    tex = out.with_suffix(".tex")
    svg = out.with_suffix(".svg")
    write_text_lf(md, render_release_markdown(card))
    write_text_lf(tex, render_release_tex(card))
    write_text_lf(svg, render_release_svg(card))
    return {"json": str(out), "markdown": str(md), "tex": str(tex), "svg": str(svg)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a deterministic TRICE release card.")
    ap.add_argument("--out", type=Path, default=Path("trice_release_card.json"))
    ap.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    ap.add_argument("--install", type=Path, default=DEFAULT_INSTALL)
    ap.add_argument("--offline", action="store_true", help="Skip public HTTP checks in the doctor snapshot.")
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--format", choices=["json", "markdown", "tex"], default="json")
    args = ap.parse_args(argv)
    card = build_release_card(contract_path=args.contract, install_path=args.install, offline=args.offline, timeout_s=args.timeout_s)
    outputs = write_release_outputs(card, args.out)
    if args.format == "markdown":
        print(render_release_markdown(card))
    elif args.format == "tex":
        print(render_release_tex(card))
    else:
        print(json.dumps({"release_card": card, "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if card["release_level"] != "not_release_ready" else 1


def verify_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a deterministic TRICE release card.")
    ap.add_argument("release_card", type=Path)
    args = ap.parse_args(argv)
    verdict = verify_release_card_file(args.release_card)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def _doctor_check(name: str, doctor: dict[str, Any], required: str) -> dict[str, Any]:
    checks = doctor.get("checks") if isinstance(doctor.get("checks"), dict) else {}
    row = checks.get(name) if isinstance(checks.get(name), dict) else {}
    return _check(name, row.get("ok") is True, f"{row.get('status')}: {row.get('detail')}", required)


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def _safe_verify(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        verdict = fn()
    except Exception as exc:
        return {"ok": False, "errors": [str(exc)]}
    return verdict if isinstance(verdict, dict) else {"ok": False, "errors": ["verifier did not return a dict"]}


def _compact_verdict(verdict: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(verdict.get("ok")),
        "errors": list(verdict.get("errors") or [])[:10],
        **{k: verdict[k] for k in ("artifact_level", "reproduction_level", "contract_level", "install_level", "artifact_review_score", "reproduction_score", "contract_score", "install_score", "claim_allowed") if k in verdict},
    }


def _input_row(path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _docs_present(paths: dict[str, Path]) -> bool:
    return all(paths[name].is_file() for name in ("readme", "public_trust_matrix", "release_checklist", "pyproject", "contract_card", "install_card"))


def _present_count(paths: dict[str, Path]) -> str:
    return f"{sum(1 for path in paths.values() if path.is_file())}/{len(paths)}"


def _release_level(checks: list[dict[str, Any]]) -> str:
    if _public_release_ready(checks):
        return "public_release_ready"
    required_local = {"local_package", "bundled_cli", "schemas", "artifact_card_verifies", "reproduction_card_verifies", "contract_card_verifies", "install_card_verifies", "release_docs_present"}
    passed = {row["name"] for row in checks if row["passed"]}
    if required_local.issubset(passed):
        return "local_release_candidate"
    return "not_release_ready"


def _public_release_ready(checks: list[dict[str, Any]]) -> bool:
    required = {
        "local_package",
        "bundled_cli",
        "schemas",
        "artifact_card_verifies",
        "reproduction_card_verifies",
        "contract_card_verifies",
        "install_card_verifies",
        "release_docs_present",
        "pypi",
        "github_tag",
        "github_actions",
        "openssf_scorecard",
    }
    passed = {row["name"] for row in checks if row["passed"]}
    return required.issubset(passed)


def _release_score(checks: list[dict[str, Any]]) -> int:
    weights = {
        "local_package": 8,
        "bundled_cli": 11,
        "schemas": 6,
        "artifact_card_verifies": 9,
        "reproduction_card_verifies": 8,
        "contract_card_verifies": 7,
        "install_card_verifies": 12,
        "release_docs_present": 5,
        "pypi": 10,
        # piwheels builds from source distributions, which are intentionally
        # not part of the 1.1 bundled-auditor contract.
        "piwheels": 0,
        # crates.io is an informational distribution check for 1.1.x. It is
        # not a GA gate until TraceRazor declares a stable public Rust API.
        "crates_io": 0,
        "github_tag": 6,
        "github_actions": 6,
        "openssf_scorecard": 4,
        "provenance_plan_documented": 3,
        "attestation_plan_documented": 3,
        "sbom_plan_documented": 2,
    }
    return min(100, sum(weights.get(row["name"], 0) for row in checks if row["passed"]))


def _next_actions(checks: list[dict[str, Any]], version: str) -> list[str]:
    missing = [row["name"] for row in checks if not row["passed"]]
    if not missing:
        return [
            "Publish the release card with the GitHub release assets.",
            "Regenerate the contract card before any public API, CLI, schema, or example change.",
            "Run the held-out claim suite before any S-tier wording.",
        ]
    actions = []
    tag_check = next((row for row in checks if row["name"] == "github_tag"), {})
    remote_tag_exists = "remote_tag=True" in str(tag_check.get("observed") or "")
    mapping = {
        "pypi": f"Publish {version} to PyPI only after local release gates pass.",
        "piwheels": "Informational only for 1.1: do not add an sdist solely for piwheels.",
        "crates_io": "Optional: publish Rust crates only after declaring a stable public Rust API; keep cargo-install claims out of the README meanwhile.",
        "github_tag": (
            f"The remote v{version} tag already exists and must not be reused; fetch it for verification, "
            "or bump the version and create a new immutable tag for the next release."
            if remote_tag_exists
            else f"Create and push the v{version} tag only after local gates pass."
        ),
        "github_actions": "Re-run and fix GitHub Actions until CI, Agent Efficiency Gate, and Release are green.",
        "openssf_scorecard": "Run and publish OpenSSF Scorecard until the public score is at least 7.0.",
        "artifact_card_verifies": "Regenerate and verify the artifact card.",
        "reproduction_card_verifies": "Regenerate and verify the reproduction card.",
        "contract_card_verifies": "Regenerate and verify the public contract card.",
        "install_card_verifies": "Rebuild the wheel, regenerate the installability card, and verify its bound hashes.",
        "provenance_plan_documented": "Document trusted publishing and OIDC in the release checklist.",
        "attestation_plan_documented": "Document GitHub release artifact attestations in the release checklist.",
        "sbom_plan_documented": "Document CycloneDX SBOM and checksum release assets.",
    }
    for name in missing:
        if name in mapping:
            actions.append(mapping[name])
    return actions or ["Fix local release prerequisites before publishing."]


def _commands() -> list[dict[str, Any]]:
    commands = [
        ("doctor", "python -m tracerazor.trice doctor --format json --timeout-s 10"),
        ("verify-contract", "python -m tracerazor.trice verify-contract docs/trice_contract_card.json"),
        ("verify-reproduction", "python -m tracerazor.trice verify-reproduction docs/trice_reproduction_card.json"),
        ("verify-artifact", "python -m tracerazor.trice verify-artifact docs/trice_artifact_card.json"),
        ("build", "bash scripts/build_platform_wheel.sh"),
        ("twine-check", "python -m twine check dist/*.whl"),
        ("pip-audit", "python -m pip_audit --progress-spinner off ."),
        ("scorecard", "scorecard --repo=github.com/ZulfaqarHafez/TraceRazor"),
    ]
    return [
        {"name": name, "command": command, "sha256": hashlib.sha256(command.encode("utf-8")).hexdigest()}
        for name, command in commands
    ]


def _check_passed(card: dict[str, Any], name: str) -> bool:
    return any(row["name"] == name and row["passed"] for row in card.get("checks", []))


def _without_release_hash(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out.pop("release_card_sha256", None)
    return out


def _required_inputs_present(paths: dict[str, Path]) -> bool:
    return all(name in paths and paths[name].is_file() for name in (
        "artifact_card",
        "reproduction_card",
        "contract_card",
        "install_card",
        "readme",
        "public_trust_matrix",
        "release_checklist",
        "pyproject",
    ))


def _resolve_bound_path(card_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    repo_candidate = REPO / candidate
    if repo_candidate.exists():
        return repo_candidate
    card_relative = card_path.parent / candidate
    if card_relative.exists():
        return card_relative
    return repo_candidate


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


if __name__ == "__main__":
    raise SystemExit(main())
