"""Top-level integrity cards for the TRICE proof graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .artifact import verify_artifact_card_file
from .contract import verify_contract_card_file
from .crates import verify_crates_card_file
from .doctor import doctor_report
from .evidence import canonical_json, sha256_file, verify_manifest, write_text_lf
from .install import verify_install_card_file
from .release import verify_release_card_file
from .release_evidence import verify_release_evidence_file
from .reproduction import verify_reproduction_card_file
from .research import verify_research_card_file
from .schemas import schema_path

INTEGRITY_CARD_SCHEMA_VERSION = "trice-integrity-card/v1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "docs" / "trice_integrity_card.json"
DEFAULT_CONTRACT = REPO / "docs" / "trice_contract_card.json"
DEFAULT_ARTIFACT = REPO / "docs" / "trice_artifact_card.json"
DEFAULT_REPRODUCTION = REPO / "docs" / "trice_reproduction_card.json"
DEFAULT_RELEASE = REPO / "docs" / "trice_release_card.json"
DEFAULT_RELEASE_EVIDENCE = REPO / "docs" / "trice_release_evidence.json"
DEFAULT_CRATES = REPO / "docs" / "trice_crates_card.json"
DEFAULT_INSTALL = REPO / "docs" / "trice_install_card.json"
DEFAULT_RESEARCH = REPO / "docs" / "trice_research_card.json"
DEFAULT_PAPER_MANIFEST = REPO / "paper" / "trice_v3_research_manifest.json"
DEFAULT_PAPER_RESULT = REPO / "benchmark" / "trice" / "results" / "v2-smoke" / "trice_v2_live_results.json"

WORKFLOW_EXPECTATIONS = {
    "release_workflow": (REPO / ".github" / "workflows" / "release.yml", ["actions/attest", "trice_release_evidence", "trice_crates_card", "trice_install_card", "trice_research_card", "pypa/gh-action-pypi-publish"]),
    "ci_workflow": (REPO / ".github" / "workflows" / "tracerazor.yml", ["tracerazor.trice research", "tracerazor.trice integrity", "tracerazor.trice verify-integrity"]),
    "scorecard_workflow": (REPO / ".github" / "workflows" / "scorecard.yml", ["ossf/scorecard-action", "publish_results: true"]),
}


def build_integrity_card(
    *,
    contract_path: str | Path = DEFAULT_CONTRACT,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    reproduction_path: str | Path = DEFAULT_REPRODUCTION,
    release_path: str | Path = DEFAULT_RELEASE,
    release_evidence_path: str | Path = DEFAULT_RELEASE_EVIDENCE,
    crates_path: str | Path = DEFAULT_CRATES,
    install_path: str | Path = DEFAULT_INSTALL,
    research_path: str | Path = DEFAULT_RESEARCH,
    paper_manifest_path: str | Path = DEFAULT_PAPER_MANIFEST,
    paper_result_path: str | Path = DEFAULT_PAPER_RESULT,
    doctor_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic integrity card over TRICE proof artifacts."""

    paths = {
        "contract_card": Path(contract_path),
        "artifact_card": Path(artifact_path),
        "reproduction_card": Path(reproduction_path),
        "release_card": Path(release_path),
        "release_evidence": Path(release_evidence_path),
        "crates_card": Path(crates_path),
        "install_card": Path(install_path),
        "research_card": Path(research_path),
        "paper_manifest": Path(paper_manifest_path),
        "paper_result": Path(paper_result_path),
    }
    doctor = doctor_snapshot or doctor_report(offline=True)
    verdicts = {
        "contract_card": _safe_verify(lambda: verify_contract_card_file(paths["contract_card"])),
        "artifact_card": _safe_verify(lambda: verify_artifact_card_file(paths["artifact_card"])),
        "reproduction_card": _safe_verify(lambda: verify_reproduction_card_file(paths["reproduction_card"])),
        "release_card": _safe_verify(lambda: verify_release_card_file(paths["release_card"])),
        "release_evidence": _safe_verify(lambda: verify_release_evidence_file(paths["release_evidence"])),
        "crates_card": _safe_verify(lambda: verify_crates_card_file(paths["crates_card"])),
        "install_card": _safe_verify(lambda: verify_install_card_file(paths["install_card"])),
        "research_card": _safe_verify(lambda: verify_research_card_file(paths["research_card"])),
        "paper_manifest": _safe_verify(lambda: verify_manifest(paths["paper_manifest"], paths["paper_result"])),
    }
    schema_rows = _schema_rows()
    workflow_rows = _workflow_rows()
    checks = [
        _check("offline_doctor_core", _offline_doctor_core_ok(doctor), _offline_doctor_summary(doctor), "local package, CLI, and schemas pass offline doctor"),
        _check("contract_card_verifies", bool(verdicts["contract_card"].get("ok")), verdicts["contract_card"].get("contract_level"), "public API/CLI/schema contract verifies"),
        _check("artifact_card_verifies", bool(verdicts["artifact_card"].get("ok")), verdicts["artifact_card"].get("artifact_level"), "artifact-review packet verifies"),
        _check("reproduction_card_verifies", bool(verdicts["reproduction_card"].get("ok")), verdicts["reproduction_card"].get("reproduction_level"), "reviewer reproduction packet verifies"),
        _check("release_card_verifies", bool(verdicts["release_card"].get("ok")), verdicts["release_card"].get("release_level"), "release trust card verifies"),
        _check("release_evidence_verifies", bool(verdicts["release_evidence"].get("ok")), verdicts["release_evidence"].get("release_evidence_level"), "release evidence packet verifies"),
        _check("crates_card_verifies", bool(verdicts["crates_card"].get("ok")), verdicts["crates_card"].get("crates_card_level"), "crates staged-publish card verifies"),
        _check("install_card_verifies", bool(verdicts["install_card"].get("ok")), verdicts["install_card"].get("install_level"), "clean-wheel installability card verifies"),
        _check("research_card_verifies", bool(verdicts["research_card"].get("ok")), verdicts["research_card"].get("research_level"), "research-basis card verifies"),
        _check("paper_manifest_verifies", bool(verdicts["paper_manifest"].get("ok")), _paper_artifact_count(verdicts["paper_manifest"]), "paper manifest and bound result verify"),
        _check("schemas_available", all(row["present"] for row in schema_rows), _present_count(schema_rows), "all shipped TRICE schemas are present"),
        _check("workflows_bound", all(row["present"] and row["markers_present"] for row in workflow_rows), _workflow_summary(workflow_rows), "CI, release, and Scorecard workflows contain integrity/provenance hooks"),
        _check("claim_honesty_bound", _claim_honesty(verdicts), _claim_honesty_summary(verdicts), "smoke evidence remains a non-S-tier claim"),
    ]
    card = {
        "schema_version": INTEGRITY_CARD_SCHEMA_VERSION,
        "scope": "TRICE proof graph integrity",
        "integrity_level": _integrity_level(checks),
        "integrity_score": _integrity_score(checks),
        "checks": checks,
        "inputs": {name: _input_row(path) for name, path in paths.items()},
        "schemas": schema_rows,
        "workflows": workflow_rows,
        "doctor_snapshot": doctor,
        "doctor_snapshot_sha256": hashlib.sha256(canonical_json(doctor).encode("utf-8")).hexdigest(),
        "verdicts": {name: _compact_verdict(verdict) for name, verdict in verdicts.items()},
        "research_basis": [
            "SLSA and in-toto make provenance useful only when the resulting subjects and predicates are inspected by a verifier.",
            "GitHub artifact attestations cover hosted release assets, while local release evidence binds checksums, SBOMs, provenance sidecars, paper artifacts, and evidence bundles.",
            "OpenSSF Scorecard is a public health signal, but local integrity must also prove the repository's proof graph is internally consistent.",
            "TRICE integrity cards make stale README, paper, workflow, schema, and proof-card drift fail as one deterministic gate.",
            "TRICE research cards make stale or under-covered literature grounding fail before paper and README claims drift.",
        ],
        "next_actions": _next_actions(checks),
    }
    card["integrity_card_sha256"] = hashlib.sha256(canonical_json(_without_integrity_hash(card)).encode("utf-8")).hexdigest()
    return card


def verify_integrity_card_file(path: str | Path) -> dict[str, Any]:
    """Verify an integrity card self hash, bound inputs, and deterministic rebuild."""

    card_path = Path(path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if card.get("schema_version") != INTEGRITY_CARD_SCHEMA_VERSION:
        errors.append(f"schema_version must be {INTEGRITY_CARD_SCHEMA_VERSION}")
    expected_hash = str(card.get("integrity_card_sha256") or "")
    actual_hash = hashlib.sha256(canonical_json(_without_integrity_hash(card)).encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        errors.append("integrity_card_sha256 mismatch")

    checked_inputs = _verify_bound_rows(card_path, card.get("inputs"), errors)
    checked_schemas = _verify_bound_list(card_path, card.get("schemas"), "schemas", errors)
    checked_workflows = _verify_bound_list(card_path, card.get("workflows"), "workflows", errors)
    doctor = card.get("doctor_snapshot") if isinstance(card.get("doctor_snapshot"), dict) else {}
    if hashlib.sha256(canonical_json(doctor).encode("utf-8")).hexdigest() != card.get("doctor_snapshot_sha256"):
        errors.append("doctor_snapshot_sha256 mismatch")

    resolved = _resolved_input_paths(card_path, card.get("inputs"))
    if _required_inputs_present(resolved) and doctor:
        rebuilt = build_integrity_card(
            contract_path=resolved["contract_card"],
            artifact_path=resolved["artifact_card"],
            reproduction_path=resolved["reproduction_card"],
            release_path=resolved["release_card"],
            release_evidence_path=resolved["release_evidence"],
            crates_path=resolved["crates_card"],
            install_path=resolved["install_card"],
            research_path=resolved["research_card"],
            paper_manifest_path=resolved["paper_manifest"],
            paper_result_path=resolved["paper_result"],
            doctor_snapshot=doctor,
        )
        if canonical_json(_without_integrity_hash(rebuilt)) != canonical_json(_without_integrity_hash(card)):
            errors.append("integrity card does not match deterministic rebuild from bound inputs")

    return {
        "ok": not errors,
        "schema_version": card.get("schema_version"),
        "integrity_level": card.get("integrity_level"),
        "integrity_score": card.get("integrity_score"),
        "integrity_card_sha256": expected_hash,
        "computed_integrity_card_sha256": actual_hash,
        "checked_inputs": checked_inputs,
        "checked_schemas": checked_schemas,
        "checked_workflows": checked_workflows,
        "errors": errors,
    }


def render_integrity_markdown(card: dict[str, Any]) -> str:
    lines = [
        "# TRICE Integrity Card",
        "",
        f"- Scope: `{card['scope']}`",
        f"- Integrity level: `{card['integrity_level']}`",
        f"- Integrity score: **{card['integrity_score']}/100**",
        "",
        "## Checks",
        "",
        "| Check | Passed | Observed | Required |",
        "|---|---:|---|---|",
    ]
    for row in card["checks"]:
        lines.append(f"| {row['name']} | {'yes' if row['passed'] else 'no'} | {_md(row['observed'])} | {_md(row['required'])} |")
    lines.extend(["", "## Workflows", "", "| Workflow | Present | Markers | Path |", "|---|---:|---|---|"])
    for row in card["workflows"]:
        lines.append(f"| {row['name']} | {'yes' if row['present'] else 'no'} | {'yes' if row['markers_present'] else 'no'} | `{row['path']}` |")
    lines.extend(["", "## Next Actions", ""])
    for item in card["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Hash", "", f"- integrity card: `{card['integrity_card_sha256']}`", ""])
    return "\n".join(lines)


def render_integrity_tex(card: dict[str, Any]) -> str:
    rows = "\n".join(
        f"{_tex(row['name'])} & {'yes' if row['passed'] else 'no'} & {_tex(row['required'])} \\\\"
        for row in card["checks"]
    )
    return (
        "\\section{Integrity Card}\n"
        f"Integrity level: \\texttt{{{_tex(card['integrity_level'])}}}; "
        f"score: {card['integrity_score']}/100.\n\n"
        "\\begin{tabular}{lrl}\n"
        "Check & Passed & Required \\\\\n"
        "\\hline\n"
        f"{rows}\n"
        "\\end{tabular}\n"
    )


def render_integrity_svg(card: dict[str, Any]) -> str:
    stages = [
        ("doctor", _check_passed(card, "offline_doctor_core")),
        ("proof", _check_passed(card, "contract_card_verifies") and _check_passed(card, "artifact_card_verifies") and _check_passed(card, "reproduction_card_verifies")),
        ("release", _check_passed(card, "release_card_verifies") and _check_passed(card, "release_evidence_verifies") and _check_passed(card, "crates_card_verifies") and _check_passed(card, "install_card_verifies")),
        ("research", _check_passed(card, "research_card_verifies")),
        ("paper", _check_passed(card, "paper_manifest_verifies")),
        ("ci", _check_passed(card, "workflows_bound")),
    ]
    width, height = 980, 280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE proof graph integrity</text>',
        f'<text x="28" y="58" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Score {card["integrity_score"]}/100 | level {card["integrity_level"]} | hash {card["integrity_card_sha256"][:16]}...</text>',
    ]
    x0, y = 28, 96
    gap = 154
    for idx, (label, passed) in enumerate(stages):
        x = x0 + idx * gap
        fill = "#2563eb" if passed else "#e5e7eb"
        text = "#ffffff" if passed else "#374151"
        parts.append(f'<rect x="{x}" y="{y}" width="118" height="58" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{x + 59}" y="{y + 36}" text-anchor="middle" font-family="Inter,Segoe UI,Arial" font-size="15" font-weight="700" fill="{text}">{label}</text>')
        if idx < len(stages) - 1:
            parts.append(f'<line x1="{x + 128}" y1="{y + 29}" x2="{x + gap - 10}" y2="{y + 29}" stroke="#9ca3af" stroke-width="3"/>')
    parts.append('<text x="28" y="210" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">Integrity card only: S-tier still requires the held-out remote live suite and passing claim card.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_integrity_outputs(card: dict[str, Any], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(out, json.dumps(card, indent=2, sort_keys=True) + "\n")
    md = out.with_suffix(".md")
    tex = out.with_suffix(".tex")
    svg = out.with_suffix(".svg")
    write_text_lf(md, render_integrity_markdown(card))
    write_text_lf(tex, render_integrity_tex(card))
    write_text_lf(svg, render_integrity_svg(card))
    return {"json": str(out), "markdown": str(md), "tex": str(tex), "svg": str(svg)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a deterministic TRICE proof-graph integrity card.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    ap.add_argument("--reproduction", type=Path, default=DEFAULT_REPRODUCTION)
    ap.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    ap.add_argument("--release-evidence", type=Path, default=DEFAULT_RELEASE_EVIDENCE)
    ap.add_argument("--crates", type=Path, default=DEFAULT_CRATES)
    ap.add_argument("--install", type=Path, default=DEFAULT_INSTALL)
    ap.add_argument("--research", type=Path, default=DEFAULT_RESEARCH)
    ap.add_argument("--paper-manifest", type=Path, default=DEFAULT_PAPER_MANIFEST)
    ap.add_argument("--paper-result", type=Path, default=DEFAULT_PAPER_RESULT)
    ap.add_argument("--format", choices=["json", "markdown", "tex"], default="json")
    args = ap.parse_args(argv)
    card = build_integrity_card(
        contract_path=args.contract,
        artifact_path=args.artifact,
        reproduction_path=args.reproduction,
        release_path=args.release,
        release_evidence_path=args.release_evidence,
        crates_path=args.crates,
        install_path=args.install,
        research_path=args.research,
        paper_manifest_path=args.paper_manifest,
        paper_result_path=args.paper_result,
    )
    outputs = write_integrity_outputs(card, args.out)
    if args.format == "markdown":
        print(render_integrity_markdown(card))
    elif args.format == "tex":
        print(render_integrity_tex(card))
    else:
        print(json.dumps({"integrity_card": card, "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if card["integrity_level"] == "proof_graph_integrity_locked" else 1


def verify_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a deterministic TRICE integrity card.")
    ap.add_argument("integrity_card", type=Path)
    args = ap.parse_args(argv)
    verdict = verify_integrity_card_file(args.integrity_card)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def _schema_rows() -> list[dict[str, Any]]:
    names = [
        "patch",
        "evidence",
        "suite",
        "bundle",
        "adapter",
        "receipt",
        "claim",
        "readiness",
        "artifact",
        "protocol",
        "design",
        "reproduction",
        "release",
        "contract",
        "release-evidence",
        "integrity",
        "crates",
        "install",
        "research",
    ]
    return [_availability_row(schema_path(name)) for name in names]


def _workflow_rows() -> list[dict[str, Any]]:
    rows = []
    for name, (path, markers) in WORKFLOW_EXPECTATIONS.items():
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        rows.append({
            "name": name,
            "path": _display_path(path),
            "present": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": sha256_file(path) if path.is_file() else None,
            "required_markers": markers,
            "markers_present": all(marker in text for marker in markers),
        })
    return rows


def _availability_row(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "path": _display_path(path),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _input_row(path: Path) -> dict[str, Any]:
    return _availability_row(path)


def _safe_verify(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        verdict = fn()
    except Exception as exc:
        return {"ok": False, "errors": [str(exc)]}
    return verdict if isinstance(verdict, dict) else {"ok": False, "errors": ["verifier did not return a dict"]}


def _compact_verdict(verdict: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "ok",
        "contract_level",
        "artifact_level",
        "reproduction_level",
        "release_level",
        "release_evidence_level",
        "crates_card_level",
        "install_level",
        "research_level",
        "integrity_level",
        "contract_score",
        "artifact_review_score",
        "reproduction_score",
        "release_score",
        "release_evidence_score",
        "crates_publish_score",
        "install_score",
        "research_score",
        "public_release_ready",
        "local_publish_plan_locked",
        "cargo_install_claim_allowed",
        "claim_allowed",
    )
    out = {key: verdict[key] for key in keys if key in verdict}
    out["errors"] = list(verdict.get("errors") or [])[:10]
    return out


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def _integrity_level(checks: list[dict[str, Any]]) -> str:
    if all(row["passed"] for row in checks):
        return "proof_graph_integrity_locked"
    required = {"offline_doctor_core", "contract_card_verifies", "artifact_card_verifies", "release_card_verifies"}
    passed = {row["name"] for row in checks if row["passed"]}
    return "partial_integrity" if required.issubset(passed) else "integrity_missing"


def _integrity_score(checks: list[dict[str, Any]]) -> int:
    weights = {
        "offline_doctor_core": 8,
        "contract_card_verifies": 10,
        "artifact_card_verifies": 10,
        "reproduction_card_verifies": 8,
        "release_card_verifies": 8,
        "release_evidence_verifies": 10,
        "crates_card_verifies": 7,
        "install_card_verifies": 7,
        "research_card_verifies": 8,
        "paper_manifest_verifies": 8,
        "schemas_available": 7,
        "workflows_bound": 6,
        "claim_honesty_bound": 3,
    }
    return min(100, sum(weights.get(row["name"], 0) for row in checks if row["passed"]))


def _offline_doctor_core_ok(doctor: dict[str, Any]) -> bool:
    checks = doctor.get("checks") if isinstance(doctor.get("checks"), dict) else {}
    return all((checks.get(name) or {}).get("ok") is True for name in ("local_package", "bundled_cli", "schemas"))


def _offline_doctor_summary(doctor: dict[str, Any]) -> str:
    checks = doctor.get("checks") if isinstance(doctor.get("checks"), dict) else {}
    return "; ".join(f"{name}={(checks.get(name) or {}).get('status')}" for name in ("local_package", "bundled_cli", "schemas"))


def _paper_artifact_count(verdict: dict[str, Any]) -> int:
    manifest = verdict.get("manifest") if isinstance(verdict.get("manifest"), dict) else {}
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    return len(artifacts)


def _present_count(rows: list[dict[str, Any]]) -> str:
    return f"{sum(1 for row in rows if row.get('present'))}/{len(rows)}"


def _workflow_summary(rows: list[dict[str, Any]]) -> str:
    return "; ".join(f"{row['name']}={'ok' if row['present'] and row['markers_present'] else 'missing'}" for row in rows)


def _claim_honesty(verdicts: dict[str, dict[str, Any]]) -> bool:
    artifact = verdicts.get("artifact_card", {})
    return artifact.get("claim_allowed") is False


def _claim_honesty_summary(verdicts: dict[str, dict[str, Any]]) -> str:
    artifact = verdicts.get("artifact_card", {})
    release = verdicts.get("release_card", {})
    return f"claim_allowed={artifact.get('claim_allowed')} public_release_ready={release.get('public_release_ready')}"


def _next_actions(checks: list[dict[str, Any]]) -> list[str]:
    missing = [row["name"] for row in checks if not row["passed"]]
    if not missing:
        return [
            "Run this card in CI after building release artifacts.",
            "Regenerate the integrity card after changing proof cards, schemas, workflows, README, paper, or release evidence.",
            "Keep S-tier wording blocked until the held-out remote claim card passes.",
        ]
    mapping = {
        "offline_doctor_core": "Fix local package, CLI, or schema availability before publishing.",
        "contract_card_verifies": "Regenerate and verify the public contract card.",
        "artifact_card_verifies": "Regenerate and verify the artifact card.",
        "reproduction_card_verifies": "Regenerate and verify the reproduction card.",
        "release_card_verifies": "Regenerate and verify the release card.",
        "release_evidence_verifies": "Regenerate release evidence from current dist and CLI artifacts.",
        "crates_card_verifies": "Regenerate and verify the crates publish card.",
        "install_card_verifies": "Rebuild the wheel, regenerate the installability card, and verify its bound hashes.",
        "research_card_verifies": "Regenerate and verify the research card from the current ledger.",
        "paper_manifest_verifies": "Regenerate the paper and evidence manifest.",
        "schemas_available": "Ship every TRICE schema in the package.",
        "workflows_bound": "Restore release attestation, Scorecard, and integrity workflow hooks.",
        "claim_honesty_bound": "Keep release/S-tier claims false until public gates and held-out evidence pass.",
    }
    return [mapping[name] for name in missing if name in mapping] or ["Repair the proof graph and rerun the integrity verifier."]


def _verify_bound_rows(card_path: Path, rows: Any, errors: list[str]) -> list[str]:
    if not isinstance(rows, dict):
        errors.append("inputs must be an object")
        return []
    checked = []
    for name, row in sorted(rows.items()):
        if isinstance(row, dict) and _verify_row(card_path, row, f"input {name}", errors):
            checked.append(name)
    return checked


def _verify_bound_list(card_path: Path, rows: Any, label: str, errors: list[str]) -> list[str]:
    if not isinstance(rows, list):
        errors.append(f"{label} must be a list")
        return []
    checked = []
    for row in rows:
        if isinstance(row, dict) and _verify_row(card_path, row, str(row.get("name") or label), errors):
            checked.append(str(row.get("name") or row.get("path")))
    return checked


def _verify_row(card_path: Path, row: dict[str, Any], label: str, errors: list[str]) -> bool:
    raw_path = row.get("path")
    if not raw_path:
        errors.append(f"{label} path is missing")
        return False
    path = _resolve_path(card_path, str(raw_path))
    if not path.is_file():
        errors.append(f"{label} file not found: {raw_path}")
        return False
    ok = True
    if path.stat().st_size != int(row.get("bytes") or 0):
        errors.append(f"{label} byte count mismatch")
        ok = False
    if sha256_file(path) != row.get("sha256"):
        errors.append(f"{label} sha256 mismatch")
        ok = False
    return ok


def _resolved_input_paths(card_path: Path, rows: Any) -> dict[str, Path]:
    if not isinstance(rows, dict):
        return {}
    resolved = {}
    for name, row in rows.items():
        if isinstance(row, dict) and row.get("path"):
            resolved[name] = _resolve_path(card_path, str(row["path"]))
    return resolved


def _required_inputs_present(paths: dict[str, Path]) -> bool:
    required = {"contract_card", "artifact_card", "reproduction_card", "release_card", "release_evidence", "crates_card", "install_card", "research_card", "paper_manifest", "paper_result"}
    return required.issubset(paths) and all(paths[name].is_file() for name in required)


def _without_integrity_hash(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out.pop("integrity_card_sha256", None)
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


def _check_passed(card: dict[str, Any], name: str) -> bool:
    return any(row["name"] == name and row["passed"] for row in card.get("checks", []))


def _md(value: Any) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return text.replace("|", "\\|")


def _tex(value: Any) -> str:
    return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


if __name__ == "__main__":
    raise SystemExit(main())
