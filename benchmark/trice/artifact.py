"""Artifact review cards for TRICE public evidence packages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .bundle import verify_evidence_bundle
from .claim import verify_claim_card_file
from .contract import verify_contract_card_file
from .design import verify_design_card_file
from .evidence import canonical_json, sha256_file, verify_manifest
from .install import verify_install_card_file
from .protocol import verify_protocol_lock_file
from .readiness import verify_readiness_file
from .reproduction import verify_reproduction_card_file
from .research import verify_research_card_file

ARTIFACT_CARD_SCHEMA_VERSION = "trice-artifact-card/v1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_READINESS = REPO / "docs" / "trice_suite_readiness.json"
DEFAULT_PROTOCOL = REPO / "docs" / "trice_protocol_lock.json"
DEFAULT_DESIGN = REPO / "docs" / "trice_design_card.json"
DEFAULT_REPRODUCTION = REPO / "docs" / "trice_reproduction_card.json"
DEFAULT_CONTRACT = REPO / "docs" / "trice_contract_card.json"
DEFAULT_INSTALL = REPO / "docs" / "trice_install_card.json"
DEFAULT_RESEARCH = REPO / "docs" / "trice_research_card.json"
DEFAULT_CLAIM = REPO / "docs" / "trice_claim_card.json"
DEFAULT_REMOTE_CLAIM = REPO / "docs" / "trice_remote_smoke_claim_card.json"
DEFAULT_BROAD_BUNDLE = REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_broad_smoke_evidence.trice.zip"
DEFAULT_REMOTE_BUNDLE = REPO / "benchmark" / "trice" / "results" / "v2-remote-smoke" / "trice_remote_smoke_evidence.trice.zip"
DEFAULT_PAPER_MANIFEST = REPO / "paper" / "trice_v3_research_manifest.json"
DEFAULT_PAPER_RESULT = REPO / "benchmark" / "trice" / "results" / "v2-smoke" / "trice_v2_live_results.json"
DEFAULT_PAPER_TEX = REPO / "paper" / "trice_v3_research_paper.tex"
DEFAULT_PAPER_PDF = REPO / "paper" / "trice_v3_research_paper.pdf"
DEFAULT_README = REPO / "README.md"
DEFAULT_LIBRARY_DOC = REPO / "docs" / "trice_library.md"
DEFAULT_SCOPE = "TRICE deterministic context-control evidence package"


def build_artifact_card(
    *,
    readiness_path: str | Path = DEFAULT_READINESS,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    design_path: str | Path = DEFAULT_DESIGN,
    reproduction_path: str | Path = DEFAULT_REPRODUCTION,
    contract_path: str | Path = DEFAULT_CONTRACT,
    install_path: str | Path = DEFAULT_INSTALL,
    research_path: str | Path = DEFAULT_RESEARCH,
    claim_path: str | Path = DEFAULT_CLAIM,
    remote_claim_path: str | Path = DEFAULT_REMOTE_CLAIM,
    bundle_path: str | Path = DEFAULT_BROAD_BUNDLE,
    remote_bundle_path: str | Path = DEFAULT_REMOTE_BUNDLE,
    paper_manifest_path: str | Path = DEFAULT_PAPER_MANIFEST,
    paper_result_path: str | Path = DEFAULT_PAPER_RESULT,
    paper_tex_path: str | Path = DEFAULT_PAPER_TEX,
    paper_pdf_path: str | Path = DEFAULT_PAPER_PDF,
    readme_path: str | Path = DEFAULT_README,
    library_doc_path: str | Path = DEFAULT_LIBRARY_DOC,
    scope: str = DEFAULT_SCOPE,
) -> dict[str, Any]:
    """Build a deterministic public artifact review card."""

    paths = {
        "readiness": Path(readiness_path),
        "protocol_lock": Path(protocol_path),
        "design_card": Path(design_path),
        "reproduction_card": Path(reproduction_path),
        "contract_card": Path(contract_path),
        "install_card": Path(install_path),
        "research_card": Path(research_path),
        "claim": Path(claim_path),
        "remote_smoke_claim": Path(remote_claim_path),
        "evidence_bundle": Path(bundle_path),
        "remote_smoke_bundle": Path(remote_bundle_path),
        "paper_manifest": Path(paper_manifest_path),
        "paper_result": Path(paper_result_path),
        "paper_tex": Path(paper_tex_path),
        "paper_pdf": Path(paper_pdf_path),
        "readme": Path(readme_path),
        "library_doc": Path(library_doc_path),
    }
    availability = [_availability_row(name, path) for name, path in paths.items()]
    readiness_verdict = _safe_verify(lambda: verify_readiness_file(paths["readiness"]))
    protocol_verdict = _safe_verify(lambda: verify_protocol_lock_file(paths["protocol_lock"]))
    design_verdict = _safe_verify(lambda: verify_design_card_file(paths["design_card"]))
    reproduction_verdict = _safe_verify(lambda: verify_reproduction_card_file(paths["reproduction_card"]))
    contract_verdict = _safe_verify(lambda: verify_contract_card_file(paths["contract_card"]))
    install_verdict = _safe_verify(lambda: verify_install_card_file(paths["install_card"]))
    research_verdict = _safe_verify(lambda: verify_research_card_file(paths["research_card"]))
    claim_verdict = _safe_verify(lambda: verify_claim_card_file(paths["claim"]))
    remote_claim_verdict = _safe_verify(lambda: verify_claim_card_file(paths["remote_smoke_claim"]))
    bundle_verdict = _safe_verify(lambda: verify_evidence_bundle(paths["evidence_bundle"]))
    remote_bundle_verdict = _safe_verify(lambda: verify_evidence_bundle(paths["remote_smoke_bundle"]))
    paper_verdict = _safe_verify(lambda: verify_manifest(paths["paper_manifest"], paths["paper_result"]))
    schema_rows = _schema_rows()
    checks = [
        _check("artifacts_available", all(row["present"] for row in availability), _present_count(availability), f"{len(availability)}/{len(availability)} present"),
        _check("readiness_verifies", bool(readiness_verdict.get("ok")), readiness_verdict.get("readiness_level"), "readiness hash and suite manifest hash verify"),
        _check("protocol_lock_verifies", bool(protocol_verdict.get("ok")), protocol_verdict.get("protocol_level"), "protocol hash and deterministic suite rebuild verify"),
        _check("design_card_verifies", bool(design_verdict.get("ok")), design_verdict.get("design_level"), "design-card hash and deterministic protocol/result rebuild verify"),
        _check("reproduction_card_verifies", bool(reproduction_verdict.get("ok")), reproduction_verdict.get("reproduction_level"), "reproduction-card hash, input hashes, and deterministic rebuild verify"),
        _check("contract_card_verifies", bool(contract_verdict.get("ok")), contract_verdict.get("contract_level"), "public API/CLI/schema contract card verifies"),
        _check("install_card_verifies", bool(install_verdict.get("ok")), install_verdict.get("install_level"), "clean-wheel installability card verifies"),
        _check("research_card_verifies", bool(research_verdict.get("ok")), research_verdict.get("research_level"), "research-basis card verifies"),
        _check("claim_card_verifies", bool(claim_verdict.get("ok")), claim_verdict.get("claim_level"), "claim-card hash and bound suite hashes verify"),
        _check("remote_smoke_claim_verifies", bool(remote_claim_verdict.get("ok")), remote_claim_verdict.get("claim_level"), "remote-git smoke claim card hash and bound suite hashes verify"),
        _check("evidence_bundle_verifies", bool(bundle_verdict.get("ok")), bundle_verdict.get("entry_count"), "bundle hashes and child manifests verify"),
        _check("remote_smoke_bundle_verifies", bool(remote_bundle_verdict.get("ok")), remote_bundle_verdict.get("entry_count"), "remote-git smoke bundle hashes and child manifest verify"),
        _check("paper_manifest_verifies", bool(paper_verdict.get("ok")), _paper_artifact_count(paper_verdict), "paper artifacts and result hash verify"),
        _check("schemas_available", all(row["present"] for row in schema_rows), _present_count(schema_rows), f"{len(schema_rows)}/{len(schema_rows)} schemas present"),
        _check("claim_honesty", claim_verdict.get("claim_level") != "s_tier" and claim_verdict.get("claim_allowed") is False, claim_verdict.get("claim_level"), "non-S-tier evidence must not allow S-tier claim"),
        _check("remote_smoke_honesty", remote_claim_verdict.get("claim_level") != "s_tier" and remote_claim_verdict.get("claim_allowed") is False, remote_claim_verdict.get("claim_level"), "remote smoke evidence must not allow S-tier claim"),
        _check("readiness_honesty", readiness_verdict.get("readiness_level") != "claim_ready" and readiness_verdict.get("claim_execution_ready") is False, readiness_verdict.get("readiness_level"), "smoke package must not be claim-ready"),
    ]
    card = {
        "schema_version": ARTIFACT_CARD_SCHEMA_VERSION,
        "scope": scope,
        "artifact_level": _artifact_level(checks),
        "artifact_review_score": _artifact_score(checks),
        "claim_allowed": bool(claim_verdict.get("claim_allowed")),
        "readiness_level": readiness_verdict.get("readiness_level"),
        "checks": checks,
        "availability": availability,
        "schemas": schema_rows,
        "verdicts": {
            "readiness": _compact_verdict(readiness_verdict),
            "protocol_lock": _compact_verdict(protocol_verdict),
            "design_card": _compact_verdict(design_verdict),
            "reproduction_card": _compact_verdict(reproduction_verdict),
            "contract_card": _compact_verdict(contract_verdict),
            "install_card": _compact_verdict(install_verdict),
            "research_card": _compact_verdict(research_verdict),
            "claim": _compact_verdict(claim_verdict),
            "remote_smoke_claim": _compact_verdict(remote_claim_verdict),
            "evidence_bundle": _compact_verdict(bundle_verdict),
            "remote_smoke_bundle": _compact_verdict(remote_bundle_verdict),
            "paper_manifest": _compact_verdict(paper_verdict),
        },
        "input_sha256": {
            name: sha256_file(path) if path.is_file() else None
            for name, path in paths.items()
        },
        "research_basis": [
            "ACM-style artifact review: availability, functionality, reusability, and reproducibility must be explicit.",
            "Agent-evaluation research: cost, repeated trials, and verifiable environments are part of the benchmark, not an appendix.",
            "Benchmark-submission practice: locked inputs, fixed revisions, and portable evidence bundles make small public smoke runs reviewable without inflating them into broad claims.",
            "Contract-card practice: SemVer, public imports, CLI commands, schemas, examples, and docs must be declared before users can trust a library compatibility promise.",
            "TRICE rule: a paper claim is reviewable only when the protocol lock, design card, reproduction card, contract card, installability card, README, paper, readiness card, claim card, bundle, and schemas verify together.",
            "Research-card practice: the literature ledger is a versioned product input, so paper and README claims should fail review when the source basis drifts.",
        ],
        "next_actions": _next_actions(checks),
    }
    card["artifact_card_sha256"] = hashlib.sha256(canonical_json(_without_card_hash(card)).encode("utf-8")).hexdigest()
    return card


def verify_artifact_card_file(artifact_card_path: str | Path) -> dict[str, Any]:
    """Verify an artifact card self hash and every bound artifact/schema hash."""

    card_path = Path(artifact_card_path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if card.get("schema_version") != ARTIFACT_CARD_SCHEMA_VERSION:
        errors.append(f"schema_version must be {ARTIFACT_CARD_SCHEMA_VERSION}")

    expected_card_hash = str(card.get("artifact_card_sha256") or "")
    actual_card_hash = hashlib.sha256(canonical_json(_without_card_hash(card)).encode("utf-8")).hexdigest()
    if actual_card_hash != expected_card_hash:
        errors.append("artifact_card_sha256 mismatch")

    checked_inputs = _verify_bound_rows(card_path, card.get("availability"), "availability", errors)
    checked_schemas = _verify_bound_rows(card_path, card.get("schemas"), "schemas", errors)
    _verify_input_hashes(card_path, card, errors)

    return {
        "ok": not errors,
        "schema_version": card.get("schema_version"),
        "artifact_level": card.get("artifact_level"),
        "artifact_review_score": card.get("artifact_review_score"),
        "claim_allowed": bool(card.get("claim_allowed")),
        "artifact_card_sha256": expected_card_hash,
        "computed_artifact_card_sha256": actual_card_hash,
        "checked_inputs": checked_inputs,
        "checked_schemas": checked_schemas,
        "errors": errors,
    }


def render_artifact_markdown(card: dict[str, Any]) -> str:
    lines = [
        "# TRICE Artifact Card",
        "",
        f"- Scope: `{card['scope']}`",
        f"- Artifact level: `{card['artifact_level']}`",
        f"- Artifact review score: **{card['artifact_review_score']}/100**",
        f"- Readiness level: `{card['readiness_level']}`",
        f"- Claim allowed: `{str(card['claim_allowed']).lower()}`",
        "",
        "## Checks",
        "",
        "| Check | Passed | Observed | Required |",
        "|---|---:|---|---|",
    ]
    for row in card["checks"]:
        lines.append(f"| {row['name']} | {'yes' if row['passed'] else 'no'} | {_md(row['observed'])} | {_md(row['required'])} |")
    lines.extend(["", "## Availability", "", "| Artifact | Present | Path | SHA-256 |", "|---|---:|---|---|"])
    for row in card["availability"]:
        lines.append(f"| {row['name']} | {'yes' if row['present'] else 'no'} | `{row['path']}` | `{row['sha256']}` |")
    lines.extend(["", "## Next Actions", ""])
    for item in card["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Hash", "", f"- artifact card: `{card['artifact_card_sha256']}`", ""])
    return "\n".join(lines)


def render_artifact_tex(card: dict[str, Any]) -> str:
    rows = "\n".join(
        f"{_tex(row['name'])} & {'yes' if row['passed'] else 'no'} & {_tex(row['required'])} \\\\"
        for row in card["checks"]
    )
    next_actions = "\n".join(f"\\item {_tex(item)}" for item in card["next_actions"])
    return (
        "\\section{Artifact Review Card}\n"
        f"Artifact level: \\texttt{{{_tex(card['artifact_level'])}}}; "
        f"review score: {card['artifact_review_score']}/100; "
        f"claim allowed: {'yes' if card['claim_allowed'] else 'no'}.\n\n"
        "\\begin{table}[h]\n\\centering\n"
        "\\begin{tabular}{lrl}\n\\toprule\nCheck & Passed & Required \\\\\n\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{Public artifact-review checks for TRICE evidence, paper, schemas, and claim boundaries.}\n"
        "\\end{table}\n\n"
        "\\noindent Next actions:\n\\begin{itemize}\n"
        f"{next_actions}\n"
        "\\end{itemize}\n"
    )


def render_artifact_svg(card: dict[str, Any]) -> str:
    stages = [
        ("available", _check_passed(card, "artifacts_available")),
        ("functional", _check_passed(card, "readiness_verifies") and _check_passed(card, "protocol_lock_verifies") and _check_passed(card, "design_card_verifies") and _check_passed(card, "contract_card_verifies") and _check_passed(card, "claim_card_verifies") and _check_passed(card, "evidence_bundle_verifies")),
        ("replayable", _check_passed(card, "reproduction_card_verifies")),
        ("reusable", _check_passed(card, "schemas_available")),
        ("honest", _check_passed(card, "claim_honesty") and _check_passed(card, "readiness_honesty")),
    ]
    width, height = 980, 280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE public artifact review card</text>',
        f'<text x="28" y="58" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Score {card["artifact_review_score"]}/100 | level {card["artifact_level"]} | claim allowed {str(card["claim_allowed"]).lower()}</text>',
    ]
    x0, y = 36, 96
    for idx, (label, passed) in enumerate(stages):
        x = x0 + idx * 184
        fill = "#059669" if passed else "#e5e7eb"
        text = "#ffffff" if passed else "#374151"
        parts.append(f'<rect x="{x}" y="{y}" width="138" height="58" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{x + 69}" y="{y + 36}" text-anchor="middle" font-family="Inter,Segoe UI,Arial" font-size="16" font-weight="700" fill="{text}">{label}</text>')
        if idx < len(stages) - 1:
            parts.append(f'<line x1="{x + 148}" y1="{y + 29}" x2="{x + 176}" y2="{y + 29}" stroke="#9ca3af" stroke-width="3"/>')
    parts.append(f'<text x="28" y="202" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#111827">Readiness {card["readiness_level"]} | artifact card hash {card["artifact_card_sha256"][:16]}...</text>')
    parts.append('<text x="28" y="228" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">Review packet only: S-tier still requires held-out remote live suite results and a passing claim card.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_artifact_outputs(card: dict[str, Any], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = out.with_suffix(".md")
    tex = out.with_suffix(".tex")
    svg = out.with_suffix(".svg")
    md.write_text(render_artifact_markdown(card), encoding="utf-8")
    tex.write_text(render_artifact_tex(card), encoding="utf-8")
    svg.write_text(render_artifact_svg(card), encoding="utf-8")
    return {"json": str(out), "markdown": str(md), "tex": str(tex), "svg": str(svg)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a deterministic TRICE public artifact review card.")
    ap.add_argument("--out", type=Path, default=Path("trice_artifact_card.json"))
    ap.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    ap.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    ap.add_argument("--reproduction", type=Path, default=DEFAULT_REPRODUCTION)
    ap.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    ap.add_argument("--install", type=Path, default=DEFAULT_INSTALL)
    ap.add_argument("--research", type=Path, default=DEFAULT_RESEARCH)
    ap.add_argument("--claim", type=Path, default=DEFAULT_CLAIM)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BROAD_BUNDLE)
    ap.add_argument("--paper-manifest", type=Path, default=DEFAULT_PAPER_MANIFEST)
    ap.add_argument("--paper-result", type=Path, default=DEFAULT_PAPER_RESULT)
    ap.add_argument("--paper-tex", type=Path, default=DEFAULT_PAPER_TEX)
    ap.add_argument("--paper-pdf", type=Path, default=DEFAULT_PAPER_PDF)
    ap.add_argument("--readme", type=Path, default=DEFAULT_README)
    ap.add_argument("--library-doc", type=Path, default=DEFAULT_LIBRARY_DOC)
    ap.add_argument("--format", choices=["json", "markdown", "tex"], default="json")
    args = ap.parse_args(argv)
    card = build_artifact_card(
        readiness_path=args.readiness,
        protocol_path=args.protocol,
        design_path=args.design,
        reproduction_path=args.reproduction,
        contract_path=args.contract,
        install_path=args.install,
        research_path=args.research,
        claim_path=args.claim,
        bundle_path=args.bundle,
        paper_manifest_path=args.paper_manifest,
        paper_result_path=args.paper_result,
        paper_tex_path=args.paper_tex,
        paper_pdf_path=args.paper_pdf,
        readme_path=args.readme,
        library_doc_path=args.library_doc,
    )
    outputs = write_artifact_outputs(card, args.out)
    if args.format == "markdown":
        print(render_artifact_markdown(card))
    elif args.format == "tex":
        print(render_artifact_tex(card))
    else:
        print(json.dumps({"artifact_card": card, "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if _check_passed(card, "artifacts_available") and _check_passed(card, "claim_honesty") else 1


def verify_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a deterministic TRICE public artifact review card.")
    ap.add_argument("artifact_card", type=Path)
    args = ap.parse_args(argv)
    verdict = verify_artifact_card_file(args.artifact_card)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def _availability_row(name: str, path: Path) -> dict[str, Any]:
    return {
        "name": name,
        "path": _display_path(path),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _schema_rows() -> list[dict[str, Any]]:
    names = [
        "trice_patch_spec.schema.json",
        "trice_evidence_manifest.schema.json",
        "trice_suite_manifest.schema.json",
        "trice_bundle_manifest.schema.json",
        "trice_adapter_profile.schema.json",
        "trice_run_receipt.schema.json",
        "trice_claim_card.schema.json",
        "trice_suite_readiness.schema.json",
        "trice_artifact_card.schema.json",
        "trice_protocol_lock.schema.json",
        "trice_design_card.schema.json",
        "trice_reproduction_card.schema.json",
        "trice_release_card.schema.json",
        "trice_contract_card.schema.json",
        "trice_release_evidence.schema.json",
        "trice_integrity_card.schema.json",
        "trice_crates_card.schema.json",
        "trice_install_card.schema.json",
        "trice_research_card.schema.json",
    ]
    return [_availability_row(name, REPO / "schemas" / name) for name in names]


def _safe_verify(fn) -> dict[str, Any]:
    try:
        verdict = fn()
    except Exception as exc:
        return {"ok": False, "errors": [str(exc)]}
    return verdict if isinstance(verdict, dict) else {"ok": False, "errors": ["verifier did not return a dict"]}


def _compact_verdict(verdict: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(verdict.get("ok")),
        "errors": list(verdict.get("errors") or [])[:10],
        **{k: verdict[k] for k in ("readiness_level", "protocol_level", "design_level", "reproduction_level", "claim_level", "claim_allowed", "entry_count") if k in verdict},
        **{k: verdict[k] for k in ("contract_level", "contract_score") if k in verdict},
        **{k: verdict[k] for k in ("install_level", "install_score") if k in verdict},
        **{k: verdict[k] for k in ("research_level", "research_score", "source_count") if k in verdict},
    }


def _paper_artifact_count(verdict: dict[str, Any]) -> int:
    manifest = verdict.get("manifest") if isinstance(verdict.get("manifest"), dict) else {}
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    return len(artifacts)


def _artifact_level(checks: list[dict[str, Any]]) -> str:
    passed = {row["name"]: bool(row["passed"]) for row in checks}
    if passed.get("artifacts_available") and passed.get("readiness_verifies") and passed.get("protocol_lock_verifies") and passed.get("design_card_verifies") and passed.get("reproduction_card_verifies") and passed.get("contract_card_verifies") and passed.get("install_card_verifies") and passed.get("research_card_verifies") and passed.get("claim_card_verifies") and passed.get("evidence_bundle_verifies") and passed.get("paper_manifest_verifies") and passed.get("schemas_available") and passed.get("claim_honesty") and passed.get("readiness_honesty"):
        return "review_ready_smoke"
    if passed.get("artifacts_available") and passed.get("readiness_verifies") and passed.get("protocol_lock_verifies") and passed.get("design_card_verifies") and passed.get("contract_card_verifies") and passed.get("install_card_verifies") and passed.get("research_card_verifies") and passed.get("claim_card_verifies"):
        return "partial_review_packet"
    return "not_review_ready"


def _artifact_score(checks: list[dict[str, Any]]) -> int:
    weights = {
        "artifacts_available": 8,
        "readiness_verifies": 7,
        "protocol_lock_verifies": 7,
        "design_card_verifies": 8,
        "reproduction_card_verifies": 8,
        "install_card_verifies": 7,
        "research_card_verifies": 13,
        "claim_card_verifies": 9,
        "contract_card_verifies": 8,
        "evidence_bundle_verifies": 8,
        "paper_manifest_verifies": 8,
        "schemas_available": 7,
        "claim_honesty": 6,
        "readiness_honesty": 4,
    }
    return min(100, sum(weights.get(row["name"], 0) for row in checks if row["passed"]))


def _next_actions(checks: list[dict[str, Any]]) -> list[str]:
    missing = [row["name"] for row in checks if not row["passed"]]
    if not missing:
        return [
            "Publish the artifact card with the release assets.",
            "Run the held-out remote pilot, then regenerate readiness, protocol, design, reproduction, contract, claim, bundle, paper, and artifact cards.",
            "Do not upgrade the README to S-tier passed until claim_allowed is true on held-out evidence.",
        ]
    actions = []
    if "evidence_bundle_verifies" in missing:
        actions.append("Regenerate and verify the .trice.zip evidence bundle.")
    if "paper_manifest_verifies" in missing:
        actions.append("Regenerate the LaTeX/PDF paper manifest from deterministic evidence.")
    if "protocol_lock_verifies" in missing:
        actions.append("Regenerate and verify the TRICE protocol lock before publishing the evidence packet.")
    if "design_card_verifies" in missing:
        actions.append("Regenerate and verify the TRICE statistical design card before publishing the evidence packet.")
    if "reproduction_card_verifies" in missing:
        actions.append("Regenerate and verify the TRICE reproduction card before publishing the evidence packet.")
    if "contract_card_verifies" in missing:
        actions.append("Regenerate and verify the TRICE public contract card before publishing the evidence packet.")
    if "install_card_verifies" in missing:
        actions.append("Rebuild the wheel, regenerate the installability card, and verify its bound hashes before publishing the evidence packet.")
    if "research_card_verifies" in missing:
        actions.append("Regenerate and verify the TRICE research card before publishing the paper or artifact packet.")
    if "schemas_available" in missing:
        actions.append("Ship all TRICE public contract schemas in the wheel.")
    if "claim_honesty" in missing or "readiness_honesty" in missing:
        actions.append("Fix public claim wording before release.")
    return actions or ["Fix missing artifacts before release."]


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def _check_passed(card: dict[str, Any], name: str) -> bool:
    return any(row["name"] == name and row["passed"] for row in card.get("checks", []))


def _present_count(rows: list[dict[str, Any]]) -> str:
    return f"{sum(1 for row in rows if row['present'])}/{len(rows)}"


def _without_card_hash(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out.pop("artifact_card_sha256", None)
    return out


def _verify_bound_rows(
    card_path: Path,
    raw_rows: Any,
    label: str,
    errors: list[str],
) -> list[str]:
    if not isinstance(raw_rows, list):
        errors.append(f"{label} must be a list")
        return []
    checked: list[str] = []
    for idx, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            errors.append(f"{label}[{idx}] must be an object")
            continue
        name = str(raw.get("name") or f"{label}[{idx}]")
        raw_path = raw.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"{name} path is missing")
            continue
        path = _resolve_bound_path(card_path, raw_path)
        present = path.is_file()
        expected_present = bool(raw.get("present"))
        row_errors = False
        if present != expected_present:
            errors.append(f"{name} present mismatch")
            row_errors = True
        actual_bytes = path.stat().st_size if present else 0
        if int(raw.get("bytes") or 0) != actual_bytes:
            errors.append(f"{name} byte count mismatch")
            row_errors = True
        expected_sha = raw.get("sha256")
        actual_sha = sha256_file(path) if present else None
        if expected_sha != actual_sha:
            errors.append(f"{name} sha256 mismatch")
            row_errors = True
        if not row_errors:
            checked.append(name)
    return checked


def _verify_input_hashes(card_path: Path, card: dict[str, Any], errors: list[str]) -> None:
    input_hashes = card.get("input_sha256")
    if not isinstance(input_hashes, dict):
        errors.append("input_sha256 must be an object")
        return
    availability = card.get("availability") if isinstance(card.get("availability"), list) else []
    paths_by_name = {row.get("name"): row.get("path") for row in availability if isinstance(row, dict)}
    for name, expected_sha in sorted(input_hashes.items()):
        raw_path = paths_by_name.get(name)
        if not raw_path:
            errors.append(f"input_sha256 {name} has no availability path")
            continue
        path = _resolve_bound_path(card_path, str(raw_path))
        actual_sha = sha256_file(path) if path.is_file() else None
        if expected_sha != actual_sha:
            errors.append(f"input_sha256 {name} mismatch")


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
