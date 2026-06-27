"""Reproduction cards for TRICE public evidence packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .bundle import verify_evidence_bundle
from .claim import verify_claim_card_file
from .design import verify_design_card_file
from .evidence import canonical_json, sha256_file, verify_manifest
from .protocol import verify_protocol_lock_file
from .readiness import verify_readiness_file

REPRODUCTION_SCHEMA_VERSION = "trice-reproduction-card/v1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_READINESS = REPO / "docs" / "trice_suite_readiness.json"
DEFAULT_SUITE_MANIFEST = REPO / "examples" / "trice_suite_bundled_live.json"
DEFAULT_PROTOCOL = REPO / "docs" / "trice_protocol_lock.json"
DEFAULT_DESIGN = REPO / "docs" / "trice_design_card.json"
DEFAULT_BROAD_RESULT = REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_suite_results.json"
DEFAULT_BROAD_MANIFEST = REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_suite_evidence_manifest.json"
DEFAULT_BROAD_BUNDLE = REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_broad_smoke_evidence.trice.zip"
DEFAULT_CLAIM = REPO / "docs" / "trice_claim_card.json"
DEFAULT_PAPER_MANIFEST = REPO / "paper" / "trice_v3_research_manifest.json"
DEFAULT_PAPER_RESULT = REPO / "benchmark" / "trice" / "results" / "v2-smoke" / "trice_v2_live_results.json"


def build_reproduction_card(
    *,
    readiness_path: str | Path = DEFAULT_READINESS,
    suite_manifest_path: str | Path = DEFAULT_SUITE_MANIFEST,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    design_path: str | Path = DEFAULT_DESIGN,
    broad_result_path: str | Path = DEFAULT_BROAD_RESULT,
    broad_manifest_path: str | Path = DEFAULT_BROAD_MANIFEST,
    broad_bundle_path: str | Path = DEFAULT_BROAD_BUNDLE,
    claim_path: str | Path = DEFAULT_CLAIM,
    paper_manifest_path: str | Path = DEFAULT_PAPER_MANIFEST,
    paper_result_path: str | Path = DEFAULT_PAPER_RESULT,
) -> dict[str, Any]:
    """Build a deterministic reviewer runbook for the TRICE evidence packet."""

    paths = {
        "readiness": Path(readiness_path),
        "suite_manifest": Path(suite_manifest_path),
        "protocol_lock": Path(protocol_path),
        "design_card": Path(design_path),
        "broad_suite_result": Path(broad_result_path),
        "broad_suite_manifest": Path(broad_manifest_path),
        "broad_evidence_bundle": Path(broad_bundle_path),
        "claim_card": Path(claim_path),
        "paper_manifest": Path(paper_manifest_path),
        "paper_result": Path(paper_result_path),
    }
    verdicts = {
        "readiness": _safe_verify(lambda: verify_readiness_file(paths["readiness"], manifest_path=paths["suite_manifest"])),
        "protocol_lock": _safe_verify(lambda: verify_protocol_lock_file(paths["protocol_lock"], manifest_path=paths["suite_manifest"])),
        "design_card": _safe_verify(lambda: verify_design_card_file(paths["design_card"], protocol_path=paths["protocol_lock"], suite_result_path=paths["broad_suite_result"])),
        "claim_card": _safe_verify(lambda: verify_claim_card_file(paths["claim_card"], suite_result_path=paths["broad_suite_result"], manifest_path=paths["broad_suite_manifest"])),
        "evidence_bundle": _safe_verify(lambda: verify_evidence_bundle(paths["broad_evidence_bundle"])),
        "paper_manifest": _safe_verify(lambda: verify_manifest(paths["paper_manifest"], paths["paper_result"])),
    }
    checks = [
        _check("inputs_available", all(path.is_file() for path in paths.values()), _present_count(paths), f"{len(paths)}/{len(paths)} inputs present"),
        _check("readiness_reproduces", bool(verdicts["readiness"].get("ok")), verdicts["readiness"].get("readiness_level"), "readiness verifier ok"),
        _check("protocol_reproduces", bool(verdicts["protocol_lock"].get("ok")), verdicts["protocol_lock"].get("protocol_level"), "protocol verifier ok"),
        _check("design_reproduces", bool(verdicts["design_card"].get("ok")), verdicts["design_card"].get("design_level"), "design verifier ok"),
        _check("claim_reproduces", bool(verdicts["claim_card"].get("ok")), verdicts["claim_card"].get("claim_level"), "claim-card verifier ok"),
        _check("bundle_reproduces", bool(verdicts["evidence_bundle"].get("ok")), verdicts["evidence_bundle"].get("entry_count"), "bundle verifier ok"),
        _check("paper_reproduces", bool(verdicts["paper_manifest"].get("ok")), _paper_artifact_count(verdicts["paper_manifest"]), "paper manifest verifier ok"),
    ]
    card = {
        "schema_version": REPRODUCTION_SCHEMA_VERSION,
        "reproduction_level": _reproduction_level(checks),
        "reproduction_score": _reproduction_score(checks),
        "claim_allowed": False,
        "python_executable": "python",
        "inputs": {
            name: {
                "path": _display_path(path),
                "present": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "commands": _commands(paths),
        "checks": checks,
        "verdicts": {name: _compact_verdict(verdict) for name, verdict in verdicts.items()},
        "research_basis": [
            "Reproduction should be possible from documented commands, not only from prose.",
            "FAIR and RO-Crate-style packaging motivate explicit inventories, metadata, and machine-checkable paths.",
            "Artifact-review practice separates availability, functionality, and independent reproduction.",
            "TRICE reproduction cards bind commands and input hashes while leaving S-tier outcome claims to claim cards.",
        ],
        "next_actions": _next_actions(checks),
    }
    card["reproduction_card_sha256"] = hashlib.sha256(canonical_json(_without_reproduction_hash(card)).encode("utf-8")).hexdigest()
    return card


def verify_reproduction_card_file(path: str | Path) -> dict[str, Any]:
    """Verify a reproduction card's self hash, input hashes, and verifier outputs."""

    card_path = Path(path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if card.get("schema_version") != REPRODUCTION_SCHEMA_VERSION:
        errors.append(f"schema_version must be {REPRODUCTION_SCHEMA_VERSION}")
    expected_hash = str(card.get("reproduction_card_sha256") or "")
    actual_hash = hashlib.sha256(canonical_json(_without_reproduction_hash(card)).encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        errors.append("reproduction_card_sha256 mismatch")

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

    if _required_inputs_present(resolved):
        rebuilt = build_reproduction_card(
            readiness_path=resolved["readiness"],
            suite_manifest_path=resolved["suite_manifest"],
            protocol_path=resolved["protocol_lock"],
            design_path=resolved["design_card"],
            broad_result_path=resolved["broad_suite_result"],
            broad_manifest_path=resolved["broad_suite_manifest"],
            broad_bundle_path=resolved["broad_evidence_bundle"],
            claim_path=resolved["claim_card"],
            paper_manifest_path=resolved["paper_manifest"],
            paper_result_path=resolved["paper_result"],
        )
        if canonical_json(_without_reproduction_hash(rebuilt)) != canonical_json(_without_reproduction_hash(card)):
            errors.append("reproduction card does not match deterministic rebuild from bound inputs")

    return {
        "ok": not errors,
        "schema_version": card.get("schema_version"),
        "reproduction_level": card.get("reproduction_level"),
        "reproduction_score": card.get("reproduction_score"),
        "claim_allowed": bool(card.get("claim_allowed")),
        "reproduction_card_sha256": expected_hash,
        "computed_reproduction_card_sha256": actual_hash,
        "checked_inputs": checked_inputs,
        "errors": errors,
    }


def render_reproduction_markdown(card: dict[str, Any]) -> str:
    lines = [
        "# TRICE Reproduction Card",
        "",
        f"- Reproduction level: `{card['reproduction_level']}`",
        f"- Reproduction score: **{card['reproduction_score']}/100**",
        f"- Claim allowed: `{str(card['claim_allowed']).lower()}`",
        "",
        "## Checks",
        "",
        "| Check | Passed | Observed | Required |",
        "|---|---:|---|---|",
    ]
    for row in card["checks"]:
        lines.append(f"| {row['name']} | {'yes' if row['passed'] else 'no'} | {_md(row['observed'])} | {_md(row['required'])} |")
    lines.extend(["", "## Commands", ""])
    for command in card["commands"]:
        lines.append(f"- `{command['name']}`: `{command['command']}`")
    lines.extend(["", "## Hash", "", f"- reproduction card: `{card['reproduction_card_sha256']}`", ""])
    return "\n".join(lines)


def render_reproduction_tex(card: dict[str, Any]) -> str:
    rows = "\n".join(
        f"{_tex(row['name'])} & {'yes' if row['passed'] else 'no'} & {_tex(row['required'])} \\\\"
        for row in card["checks"]
    )
    return (
        "\\section{Reproduction Card}\n"
        f"Reproduction level: \\texttt{{{_tex(card['reproduction_level'])}}}; "
        f"score: {card['reproduction_score']}/100; "
        f"claim allowed: {'yes' if card['claim_allowed'] else 'no'}.\n\n"
        "\\begin{table}[h]\n\\centering\n"
        "\\begin{tabular}{lrl}\n\\toprule\nCheck & Passed & Required \\\\\n\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{TRICE reviewer commands and bound-input reproduction checks.}\n"
        "\\end{table}\n"
    )


def render_reproduction_svg(card: dict[str, Any]) -> str:
    stages = [
        ("inputs", _check_passed(card, "inputs_available")),
        ("cards", _check_passed(card, "readiness_reproduces") and _check_passed(card, "protocol_reproduces") and _check_passed(card, "design_reproduces") and _check_passed(card, "claim_reproduces")),
        ("bundle", _check_passed(card, "bundle_reproduces")),
        ("paper", _check_passed(card, "paper_reproduces")),
    ]
    width, height = 940, 280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE reproduction card</text>',
        f'<text x="28" y="58" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Score {card["reproduction_score"]}/100 | level {card["reproduction_level"]} | claim allowed {str(card["claim_allowed"]).lower()}</text>',
    ]
    x0, y = 52, 96
    for idx, (label, passed) in enumerate(stages):
        x = x0 + idx * 216
        fill = "#4338ca" if passed else "#e5e7eb"
        text = "#ffffff" if passed else "#374151"
        parts.append(f'<rect x="{x}" y="{y}" width="164" height="58" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{x + 82}" y="{y + 36}" text-anchor="middle" font-family="Inter,Segoe UI,Arial" font-size="17" font-weight="700" fill="{text}">{label}</text>')
        if idx < len(stages) - 1:
            parts.append(f'<line x1="{x + 174}" y1="{y + 29}" x2="{x + 208}" y2="{y + 29}" stroke="#9ca3af" stroke-width="3"/>')
    parts.append(f'<text x="28" y="202" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#111827">Commands {len(card["commands"])} | inputs {len(card["inputs"])} | hash {card["reproduction_card_sha256"][:16]}...</text>')
    parts.append('<text x="28" y="228" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">Reproduction packet only: S-tier requires held-out live claim results and a passing claim card.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_reproduction_outputs(card: dict[str, Any], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = out.with_suffix(".md")
    tex = out.with_suffix(".tex")
    svg = out.with_suffix(".svg")
    md.write_text(render_reproduction_markdown(card), encoding="utf-8")
    tex.write_text(render_reproduction_tex(card), encoding="utf-8")
    svg.write_text(render_reproduction_svg(card), encoding="utf-8")
    return {"json": str(out), "markdown": str(md), "tex": str(tex), "svg": str(svg)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a deterministic TRICE reproduction card.")
    ap.add_argument("--out", type=Path, default=Path("trice_reproduction_card.json"))
    ap.add_argument("--format", choices=["json", "markdown", "tex"], default="json")
    args = ap.parse_args(argv)
    card = build_reproduction_card()
    outputs = write_reproduction_outputs(card, args.out)
    if args.format == "markdown":
        print(render_reproduction_markdown(card))
    elif args.format == "tex":
        print(render_reproduction_tex(card))
    else:
        print(json.dumps({"reproduction_card": card, "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if card["reproduction_level"] != "not_reproducible" else 1


def verify_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a deterministic TRICE reproduction card.")
    ap.add_argument("reproduction_card", type=Path)
    args = ap.parse_args(argv)
    verdict = verify_reproduction_card_file(args.reproduction_card)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def _commands(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return [
        _command("verify-readiness", f"python -m tracerazor.trice suite verify-readiness {_display_path(paths['readiness'])} --manifest {_display_path(paths['suite_manifest'])}", "Verify suite readiness card."),
        _command("verify-protocol", f"python -m tracerazor.trice verify-protocol {_display_path(paths['protocol_lock'])} --manifest {_display_path(paths['suite_manifest'])}", "Verify protocol lock."),
        _command("verify-design", f"python -m tracerazor.trice verify-design {_display_path(paths['design_card'])} --protocol {_display_path(paths['protocol_lock'])} --suite-result {_display_path(paths['broad_suite_result'])}", "Verify statistical design card."),
        _command("verify-claim", f"python -m tracerazor.trice verify-claim {_display_path(paths['claim_card'])} --suite-result {_display_path(paths['broad_suite_result'])} --manifest {_display_path(paths['broad_suite_manifest'])}", "Verify deterministic claim card."),
        _command("verify-bundle", f"python -m tracerazor.trice verify-bundle {_display_path(paths['broad_evidence_bundle'])}", "Verify portable evidence bundle."),
        _command("verify-paper", f"python -m tracerazor.trice verify {_display_path(paths['paper_manifest'])} --result {_display_path(paths['paper_result'])}", "Verify paper manifest and generated artifacts."),
        _command("verify-artifact", "python -m tracerazor.trice verify-artifact docs/trice_artifact_card.json", "Verify final artifact card after it has been regenerated."),
    ]


def _command(name: str, command: str, purpose: str) -> dict[str, Any]:
    return {
        "name": name,
        "command": command,
        "purpose": purpose,
        "sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
    }


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
        **{k: verdict[k] for k in ("readiness_level", "protocol_level", "design_level", "claim_level", "entry_count") if k in verdict},
    }


def _paper_artifact_count(verdict: dict[str, Any]) -> int:
    manifest = verdict.get("manifest") if isinstance(verdict.get("manifest"), dict) else {}
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    return len(artifacts)


def _reproduction_level(checks: list[dict[str, Any]]) -> str:
    passed = {row["name"]: bool(row["passed"]) for row in checks}
    if all(passed.values()):
        return "reviewer_replay_ready_smoke"
    if passed.get("inputs_available") and passed.get("bundle_reproduces"):
        return "partial_reproduction_packet"
    return "not_reproducible"


def _reproduction_score(checks: list[dict[str, Any]]) -> int:
    weights = {
        "inputs_available": 16,
        "readiness_reproduces": 12,
        "protocol_reproduces": 13,
        "design_reproduces": 13,
        "claim_reproduces": 13,
        "bundle_reproduces": 17,
        "paper_reproduces": 16,
    }
    return min(100, sum(weights.get(row["name"], 0) for row in checks if row["passed"]))


def _next_actions(checks: list[dict[str, Any]]) -> list[str]:
    missing = [row["name"] for row in checks if not row["passed"]]
    if not missing:
        return [
            "Publish the reproduction card with release assets.",
            "Run the listed commands in a clean wheel install before public release.",
            "Do not claim S-tier until held-out claim evidence and claim card pass.",
        ]
    return [f"Fix reproduction check: {name}" for name in missing]


def _required_inputs_present(paths: dict[str, Path]) -> bool:
    return all(name in paths and paths[name].is_file() for name in (
        "readiness",
        "suite_manifest",
        "protocol_lock",
        "design_card",
        "broad_suite_result",
        "broad_suite_manifest",
        "broad_evidence_bundle",
        "claim_card",
        "paper_manifest",
        "paper_result",
    ))


def _present_count(paths: dict[str, Path]) -> str:
    return f"{sum(1 for path in paths.values() if path.is_file())}/{len(paths)}"


def _check_passed(card: dict[str, Any], name: str) -> bool:
    return any(row["name"] == name and row["passed"] for row in card.get("checks", []))


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def _without_reproduction_hash(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out.pop("reproduction_card_sha256", None)
    return out


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
