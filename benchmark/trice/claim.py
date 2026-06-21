"""Deterministic claim cards for TRICE evidence bundles and suites."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .evidence import canonical_json
from .suite import verify_suite_evidence

CLAIM_SCHEMA_VERSION = "trice-claim-card/v1"
DEFAULT_SCOPE = "python software-repair/context-control tasks on held-out Git repositories"
REPO = Path(__file__).resolve().parents[2]


def build_claim_card(
    suite_result_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    scope: str = DEFAULT_SCOPE,
) -> dict[str, Any]:
    """Build a deterministic, machine-readable claim boundary from suite evidence."""

    result_path = Path(suite_result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    gate = result.get("claim_gate") or {}
    s_gate = gate.get("s_tier_gate") or {}
    manifest = _resolve_manifest(result_path, result, manifest_path)
    verification = verify_suite_evidence(manifest) if manifest and manifest.is_file() else {"ok": False, "errors": ["missing suite evidence manifest"]}
    requirements = s_gate.get("requirements") if isinstance(s_gate.get("requirements"), dict) else {}
    requirement_rows = _requirement_rows(requirements)
    s_tier_passed = bool(s_gate.get("passed")) and bool(verification.get("ok"))
    claim_level = _claim_level(gate, s_gate, verification)
    metrics = _metrics(gate)
    score = _determinism_contract_score(requirement_rows, gate, verification)
    card = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "scope": scope,
        "claim_level": claim_level,
        "claim_allowed": s_tier_passed,
        "determinism_contract_score": score,
        "input_sha256": {
            "suite_result": _sha256_file(result_path),
            "suite_manifest": _sha256_file(manifest) if manifest and manifest.is_file() else None,
        },
        "source": {
            "suite_result_path": _display_path(result_path),
            "suite_manifest_path": _display_path(manifest) if manifest else None,
            "suite_name": (result.get("suite") or {}).get("name"),
            "algorithm": result.get("algorithm"),
        },
        "metrics": metrics,
        "requirements": requirement_rows,
        "verification": {
            "ok": bool(verification.get("ok")),
            "error_count": len(verification.get("errors") or []),
            "errors": list(verification.get("errors") or [])[:10],
        },
        "non_claims": _non_claims(claim_level, s_gate, verification),
    }
    card["claim_card_sha256"] = hashlib.sha256(canonical_json(_without_card_hash(card)).encode("utf-8")).hexdigest()
    return card


def verify_claim_card_file(
    claim_card_path: str | Path,
    *,
    suite_result_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a claim card's self hash and, when available, bound input hashes."""

    card_path = Path(claim_card_path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if card.get("schema_version") != CLAIM_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CLAIM_SCHEMA_VERSION}")

    expected_card_hash = str(card.get("claim_card_sha256") or "")
    actual_card_hash = hashlib.sha256(canonical_json(_without_card_hash(card)).encode("utf-8")).hexdigest()
    if actual_card_hash != expected_card_hash:
        errors.append("claim_card_sha256 mismatch")

    checked_inputs: list[str] = []
    result_file = _resolve_bound_path(card_path, card, "suite_result_path", suite_result_path)
    if result_file is not None:
        expected = ((card.get("input_sha256") or {}).get("suite_result") or "")
        if not result_file.is_file():
            errors.append(f"suite result file not found: {_display_path(result_file)}")
        elif _sha256_file(result_file) != expected:
            errors.append("suite_result sha256 mismatch")
        else:
            checked_inputs.append("suite_result")

    manifest_file = _resolve_bound_path(card_path, card, "suite_manifest_path", manifest_path)
    expected_manifest = (card.get("input_sha256") or {}).get("suite_manifest")
    if manifest_file is not None or expected_manifest:
        if manifest_file is None:
            errors.append("suite manifest path is missing")
        elif not manifest_file.is_file():
            errors.append(f"suite manifest file not found: {_display_path(manifest_file)}")
        elif _sha256_file(manifest_file) != expected_manifest:
            errors.append("suite_manifest sha256 mismatch")
        else:
            checked_inputs.append("suite_manifest")

    return {
        "ok": not errors,
        "schema_version": card.get("schema_version"),
        "claim_level": card.get("claim_level"),
        "claim_allowed": bool(card.get("claim_allowed")),
        "claim_card_sha256": expected_card_hash,
        "computed_claim_card_sha256": actual_card_hash,
        "checked_inputs": checked_inputs,
        "errors": errors,
    }


def render_claim_card_markdown(card: dict[str, Any]) -> str:
    metrics = card["metrics"]
    lines = [
        "# TRICE Claim Card",
        "",
        f"- Scope: `{card['scope']}`",
        f"- Claim level: `{card['claim_level']}`",
        f"- Claim allowed: `{str(card['claim_allowed']).lower()}`",
        f"- Determinism contract score: **{card['determinism_contract_score']}/100**",
        f"- Mean input-token savings: **{100 * metrics['mean_input_token_savings']:.1f}%**",
        f"- Clustered CI lower bound: **{100 * metrics['clustered_savings_ci_low']:.1f}%**",
        f"- Pass regressions: **{metrics['pass_regressions']}**",
        f"- Accepted runs: **{metrics['accepted_rounds']}/{metrics['total_rounds']}**",
        f"- Evidence verification: **{'ok' if card['verification']['ok'] else 'failed'}**",
        "",
        "## Requirements",
        "",
        "| Requirement | Passed | Observed | Required |",
        "|---|---:|---|---|",
    ]
    for row in card["requirements"]:
        lines.append(
            f"| {row['name']} | {'yes' if row['passed'] else 'no'} | "
            f"{_md(row['observed'])} | {_md(row['required'])} |"
        )
    lines.extend(["", "## Non-Claims", ""])
    for item in card["non_claims"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Hashes",
            "",
            f"- suite result: `{card['input_sha256']['suite_result']}`",
            f"- suite manifest: `{card['input_sha256']['suite_manifest']}`",
            f"- claim card: `{card['claim_card_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_claim_card_tex(card: dict[str, Any]) -> str:
    metrics = card["metrics"]
    rows = "\n".join(
        f"{_tex(row['name'])} & {'yes' if row['passed'] else 'no'} & {_tex(row['required'])} \\\\"
        for row in card["requirements"]
    )
    non_claims = "\n".join(f"\\item {_tex(item)}" for item in card["non_claims"])
    return (
        "\\section{Deterministic Claim Card}\n"
        f"Scope: \\texttt{{{_tex(card['scope'])}}}. "
        f"Claim level: \\texttt{{{_tex(card['claim_level'])}}}. "
        f"Claim allowed: {'yes' if card['claim_allowed'] else 'no'}. "
        f"Determinism contract score: {card['determinism_contract_score']}/100.\n\n"
        "\\begin{table}[h]\n\\centering\n"
        "\\begin{tabular}{lrl}\n\\toprule\nRequirement & Passed & Required \\\\\n\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{Machine-readable claim gate distilled from the suite result and evidence manifest.}\n"
        "\\end{table}\n\n"
        f"Mean input-token savings is {100 * metrics['mean_input_token_savings']:.1f}\\%; "
        f"clustered CI lower bound is {100 * metrics['clustered_savings_ci_low']:.1f}\\%; "
        f"pass regressions: {metrics['pass_regressions']}. "
        f"Claim-card hash: \\texttt{{{card['claim_card_sha256'][:16]}...}}.\n\n"
        "\\noindent Non-claims:\n\\begin{itemize}\n"
        f"{non_claims}\n"
        "\\end{itemize}\n"
    )


def render_claim_ladder_svg(card: dict[str, Any]) -> str:
    stages = [
        ("smoke", card["metrics"]["total_rounds"] >= 1 and card["metrics"]["pass_regressions"] == 0),
        ("pilot", card["metrics"]["task_cluster_count"] >= 10 and card["metrics"]["replicate_count"] >= 20),
        ("s-tier", bool(card["claim_allowed"])),
    ]
    width, height = 820, 260
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE deterministic claim ladder</text>',
        f'<text x="28" y="58" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Score {card["determinism_contract_score"]}/100 | claim level {card["claim_level"]}</text>',
    ]
    x0, y = 70, 98
    for idx, (label, passed) in enumerate(stages):
        x = x0 + idx * 230
        fill = "#059669" if passed else "#e5e7eb"
        text = "#ffffff" if passed else "#374151"
        parts.append(f'<rect x="{x}" y="{y}" width="170" height="56" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{x + 85}" y="{y + 35}" text-anchor="middle" font-family="Inter,Segoe UI,Arial" font-size="18" font-weight="700" fill="{text}">{label}</text>')
        if idx < len(stages) - 1:
            parts.append(f'<line x1="{x + 178}" y1="{y + 28}" x2="{x + 220}" y2="{y + 28}" stroke="#9ca3af" stroke-width="3"/>')
    metrics = card["metrics"]
    parts.append(f'<text x="28" y="196" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#111827">Mean savings {100 * metrics["mean_input_token_savings"]:.1f}% | clustered lower CI {100 * metrics["clustered_savings_ci_low"]:.1f}% | pass regressions {metrics["pass_regressions"]}</text>')
    parts.append(f'<text x="28" y="220" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">S-tier requires 50 remote Git task clusters, 3 replicates each, adapter profiles, valid receipts, and verified evidence.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_claim_outputs(card: dict[str, Any], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = out.with_suffix(".md")
    tex = out.with_suffix(".tex")
    svg = out.with_suffix(".svg")
    md.write_text(render_claim_card_markdown(card), encoding="utf-8")
    tex.write_text(render_claim_card_tex(card), encoding="utf-8")
    svg.write_text(render_claim_ladder_svg(card), encoding="utf-8")
    return {"json": str(out), "markdown": str(md), "tex": str(tex), "svg": str(svg)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a deterministic TRICE claim card.")
    ap.add_argument("--suite-result", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("trice_claim_card.json"))
    ap.add_argument("--scope", default=DEFAULT_SCOPE)
    ap.add_argument("--format", choices=["json", "markdown", "tex"], default="json")
    args = ap.parse_args(argv)
    card = build_claim_card(args.suite_result, manifest_path=args.manifest, scope=args.scope)
    outputs = write_claim_outputs(card, args.out)
    if args.format == "markdown":
        print(render_claim_card_markdown(card))
    elif args.format == "tex":
        print(render_claim_card_tex(card))
    else:
        print(json.dumps({"claim_card": card, "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if card["verification"]["ok"] else 1


def _resolve_manifest(result_path: Path, result: dict[str, Any], manifest_path: str | Path | None) -> Path | None:
    if manifest_path is not None:
        return Path(manifest_path)
    rel = result.get("manifest_path")
    if isinstance(rel, str) and rel:
        p = Path(rel)
        return p if p.is_absolute() else result_path.parent / p
    return None


def _metrics(gate: dict[str, Any]) -> dict[str, Any]:
    clustered = gate.get("clustered_savings_ci") or gate.get("savings_ci") or {}
    return {
        "mean_input_token_savings": float(gate.get("mean_savings") or 0.0),
        "clustered_savings_ci_low": float(clustered.get("low") or 0.0),
        "clustered_savings_ci_high": float(clustered.get("high") or 0.0),
        "target_savings": float(gate.get("target_savings") or 0.60),
        "pass_regressions": int(gate.get("pass_regressions") or 0),
        "accepted_rounds": int(gate.get("accepted_rounds") or 0),
        "total_rounds": int(gate.get("total_rounds") or 0),
        "replicate_count": int(gate.get("replicate_count") or 0),
        "task_cluster_count": int(gate.get("task_cluster_count") or 0),
        "trice_pass_rate": float(gate.get("trice_pass_rate") or 0.0),
    }


def _requirement_rows(requirements: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, raw in sorted(requirements.items()):
        if isinstance(raw, dict):
            rows.append(
                {
                    "name": name,
                    "passed": bool(raw.get("passed")),
                    "observed": raw.get("observed"),
                    "required": raw.get("required"),
                }
            )
    return rows


def _claim_level(gate: dict[str, Any], s_gate: dict[str, Any], verification: dict[str, Any]) -> str:
    if bool(s_gate.get("passed")) and bool(verification.get("ok")):
        return "s_tier"
    if int(gate.get("task_cluster_count") or 0) >= 10 and int(gate.get("pass_regressions") or 0) == 0:
        return "pilot"
    if bool(gate.get("smoke_gate_passed")) and bool(verification.get("ok")):
        return "smoke"
    return "failed"


def _determinism_contract_score(rows: list[dict[str, Any]], gate: dict[str, Any], verification: dict[str, Any]) -> int:
    checks = {
        "evidence_verifies": bool(verification.get("ok")),
        "zero_pass_regressions": int(gate.get("pass_regressions") or 0) == 0,
        "all_runs_accepted": int(gate.get("accepted_rounds") or 0) == int(gate.get("total_rounds") or -1),
        "target_savings_met": float(gate.get("mean_savings") or 0.0) >= float(gate.get("target_savings") or 0.60),
        "clustered_ci_met": float((gate.get("clustered_savings_ci") or gate.get("savings_ci") or {}).get("low") or 0.0)
        >= float(gate.get("target_savings") or 0.60),
    }
    score = sum(12 for passed in checks.values() if passed)
    score += sum(4 for row in rows if row["passed"])
    return min(100, score)


def _non_claims(claim_level: str, s_gate: dict[str, Any], verification: dict[str, Any]) -> list[str]:
    items = []
    if claim_level != "s_tier":
        missing = s_gate.get("missing_requirements") or []
        if missing:
            items.append("Not an S-tier claim; missing " + ", ".join(str(x) for x in missing) + ".")
        else:
            items.append("Not an S-tier claim; suite gate did not pass.")
    if not verification.get("ok"):
        items.append("Evidence verification did not pass.")
    items.append("Does not claim universal all-language or all-agent performance.")
    items.append("Does not certify replay-only savings as live savings.")
    return items


def _without_card_hash(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out.pop("claim_card_sha256", None)
    return out


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_bound_path(
    card_path: Path,
    card: dict[str, Any],
    source_key: str,
    explicit: str | Path | None,
) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    source = card.get("source") if isinstance(card.get("source"), dict) else {}
    raw = source.get(source_key)
    if not raw:
        return None
    candidate = Path(str(raw))
    if candidate.is_absolute():
        return candidate
    repo_candidate = REPO / candidate
    if repo_candidate.exists():
        return repo_candidate
    card_relative = card_path.parent / candidate
    if card_relative.exists():
        return card_relative
    return repo_candidate


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
