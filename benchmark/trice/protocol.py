"""Deterministic protocol locks for TRICE live evaluation claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .evidence import canonical_json
from .readiness import build_suite_readiness
from .suite import load_suite_manifest

PROTOCOL_SCHEMA_VERSION = "trice-protocol-lock/v1"
DEFAULT_SCOPE = "python software-repair/context-control tasks on held-out Git repositories"
REPO = Path(__file__).resolve().parents[2]


def build_protocol_lock(
    manifest_path: str | Path,
    *,
    scope: str = DEFAULT_SCOPE,
    protocol_id: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic pre-registration contract for a TRICE suite."""

    path = Path(manifest_path)
    manifest = load_suite_manifest(path)
    readiness = build_suite_readiness(path, scope=scope)
    gate = _gate_config(manifest)
    tasks = manifest["tasks"]
    replicates = readiness["planned_execution"]["replicates_per_task"]
    checks = _checks(manifest, readiness, gate)
    card = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": protocol_id or _default_protocol_id(path, manifest),
        "scope": scope,
        "protocol_level": _protocol_level(checks),
        "protocol_score": _protocol_score(checks),
        "claim_allowed_by_protocol": _protocol_level(checks) == "claim_protocol_ready",
        "input_sha256": {
            "suite_manifest": _sha256_file(path),
            "readiness_preflight": readiness.get("readiness_sha256"),
        },
        "source": {
            "suite_manifest_path": _display_path(path),
            "suite_name": manifest.get("name") or path.stem,
            "suite_schema_version": manifest.get("schema_version"),
        },
        "suite_shape": {
            "task_clusters": len(tasks),
            "planned_runs": readiness["planned_execution"]["planned_runs"],
            "rounds_per_run": readiness["planned_execution"]["rounds_per_run"],
            "replicates_per_task": replicates,
            "remote_git_task_count": readiness["planned_execution"]["remote_git_task_count"],
            "adapter_profile_count": readiness["planned_execution"]["adapter_profile_count"],
        },
        "evaluation_contract": _evaluation_contract(gate),
        "checks": checks,
        "research_basis": [
            "Agent benchmarks must evaluate cost and task success together, not accuracy alone.",
            "Held-out real repositories and locked commits reduce benchmark shortcutting and overfitting.",
            "Artifact review expects documented, complete, exercisable, and independently checkable evidence.",
            "The protocol lock is pre-outcome evidence: it freezes thresholds before a claim run is executed.",
        ],
        "non_claims": _non_claims(checks),
        "next_actions": _next_actions(checks, gate),
    }
    card["protocol_lock_sha256"] = hashlib.sha256(canonical_json(_without_protocol_hash(card)).encode("utf-8")).hexdigest()
    return card


def verify_protocol_lock_file(
    protocol_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a protocol lock's self hash and bound suite manifest hash."""

    path = Path(protocol_path)
    card = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if card.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PROTOCOL_SCHEMA_VERSION}")

    expected_hash = str(card.get("protocol_lock_sha256") or "")
    actual_hash = hashlib.sha256(canonical_json(_without_protocol_hash(card)).encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        errors.append("protocol_lock_sha256 mismatch")

    checked_inputs: list[str] = []
    suite_manifest = _resolve_bound_manifest(path, card, manifest_path)
    expected_manifest = (card.get("input_sha256") or {}).get("suite_manifest")
    if suite_manifest is None:
        errors.append("suite manifest path is missing")
    elif not suite_manifest.is_file():
        errors.append(f"suite manifest file not found: {_display_path(suite_manifest)}")
    elif _sha256_file(suite_manifest) != expected_manifest:
        errors.append("suite_manifest sha256 mismatch")
    else:
        checked_inputs.append("suite_manifest")
        rebuilt = build_protocol_lock(
            suite_manifest,
            scope=str(card.get("scope") or DEFAULT_SCOPE),
            protocol_id=str(card.get("protocol_id") or ""),
        )
        if canonical_json(_without_protocol_hash(rebuilt)) != canonical_json(_without_protocol_hash(card)):
            errors.append("protocol lock does not match deterministic rebuild from suite manifest")

    return {
        "ok": not errors,
        "schema_version": card.get("schema_version"),
        "protocol_id": card.get("protocol_id"),
        "protocol_level": card.get("protocol_level"),
        "protocol_score": card.get("protocol_score"),
        "claim_allowed_by_protocol": bool(card.get("claim_allowed_by_protocol")),
        "protocol_lock_sha256": expected_hash,
        "computed_protocol_lock_sha256": actual_hash,
        "checked_inputs": checked_inputs,
        "errors": errors,
    }


def render_protocol_markdown(card: dict[str, Any]) -> str:
    contract = card["evaluation_contract"]
    shape = card["suite_shape"]
    lines = [
        "# TRICE Protocol Lock",
        "",
        f"- Protocol id: `{card['protocol_id']}`",
        f"- Scope: `{card['scope']}`",
        f"- Protocol level: `{card['protocol_level']}`",
        f"- Protocol score: **{card['protocol_score']}/100**",
        f"- Claim allowed by protocol: `{str(card['claim_allowed_by_protocol']).lower()}`",
        f"- Task clusters: **{shape['task_clusters']}**",
        f"- Planned runs: **{shape['planned_runs']}**",
        f"- Primary metric: `{contract['primary_metric']}`",
        f"- Target savings: **{100 * contract['target_mean_input_token_savings']:.1f}%**",
        "",
        "## Checks",
        "",
        "| Check | Passed | Observed | Required |",
        "|---|---:|---|---|",
    ]
    for row in card["checks"]:
        lines.append(f"| {row['name']} | {'yes' if row['passed'] else 'no'} | {_md(row['observed'])} | {_md(row['required'])} |")
    lines.extend(["", "## Non-Claims", ""])
    for item in card["non_claims"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Actions", ""])
    for item in card["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Hashes",
            "",
            f"- suite manifest: `{card['input_sha256']['suite_manifest']}`",
            f"- readiness preflight: `{card['input_sha256']['readiness_preflight']}`",
            f"- protocol lock: `{card['protocol_lock_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_protocol_tex(card: dict[str, Any]) -> str:
    contract = card["evaluation_contract"]
    shape = card["suite_shape"]
    rows = "\n".join(
        f"{_tex(row['name'])} & {'yes' if row['passed'] else 'no'} & {_tex(row['required'])} \\\\"
        for row in card["checks"]
    )
    non_claims = "\n".join(f"\\item {_tex(item)}" for item in card["non_claims"])
    return (
        "\\section{Protocol Lock}\n"
        f"Protocol \\texttt{{{_tex(card['protocol_id'])}}} is "
        f"\\texttt{{{_tex(card['protocol_level'])}}} with score {card['protocol_score']}/100. "
        f"It freezes {shape['planned_runs']} planned run(s) across {shape['task_clusters']} task cluster(s). "
        f"The primary metric is \\texttt{{{_tex(contract['primary_metric'])}}}; "
        f"target mean input-token savings is {100 * contract['target_mean_input_token_savings']:.1f}\\% and "
        f"clustered CI lower-bound target is {100 * contract['target_clustered_ci_low']:.1f}\\%.\n\n"
        "\\begin{table}[h]\n\\centering\n"
        "\\begin{tabular}{lrl}\n\\toprule\nCheck & Passed & Required \\\\\n\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{Pre-registered TRICE protocol checks locked before live outcome claims.}\n"
        "\\end{table}\n\n"
        "\\noindent Non-claims:\n\\begin{itemize}\n"
        f"{non_claims}\n"
        "\\end{itemize}\n"
    )


def render_protocol_svg(card: dict[str, Any]) -> str:
    stages = [
        ("metric", _check_passed(card, "primary_metric")),
        ("holdout", _check_passed(card, "remote_git_sources") and _check_passed(card, "commit_sha_revisions")),
        ("replicate", _check_passed(card, "claim_replicates_per_task")),
        ("receipts", _check_passed(card, "receipt_validation_gate")),
    ]
    width, height = 940, 280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE protocol lock</text>',
        f'<text x="28" y="58" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Score {card["protocol_score"]}/100 | level {card["protocol_level"]} | claim protocol {str(card["claim_allowed_by_protocol"]).lower()}</text>',
    ]
    x0, y = 52, 96
    for idx, (label, passed) in enumerate(stages):
        x = x0 + idx * 216
        fill = "#7c3aed" if passed else "#e5e7eb"
        text = "#ffffff" if passed else "#374151"
        parts.append(f'<rect x="{x}" y="{y}" width="164" height="58" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{x + 82}" y="{y + 36}" text-anchor="middle" font-family="Inter,Segoe UI,Arial" font-size="17" font-weight="700" fill="{text}">{label}</text>')
        if idx < len(stages) - 1:
            parts.append(f'<line x1="{x + 174}" y1="{y + 29}" x2="{x + 208}" y2="{y + 29}" stroke="#9ca3af" stroke-width="3"/>')
    shape = card["suite_shape"]
    contract = card["evaluation_contract"]
    parts.append(f'<text x="28" y="202" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#111827">Task clusters {shape["task_clusters"]} | planned runs {shape["planned_runs"]} | target {100 * contract["target_mean_input_token_savings"]:.1f}%</text>')
    parts.append('<text x="28" y="228" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">Pre-outcome contract only: S-tier still requires held-out live results, verified bundles, and claim card pass.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_protocol_outputs(card: dict[str, Any], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = out.with_suffix(".md")
    tex = out.with_suffix(".tex")
    svg = out.with_suffix(".svg")
    md.write_text(render_protocol_markdown(card), encoding="utf-8")
    tex.write_text(render_protocol_tex(card), encoding="utf-8")
    svg.write_text(render_protocol_svg(card), encoding="utf-8")
    return {"json": str(out), "markdown": str(md), "tex": str(tex), "svg": str(svg)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a deterministic TRICE protocol lock for a suite manifest.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("trice_protocol_lock.json"))
    ap.add_argument("--scope", default=DEFAULT_SCOPE)
    ap.add_argument("--protocol-id", default=None)
    ap.add_argument("--format", choices=["json", "markdown", "tex"], default="json")
    args = ap.parse_args(argv)
    card = build_protocol_lock(args.manifest, scope=args.scope, protocol_id=args.protocol_id)
    outputs = write_protocol_outputs(card, args.out)
    if args.format == "markdown":
        print(render_protocol_markdown(card))
    elif args.format == "tex":
        print(render_protocol_tex(card))
    else:
        print(json.dumps({"protocol_lock": card, "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if card["protocol_level"] != "invalid_protocol" else 1


def verify_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a deterministic TRICE protocol lock.")
    ap.add_argument("protocol_lock", type=Path)
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args(argv)
    verdict = verify_protocol_lock_file(args.protocol_lock, manifest_path=args.manifest)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def _checks(manifest: dict[str, Any], readiness: dict[str, Any], gate: dict[str, Any]) -> list[dict[str, Any]]:
    readiness_checks = {row["name"]: row for row in readiness["checks"]}
    return [
        _lift(readiness_checks, "manifest_valid"),
        _lift(readiness_checks, "unique_task_ids"),
        _lift(readiness_checks, "interventions", name="deterministic_interventions"),
        _check("primary_metric", True, "input_token_savings", "input_token_savings"),
        _check("cost_quality_joint_gate", True, ["input_token_savings", "pass_regressions"], "savings measured with pass preservation"),
        _lift(readiness_checks, "target_savings", name="target_mean_savings"),
        _check("clustered_ci_target", gate["min_clustered_savings_ci_low"] >= gate["min_mean_savings"], gate["min_clustered_savings_ci_low"], f">= {gate['min_mean_savings']:.3f}"),
        _lift(readiness_checks, "evidence_recall_gate"),
        _lift(readiness_checks, "pass_regression_gate"),
        _lift(readiness_checks, "receipt_validation_gate"),
        _lift(readiness_checks, "remote_git_sources"),
        _lift(readiness_checks, "commit_sha_revisions"),
        _lift(readiness_checks, "adapter_profiles"),
        _lift(readiness_checks, "pilot_task_clusters"),
        _lift(readiness_checks, "pilot_replicates_per_task"),
        _lift(readiness_checks, "claim_task_clusters"),
        _lift(readiness_checks, "claim_replicates_per_task"),
        _check("all_runs_accepted_gate", True, True, "all optimized runs accepted"),
        _check("evidence_bundle_required", True, ".trice.zip", "portable evidence bundle verifies"),
        _check("claim_card_required", True, "trice-claim-card/v1", "claim card verifies before README S-tier wording"),
        _check("artifact_card_required", True, "trice-artifact-card/v1", "artifact card verifies before release claim"),
    ]


def _evaluation_contract(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_metric": "input_token_savings",
        "quality_guardrail": "zero pass regressions versus baseline verifier outcome",
        "confidence_method": "clustered bootstrap by task_id",
        "unit_of_analysis": "task_id cluster",
        "conditions": ["baseline_full_context", "trice_context_control"],
        "target_mean_input_token_savings": gate["min_mean_savings"],
        "target_clustered_ci_low": gate["min_clustered_savings_ci_low"],
        "max_pass_regressions": gate["max_pass_regressions"],
        "min_task_clusters": gate["min_task_clusters"],
        "min_replicates_per_task": gate["min_replicates_per_task"],
        "evidence_recall_min": gate["min_evidence_recall"],
        "require_remote_git_sources": gate["require_remote_git_sources"],
        "require_locked_git_sources": gate["require_locked_git_sources"],
        "require_adapter_profiles": gate["require_adapter_profiles"],
        "require_receipt_validation": gate["require_receipt_validation"],
        "require_verified_bundle": True,
        "require_claim_card": True,
        "require_artifact_card": True,
    }


def _gate_config(manifest: dict[str, Any]) -> dict[str, Any]:
    raw = manifest.get("s_tier_gate") if isinstance(manifest.get("s_tier_gate"), dict) else {}
    target = float(manifest.get("target_savings") or 0.60)
    return {
        "min_task_clusters": int(raw.get("min_task_clusters", 50)),
        "min_replicates_per_task": int(raw.get("min_replicates_per_task", 3)),
        "min_mean_savings": float(raw.get("min_mean_savings", target)),
        "min_clustered_savings_ci_low": float(raw.get("min_clustered_savings_ci_low", target)),
        "min_evidence_recall": float(raw.get("min_evidence_recall", 0.95)),
        "max_pass_regressions": int(raw.get("max_pass_regressions", 0)),
        "require_locked_git_sources": bool(raw.get("require_locked_git_sources", True)),
        "require_remote_git_sources": bool(raw.get("require_remote_git_sources", True)),
        "require_adapter_profiles": bool(raw.get("require_adapter_profiles", True)),
        "require_receipt_validation": bool(raw.get("require_receipt_validation", True)),
    }


def _protocol_level(checks: list[dict[str, Any]]) -> str:
    if _all_pass(checks, ["manifest_valid", "unique_task_ids", "deterministic_interventions", "primary_metric", "cost_quality_joint_gate"]):
        if _all_pass(
            checks,
            [
                "claim_task_clusters",
                "claim_replicates_per_task",
                "remote_git_sources",
                "commit_sha_revisions",
                "adapter_profiles",
                "target_mean_savings",
                "clustered_ci_target",
                "evidence_recall_gate",
                "pass_regression_gate",
                "receipt_validation_gate",
            ],
        ):
            return "claim_protocol_ready"
        if _all_pass(checks, ["pilot_task_clusters", "pilot_replicates_per_task", "remote_git_sources", "commit_sha_revisions", "adapter_profiles"]):
            return "pilot_protocol_ready"
        return "smoke_protocol_locked"
    return "invalid_protocol"


def _protocol_score(checks: list[dict[str, Any]]) -> int:
    weights = {
        "manifest_valid": 8,
        "unique_task_ids": 6,
        "deterministic_interventions": 6,
        "primary_metric": 6,
        "cost_quality_joint_gate": 7,
        "target_mean_savings": 6,
        "clustered_ci_target": 6,
        "evidence_recall_gate": 6,
        "pass_regression_gate": 6,
        "receipt_validation_gate": 6,
        "remote_git_sources": 7,
        "commit_sha_revisions": 7,
        "adapter_profiles": 6,
        "pilot_task_clusters": 5,
        "pilot_replicates_per_task": 5,
        "claim_task_clusters": 5,
        "claim_replicates_per_task": 5,
        "all_runs_accepted_gate": 3,
        "evidence_bundle_required": 3,
        "claim_card_required": 3,
        "artifact_card_required": 3,
    }
    return min(100, sum(weights.get(row["name"], 0) for row in checks if row["passed"]))


def _non_claims(checks: list[dict[str, Any]]) -> list[str]:
    missing = [row["name"] for row in checks if not row["passed"]]
    items = ["Protocol lock is not outcome evidence; it does not claim measured savings or task success."]
    if "claim_task_clusters" in missing or "claim_replicates_per_task" in missing:
        items.append("Not a claim-ready protocol until the suite has 50 task clusters and 3 replicates per task.")
    if "remote_git_sources" in missing or "commit_sha_revisions" in missing:
        items.append("Not a held-out remote-repo protocol until every task uses a locked remote Git commit.")
    if "adapter_profiles" in missing:
        items.append("Not an adapter-profiled claim protocol until every task uses a versioned adapter profile.")
    items.append("Does not permit README S-tier wording without a passing claim card and verified artifact card.")
    return items


def _next_actions(checks: list[dict[str, Any]], gate: dict[str, Any]) -> list[str]:
    missing = [row["name"] for row in checks if not row["passed"]]
    actions = []
    if "remote_git_sources" in missing or "commit_sha_revisions" in missing:
        actions.append("Replace local suite tasks with remote Git URLs pinned to immutable 40-hex commits.")
    if "adapter_profiles" in missing:
        actions.append("Move interventions into adapter_profile files so agent behavior is versioned and reusable.")
    if "pilot_task_clusters" in missing or "pilot_replicates_per_task" in missing:
        actions.append("Build the 10-task x 2-replicate pilot protocol before the claim run.")
    if "claim_task_clusters" in missing or "claim_replicates_per_task" in missing:
        actions.append(f"Scale to {gate['min_task_clusters']} task clusters and {gate['min_replicates_per_task']} replicates per task for the S-tier protocol.")
    return actions or ["Run the locked suite, verify the evidence bundle, generate a claim card, then regenerate the artifact card."]


def _default_protocol_id(path: Path, manifest: dict[str, Any]) -> str:
    name = str(manifest.get("name") or path.stem)
    return f"{name}:{_sha256_file(path)[:16]}"


def _resolve_bound_manifest(card_path: Path, card: dict[str, Any], explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    source = card.get("source") if isinstance(card.get("source"), dict) else {}
    raw = source.get("suite_manifest_path")
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


def _lift(rows: dict[str, dict[str, Any]], source: str, *, name: str | None = None) -> dict[str, Any]:
    row = rows[source]
    return _check(name or source, bool(row["passed"]), row.get("observed"), row.get("required"))


def _all_pass(checks: list[dict[str, Any]], names: list[str]) -> bool:
    by_name = {row["name"]: row for row in checks}
    return all(bool(by_name[name]["passed"]) for name in names)


def _check_passed(card: dict[str, Any], name: str) -> bool:
    return any(row["name"] == name and row["passed"] for row in card.get("checks", []))


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def _without_protocol_hash(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out.pop("protocol_lock_sha256", None)
    return out


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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
