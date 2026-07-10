"""Statistical design cards for TRICE claim planning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .evidence import canonical_json, sha256_file, write_text_lf
from .protocol import verify_protocol_lock_file

DESIGN_SCHEMA_VERSION = "trice-design-card/v1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = REPO / "docs" / "trice_protocol_lock.json"
DEFAULT_SUITE_RESULT = REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_suite_results.json"
Z95 = 1.959963984540054


def build_design_card(
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    *,
    suite_result_path: str | Path = DEFAULT_SUITE_RESULT,
) -> dict[str, Any]:
    """Build a deterministic statistical-design review for a TRICE suite."""

    protocol_file = Path(protocol_path)
    result_file = Path(suite_result_path)
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    result = json.loads(result_file.read_text(encoding="utf-8"))
    protocol_verdict = verify_protocol_lock_file(protocol_file)
    contract = protocol.get("evaluation_contract") or {}
    gate = result.get("claim_gate") or {}
    target = float(contract.get("target_mean_input_token_savings") or gate.get("target_savings") or 0.60)
    min_clusters = int(contract.get("min_task_clusters") or 50)
    min_replicates = int(contract.get("min_replicates_per_task") or 3)
    clusters = _cluster_savings(result)
    cluster_means = {task_id: round(mean(values), 6) for task_id, values in sorted(clusters.items())}
    flat = [value for values in clusters.values() for value in values]
    observed_mean = float(gate.get("mean_savings") or (mean(flat) if flat else 0.0))
    clustered_ci = gate.get("clustered_savings_ci") or gate.get("savings_ci") or {}
    cluster_sd = _sample_sd(list(cluster_means.values()))
    projected = _projection(observed_mean, cluster_sd, target, min_clusters)
    checks = _checks(protocol, protocol_verdict, result, target, min_clusters, min_replicates, projected)
    card = {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "design_level": _design_level(checks, protocol),
        "design_score": _design_score(checks),
        "claim_design_ready": False,
        "input_sha256": {
            "protocol_lock": sha256_file(protocol_file),
            "suite_result": sha256_file(result_file),
        },
        "source": {
            "protocol_lock_path": _display_path(protocol_file),
            "suite_result_path": _display_path(result_file),
            "suite_name": (result.get("suite") or {}).get("name"),
            "protocol_id": protocol.get("protocol_id"),
        },
        "observed": {
            "task_clusters": len(clusters),
            "total_runs": int(gate.get("total_rounds") or len(flat)),
            "replicate_count": int(gate.get("replicate_count") or len(flat)),
            "cluster_means": cluster_means,
            "cluster_sample_sd": round(cluster_sd, 6),
            "mean_input_token_savings": round(observed_mean, 6),
            "clustered_ci_low": round(float(clustered_ci.get("low") or 0.0), 6),
            "clustered_ci_high": round(float(clustered_ci.get("high") or 0.0), 6),
            "pass_regressions": int(gate.get("pass_regressions") or 0),
            "accepted_rounds": int(gate.get("accepted_rounds") or 0),
            "total_rounds": int(gate.get("total_rounds") or len(flat)),
        },
        "projection": projected,
        "checks": checks,
        "research_basis": [
            "Claim design must be separated from outcome scoring to avoid post-hoc metric selection.",
            "Clustered task-level uncertainty is the relevant unit when repeated runs share a repository/task.",
            "A positive smoke mean is not enough; the projected claim-run lower bound must clear the target.",
            "Statistical readiness does not override external-validity requirements such as held-out remote commits.",
        ],
        "non_claims": _non_claims(checks),
        "next_actions": _next_actions(checks, min_clusters, min_replicates),
    }
    card["claim_design_ready"] = card["design_level"] == "claim_design_ready"
    card["design_card_sha256"] = hashlib.sha256(canonical_json(_without_design_hash(card)).encode("utf-8")).hexdigest()
    return card


def verify_design_card_file(
    design_path: str | Path,
    *,
    protocol_path: str | Path | None = None,
    suite_result_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a design card self hash and deterministic rebuild from bound inputs."""

    path = Path(design_path)
    card = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if card.get("schema_version") != DESIGN_SCHEMA_VERSION:
        errors.append(f"schema_version must be {DESIGN_SCHEMA_VERSION}")
    expected_hash = str(card.get("design_card_sha256") or "")
    actual_hash = hashlib.sha256(canonical_json(_without_design_hash(card)).encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        errors.append("design_card_sha256 mismatch")

    protocol_file = _resolve_bound_path(path, card, "protocol_lock_path", protocol_path)
    result_file = _resolve_bound_path(path, card, "suite_result_path", suite_result_path)
    checked_inputs: list[str] = []
    input_hashes = card.get("input_sha256") if isinstance(card.get("input_sha256"), dict) else {}
    if protocol_file is None:
        errors.append("protocol lock path is missing")
    elif not protocol_file.is_file():
        errors.append(f"protocol lock file not found: {_display_path(protocol_file)}")
    elif sha256_file(protocol_file) != input_hashes.get("protocol_lock"):
        errors.append("protocol_lock sha256 mismatch")
    else:
        checked_inputs.append("protocol_lock")
    if result_file is None:
        errors.append("suite result path is missing")
    elif not result_file.is_file():
        errors.append(f"suite result file not found: {_display_path(result_file)}")
    elif sha256_file(result_file) != input_hashes.get("suite_result"):
        errors.append("suite_result sha256 mismatch")
    else:
        checked_inputs.append("suite_result")
    if protocol_file and protocol_file.is_file() and result_file and result_file.is_file():
        rebuilt = build_design_card(protocol_file, suite_result_path=result_file)
        if canonical_json(_without_design_hash(rebuilt)) != canonical_json(_without_design_hash(card)):
            errors.append("design card does not match deterministic rebuild from bound inputs")
    return {
        "ok": not errors,
        "schema_version": card.get("schema_version"),
        "design_level": card.get("design_level"),
        "design_score": card.get("design_score"),
        "claim_design_ready": bool(card.get("claim_design_ready")),
        "design_card_sha256": expected_hash,
        "computed_design_card_sha256": actual_hash,
        "checked_inputs": checked_inputs,
        "errors": errors,
    }


def render_design_markdown(card: dict[str, Any]) -> str:
    observed = card["observed"]
    projection = card["projection"]
    lines = [
        "# TRICE Design Card",
        "",
        f"- Design level: `{card['design_level']}`",
        f"- Design score: **{card['design_score']}/100**",
        f"- Claim design ready: `{str(card['claim_design_ready']).lower()}`",
        f"- Observed mean savings: **{100 * observed['mean_input_token_savings']:.1f}%**",
        f"- Observed clustered CI low: **{100 * observed['clustered_ci_low']:.1f}%**",
        f"- Projected claim lower bound: **{100 * projection['projected_ci_low_at_claim_n']:.1f}%**",
        f"- Projected clusters required by variance: **{projection['projected_clusters_for_target']}**",
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
    lines.extend(["", "## Hashes", ""])
    lines.append(f"- protocol lock: `{card['input_sha256']['protocol_lock']}`")
    lines.append(f"- suite result: `{card['input_sha256']['suite_result']}`")
    lines.append(f"- design card: `{card['design_card_sha256']}`")
    lines.append("")
    return "\n".join(lines)


def render_design_tex(card: dict[str, Any]) -> str:
    observed = card["observed"]
    projection = card["projection"]
    rows = "\n".join(
        f"{_tex(row['name'])} & {'yes' if row['passed'] else 'no'} & {_tex(row['required'])} \\\\"
        for row in card["checks"]
    )
    return (
        "\\section{Statistical Design Card}\n"
        f"Design level: \\texttt{{{_tex(card['design_level'])}}}; "
        f"score: {card['design_score']}/100; claim-design ready: "
        f"{'yes' if card['claim_design_ready'] else 'no'}. "
        f"Observed mean input-token savings is {100 * observed['mean_input_token_savings']:.1f}\\%; "
        f"observed clustered CI lower bound is {100 * observed['clustered_ci_low']:.1f}\\%; "
        f"projected claim lower bound at the locked claim sample size is "
        f"{100 * projection['projected_ci_low_at_claim_n']:.1f}\\%.\n\n"
        "\\begin{table}[h]\n\\centering\n"
        "\\begin{tabular}{lrl}\n\\toprule\nCheck & Passed & Required \\\\\n\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{TRICE statistical-design checks before interpreting a live claim run.}\n"
        "\\end{table}\n"
    )


def render_design_svg(card: dict[str, Any]) -> str:
    stages = [
        ("signal", _check_passed(card, "observed_mean_above_target")),
        ("ci", _check_passed(card, "observed_clustered_ci_above_target")),
        ("power", _check_passed(card, "projected_claim_ci_above_target")),
        ("external", _check_passed(card, "protocol_claim_ready")),
    ]
    width, height = 940, 280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE statistical design card</text>',
        f'<text x="28" y="58" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Score {card["design_score"]}/100 | level {card["design_level"]} | claim design {str(card["claim_design_ready"]).lower()}</text>',
    ]
    x0, y = 52, 96
    for idx, (label, passed) in enumerate(stages):
        x = x0 + idx * 216
        fill = "#0f766e" if passed else "#e5e7eb"
        text = "#ffffff" if passed else "#374151"
        parts.append(f'<rect x="{x}" y="{y}" width="164" height="58" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{x + 82}" y="{y + 36}" text-anchor="middle" font-family="Inter,Segoe UI,Arial" font-size="17" font-weight="700" fill="{text}">{label}</text>')
        if idx < len(stages) - 1:
            parts.append(f'<line x1="{x + 174}" y1="{y + 29}" x2="{x + 208}" y2="{y + 29}" stroke="#9ca3af" stroke-width="3"/>')
    obs = card["observed"]
    proj = card["projection"]
    parts.append(f'<text x="28" y="202" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#111827">Observed mean {100 * obs["mean_input_token_savings"]:.1f}% | clustered low {100 * obs["clustered_ci_low"]:.1f}% | projected claim low {100 * proj["projected_ci_low_at_claim_n"]:.1f}%</text>')
    parts.append('<text x="28" y="228" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">Design review only: external validity still requires held-out locked remote repositories and claim evidence.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_design_outputs(card: dict[str, Any], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(out, json.dumps(card, indent=2, sort_keys=True) + "\n")
    md = out.with_suffix(".md")
    tex = out.with_suffix(".tex")
    svg = out.with_suffix(".svg")
    write_text_lf(md, render_design_markdown(card))
    write_text_lf(tex, render_design_tex(card))
    write_text_lf(svg, render_design_svg(card))
    return {"json": str(out), "markdown": str(md), "tex": str(tex), "svg": str(svg)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a deterministic TRICE statistical design card.")
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    ap.add_argument("--suite-result", type=Path, default=DEFAULT_SUITE_RESULT)
    ap.add_argument("--out", type=Path, default=Path("trice_design_card.json"))
    ap.add_argument("--format", choices=["json", "markdown", "tex"], default="json")
    args = ap.parse_args(argv)
    card = build_design_card(args.protocol, suite_result_path=args.suite_result)
    outputs = write_design_outputs(card, args.out)
    if args.format == "markdown":
        print(render_design_markdown(card))
    elif args.format == "tex":
        print(render_design_tex(card))
    else:
        print(json.dumps({"design_card": card, "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if card["design_level"] != "not_design_ready" else 1


def verify_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a deterministic TRICE statistical design card.")
    ap.add_argument("design_card", type=Path)
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--suite-result", type=Path, default=None)
    args = ap.parse_args(argv)
    verdict = verify_design_card_file(args.design_card, protocol_path=args.protocol, suite_result_path=args.suite_result)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def _checks(
    protocol: dict[str, Any],
    protocol_verdict: dict[str, Any],
    result: dict[str, Any],
    target: float,
    min_clusters: int,
    min_replicates: int,
    projection: dict[str, Any],
) -> list[dict[str, Any]]:
    gate = result.get("claim_gate") or {}
    observed = gate.get("s_tier_gate", {}).get("requirements", {})
    clustered_ci = gate.get("clustered_savings_ci") or gate.get("savings_ci") or {}
    task_clusters = int(gate.get("task_cluster_count") or 0)
    replicate_count = int(gate.get("replicate_count") or 0)
    return [
        _check("protocol_verifies", bool(protocol_verdict.get("ok")), protocol_verdict.get("protocol_level"), "protocol lock verifies"),
        _check("suite_result_schema", result.get("schema_version") == "trice-suite-result/v1", result.get("schema_version"), "trice-suite-result/v1"),
        _check("primary_metric_locked", (protocol.get("evaluation_contract") or {}).get("primary_metric") == "input_token_savings", (protocol.get("evaluation_contract") or {}).get("primary_metric"), "input_token_savings"),
        _check("observed_mean_above_target", float(gate.get("mean_savings") or 0.0) >= target, gate.get("mean_savings"), f">= {target:.3f}"),
        _check("observed_clustered_ci_above_target", float(clustered_ci.get("low") or 0.0) >= target, clustered_ci.get("low"), f">= {target:.3f}"),
        _check("zero_pass_regressions", int(gate.get("pass_regressions") or 0) == 0, gate.get("pass_regressions"), "0"),
        _check("all_runs_accepted", int(gate.get("accepted_rounds") or 0) == int(gate.get("total_rounds") or -1), f"{gate.get('accepted_rounds')}/{gate.get('total_rounds')}", "all runs accepted"),
        _check("pilot_task_clusters", task_clusters >= 10, task_clusters, ">= 10"),
        _check("claim_task_clusters", task_clusters >= min_clusters, task_clusters, f">= {min_clusters}"),
        _check("claim_replicates", _replicate_requirement_passed(observed, min_replicates, replicate_count, task_clusters), _replicate_observed(observed, replicate_count, task_clusters), f"each task >= {min_replicates}"),
        _check("projected_claim_ci_above_target", bool(projection["projected_ci_passes_target"]), projection["projected_ci_low_at_claim_n"], f">= {target:.3f}"),
        _check("protocol_claim_ready", protocol.get("protocol_level") == "claim_protocol_ready", protocol.get("protocol_level"), "claim_protocol_ready"),
    ]


def _projection(observed_mean: float, cluster_sd: float, target: float, min_clusters: int) -> dict[str, Any]:
    margin = observed_mean - target
    if cluster_sd <= 0:
        projected_clusters = 1 if margin >= 0 else None
    elif margin <= 0:
        projected_clusters = None
    else:
        projected_clusters = int(math.ceil((Z95 * cluster_sd / margin) ** 2))
    claim_n = max(min_clusters, projected_clusters or min_clusters)
    projected_low = observed_mean - (Z95 * cluster_sd / math.sqrt(max(1, min_clusters)))
    return {
        "target": round(target, 6),
        "observed_margin_to_target": round(margin, 6),
        "projected_clusters_for_target": projected_clusters,
        "protocol_min_task_clusters": min_clusters,
        "effective_required_task_clusters": claim_n,
        "projected_ci_low_at_claim_n": round(projected_low, 6),
        "projected_ci_passes_target": projected_low >= target,
        "z": Z95,
        "method": "normal approximation over observed task-cluster means",
    }


def _cluster_savings(result: dict[str, Any]) -> dict[str, list[float]]:
    clusters: dict[str, list[float]] = {}
    for task in result.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "")
        if not task_id:
            continue
        clusters.setdefault(task_id, []).append(float(task.get("mean_savings") or 0.0))
    return clusters


def _design_level(checks: list[dict[str, Any]], protocol: dict[str, Any]) -> str:
    passed = {row["name"]: bool(row["passed"]) for row in checks}
    if not passed.get("protocol_verifies") or not passed.get("suite_result_schema"):
        return "not_design_ready"
    if passed.get("protocol_claim_ready") and passed.get("projected_claim_ci_above_target") and passed.get("claim_task_clusters") and passed.get("claim_replicates"):
        return "claim_design_ready"
    if passed.get("pilot_task_clusters") and protocol.get("protocol_level") in {"pilot_protocol_ready", "claim_protocol_ready"}:
        return "pilot_design_ready"
    return "smoke_design_observed"


def _design_score(checks: list[dict[str, Any]]) -> int:
    weights = {
        "protocol_verifies": 10,
        "suite_result_schema": 6,
        "primary_metric_locked": 7,
        "observed_mean_above_target": 9,
        "observed_clustered_ci_above_target": 9,
        "zero_pass_regressions": 9,
        "all_runs_accepted": 7,
        "pilot_task_clusters": 6,
        "claim_task_clusters": 7,
        "claim_replicates": 7,
        "projected_claim_ci_above_target": 8,
        "protocol_claim_ready": 15,
    }
    return min(100, sum(weights.get(row["name"], 0) for row in checks if row["passed"]))


def _non_claims(checks: list[dict[str, Any]]) -> list[str]:
    missing = [row["name"] for row in checks if not row["passed"]]
    items = ["Design card is not outcome evidence and does not claim S-tier performance."]
    if "protocol_claim_ready" in missing:
        items.append("The protocol is not claim-ready, so statistical signal cannot override missing held-out design requirements.")
    if "claim_task_clusters" in missing or "claim_replicates" in missing:
        items.append("The observed suite does not have enough task clusters or replicates for the S-tier claim design.")
    if "projected_claim_ci_above_target" in missing:
        items.append("Observed variance does not project a claim-sample lower bound above target.")
    return items


def _next_actions(checks: list[dict[str, Any]], min_clusters: int, min_replicates: int) -> list[str]:
    missing = [row["name"] for row in checks if not row["passed"]]
    actions = []
    if "pilot_task_clusters" in missing:
        actions.append("Run or curate at least 10 task clusters before interpreting pilot design.")
    if "claim_task_clusters" in missing:
        actions.append(f"Scale the claim suite to at least {min_clusters} held-out task clusters.")
    if "claim_replicates" in missing:
        actions.append(f"Run at least {min_replicates} replicates per task cluster for the claim suite.")
    if "protocol_claim_ready" in missing:
        actions.append("Regenerate a claim-ready protocol lock with held-out remote Git commits and adapter profiles.")
    return actions or ["Run the locked claim suite, verify the bundle, then regenerate the claim and artifact cards."]


def _replicate_requirement_passed(observed: dict[str, Any], min_replicates: int, replicate_count: int, task_clusters: int) -> bool:
    row = observed.get("replicates_per_task") if isinstance(observed, dict) else None
    if isinstance(row, dict) and isinstance(row.get("passed"), bool):
        return bool(row.get("passed"))
    return task_clusters > 0 and replicate_count >= task_clusters * min_replicates


def _replicate_observed(observed: dict[str, Any], replicate_count: int, task_clusters: int) -> Any:
    row = observed.get("replicates_per_task") if isinstance(observed, dict) else None
    if isinstance(row, dict) and "observed" in row:
        return row["observed"]
    return {"replicate_count": replicate_count, "task_clusters": task_clusters}


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(stdev(values))


def _resolve_bound_path(card_path: Path, card: dict[str, Any], source_key: str, explicit: str | Path | None) -> Path | None:
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


def _check_passed(card: dict[str, Any], name: str) -> bool:
    return any(row["name"] == name and row["passed"] for row in card.get("checks", []))


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def _without_design_hash(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out.pop("design_card_sha256", None)
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


if __name__ == "__main__":
    raise SystemExit(main())
