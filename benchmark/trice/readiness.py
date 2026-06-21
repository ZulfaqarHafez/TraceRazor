"""Deterministic preflight readiness reports for TRICE live suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .evidence import canonical_json
from .suite import SUITE_SCHEMA_VERSION, load_suite_manifest

READINESS_SCHEMA_VERSION = "trice-suite-readiness/v1"
DEFAULT_SCOPE = "python software-repair/context-control tasks on held-out Git repositories"
REPO = Path(__file__).resolve().parents[2]


def build_suite_readiness(
    manifest_path: str | Path,
    *,
    scope: str = DEFAULT_SCOPE,
    pilot_task_clusters: int = 10,
    pilot_replicates_per_task: int = 2,
) -> dict[str, Any]:
    """Build a deterministic preflight report for a TRICE suite manifest.

    This is a no-execution gate. It tells users whether a manifest is eligible
    for smoke, pilot, or S-tier claim execution before spending live-run tokens.
    """

    path = Path(manifest_path)
    manifest = load_suite_manifest(path)
    tasks = manifest["tasks"]
    gate = _gate_config(manifest)
    per_task_replicates = _per_task_replicates(manifest)
    task_ids = [str(task.get("task_id") or "") for task in tasks]
    rounds = int(manifest.get("rounds") or 1)
    planned_runs = sum(per_task_replicates.values())
    checks = _checks(manifest, gate, per_task_replicates, pilot_task_clusters, pilot_replicates_per_task)
    levels = {
        "smoke_ready": _passed(checks, ["manifest_valid", "unique_task_ids", "prompts", "verify_commands", "interventions"]),
        "pilot_ready": _passed(
            checks,
            [
                "manifest_valid",
                "unique_task_ids",
                "prompts",
                "verify_commands",
                "interventions",
                "pilot_task_clusters",
                "pilot_replicates_per_task",
                "remote_git_sources",
                "commit_sha_revisions",
                "adapter_profiles",
            ],
        ),
        "claim_ready": _passed(
            checks,
            [
                "manifest_valid",
                "unique_task_ids",
                "prompts",
                "verify_commands",
                "interventions",
                "claim_task_clusters",
                "claim_replicates_per_task",
                "remote_git_sources",
                "commit_sha_revisions",
                "adapter_profiles",
                "target_savings",
                "pass_regression_gate",
                "receipt_validation_gate",
            ],
        ),
    }
    readiness_level = (
        "claim_ready"
        if levels["claim_ready"]
        else "pilot_ready"
        if levels["pilot_ready"]
        else "smoke_ready"
        if levels["smoke_ready"]
        else "not_ready"
    )
    card = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "scope": scope,
        "readiness_level": readiness_level,
        "claim_execution_ready": bool(levels["claim_ready"]),
        "pilot_execution_ready": bool(levels["pilot_ready"]),
        "manifest_sha256": _sha256_file(path),
        "source": {
            "suite_manifest_path": _display_path(path),
            "suite_name": manifest.get("name") or path.stem,
            "suite_schema_version": manifest.get("schema_version"),
        },
        "planned_execution": {
            "task_clusters": len(tasks),
            "planned_runs": planned_runs,
            "rounds_per_run": rounds,
            "baseline_plus_trice_conditions": planned_runs * rounds * 2,
            "verify_command_invocations_min": planned_runs * rounds * 2,
            "replicates_per_task": per_task_replicates,
            "adapter_profile_count": sum(1 for task in tasks if task.get("adapter_profile")),
            "remote_git_task_count": sum(1 for task in tasks if isinstance(task.get("git"), dict)),
        },
        "gate_config": gate,
        "checks": checks,
        "missing_for_pilot": _missing(
            checks,
            [
                "pilot_task_clusters",
                "pilot_replicates_per_task",
                "remote_git_sources",
                "commit_sha_revisions",
                "adapter_profiles",
            ],
        ),
        "missing_for_claim": _missing(
            checks,
            [
                "claim_task_clusters",
                "claim_replicates_per_task",
                "remote_git_sources",
                "commit_sha_revisions",
                "adapter_profiles",
                "target_savings",
                "pass_regression_gate",
                "receipt_validation_gate",
            ],
        ),
        "research_contract": [
            "Preflight only: no savings, pass-rate, or S-tier result is claimed.",
            "Claim execution requires held-out remote Git tasks, immutable revisions, fixed adapter profiles, and repeated live runs.",
            "Outcome evidence must come from suite results, evidence manifests, claim cards, and bundle verification.",
        ],
        "recommendations": _recommendations(checks, gate, task_ids),
    }
    card["readiness_score"] = _readiness_score(card)
    card["readiness_sha256"] = hashlib.sha256(canonical_json(_without_readiness_hash(card)).encode("utf-8")).hexdigest()
    return card


def verify_readiness_file(readiness_path: str | Path, *, manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Verify a readiness report's self hash and bound suite manifest hash."""

    path = Path(readiness_path)
    card = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if card.get("schema_version") != READINESS_SCHEMA_VERSION:
        errors.append(f"schema_version must be {READINESS_SCHEMA_VERSION}")
    expected_hash = str(card.get("readiness_sha256") or "")
    actual_hash = hashlib.sha256(canonical_json(_without_readiness_hash(card)).encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        errors.append("readiness_sha256 mismatch")

    checked_inputs: list[str] = []
    bound_manifest = _resolve_bound_manifest(path, card, manifest_path)
    if bound_manifest is None:
        errors.append("suite manifest path is missing")
    elif not bound_manifest.is_file():
        errors.append(f"suite manifest file not found: {_display_path(bound_manifest)}")
    elif _sha256_file(bound_manifest) != card.get("manifest_sha256"):
        errors.append("suite manifest sha256 mismatch")
    else:
        checked_inputs.append("suite_manifest")

    return {
        "ok": not errors,
        "schema_version": card.get("schema_version"),
        "readiness_level": card.get("readiness_level"),
        "pilot_execution_ready": bool(card.get("pilot_execution_ready")),
        "claim_execution_ready": bool(card.get("claim_execution_ready")),
        "readiness_sha256": expected_hash,
        "computed_readiness_sha256": actual_hash,
        "checked_inputs": checked_inputs,
        "errors": errors,
    }


def render_readiness_markdown(card: dict[str, Any]) -> str:
    planned = card["planned_execution"]
    lines = [
        "# TRICE Suite Readiness",
        "",
        f"- Scope: `{card['scope']}`",
        f"- Suite: `{card['source']['suite_name']}`",
        f"- Readiness level: `{card['readiness_level']}`",
        f"- Pilot execution ready: `{str(card['pilot_execution_ready']).lower()}`",
        f"- Claim execution ready: `{str(card['claim_execution_ready']).lower()}`",
        f"- Readiness score: **{card['readiness_score']}/100**",
        f"- Task clusters: **{planned['task_clusters']}**",
        f"- Planned runs: **{planned['planned_runs']}**",
        f"- Minimum verifier invocations: **{planned['verify_command_invocations_min']}**",
        "",
        "## Checks",
        "",
        "| Check | Passed | Observed | Required |",
        "|---|---:|---|---|",
    ]
    for row in card["checks"]:
        lines.append(
            f"| {row['name']} | {'yes' if row['passed'] else 'no'} | "
            f"{_md(row['observed'])} | {_md(row['required'])} |"
        )
    lines.extend(["", "## Recommendations", ""])
    for item in card["recommendations"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Contract",
            "",
        ]
    )
    for item in card["research_contract"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Hashes",
            "",
            f"- suite manifest: `{card['manifest_sha256']}`",
            f"- readiness report: `{card['readiness_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_readiness_tex(card: dict[str, Any]) -> str:
    planned = card["planned_execution"]
    rows = "\n".join(
        f"{_tex(row['name'])} & {'yes' if row['passed'] else 'no'} & {_tex(row['required'])} \\\\"
        for row in card["checks"]
    )
    recommendations = "\n".join(f"\\item {_tex(item)}" for item in card["recommendations"])
    return (
        "\\section{Suite Readiness Preflight}\n"
        f"Suite \\texttt{{{_tex(card['source']['suite_name'])}}} is "
        f"\\texttt{{{_tex(card['readiness_level'])}}} with readiness score "
        f"{card['readiness_score']}/100. It plans {planned['planned_runs']} live run(s) "
        f"across {planned['task_clusters']} task cluster(s), requiring at least "
        f"{planned['verify_command_invocations_min']} verifier invocations before retries.\n\n"
        "\\begin{table}[h]\n\\centering\n"
        "\\begin{tabular}{lrl}\n\\toprule\nCheck & Passed & Required \\\\\n\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{No-execution readiness checks for the held-out TRICE suite protocol.}\n"
        "\\end{table}\n\n"
        "\\noindent Recommendations:\n\\begin{itemize}\n"
        f"{recommendations}\n"
        "\\end{itemize}\n"
    )


def render_readiness_svg(card: dict[str, Any]) -> str:
    stages = [
        ("smoke", card["readiness_level"] in {"smoke_ready", "pilot_ready", "claim_ready"}),
        ("pilot", bool(card["pilot_execution_ready"])),
        ("claim", bool(card["claim_execution_ready"])),
    ]
    width, height = 840, 280
    planned = card["planned_execution"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE suite readiness preflight</text>',
        f'<text x="28" y="58" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Score {card["readiness_score"]}/100 | readiness {card["readiness_level"]}</text>',
    ]
    x0, y = 70, 96
    for idx, (label, passed) in enumerate(stages):
        x = x0 + idx * 232
        fill = "#2563eb" if passed else "#e5e7eb"
        text = "#ffffff" if passed else "#374151"
        parts.append(f'<rect x="{x}" y="{y}" width="174" height="58" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{x + 87}" y="{y + 36}" text-anchor="middle" font-family="Inter,Segoe UI,Arial" font-size="18" font-weight="700" fill="{text}">{label}</text>')
        if idx < len(stages) - 1:
            parts.append(f'<line x1="{x + 184}" y1="{y + 29}" x2="{x + 224}" y2="{y + 29}" stroke="#9ca3af" stroke-width="3"/>')
    parts.append(f'<text x="28" y="202" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#111827">Task clusters {planned["task_clusters"]} | planned runs {planned["planned_runs"]} | min verifier calls {planned["verify_command_invocations_min"]}</text>')
    parts.append('<text x="28" y="228" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">Preflight only: outcome claims still require live suite results, verified manifests, and a claim card.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_readiness_outputs(card: dict[str, Any], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = out.with_suffix(".md")
    tex = out.with_suffix(".tex")
    svg = out.with_suffix(".svg")
    md.write_text(render_readiness_markdown(card), encoding="utf-8")
    tex.write_text(render_readiness_tex(card), encoding="utf-8")
    svg.write_text(render_readiness_svg(card), encoding="utf-8")
    return {"json": str(out), "markdown": str(md), "tex": str(tex), "svg": str(svg)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preflight a TRICE live suite manifest for smoke, pilot, and claim readiness.")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, default=Path("trice_suite_readiness.json"))
    ap.add_argument("--format", choices=["json", "markdown", "tex"], default="json")
    ap.add_argument("--scope", default=DEFAULT_SCOPE)
    args = ap.parse_args(argv)
    card = build_suite_readiness(args.manifest, scope=args.scope)
    outputs = write_readiness_outputs(card, args.out)
    if args.format == "markdown":
        print(render_readiness_markdown(card))
    elif args.format == "tex":
        print(render_readiness_tex(card))
    else:
        print(json.dumps({"readiness": card, "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if card["readiness_level"] != "not_ready" else 1


def verify_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a deterministic TRICE suite readiness report.")
    ap.add_argument("readiness", type=Path)
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args(argv)
    verdict = verify_readiness_file(args.readiness, manifest_path=args.manifest)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def _checks(
    manifest: dict[str, Any],
    gate: dict[str, Any],
    per_task_replicates: dict[str, int],
    pilot_task_clusters: int,
    pilot_replicates_per_task: int,
) -> list[dict[str, Any]]:
    tasks = manifest["tasks"]
    task_ids = [str(task.get("task_id") or "") for task in tasks]
    interventions = [_intervention_count(task) for task in tasks]
    git_tasks = [task.get("git") if isinstance(task.get("git"), dict) else None for task in tasks]
    min_replicates = min(per_task_replicates.values()) if per_task_replicates else 0
    max_pass_regressions = int(gate.get("max_pass_regressions", 0))
    return [
        _check("manifest_valid", manifest.get("schema_version") == SUITE_SCHEMA_VERSION, manifest.get("schema_version"), SUITE_SCHEMA_VERSION),
        _check("unique_task_ids", len(task_ids) == len(set(task_ids)), task_ids, "all task_id values unique"),
        _check("prompts", all(str(task.get("prompt") or "").strip() for task in tasks), [bool(str(task.get("prompt") or "").strip()) for task in tasks], "every task has prompt"),
        _check("verify_commands", all(task.get("verify_cmd") for task in tasks), [bool(task.get("verify_cmd")) for task in tasks], "every task has verify_cmd"),
        _check("interventions", all(count == 1 for count in interventions), interventions, "exactly one intervention per task"),
        _check("pilot_task_clusters", len(tasks) >= pilot_task_clusters, len(tasks), f">= {pilot_task_clusters}"),
        _check("pilot_replicates_per_task", bool(per_task_replicates) and min_replicates >= pilot_replicates_per_task, per_task_replicates, f"each task >= {pilot_replicates_per_task}"),
        _check("claim_task_clusters", len(tasks) >= int(gate.get("min_task_clusters", 50)), len(tasks), f">= {int(gate.get('min_task_clusters', 50))}"),
        _check("claim_replicates_per_task", bool(per_task_replicates) and min_replicates >= int(gate.get("min_replicates_per_task", 3)), per_task_replicates, f"each task >= {int(gate.get('min_replicates_per_task', 3))}"),
        _check("remote_git_sources", all(_is_remote_git(task) for task in git_tasks), [_git_url(task) for task in git_tasks], "all tasks use remote Git URLs"),
        _check("commit_sha_revisions", all(_is_full_commit_sha((task or {}).get("rev")) for task in git_tasks), [(task or {}).get("rev") for task in git_tasks], "all git.rev values are 40-hex commit SHA"),
        _check("adapter_profiles", all(task.get("adapter_profile") for task in tasks), [task.get("adapter_profile") for task in tasks], "all tasks use adapter_profile"),
        _check("target_savings", float(manifest.get("target_savings") or 0.0) >= float(gate.get("min_mean_savings", 0.60)), manifest.get("target_savings"), f">= {float(gate.get('min_mean_savings', 0.60)):.3f}"),
        _check("pass_regression_gate", max_pass_regressions == 0, max_pass_regressions, "0"),
        _check("receipt_validation_gate", bool(gate.get("require_receipt_validation", True)), gate.get("require_receipt_validation", True), "true"),
    ]


def _gate_config(manifest: dict[str, Any]) -> dict[str, Any]:
    raw = manifest.get("s_tier_gate") if isinstance(manifest.get("s_tier_gate"), dict) else {}
    return {
        "min_task_clusters": int(raw.get("min_task_clusters", 50)),
        "min_replicates_per_task": int(raw.get("min_replicates_per_task", 3)),
        "min_mean_savings": float(raw.get("min_mean_savings", manifest.get("target_savings") or 0.60)),
        "min_clustered_savings_ci_low": float(raw.get("min_clustered_savings_ci_low", manifest.get("target_savings") or 0.60)),
        "max_pass_regressions": int(raw.get("max_pass_regressions", 0)),
        "require_locked_git_sources": bool(raw.get("require_locked_git_sources", True)),
        "require_remote_git_sources": bool(raw.get("require_remote_git_sources", True)),
        "require_adapter_profiles": bool(raw.get("require_adapter_profiles", True)),
        "require_receipt_validation": bool(raw.get("require_receipt_validation", True)),
    }


def _per_task_replicates(manifest: dict[str, Any]) -> dict[str, int]:
    default = int(manifest.get("replicates") or 1)
    return {
        str(task.get("task_id") or f"task-{idx + 1}"): int(task.get("replicates") or default)
        for idx, task in enumerate(manifest["tasks"])
    }


def _readiness_score(card: dict[str, Any]) -> int:
    base = sum(5 for row in card["checks"] if row["passed"])
    if card["readiness_level"] == "smoke_ready":
        base += 10
    elif card["readiness_level"] == "pilot_ready":
        base += 18
    elif card["readiness_level"] == "claim_ready":
        base += 25
    return min(100, base)


def _recommendations(checks: list[dict[str, Any]], gate: dict[str, Any], task_ids: list[str]) -> list[str]:
    by_name = {row["name"]: row for row in checks}
    recs = []
    if not by_name["remote_git_sources"]["passed"]:
        recs.append("Use locked remote Git sources instead of local paths for pilot and claim suites.")
    if not by_name["commit_sha_revisions"]["passed"]:
        recs.append("Pin every git.rev to an immutable 40-hex commit SHA before running held-out evidence.")
    if not by_name["adapter_profiles"]["passed"]:
        recs.append("Move ad hoc patch or repair commands into adapter_profile files so adapter behavior is versioned.")
    if not by_name["pilot_task_clusters"]["passed"]:
        recs.append("Add held-out task clusters until the pilot has at least 10 distinct task_id values.")
    if not by_name["claim_task_clusters"]["passed"]:
        recs.append(f"Add held-out task clusters until the claim suite has at least {gate['min_task_clusters']} distinct task_id values.")
    if not by_name["claim_replicates_per_task"]["passed"]:
        recs.append(f"Set replicates to at least {gate['min_replicates_per_task']} for every claim task.")
    if len(task_ids) != len(set(task_ids)):
        recs.append("Rename duplicate task_id values; clustered statistics depend on unique task clusters.")
    if not recs:
        recs.append("Manifest is preflight-ready; run the suite, verify the evidence bundle, then generate a claim card.")
    return recs


def _missing(checks: list[dict[str, Any]], names: list[str]) -> list[str]:
    by_name = {row["name"]: row for row in checks}
    return [name for name in names if not by_name[name]["passed"]]


def _passed(checks: list[dict[str, Any]], names: list[str]) -> bool:
    by_name = {row["name"]: row for row in checks}
    return all(bool(by_name[name]["passed"]) for name in names)


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def _intervention_count(task: dict[str, Any]) -> int:
    return sum(int(bool(task.get(field))) for field in ("patch_spec", "repair_cmd", "adapter_profile"))


def _is_remote_git(task: dict[str, Any] | None) -> bool:
    if not task:
        return False
    url = str(task.get("url") or "").strip().lower()
    return url.startswith(("https://", "http://", "ssh://", "git://")) or url.startswith("git@")


def _git_url(task: dict[str, Any] | None) -> str | None:
    return str(task.get("url")) if task else None


def _is_full_commit_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{40}", value.strip()))


def _without_readiness_hash(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out.pop("readiness_sha256", None)
    return out


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
