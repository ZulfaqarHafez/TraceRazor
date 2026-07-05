"""Manifest-driven TRICE live suites.

A suite is the user-facing way to run TRICE against multiple real repositories
with deterministic adapters. Each task still gets its own live evidence bundle;
the suite result records an aggregate gate and hashes the child manifests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .adapters import CommandRepairAdapter, JsonPatchAdapter, RepairAdapter
from .evidence import build_manifest, resolve_contained_path, verify_manifest, write_manifest
from .live import DEFAULT_OUT_DIR, VERIFY_CMD, LiveRolloutResult, LiveTask, run_live_learning_loop
from .provenance import fingerprint_tree, hash_file
from .stats import clustered_bootstrap_mean_ci, claim_gate_from_rounds

SUITE_SCHEMA_VERSION = "trice-suite/v1"
SUITE_RESULT_VERSION = "trice-suite-result/v1"
DEFAULT_SUITE_OUT_DIR = DEFAULT_OUT_DIR.parent / "v2-suite"
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


@dataclass(frozen=True)
class SuiteTaskSpec:
    task_id: str
    repo: Path
    repo_ref: str
    patch_spec: Path | None = None
    patch_spec_ref: str | None = None
    repair_cmd: tuple[str, ...] | None = None
    repair_cmd_ref: str | list[str] | None = None
    adapter_profile: Path | None = None
    adapter_profile_ref: str | None = None
    repair_timeout_s: int = 600
    allow_test_edits: bool = False
    source_type: str = "local"
    git_source: dict[str, Any] | None = None
    prompt: str | None = None
    verify_cmd: tuple[str, ...] = tuple(VERIFY_CMD)
    user_feedback: str | None = None
    rounds: int | None = None
    replicates: int = 1


@dataclass
class SuiteTaskRun:
    task_id: str
    replicate_index: int
    repo: str
    patch_spec: str | None
    repair_cmd: str | list[str] | None
    adapter_profile: str | None
    source: dict[str, Any]
    result_path: str
    report_path: str
    manifest_path: str
    rounds: int
    mean_savings: float
    accepted_rounds: int
    pass_regressions: int
    smoke_gate_passed: bool
    evidence_recall: float | None = None
    evidence_recall_passed: bool | None = None


@dataclass
class SuiteRunResult:
    schema_version: str
    algorithm: str
    suite: dict[str, Any]
    tasks: list[SuiteTaskRun] = field(default_factory=list)
    claim_gate: dict[str, Any] | None = None
    report_path: str | None = None
    result_path: str | None = None
    manifest_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "suite": self.suite,
            "tasks": [asdict(t) for t in self.tasks],
            "claim_gate": self.claim_gate,
            "report_path": self.report_path,
            "result_path": self.result_path,
            "manifest_path": self.manifest_path,
        }


def load_suite_manifest(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != SUITE_SCHEMA_VERSION:
        raise ValueError(f"suite manifest schema_version must be {SUITE_SCHEMA_VERSION!r}")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("suite manifest requires at least one task")
    return data


def scaffold_suite_manifest(source_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Build a locked remote-git TRICE suite manifest from a compact source list."""

    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    if isinstance(source, list):
        source = {"tasks": source}
    if not isinstance(source, dict):
        raise ValueError("remote git source list must be a JSON object")
    tasks = source.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("remote git source list requires a non-empty tasks array")

    default_verify = _parse_verify_cmd(source.get("verify_cmd"))
    manifest: dict[str, Any] = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "name": str(source.get("name") or "trice-remote-git-suite"),
        "user_feedback": str(source.get("user_feedback") or "real runs, not replay; target 60% savings"),
        "target_savings": float(source.get("target_savings") or 0.60),
        "rounds": int(source.get("rounds") or 1),
        "replicates": int(source.get("replicates") or 1),
        "s_tier_gate": dict(
            source.get("s_tier_gate")
            or {
                "min_task_clusters": 50,
                "min_replicates_per_task": 3,
                "require_locked_git_sources": True,
                "require_remote_git_sources": True,
                "require_adapter_profiles": True,
                "min_mean_savings": 0.60,
                "min_clustered_savings_ci_low": 0.60,
                "min_evidence_recall": 0.95,
                "max_pass_regressions": 0,
            }
        ),
        "tasks": [],
    }

    top_level_intervention = {
        field: source.get(field)
        for field in ("adapter_profile", "patch_spec", "repair_cmd")
        if source.get(field)
    }
    for raw in tasks:
        if not isinstance(raw, dict):
            raise ValueError("each remote git task must be an object")
        task_id = str(raw.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("remote git task requires task_id")
        git_info = _scaffold_git_source(raw)
        intervention = _scaffold_intervention(raw, top_level_intervention)
        verify_cmd = list(_parse_verify_cmd(raw.get("verify_cmd"))) if raw.get("verify_cmd") is not None else list(default_verify)
        prompt = str(raw.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"remote git task {task_id} requires prompt")
        task: dict[str, Any] = {
            "task_id": task_id,
            "git": git_info,
            "prompt": prompt,
            "verify_cmd": verify_cmd,
            **intervention,
        }
        for field in ("rounds", "replicates", "user_feedback", "repair_timeout_s", "allow_test_edits"):
            if raw.get(field) is not None:
                task[field] = raw[field]
        manifest["tasks"].append(task)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def suite_task_specs(manifest_path: str | Path, source_root: str | Path | None = None) -> list[SuiteTaskSpec]:
    manifest = load_suite_manifest(manifest_path)
    base = Path(manifest_path).resolve().parent
    source_root_path = Path(source_root).resolve() if source_root is not None else None
    specs: list[SuiteTaskSpec] = []
    for raw in manifest["tasks"]:
        if not isinstance(raw, dict):
            raise ValueError("each suite task must be an object")
        task_id = str(raw.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("suite task requires task_id")
        _safe_task_id(task_id)
        repo, repo_ref, source_type, git_source = _resolve_task_repo(base, raw, task_id, source_root_path)
        has_patch = bool(raw.get("patch_spec"))
        has_repair_cmd = bool(raw.get("repair_cmd"))
        has_adapter_profile = bool(raw.get("adapter_profile"))
        if sum(int(v) for v in (has_patch, has_repair_cmd, has_adapter_profile)) != 1:
            raise ValueError("suite task requires exactly one of patch_spec, repair_cmd, or adapter_profile")
        patch_spec = _resolve_manifest_path(base, raw.get("patch_spec"), "patch_spec") if has_patch else None
        repair_cmd = _parse_cmd(raw.get("repair_cmd")) if has_repair_cmd else None
        adapter_profile = _resolve_manifest_path(base, raw.get("adapter_profile"), "adapter_profile") if has_adapter_profile else None
        if not repo.is_dir():
            raise FileNotFoundError(f"suite task repo not found: {repo}")
        if patch_spec is not None and not patch_spec.is_file():
            raise FileNotFoundError(f"suite task patch_spec not found: {patch_spec}")
        if adapter_profile is not None and not adapter_profile.is_file():
            raise FileNotFoundError(f"suite task adapter_profile not found: {adapter_profile}")
        specs.append(
            SuiteTaskSpec(
                task_id=task_id,
                repo=repo,
                repo_ref=repo_ref,
                patch_spec=patch_spec,
                patch_spec_ref=str(raw["patch_spec"]) if has_patch else None,
                repair_cmd=repair_cmd,
                repair_cmd_ref=raw.get("repair_cmd") if has_repair_cmd else None,
                adapter_profile=adapter_profile,
                adapter_profile_ref=str(raw["adapter_profile"]) if has_adapter_profile else None,
                repair_timeout_s=int(raw.get("repair_timeout_s") or 600),
                allow_test_edits=bool(raw.get("allow_test_edits", False)),
                source_type=source_type,
                git_source=git_source,
                prompt=raw.get("prompt"),
                verify_cmd=_parse_verify_cmd(raw.get("verify_cmd")),
                user_feedback=raw.get("user_feedback"),
                rounds=int(raw["rounds"]) if raw.get("rounds") is not None else None,
                replicates=int(raw["replicates"]) if raw.get("replicates") is not None else int(manifest.get("replicates") or 1),
            )
        )
    return specs


def run_suite_manifest(
    manifest_path: str | Path,
    *,
    out_dir: str | Path = DEFAULT_SUITE_OUT_DIR,
    rounds: int | None = None,
    replicates: int | None = None,
    user_feedback: str | None = None,
) -> SuiteRunResult:
    manifest_path = Path(manifest_path)
    manifest = load_suite_manifest(manifest_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = suite_task_specs(manifest_path, source_root=out / "_sources")

    snapshot_path = out / "trice_suite_manifest.snapshot.json"
    sources_path = out / "trice_suite_sources.json"
    snapshot_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    suite_feedback = user_feedback or manifest.get("user_feedback")
    result = SuiteRunResult(
        schema_version=SUITE_RESULT_VERSION,
        algorithm="trice-v2-suite-live-user-conditioned-rollout",
        suite={
            "name": manifest.get("name") or manifest_path.stem,
            "manifest_snapshot": snapshot_path.name,
            "source_manifest": sources_path.name,
            "task_count": len(specs),
            "replicate_count": sum(replicates if replicates is not None else spec.replicates for spec in specs),
            "clustered_by": "task_id",
            "user_feedback": suite_feedback,
        },
    )

    child_results: list[LiveRolloutResult] = []
    cluster_savings: dict[str, list[float]] = {}
    source_records: list[dict[str, Any]] = []
    for spec in specs:
        replicate_count = replicates if replicates is not None else spec.replicates
        if replicate_count < 1:
            raise ValueError("replicates must be >= 1")
        source = _source_record(spec)
        source_records.append(source)
        for replicate_index in range(1, replicate_count + 1):
            task_out = out / "tasks" / _safe_task_id(spec.task_id) / f"replicate-{replicate_index}"
            feedback = user_feedback or spec.user_feedback or manifest.get("user_feedback")
            task = LiveTask.from_repo(
                spec.repo,
                task_id=spec.task_id,
                prompt=spec.prompt,
                verify_cmd=spec.verify_cmd,
            )
            live_result = run_live_learning_loop(
                [task],
                out_dir=task_out,
                user_feedback=feedback,
                rounds=rounds if rounds is not None else spec.rounds or manifest.get("rounds"),
                adapter=_adapter_for_spec(spec),
            )
            child_results.append(live_result)
            gate = live_result.claim_gate or {}
            run_savings = round(float(gate.get("mean_savings", 0.0)), 6)
            recall_value = gate.get("evidence_recall_minimum")
            recall_passed = int(gate.get("evidence_recall_failures", 0)) == 0
            cluster_savings.setdefault(spec.task_id, []).append(run_savings)
            result.tasks.append(
                SuiteTaskRun(
                    task_id=spec.task_id,
                    replicate_index=replicate_index,
                    repo=spec.repo_ref,
                    patch_spec=spec.patch_spec_ref,
                    repair_cmd=spec.repair_cmd_ref,
                    adapter_profile=spec.adapter_profile_ref,
                    source=source,
                    result_path=_rel(live_result.result_path, out),
                    report_path=_rel(live_result.report_path, out),
                    manifest_path=_rel(live_result.manifest_path, out),
                    rounds=len(live_result.rounds),
                    mean_savings=run_savings,
                    accepted_rounds=int(gate.get("accepted_rounds", 0)),
                    pass_regressions=int(gate.get("pass_regressions", 0)),
                    smoke_gate_passed=bool(gate.get("smoke_gate_passed", False)),
                    evidence_recall=round(float(recall_value), 6) if recall_value is not None else None,
                    evidence_recall_passed=recall_passed,
                )
            )

    all_rounds = [live_round for child in child_results for live_round in child.rounds]
    target = float(manifest.get("target_savings") or 0.60)
    result.claim_gate = claim_gate_from_rounds(all_rounds, target).to_dict()
    result.claim_gate["clustered_savings_ci"] = clustered_bootstrap_mean_ci(cluster_savings).to_dict()
    result.claim_gate["task_cluster_count"] = len(cluster_savings)
    result.claim_gate["replicate_count"] = len(result.tasks)
    result.claim_gate["clustered_by"] = "task_id"
    result.claim_gate["adapter_breakdown"] = _adapter_breakdown(result.tasks)
    result.claim_gate["failure_breakdown"] = _failure_breakdown(result.tasks)
    result.claim_gate["s_tier_gate"] = _s_tier_gate(
        result,
        specs,
        cluster_savings,
        target,
        manifest.get("s_tier_gate") or {},
    )

    report_path = out / "trice_suite_report.md"
    result_path = out / "trice_suite_results.json"
    evidence_path = out / "trice_suite_evidence_manifest.json"
    result.report_path = report_path.name
    result.result_path = result_path.name
    result.manifest_path = evidence_path.name
    sources_path.write_text(json.dumps({"schema_version": "trice-suite-sources/v1", "sources": source_records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_suite_report(result), encoding="utf-8")
    result_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = [snapshot_path, sources_path, report_path]
    artifacts.extend(out / task.manifest_path for task in result.tasks)
    manifest_out = build_manifest(
        result.to_dict(),
        result_path=result_path,
        artifact_paths=artifacts,
        algorithm=result.algorithm,
        notes=[
            "Manifest-driven TRICE suite",
            "Each child task is a real live rollout with its own evidence manifest",
            "Replicates use fresh child output bundles",
            "Repo tree fingerprints and intervention provenance captured before execution",
            "Aggregate manifest hashes suite snapshot and child manifests",
        ],
    )
    write_manifest(manifest_out, evidence_path)
    return result


def render_suite_report(result: SuiteRunResult) -> str:
    gate = result.claim_gate or {}
    ci = gate.get("savings_ci") or {}
    clustered_ci = gate.get("clustered_savings_ci") or {}
    lines = [
        "# TRICE Live Suite Report",
        "",
        f"Suite: `{result.suite.get('name', 'unnamed')}`",
        f"Algorithm: `{result.algorithm}`",
        f"Evidence manifest: `{Path(result.manifest_path).name if result.manifest_path else 'pending'}`",
        "",
        "## Tasks",
        "",
            "| Task | Replicate | Rounds | Mean savings | Accepted | Pass regressions | Child manifest |",
            "|---|---:|---:|---:|---:|---:|---|",
    ]
    for task in result.tasks:
        lines.append(
            "| {task} | {replicate} | {rounds} | {savings:.1%} | {accepted} | {regressions} | `{manifest}` |".format(
                task=task.task_id,
                replicate=task.replicate_index,
                rounds=task.rounds,
                savings=task.mean_savings,
                accepted=task.accepted_rounds,
                regressions=task.pass_regressions,
                manifest=task.manifest_path,
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate Gate",
            "",
            f"- Mean savings: {gate.get('mean_savings', 0.0):.1%}",
            f"- Savings 95% bootstrap CI: [{ci.get('low', 0.0):.1%}, {ci.get('high', 0.0):.1%}]",
            f"- Clustered-by-task savings 95% CI: [{clustered_ci.get('low', 0.0):.1%}, {clustered_ci.get('high', 0.0):.1%}]",
            f"- Task clusters: {gate.get('task_cluster_count', 0)}",
            f"- Replicates: {gate.get('replicate_count', 0)}",
            f"- Pass regressions: {gate.get('pass_regressions', 0)}",
            f"- Evidence recall minimum: {gate.get('evidence_recall_minimum', 0.0):.1%}",
            f"- Evidence recall failures: {gate.get('evidence_recall_failures', 0)}",
            f"- Local smoke gate passed: {'yes' if gate.get('smoke_gate_passed') else 'no'}",
            f"- Broad claim allowed: {'yes' if gate.get('broad_claim_allowed') else 'no'}",
            f"- Rationale: {gate.get('rationale', 'not computed')}",
            f"- S-tier gate passed: {'yes' if (gate.get('s_tier_gate') or {}).get('passed') else 'no'}",
            "",
            "## Adapter Breakdown",
            "",
            "| Adapter | Runs | Mean savings | Pass regressions |",
            "|---|---:|---:|---:|",
        ]
    )
    for adapter_type, row in sorted((gate.get("adapter_breakdown") or {}).items()):
        lines.append(
            f"| {adapter_type} | {row.get('runs', 0)} | {row.get('mean_savings', 0.0):.1%} | {row.get('pass_regressions', 0)} |"
        )
    failures = gate.get("failure_breakdown") or {}
    s_tier = gate.get("s_tier_gate") or {}
    lines.extend(
        [
            "",
            "## Failure Modes",
            "",
            f"- Pass regression runs: {failures.get('pass_regression_runs', 0)}",
            f"- Unaccepted runs: {failures.get('unaccepted_runs', 0)}",
            f"- Failed smoke-gate runs: {failures.get('failed_smoke_gate_runs', 0)}",
            "",
            "## S-Tier Gate",
            "",
            f"- Claim level: `{s_tier.get('claim_level', 'not_evaluated')}`",
            f"- Passed: {'yes' if s_tier.get('passed') else 'no'}",
            f"- Missing requirements: {', '.join(s_tier.get('missing_requirements') or []) or 'none'}",
            f"- Rationale: {s_tier.get('rationale', 'not evaluated')}",
            "",
            "## Interpretation",
            "",
            "A suite report is aggregate evidence. Each child task manifest must also",
            "verify, because the suite manifest intentionally hashes child manifests",
            "rather than duplicating every trace and context artifact.",
            "Repo tree fingerprints and intervention provenance are recorded in",
            "`trice_suite_sources.json` before live execution. JSON patch tasks",
            "record patch-spec SHA-256 hashes; command tasks record argv, timeout,",
            "and test-edit policy; adapter-profile tasks record profile SHA-256.",
            "",
        ]
    )
    return "\n".join(lines)


def _adapter_breakdown(tasks: list[SuiteTaskRun]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[SuiteTaskRun]] = {}
    for task in tasks:
        adapter_type = str(task.source.get("adapter_type") or "unknown")
        groups.setdefault(adapter_type, []).append(task)
    out: dict[str, dict[str, Any]] = {}
    for adapter_type, rows in groups.items():
        out[adapter_type] = {
            "runs": len(rows),
            "mean_savings": round(sum(row.mean_savings for row in rows) / max(1, len(rows)), 6),
            "pass_regressions": sum(row.pass_regressions for row in rows),
            "accepted_runs": sum(1 for row in rows if row.accepted_rounds > 0),
        }
    return out


def _failure_breakdown(tasks: list[SuiteTaskRun]) -> dict[str, int]:
    return {
        "pass_regression_runs": sum(1 for task in tasks if task.pass_regressions > 0),
        "unaccepted_runs": sum(1 for task in tasks if task.accepted_rounds == 0),
        "failed_smoke_gate_runs": sum(1 for task in tasks if not task.smoke_gate_passed),
    }


def _s_tier_gate(
    result: SuiteRunResult,
    specs: list[SuiteTaskSpec],
    cluster_savings: dict[str, list[float]],
    target_savings: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = result.claim_gate or {}
    clustered_ci = gate.get("clustered_savings_ci") or {}
    failures = gate.get("failure_breakdown") or {}
    task_counts = {task_id: len(values) for task_id, values in cluster_savings.items()}
    min_task_clusters = int(config.get("min_task_clusters", 50))
    min_replicates_per_task = int(config.get("min_replicates_per_task", 3))
    min_mean_savings = float(config.get("min_mean_savings", target_savings))
    min_clustered_ci_low = float(config.get("min_clustered_savings_ci_low", target_savings))
    max_pass_regressions = int(config.get("max_pass_regressions", 0))
    min_evidence_recall = float(config.get("min_evidence_recall", 0.95))
    require_locked_git_sources = bool(config.get("require_locked_git_sources", True))
    require_remote_git_sources = bool(config.get("require_remote_git_sources", True))
    require_adapter_profiles = bool(config.get("require_adapter_profiles", True))
    require_receipt_validation = bool(config.get("require_receipt_validation", True))

    checks = {
        "mean_savings": _gate_check(
            float(gate.get("mean_savings", 0.0)) >= min_mean_savings,
            observed=gate.get("mean_savings", 0.0),
            required=f">= {min_mean_savings:.3f}",
        ),
        "clustered_savings_ci_low": _gate_check(
            float(clustered_ci.get("low", 0.0)) >= min_clustered_ci_low,
            observed=clustered_ci.get("low", 0.0),
            required=f">= {min_clustered_ci_low:.3f}",
        ),
        "pass_regressions": _gate_check(
            int(gate.get("pass_regressions", 0)) <= max_pass_regressions,
            observed=gate.get("pass_regressions", 0),
            required=f"<= {max_pass_regressions}",
        ),
        "accepted_runs": _gate_check(
            int(failures.get("unaccepted_runs", 0)) == 0,
            observed=failures.get("unaccepted_runs", 0),
            required="0 unaccepted runs",
        ),
        "evidence_recall": _gate_check(
            float(gate.get("evidence_recall_minimum", 0.0)) >= min_evidence_recall
            and int(gate.get("evidence_recall_failures", 0)) == 0
            and all(task.evidence_recall is not None for task in result.tasks),
            observed={
                "minimum": gate.get("evidence_recall_minimum"),
                "failures": gate.get("evidence_recall_failures"),
            },
            required=f">= {min_evidence_recall:.3f} on every accepted optimized run",
        ),
        "task_clusters": _gate_check(
            len(cluster_savings) >= min_task_clusters,
            observed=len(cluster_savings),
            required=f">= {min_task_clusters}",
        ),
        "replicates_per_task": _gate_check(
            bool(task_counts) and all(count >= min_replicates_per_task for count in task_counts.values()),
            observed=task_counts,
            required=f"each task >= {min_replicates_per_task}",
        ),
        "locked_git_sources": _gate_check(
            (not require_locked_git_sources)
            or all(spec.source_type == "git" and spec.git_source and spec.git_source.get("resolved_commit") for spec in specs),
            observed=[spec.source_type for spec in specs],
            required="all tasks use locked git sources" if require_locked_git_sources else "not required",
        ),
        "remote_git_sources": _gate_check(
            (not require_remote_git_sources)
            or all(_is_remote_git_source(spec) for spec in specs),
            observed=[(spec.git_source or {}).get("url") if spec.git_source else spec.source_type for spec in specs],
            required="all tasks use remote Git URLs" if require_remote_git_sources else "not required",
        ),
        "adapter_profiles": _gate_check(
            (not require_adapter_profiles) or all(spec.adapter_profile is not None for spec in specs),
            observed=[spec.adapter_profile_ref for spec in specs],
            required="all tasks use adapter_profile" if require_adapter_profiles else "not required",
        ),
        "receipt_validation": _gate_check(
            True,
            observed="enabled",
            required="enabled" if require_receipt_validation else "not required",
        ),
    }
    missing = [name for name, check in checks.items() if not check["passed"]]
    passed = not missing
    return {
        "schema_version": "trice-s-tier-gate/v1",
        "passed": passed,
        "claim_level": "s_tier" if passed else "not_s_tier",
        "requirements": checks,
        "missing_requirements": missing,
        "config": {
            "min_task_clusters": min_task_clusters,
            "min_replicates_per_task": min_replicates_per_task,
            "min_mean_savings": min_mean_savings,
            "min_clustered_savings_ci_low": min_clustered_ci_low,
            "max_pass_regressions": max_pass_regressions,
            "min_evidence_recall": min_evidence_recall,
            "require_locked_git_sources": require_locked_git_sources,
            "require_remote_git_sources": require_remote_git_sources,
            "require_adapter_profiles": require_adapter_profiles,
            "require_receipt_validation": require_receipt_validation,
        },
        "rationale": (
            "suite evidence clears the S-tier claim gate"
            if passed
            else "suite evidence is useful but not broad enough for an S-tier claim"
        ),
    }


def _gate_check(passed: bool, *, observed: Any, required: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "observed": observed, "required": required}


def _is_remote_git_source(spec: SuiteTaskSpec) -> bool:
    if spec.source_type != "git" or not spec.git_source:
        return False
    url = str(spec.git_source.get("url") or "").strip().lower()
    return url.startswith(("https://", "http://", "ssh://", "git://")) or url.startswith("git@")


def _is_remote_git_url(url: str) -> bool:
    url_l = url.strip().lower()
    return url_l.startswith(("https://", "http://", "ssh://", "git://")) or url_l.startswith("git@")


def _scaffold_git_source(raw: dict[str, Any]) -> dict[str, Any]:
    git_info = raw.get("git") if isinstance(raw.get("git"), dict) else {}
    url = str(git_info.get("url") or raw.get("url") or "").strip()
    rev = str(git_info.get("rev") or raw.get("rev") or "").strip()
    subdir = str(git_info.get("subdir") or raw.get("subdir") or "").strip()
    if not url or not rev:
        raise ValueError("remote git task requires url and rev")
    if not _is_remote_git_url(url):
        raise ValueError(f"remote git task url must be a remote git URL: {url}")
    git: dict[str, Any] = {"url": url, "rev": rev}
    if subdir:
        git["subdir"] = subdir
    return git


def _scaffold_intervention(raw: dict[str, Any], top_level: dict[str, Any]) -> dict[str, Any]:
    merged = {field: raw.get(field) if raw.get(field) is not None else top_level.get(field) for field in ("adapter_profile", "patch_spec", "repair_cmd")}
    active = {field: value for field, value in merged.items() if value}
    if len(active) != 1:
        raise ValueError("remote git task requires exactly one of adapter_profile, patch_spec, or repair_cmd")
    field, value = next(iter(active.items()))
    if field == "repair_cmd":
        return {field: list(_parse_cmd(value))}
    return {field: str(value)}


def verify_suite_evidence(manifest_path: str | Path, result_path: str | Path | None = None) -> dict[str, Any]:
    aggregate = verify_manifest(manifest_path, result_path)
    base = Path(manifest_path).parent
    resolved_result = Path(result_path) if result_path else base / "trice_suite_results.json"
    child_verdicts = []
    errors = list(aggregate["errors"])
    if resolved_result.is_file():
        data = json.loads(resolved_result.read_text(encoding="utf-8"))
        for task in data.get("tasks", []):
            raw_manifest_path = str(task.get("manifest_path", ""))
            try:
                child_manifest = resolve_contained_path(base, raw_manifest_path, "suite child manifest")
                verdict = verify_manifest(child_manifest)
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                child_manifest = base / "__invalid__"
                verdict = {"ok": False, "errors": [str(exc)]}
            child_verdicts.append(
                {
                    "task_id": task.get("task_id"),
                    "manifest_path": str(child_manifest),
                    "ok": verdict["ok"],
                    "errors": verdict["errors"],
                }
            )
            errors.extend(f"{task.get('task_id')}: {err}" for err in verdict["errors"])
    else:
        errors.append(f"missing suite result file: {resolved_result}")
    return {
        "ok": aggregate["ok"] and not errors,
        "errors": errors,
        "aggregate": aggregate,
        "children": child_verdicts,
    }


def validate_suite_manifest_file(path: str | Path) -> dict[str, Any]:
    manifest = load_suite_manifest(path)
    tasks = manifest["tasks"]
    run_count = sum(int(task.get("replicates") or manifest.get("replicates") or 1) for task in tasks)
    for raw in tasks:
        if not isinstance(raw, dict):
            raise ValueError("each suite task must be an object")
        if not raw.get("repo") and not raw.get("git"):
            raise ValueError("suite task requires repo or git")
        if raw.get("repo") and raw.get("git"):
            raise ValueError("suite task must not set both repo and git")
        if sum(int(bool(raw.get(field))) for field in ("patch_spec", "repair_cmd", "adapter_profile")) != 1:
            raise ValueError("suite task requires exactly one of patch_spec, repair_cmd, or adapter_profile")
    return {
        "ok": True,
        "name": manifest.get("name") or Path(path).stem,
        "task_count": len(tasks),
        "run_count": run_count,
        "schema_version": manifest["schema_version"],
    }


def _source_record(spec: SuiteTaskSpec) -> dict[str, Any]:
    tree = fingerprint_tree(spec.repo).to_dict()
    record = {
        "task_id": spec.task_id,
        "repo": spec.repo_ref,
        "source_type": spec.source_type,
        "git": spec.git_source,
        "repo_tree": tree,
        "verify_cmd": list(spec.verify_cmd),
    }
    if spec.patch_spec is not None:
        record.update(
            {
                "adapter_type": "json_patch",
                "patch_spec": spec.patch_spec_ref,
                "patch_sha256": hash_file(spec.patch_spec),
            }
        )
    elif spec.adapter_profile is not None:
        profile = json.loads(spec.adapter_profile.read_text(encoding="utf-8"))
        record.update(
            {
                "adapter_type": "command_profile",
                "adapter_profile": spec.adapter_profile_ref,
                "adapter_profile_sha256": hash_file(spec.adapter_profile),
                "adapter_profile_name": profile.get("name"),
                "repair_cmd": profile.get("command"),
                "repair_timeout_s": profile.get("timeout_s") or profile.get("repair_timeout_s"),
                "allow_test_edits": bool(profile.get("allow_test_edits", False)),
            }
        )
    else:
        record.update(
            {
                "adapter_type": "command",
                "repair_cmd": spec.repair_cmd_ref,
                "repair_timeout_s": spec.repair_timeout_s,
                "allow_test_edits": spec.allow_test_edits,
            }
        )
    return record


def _adapter_for_spec(spec: SuiteTaskSpec) -> RepairAdapter:
    if spec.patch_spec is not None:
        return JsonPatchAdapter.from_file(spec.patch_spec)
    if spec.adapter_profile is not None:
        return CommandRepairAdapter.from_file(spec.adapter_profile)
    if spec.repair_cmd is None:
        raise ValueError(f"suite task {spec.task_id} has no repair intervention")
    return CommandRepairAdapter(
        command=spec.repair_cmd,
        timeout_s=spec.repair_timeout_s,
        allow_test_edits=spec.allow_test_edits,
    )


def _resolve_task_repo(base: Path, raw: dict[str, Any], task_id: str, source_root: Path | None) -> tuple[Path, str, str, dict[str, Any] | None]:
    has_repo = bool(raw.get("repo"))
    has_git = bool(raw.get("git"))
    if has_repo == has_git:
        raise ValueError("suite task requires exactly one of repo or git")
    if has_repo:
        repo_ref = str(raw["repo"])
        return _resolve_manifest_path(base, repo_ref, "repo"), repo_ref, "local", None
    if source_root is None:
        raise ValueError("git suite task needs source_root for materialization")
    git_info = raw["git"]
    if not isinstance(git_info, dict):
        raise ValueError("git source must be an object")
    url = str(git_info.get("url") or "").strip()
    rev = str(git_info.get("rev") or "").strip()
    subdir = str(git_info.get("subdir") or "").strip()
    if not url or not rev:
        raise ValueError("git source requires url and rev")
    checkout = _materialize_git_source(url, rev, source_root / _safe_task_id(task_id))
    resolved_commit = _git(["rev-parse", "HEAD"], cwd=checkout).strip()
    git_dir = checkout / ".git"
    if git_dir.exists():
        _remove_tree(git_dir)
    repo = checkout / subdir if subdir else checkout
    if not repo.resolve().is_relative_to(checkout.resolve()):
        raise ValueError(f"git source subdir escapes checkout: {subdir}")
    ref = f"git+{url}@{rev}" + (f"#{subdir}" if subdir else "")
    return repo, ref, "git", {"url": url, "rev": rev, "resolved_commit": resolved_commit, "subdir": subdir or None}


def _resolve_manifest_path(base: Path, value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"suite task requires {field_name}")
    p = Path(value)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _materialize_git_source(url: str, rev: str, dest: Path) -> Path:
    if dest.exists():
        _remove_tree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(["clone", "--no-checkout", "--", url, str(dest)], cwd=None)
    _git(["checkout", "--detach", rev], cwd=dest)
    return dest.resolve()


def _git(args: list[str], cwd: Path | None) -> str:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stdout + "\n" + proc.stderr).strip())
    return proc.stdout


def _safe_task_id(task_id: str) -> str:
    if not _TASK_ID_RE.fullmatch(task_id):
        raise ValueError(
            "suite task_id must start with an ASCII letter or digit and contain only letters, digits, '_' or '-'"
        )
    return task_id


def _remove_tree(path: Path) -> None:
    def onerror(func, value, _exc):
        os.chmod(value, stat.S_IWRITE)
        func(value)

    shutil.rmtree(path, onerror=onerror)


def _parse_verify_cmd(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple(VERIFY_CMD)
    if isinstance(value, str):
        return tuple(shlex.split(value, posix=False))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError("verify_cmd must be a string or list of strings")


def _parse_cmd(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return tuple(shlex.split(value, posix=False))
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return tuple(value)
    raise ValueError("repair_cmd must be a non-empty string or list of strings")


def _rel(path: str | Path | None, base: Path) -> str:
    if path is None:
        return ""
    return Path(path).resolve().relative_to(base.resolve()).as_posix()


def main(argv: list[str] | None = None) -> int:
    if argv and argv[0] == "scaffold":
        return _scaffold_main(argv[1:])
    if argv and argv[0] == "readiness":
        from .readiness import main as readiness_main

        return readiness_main(argv[1:])
    if argv and argv[0] == "verify-readiness":
        from .readiness import verify_main as readiness_verify_main

        return readiness_verify_main(argv[1:])
    ap = argparse.ArgumentParser(description="Run a manifest-driven TRICE live suite.")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_SUITE_OUT_DIR)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--user-feedback", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verify-suite", type=Path, default=None, help="Deep-verify a TRICE suite evidence manifest and exit.")
    args = ap.parse_args(argv)

    if args.verify_suite:
        verdict = verify_suite_evidence(args.verify_suite)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1

    result = run_suite_manifest(
        args.manifest,
        out_dir=args.out_dir,
        rounds=args.rounds,
        replicates=args.replicates,
        user_feedback=args.user_feedback,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        gate = result.claim_gate or {}
        print(
            f"TRICE suite: {len(result.tasks)} run(s), "
            f"mean savings={gate.get('mean_savings', 0.0):.1%}, "
            f"smoke_gate={bool(gate.get('smoke_gate_passed'))}"
        )
        print(f"report: {args.out_dir / str(result.report_path)}")
        print(f"json  : {args.out_dir / str(result.result_path)}")
        print(f"manifest: {args.out_dir / str(result.manifest_path)}")
    return 0


def _scaffold_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a locked remote-git TRICE suite manifest.")
    ap.add_argument("--source", type=Path, required=True, help="remote-git-list JSON")
    ap.add_argument("--out", type=Path, required=True, help="suite manifest JSON to write")
    ap.add_argument("--json", action="store_true", help="Print the generated manifest.")
    args = ap.parse_args(argv)
    manifest = scaffold_suite_manifest(args.source, args.out)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.out} ({len(manifest['tasks'])} task(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
