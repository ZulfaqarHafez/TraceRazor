"""Manifest-driven TRICE live suites.

A suite is the user-facing way to run TRICE against multiple real repositories
with deterministic adapters. Each task still gets its own live evidence bundle;
the suite result records an aggregate gate and hashes the child manifests.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .adapters import JsonPatchAdapter
from .evidence import build_manifest, verify_manifest, write_manifest
from .live import DEFAULT_OUT_DIR, VERIFY_CMD, LiveRolloutResult, LiveTask, run_live_learning_loop
from .stats import clustered_bootstrap_mean_ci, claim_gate_from_rounds

SUITE_SCHEMA_VERSION = "trice-suite/v1"
SUITE_RESULT_VERSION = "trice-suite-result/v1"
DEFAULT_SUITE_OUT_DIR = DEFAULT_OUT_DIR.parent / "v2-suite"


@dataclass(frozen=True)
class SuiteTaskSpec:
    task_id: str
    repo: Path
    repo_ref: str
    patch_spec: Path
    patch_spec_ref: str
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
    patch_spec: str
    result_path: str
    report_path: str
    manifest_path: str
    rounds: int
    mean_savings: float
    accepted_rounds: int
    pass_regressions: int
    smoke_gate_passed: bool


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


def suite_task_specs(manifest_path: str | Path) -> list[SuiteTaskSpec]:
    manifest = load_suite_manifest(manifest_path)
    base = Path(manifest_path).resolve().parent
    specs: list[SuiteTaskSpec] = []
    for raw in manifest["tasks"]:
        if not isinstance(raw, dict):
            raise ValueError("each suite task must be an object")
        task_id = str(raw.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("suite task requires task_id")
        repo = _resolve_manifest_path(base, raw.get("repo"), "repo")
        patch_spec = _resolve_manifest_path(base, raw.get("patch_spec"), "patch_spec")
        if not repo.is_dir():
            raise FileNotFoundError(f"suite task repo not found: {repo}")
        if not patch_spec.is_file():
            raise FileNotFoundError(f"suite task patch_spec not found: {patch_spec}")
        specs.append(
            SuiteTaskSpec(
                task_id=task_id,
                repo=repo,
                repo_ref=str(raw["repo"]),
                patch_spec=patch_spec,
                patch_spec_ref=str(raw["patch_spec"]),
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
    specs = suite_task_specs(manifest_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    snapshot_path = out / "trice_suite_manifest.snapshot.json"
    snapshot_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    suite_feedback = user_feedback or manifest.get("user_feedback")
    result = SuiteRunResult(
        schema_version=SUITE_RESULT_VERSION,
        algorithm="trice-v2-suite-live-user-conditioned-rollout",
        suite={
            "name": manifest.get("name") or manifest_path.stem,
            "manifest_snapshot": snapshot_path.name,
            "task_count": len(specs),
            "replicate_count": sum(replicates if replicates is not None else spec.replicates for spec in specs),
            "clustered_by": "task_id",
            "user_feedback": suite_feedback,
        },
    )

    child_results: list[LiveRolloutResult] = []
    cluster_savings: dict[str, list[float]] = {}
    for spec in specs:
        replicate_count = replicates if replicates is not None else spec.replicates
        if replicate_count < 1:
            raise ValueError("replicates must be >= 1")
        for replicate_index in range(1, replicate_count + 1):
            task_out = out / "tasks" / spec.task_id / f"replicate-{replicate_index}"
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
                adapter=JsonPatchAdapter.from_file(spec.patch_spec),
            )
            child_results.append(live_result)
            gate = live_result.claim_gate or {}
            run_savings = round(float(gate.get("mean_savings", 0.0)), 6)
            cluster_savings.setdefault(spec.task_id, []).append(run_savings)
            result.tasks.append(
                SuiteTaskRun(
                    task_id=spec.task_id,
                    replicate_index=replicate_index,
                    repo=spec.repo_ref,
                    patch_spec=spec.patch_spec_ref,
                    result_path=_rel(live_result.result_path, out),
                    report_path=_rel(live_result.report_path, out),
                    manifest_path=_rel(live_result.manifest_path, out),
                    rounds=len(live_result.rounds),
                    mean_savings=run_savings,
                    accepted_rounds=int(gate.get("accepted_rounds", 0)),
                    pass_regressions=int(gate.get("pass_regressions", 0)),
                    smoke_gate_passed=bool(gate.get("smoke_gate_passed", False)),
                )
            )

    all_rounds = [live_round for child in child_results for live_round in child.rounds]
    target = float(manifest.get("target_savings") or 0.60)
    result.claim_gate = claim_gate_from_rounds(all_rounds, target).to_dict()
    result.claim_gate["clustered_savings_ci"] = clustered_bootstrap_mean_ci(cluster_savings).to_dict()
    result.claim_gate["task_cluster_count"] = len(cluster_savings)
    result.claim_gate["replicate_count"] = len(result.tasks)
    result.claim_gate["clustered_by"] = "task_id"

    report_path = out / "trice_suite_report.md"
    result_path = out / "trice_suite_results.json"
    evidence_path = out / "trice_suite_evidence_manifest.json"
    result.report_path = report_path.name
    result.result_path = result_path.name
    result.manifest_path = evidence_path.name
    report_path.write_text(render_suite_report(result), encoding="utf-8")
    result_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = [snapshot_path, report_path]
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
            f"- Local smoke gate passed: {'yes' if gate.get('smoke_gate_passed') else 'no'}",
            f"- Broad claim allowed: {'yes' if gate.get('broad_claim_allowed') else 'no'}",
            f"- Rationale: {gate.get('rationale', 'not computed')}",
            "",
            "## Interpretation",
            "",
            "A suite report is aggregate evidence. Each child task manifest must also",
            "verify, because the suite manifest intentionally hashes child manifests",
            "rather than duplicating every trace and context artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def verify_suite_evidence(manifest_path: str | Path, result_path: str | Path | None = None) -> dict[str, Any]:
    aggregate = verify_manifest(manifest_path, result_path)
    base = Path(manifest_path).parent
    resolved_result = Path(result_path) if result_path else base / "trice_suite_results.json"
    child_verdicts = []
    errors = list(aggregate["errors"])
    if resolved_result.is_file():
        data = json.loads(resolved_result.read_text(encoding="utf-8"))
        for task in data.get("tasks", []):
            child_manifest = base / str(task.get("manifest_path", ""))
            verdict = verify_manifest(child_manifest)
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
    specs = suite_task_specs(path)
    run_count = sum(spec.replicates for spec in specs)
    return {
        "ok": True,
        "name": manifest.get("name") or Path(path).stem,
        "task_count": len(specs),
        "run_count": run_count,
        "schema_version": manifest["schema_version"],
    }


def _resolve_manifest_path(base: Path, value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"suite task requires {field_name}")
    p = Path(value)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _parse_verify_cmd(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple(VERIFY_CMD)
    if isinstance(value, str):
        return tuple(shlex.split(value, posix=False))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError("verify_cmd must be a string or list of strings")


def _rel(path: str | Path | None, base: Path) -> str:
    if path is None:
        return ""
    return Path(path).resolve().relative_to(base.resolve()).as_posix()


def main(argv: list[str] | None = None) -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
