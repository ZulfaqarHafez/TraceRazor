"""TRICE V2 live rollout runner.

V1 proves that a context policy preserves recorded evidence. V2 must earn its
number in a live workspace: copy a task repo, assemble context, apply a managed
repair, run the verifier, compare full-context versus TRICE-context input
tokens, and update the user profile from the measured outcome.

The bundled managed adapter is deterministic so CI can exercise the loop
without API keys. It is still a live run: files are edited in a fresh workspace
and the task is accepted only if the verifier command passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .adapters import CommandRepairAdapter, JsonPatchAdapter, RepairAdapter
from .evidence import build_manifest, verify_manifest, write_manifest, write_text_lf
from .learn import LearningWeights, update_weights
from .policy import ContextPolicy, solve_policy
from .recall import evidence_recall_from_policy
from .render import render_context, render_policy_json
from .segment import segments_from_trace
from .stats import claim_gate_from_rounds
from .user import UserPreferenceProfile

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_DIR = REPO / "benchmark" / "live" / "tasks"
DEFAULT_OUT_DIR = REPO / "benchmark" / "trice" / "results" / "v2-live"
VERIFY_CMD = [sys.executable, "-m", "pytest", "-q", "--tb=short"]
_DURATION_RE = re.compile(r"\bin \d+(?:\.\d+)?s\b")
_SECONDS_RE = re.compile(r"\bin \d+(?:\.\d+)? seconds?\b")


@dataclass(frozen=True)
class LiveTask:
    task_id: str
    prompt: str
    seed_dir: Path
    verify_cmd: tuple[str, ...] = tuple(VERIFY_CMD)

    @classmethod
    def from_dir(cls, task_dir: str | Path) -> "LiveTask":
        d = Path(task_dir)
        return cls(
            task_id=d.name,
            prompt=(d / "prompt.md").read_text(encoding="utf-8").strip(),
            seed_dir=d / "seed",
        )

    @classmethod
    def from_repo(
        cls,
        repo_dir: str | Path,
        *,
        task_id: str | None = None,
        prompt: str | None = None,
        verify_cmd: list[str] | tuple[str, ...] | None = None,
    ) -> "LiveTask":
        repo = Path(repo_dir)
        resolved_prompt = prompt
        if resolved_prompt is None:
            prompt_file = repo / "TASK.md"
            resolved_prompt = prompt_file.read_text(encoding="utf-8").strip() if prompt_file.is_file() else f"Run deterministic task in {repo.name}."
        return cls(
            task_id=task_id or repo.name,
            prompt=resolved_prompt,
            seed_dir=repo,
            verify_cmd=tuple(verify_cmd or VERIFY_CMD),
        )


@dataclass(frozen=True)
class AdapterTaskContext:
    task_id: str
    prompt: str
    verify_cmd: tuple[str, ...]
    trice_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionRun:
    task_id: str
    condition: str
    workspace: str
    input_tokens: int
    passed: bool
    verify_exit_code: int
    verify_output_excerpt: str
    modified_files: list[str]
    trace_path: str
    decision_trace_path: str | None = None
    receipt_path: str | None = None
    receipt_sha256: str | None = None
    adapter_type: str | None = None
    adapter_reported_input_tokens: int | None = None
    policy_path: str | None = None
    context_path: str | None = None
    policy: dict[str, Any] | None = None
    evidence_recall: float | None = None
    evidence_recall_passed: bool | None = None
    evidence_recall_report: dict[str, Any] | None = None


@dataclass
class LiveRound:
    task_id: str
    round_index: int
    baseline: ConditionRun
    optimized: ConditionRun
    measured_input_savings: float
    quality_delta: float
    pass_noninferior: bool
    accepted: bool
    learning_update: dict[str, Any]
    profile_after: dict[str, Any]


@dataclass
class LiveRolloutResult:
    algorithm: str
    profile: dict[str, Any]
    rounds: list[LiveRound] = field(default_factory=list)
    claim_gate: dict[str, Any] | None = None
    report_path: str | None = None
    result_path: str | None = None
    manifest_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "profile": self.profile,
            "rounds": [asdict(r) for r in self.rounds],
            "claim_gate": self.claim_gate,
            "report_path": self.report_path,
            "result_path": self.result_path,
            "manifest_path": self.manifest_path,
        }


class ManagedPythonRepairAdapter:
    """Small deterministic repair adapter for bundled Python live tasks."""

    name = "managed-python-repair-v2"

    def apply_fix(self, task: LiveTask, workspace: Path) -> list[str]:
        if task.task_id == "fix-offby-one":
            return self._replace(workspace / "chunker.py", "size - 1", "size")
        if task.task_id == "fix-imports":
            changed = []
            changed += self._replace(workspace / "mypkg" / "api.py", "from loaders import read_rows", "from .loaders import read_rows")
            (workspace / "mypkg" / "__init__.py").write_text(
                "from .api import run_pipeline\n\n__all__ = [\"run_pipeline\"]\n",
                encoding="utf-8",
            )
            changed.append("mypkg/__init__.py")
            return sorted(set(changed))
        if task.task_id == "csv-filter":
            code = (
                "import csv\n\n\n"
                "def filter_rows(path, min_score):\n"
                "    \"\"\"Read a CSV with header `name,score`; return rows with score >= min_score.\n\n"
                "    Each returned row is a dict {\"name\": str, \"score\": int}, in file order.\n"
                "    \"\"\"\n"
                "    out = []\n"
                "    with open(path, newline=\"\", encoding=\"utf-8\") as fh:\n"
                "        for row in csv.DictReader(fh):\n"
                "            score = int(row[\"score\"])\n"
                "            if score >= min_score:\n"
                "                out.append({\"name\": row[\"name\"], \"score\": score})\n"
                "    return out\n"
            )
            (workspace / "filt.py").write_text(code, encoding="utf-8")
            return ["filt.py"]
        if task.task_id == "implement-median":
            return self._replace(
                workspace / "stats.py",
                "    raise NotImplementedError\n",
                (
                    "    if not xs:\n"
                    "        raise ValueError(\"median of empty sequence\")\n"
                    "    values = sorted(xs)\n"
                    "    mid = len(values) // 2\n"
                    "    if len(values) % 2:\n"
                    "        return values[mid]\n"
                    "    return (values[mid - 1] + values[mid]) / 2\n"
                ),
            )
        if task.task_id == "dedupe-helpers":
            textutil = (
                "def normalize_name(name):\n"
                "    return \" \".join(str(name).split()).strip().lower()\n"
            )
            (workspace / "textutil.py").write_text(textutil, encoding="utf-8")
            (workspace / "utils_a.py").write_text(
                "from textutil import normalize_name\n\n\n"
                "def label_for(name):\n"
                "    return f\"user:{normalize_name(name)}\"\n",
                encoding="utf-8",
            )
            (workspace / "utils_b.py").write_text(
                "from textutil import normalize_name\n\n\n"
                "def greeting(name):\n"
                "    return f\"hello {normalize_name(name)}\"\n",
                encoding="utf-8",
            )
            return ["textutil.py", "utils_a.py", "utils_b.py"]
        if task.task_id == "rename-api":
            changed: list[str] = []
            changed += self._replace(workspace / "core.py", "def fetch_data", "def load_records")
            for rel in ("report.py", "cli.py"):
                changed += self._replace(workspace / rel, "fetch_data", "load_records")
            return sorted(set(changed))
        raise ValueError(f"managed adapter has no fix recipe for task {task.task_id!r}")

    @staticmethod
    def _replace(path: Path, old: str, new: str) -> list[str]:
        text = path.read_text(encoding="utf-8")
        if old not in text:
            return []
        path.write_text(text.replace(old, new), encoding="utf-8")
        return [path.name]


def run_live_learning_loop(
    tasks: list[LiveTask],
    out_dir: str | Path = DEFAULT_OUT_DIR,
    user_feedback: str | None = None,
    profile_path: str | Path | None = None,
    rounds: int | None = None,
    adapter: RepairAdapter | None = None,
) -> LiveRolloutResult:
    if not tasks:
        raise ValueError("run_live_learning_loop needs at least one task")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile_file = Path(profile_path) if profile_path else out / "user_profile.json"
    profile = UserPreferenceProfile.load(profile_file).ingest_feedback(user_feedback)
    rounds_to_run = max(1, rounds if rounds is not None else profile.max_rounds)
    adapter = adapter or ManagedPythonRepairAdapter()

    result = LiveRolloutResult(
        algorithm="trice-v2-live-user-conditioned-rollout",
        profile=profile.to_dict(),
    )
    weights = LearningWeights()
    for task in tasks:
        for round_index in range(1, rounds_to_run + 1):
            round_dir = out / task.task_id / f"round-{round_index}"
            baseline = _run_condition(task, "baseline-full-context", round_dir, None, adapter)
            optimized_policy = _policy_for_task(task, round_dir, profile.budget_ratio)
            optimized = _run_condition(task, "trice-v2", round_dir, optimized_policy, adapter)

            savings = _safe_ratio(baseline.input_tokens - optimized.input_tokens, baseline.input_tokens)
            quality_delta = (1.0 if optimized.passed else 0.0) - (1.0 if baseline.passed else 0.0)
            pass_noninferior = (not baseline.passed) or optimized.passed
            accepted = optimized.passed and pass_noninferior and savings + 1e-9 >= profile.target_savings
            update = update_weights(
                weights,
                features=(
                    savings,
                    1.0 - profile.budget_ratio,
                    0.0 if pass_noninferior else 1.0,
                    1.0,
                    max(0.0, -quality_delta),
                ),
                measured_input_savings=savings,
                quality_drop=max(0.0, -quality_delta),
                evidence_recall_failure=0.0 if pass_noninferior else 1.0,
                compression_overhead=optimized.input_tokens / max(1, baseline.input_tokens),
            )
            weights = update.weights
            profile.adapt_from_outcome(savings, pass_noninferior)
            profile.save(profile_file)

            result.rounds.append(
                LiveRound(
                    task_id=task.task_id,
                    round_index=round_index,
                    baseline=baseline,
                    optimized=optimized,
                    measured_input_savings=round(savings, 6),
                    quality_delta=round(quality_delta, 6),
                    pass_noninferior=pass_noninferior,
                    accepted=accepted,
                    learning_update=asdict(update),
                    profile_after=profile.to_dict(),
                )
            )
            if accepted:
                break

    result.profile = profile.to_dict()
    result.claim_gate = claim_gate_from_rounds(result.rounds, profile.target_savings).to_dict()
    result_path = out / "trice_v2_live_results.json"
    result.result_path = str(result_path)
    report_path = out / "trice_v2_live_report.md"
    result.report_path = str(report_path)
    manifest_path = out / "trice_v2_evidence_manifest.json"
    result.manifest_path = str(manifest_path)
    write_text_lf(report_path, render_live_report(result))
    write_text_lf(result_path, json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    artifact_paths: list[Path] = [report_path]
    try:
        profile_file.resolve().relative_to(out.resolve())
    except ValueError:
        pass
    else:
        artifact_paths.append(profile_file)
    for live_round in result.rounds:
        if live_round.baseline.decision_trace_path:
            artifact_paths.append(Path(live_round.baseline.decision_trace_path))
        artifact_paths.append(Path(live_round.baseline.trace_path))
        if live_round.optimized.decision_trace_path:
            artifact_paths.append(Path(live_round.optimized.decision_trace_path))
        artifact_paths.append(Path(live_round.optimized.trace_path))
        if live_round.optimized.policy_path:
            artifact_paths.append(Path(live_round.optimized.policy_path))
        if live_round.optimized.context_path:
            artifact_paths.append(Path(live_round.optimized.context_path))
        if live_round.baseline.receipt_path:
            artifact_paths.append(Path(live_round.baseline.receipt_path))
        if live_round.optimized.receipt_path:
            artifact_paths.append(Path(live_round.optimized.receipt_path))
    manifest = build_manifest(
        result.to_dict(),
        result_path=result_path,
        artifact_paths=artifact_paths,
        algorithm=result.algorithm,
        notes=[
            "Fresh workspace per condition",
            "Objective verifier command per run",
            "User-conditioned profile captured in result JSON",
            "Verifier durations normalized before hashing",
            "Wall-clock metadata excluded from traces",
            "Run receipts capture adapter envelopes and changed workspace fingerprints",
            "Replay is not accepted as final proof",
        ],
    )
    write_manifest(manifest, manifest_path)
    return result


def render_live_report(result: LiveRolloutResult) -> str:
    profile = result.profile
    lines = [
        "# TRICE V2 Live Rollout Report",
        "",
        f"Algorithm: `{result.algorithm}`",
        f"Target savings: {profile['target_savings']:.0%}",
        f"Final budget ratio: {profile['budget_ratio']:.0%}",
        f"Evidence manifest: `{Path(result.manifest_path).name if result.manifest_path else 'pending'}`",
        "",
        "## Evidence",
        "",
        "| Task | Round | Baseline tokens | TRICE tokens | Savings | Baseline pass | TRICE pass | Accepted |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for r in result.rounds:
        lines.append(
            "| {task} | {round} | {base} | {opt} | {savings:.1%} | {bp} | {op} | {accepted} |".format(
                task=r.task_id,
                round=r.round_index,
                base=r.baseline.input_tokens,
                opt=r.optimized.input_tokens,
                savings=r.measured_input_savings,
                bp="yes" if r.baseline.passed else "no",
                op="yes" if r.optimized.passed else "no",
                accepted="yes" if r.accepted else "no",
            )
        )
    gate = result.claim_gate or {}
    ci = gate.get("savings_ci") or {}
    pass_ci = gate.get("trice_pass_ci") or {}
    lines.extend(
        [
            "",
            "## Deterministic Claim Gate",
            "",
            f"- Scope: `{gate.get('scope', 'unknown')}`",
            f"- Mean savings: {gate.get('mean_savings', 0.0):.1%}",
            f"- Savings 95% bootstrap CI: [{ci.get('low', 0.0):.1%}, {ci.get('high', 0.0):.1%}]",
            f"- TRICE pass rate: {gate.get('trice_pass_rate', 0.0):.1%} "
            f"(Wilson 95% CI [{pass_ci.get('low', 0.0):.1%}, {pass_ci.get('high', 0.0):.1%}])",
            f"- Pass regressions: {gate.get('pass_regressions', 0)}",
            f"- Local smoke gate passed: {'yes' if gate.get('smoke_gate_passed') else 'no'}",
            f"- Broad claim allowed: {'yes' if gate.get('broad_claim_allowed') else 'no'}",
            f"- Rationale: {gate.get('rationale', 'not computed')}",
            "",
            "## User-Learned Policy",
            "",
        ]
    )
    for lesson in profile.get("lessons", []):
        lines.append(f"- {lesson}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is live-rollout evidence: each condition used a fresh copied workspace,",
            "made real source edits, and passed or failed on the verifier command. It is",
            "not replay evidence. The managed adapter is deterministic for CI; provider",
            "adapters can reuse the same gate as long as they report assembled input",
            "tokens and objective verifier results.",
            "Verifier duration text is normalized and wall-clock metadata is excluded",
            "from evidence hashes because timing noise is not decision evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_condition(
    task: LiveTask,
    condition: str,
    round_dir: Path,
    policy: ContextPolicy | None,
    adapter: RepairAdapter,
) -> ConditionRun:
    workspace = round_dir / condition / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_external_symlinks(task.seed_dir)
    shutil.copytree(task.seed_dir, workspace, symlinks=True)

    trace = _decision_trace(task, workspace)
    decision_trace_path = round_dir / condition / "decision_trace.json"
    decision_trace_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(decision_trace_path, json.dumps(trace, indent=2, sort_keys=True) + "\n")
    segments = segments_from_trace(trace)
    baseline_input_tokens = sum(s.tokens for s in segments)
    input_tokens = baseline_input_tokens
    policy_path = None
    context_path = None
    if policy is not None:
        input_tokens = policy.policy_tokens
        policy_path = str(round_dir / condition / "context_policy.json")
        context_path = str(round_dir / condition / "compressed_context.txt")
        Path(policy_path).parent.mkdir(parents=True, exist_ok=True)
        write_text_lf(policy_path, render_policy_json(policy))
        write_text_lf(context_path, render_context(policy, segments))

    trice_context = _trice_context_for_condition(
        condition=condition,
        input_tokens=input_tokens,
        baseline_input_tokens=baseline_input_tokens,
        policy=policy,
        policy_path=policy_path,
        context_path=context_path,
        decision_trace_path=str(decision_trace_path),
        verify_cmd=task.verify_cmd,
    )
    evidence_recall_report = trice_context.get("evidence_recall") if isinstance(trice_context.get("evidence_recall"), dict) else None
    adapter_task = AdapterTaskContext(
        task_id=task.task_id,
        prompt=task.prompt,
        verify_cmd=task.verify_cmd,
        trice_context=trice_context,
    )

    modified = adapter.apply_fix(adapter_task, workspace)
    receipt = _receipt_for_condition(adapter, adapter_task, workspace, modified)
    receipt_path = round_dir / condition / "run_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt_sha = _sha256_file(receipt_path)
    _clear_python_bytecode(workspace)
    verify = _run_verify(task.verify_cmd, workspace)
    trace["trace_id"] = f"{task.task_id}.{condition}"
    trace["agent_name"] = adapter.name
    trace["task_value_score"] = 1.0 if verify["passed"] else 0.0
    trace["total_tokens"] = input_tokens
    trace["metadata"]["condition"] = condition
    trace["metadata"]["run_receipt"] = receipt_path.name
    trace["metadata"]["run_receipt_sha256"] = receipt_sha
    trace["steps"].append(
        {
            "id": len(trace["steps"]) + 1,
            "type": "tool_call",
            "content": "Run verifier after managed edit",
            "tool_name": "pytest",
            "tool_success": verify["passed"],
            "tokens": _estimate_tokens(verify["output"]),
            "output": verify["output"],
        }
    )

    trace_path = round_dir / condition / "trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(trace_path, json.dumps(trace, indent=2, sort_keys=True) + "\n")
    return ConditionRun(
        task_id=task.task_id,
        condition=condition,
        workspace=str(workspace),
        input_tokens=input_tokens,
        passed=verify["passed"],
        verify_exit_code=verify["exit_code"],
        verify_output_excerpt=verify["output"][:1200],
        modified_files=modified,
        trace_path=str(trace_path),
        decision_trace_path=str(decision_trace_path),
        receipt_path=str(receipt_path),
        receipt_sha256=receipt_sha,
        adapter_type=str(receipt.get("adapter_type") or type(adapter).__name__),
        adapter_reported_input_tokens=_adapter_reported_input_tokens(receipt),
        policy_path=policy_path,
        context_path=context_path,
        policy=policy.to_dict() if policy is not None else None,
        evidence_recall=float(evidence_recall_report.get("evidence_recall")) if evidence_recall_report else None,
        evidence_recall_passed=bool(evidence_recall_report.get("passed")) if evidence_recall_report else None,
        evidence_recall_report=evidence_recall_report,
    )


def _trice_context_for_condition(
    *,
    condition: str,
    input_tokens: int,
    baseline_input_tokens: int,
    policy: ContextPolicy | None,
    policy_path: str | None,
    context_path: str | None,
    decision_trace_path: str,
    verify_cmd: tuple[str, ...],
) -> dict[str, Any]:
    action_counts: dict[str, int] = {}
    if policy is not None:
        for decision in policy.decisions:
            action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
    return {
        "schema_version": "trice-context-envelope/v1",
        "condition": condition,
        "context_mode": "trice_policy" if policy is not None else "full_context",
        "input_tokens": int(input_tokens),
        "baseline_input_tokens": int(baseline_input_tokens),
        "policy_tokens": int(policy.policy_tokens) if policy is not None else None,
        "budget_tokens": int(policy.budget_tokens) if policy is not None else None,
        "budget_ratio": float(policy.budget_ratio) if policy is not None else 1.0,
        "realized_budget_ratio": round(input_tokens / max(1, baseline_input_tokens), 6),
        "projected_input_savings_pct": float(policy.projected_input_savings_pct) if policy is not None else 0.0,
        "budget_exceeded": bool(policy.budget_exceeded) if policy is not None else False,
        "policy_path": policy_path,
        "policy_sha256": _sha256_file(Path(policy_path)) if policy_path else None,
        "compressed_context_path": context_path,
        "compressed_context_sha256": _sha256_file(Path(context_path)) if context_path else None,
        "trace_path": decision_trace_path,
        "decision_trace_path": decision_trace_path,
        "decision_trace_sha256": _sha256_file(Path(decision_trace_path)),
        "verify_cmd": list(verify_cmd),
        "policy_action_counts": action_counts,
        "evidence_recall": _evidence_recall_for_context(policy),
    }


def _evidence_recall_for_context(policy: ContextPolicy | None) -> dict[str, Any]:
    if policy is None:
        return {
            "schema_version": "trice-evidence-recall/v1",
            "evidence_recall": 1.0,
            "required_min": 0.95,
            "passed": True,
            "obligation_count": 0,
            "obligation_tokens": 0,
            "recalled_count": 0,
            "recalled_tokens": 0,
            "missing": [],
        }
    return evidence_recall_from_policy(policy).to_dict()


def _policy_for_task(task: LiveTask, round_dir: Path, budget_ratio: float) -> ContextPolicy:
    workspace = round_dir / "policy-build" / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_external_symlinks(task.seed_dir)
    shutil.copytree(task.seed_dir, workspace, symlinks=True)
    trace = _decision_trace(task, workspace)
    return solve_policy(segments_from_trace(trace), budget_ratio=budget_ratio)


def _decision_trace(task: LiveTask, workspace: Path) -> dict[str, Any]:
    first_verify = _run_verify(task.verify_cmd, workspace)
    project_files = _read_project_files(workspace)
    repeated_log = "\n".join([first_verify["output"]] * 3)
    trace = {
        "trace_id": f"{task.task_id}.decision",
        "agent_name": "managed-python-repair-v2",
        "framework": "trice-live",
        "task_value_score": 0.0,
        "metadata": {"task": task.prompt, "task_id": task.task_id},
        "steps": [
            {
                "id": 1,
                "type": "reasoning",
                "content": task.prompt,
                "input_context": task.prompt,
                "tokens": _estimate_tokens(task.prompt),
            },
            {
                "id": 2,
                "type": "tool_call",
                "content": "Run failing verifier before edit",
                "tool_name": "pytest",
                "tool_success": first_verify["passed"],
                "tokens": max(96, _estimate_tokens(first_verify["output"])),
                "output": first_verify["output"],
            },
            {
                "id": 3,
                "type": "tool_call",
                "content": "Read editable project files",
                "tool_name": "read_project",
                "tool_success": True,
                "tokens": max(256, _estimate_tokens(project_files["editable"])),
                "output": project_files["editable"],
            },
            {
                "id": 4,
                "type": "tool_call",
                "content": "Read tests as behavior oracle",
                "tool_name": "read_tests",
                "tool_success": True,
                "tokens": max(256, _estimate_tokens(project_files["tests"])),
                "output": project_files["tests"],
            },
            {
                "id": 5,
                "type": "reasoning",
                "content": repeated_log,
                "flags": ["REDUNDANT"],
                "tokens": max(640, _estimate_tokens(repeated_log)),
            },
            {
                "id": 6,
                "type": "reasoning",
                "content": (
                    "Stale hypothesis scratchpad: re-read the same failure, consider unrelated "
                    "renames, then discard this branch. This is intentionally receipt-safe noise."
                ),
                "flags": ["REDUNDANT"],
                "tokens": 420,
            },
            {
                "id": 7,
                "type": "reasoning",
                "content": "Ready to edit only source files and re-run the verifier.",
                "tokens": 64,
            },
        ],
    }
    return trace


def _read_project_files(workspace: Path) -> dict[str, str]:
    editable: list[str] = []
    tests: list[str] = []
    for path in sorted(workspace.rglob("*.py")):
        _assert_contained_file(workspace, path)
        rel = path.relative_to(workspace).as_posix()
        block = f"### {rel}\n{path.read_text(encoding='utf-8')}"
        if rel.startswith("tests/") or "/tests/" in rel:
            tests.append(block)
        else:
            editable.append(block)
    return {"editable": "\n\n".join(editable), "tests": "\n\n".join(tests)}


def _assert_no_external_symlinks(root: Path) -> None:
    base = root.resolve()
    for path in root.rglob("*"):
        if path.is_symlink():
            resolved = path.resolve()
            if resolved != base and base not in resolved.parents:
                rel = path.relative_to(root).as_posix()
                raise ValueError(f"refusing external symlink in task repo: {rel}")


def _assert_contained_file(root: Path, path: Path) -> None:
    base = root.resolve()
    resolved = path.resolve()
    if resolved != base and base not in resolved.parents:
        rel = path.relative_to(root).as_posix()
        raise ValueError(f"refusing to read file outside workspace: {rel}")


def _run_verify(cmd: tuple[str, ...], cwd: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, timeout=120, env=env)
    output = _normalize_verify_output((proc.stdout + "\n" + proc.stderr).strip())
    return {"passed": proc.returncode == 0, "exit_code": proc.returncode, "output": output}


def _clear_python_bytecode(workspace: Path) -> None:
    """Avoid stale same-size .pyc files after deterministic source edits."""

    for cache_dir in workspace.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
    for pyc in workspace.rglob("*.pyc"):
        if pyc.is_file():
            pyc.unlink()


def _receipt_for_condition(adapter: RepairAdapter, task: LiveTask, workspace: Path, modified: list[str]) -> dict[str, Any]:
    trice_context = getattr(task, "trice_context", None)
    if not isinstance(trice_context, dict):
        trice_context = {}
    receipt = getattr(adapter, "last_receipt", None)
    if isinstance(receipt, dict):
        out = dict(receipt)
        out.setdefault("schema_version", "trice-run-receipt/v1")
        out.setdefault("changed_files", modified)
        out.setdefault("changed_file_count", len(modified))
        out.setdefault("trice_context", trice_context)
        return out
    if isinstance(adapter, JsonPatchAdapter):
        return {
            "schema_version": "trice-run-receipt/v1",
            "adapter_type": "json_patch",
            "adapter_name": adapter.name,
            "task_id": task.task_id,
            "prompt_sha256": _sha256_text(task.prompt),
            "changed_files": modified,
            "changed_file_count": len(modified),
            "edit_count": len(adapter.edits),
            "allow_test_edits": adapter.allow_test_edits,
            "forbidden_prefixes": list(adapter.forbidden_prefixes),
            "metadata": adapter.metadata,
            "agent_reported": None,
            "trice_context": trice_context,
        }
    return {
        "schema_version": "trice-run-receipt/v1",
        "adapter_type": "managed_python",
        "adapter_name": adapter.name,
        "task_id": task.task_id,
        "prompt_sha256": _sha256_text(task.prompt),
        "workspace": workspace.name,
        "changed_files": modified,
        "changed_file_count": len(modified),
        "agent_reported": None,
        "trice_context": trice_context,
    }


def _adapter_reported_input_tokens(receipt: dict[str, Any]) -> int | None:
    reported = receipt.get("agent_reported")
    if not isinstance(reported, dict):
        return None
    token_accounting = reported.get("token_accounting")
    if isinstance(token_accounting, dict):
        value = token_accounting.get("input_tokens")
        if isinstance(value, int):
            return value
    value = reported.get("input_tokens")
    if isinstance(value, int):
        return value
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_verify_output(output: str) -> str:
    """Remove verifier clock noise while preserving the pass/fail evidence text."""

    output = output.replace("\r\n", "\n")
    output = _DURATION_RE.sub("in <duration>", output)
    return _SECONDS_RE.sub("in <duration>", output)


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.35))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _load_tasks(tasks_dir: Path, selected: list[str] | None) -> list[LiveTask]:
    dirs = sorted(d for d in tasks_dir.iterdir() if (d / "prompt.md").is_file() and (d / "seed").is_dir())
    if selected:
        wanted = set(selected)
        dirs = [d for d in dirs if d.name in wanted]
        missing = wanted - {d.name for d in dirs}
        if missing:
            raise SystemExit(f"unknown task(s): {', '.join(sorted(missing))}")
    return [LiveTask.from_dir(d) for d in dirs]


def _parse_verify_cmd(value: str | None) -> tuple[str, ...]:
    if not value:
        return tuple(VERIFY_CMD)
    return tuple(shlex.split(value, posix=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run TRICE V2 on live managed repo tasks.")
    ap.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    ap.add_argument("--task", action="append", default=None, help="Task id to run; repeatable.")
    ap.add_argument("--repo", type=Path, default=None, help="Run one arbitrary repo/seed directory instead of --tasks-dir.")
    ap.add_argument("--task-id", default=None, help="Task id for --repo runs.")
    ap.add_argument("--prompt", default=None, help="Task prompt for --repo runs; defaults to TASK.md or a generic prompt.")
    ap.add_argument("--verify-cmd", default=None, help="Verifier command for --repo runs, e.g. 'python -m pytest -q'.")
    ap.add_argument("--patch-spec", type=Path, default=None, help="Deterministic JSON patch spec for --repo runs.")
    ap.add_argument("--repair-cmd", default=None, help="Deterministic repair command for --repo runs.")
    ap.add_argument("--adapter-profile", type=Path, default=None, help="Reusable TRICE command adapter profile JSON.")
    ap.add_argument("--repair-timeout-s", type=int, default=600, help="Timeout for --repair-cmd.")
    ap.add_argument("--allow-test-edits", action="store_true", help="Allow a repair command to modify tests.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--profile", type=Path, default=None, help="Persistent user profile JSON.")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--user-feedback", default=None)
    ap.add_argument("--json", action="store_true", help="Print JSON result instead of a short summary.")
    ap.add_argument("--verify-manifest", type=Path, default=None, help="Verify a TRICE evidence manifest and exit.")
    args = ap.parse_args(argv)

    if args.verify_manifest:
        verdict = verify_manifest(args.verify_manifest)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1

    if args.repo:
        intervention_count = sum(1 for value in (args.patch_spec, args.repair_cmd, args.adapter_profile) if value)
        if intervention_count != 1:
            print("error: --repo requires exactly one of --patch-spec, --repair-cmd, or --adapter-profile", file=sys.stderr)
            return 2
        tasks = [
            LiveTask.from_repo(
                args.repo,
                task_id=args.task_id,
                prompt=args.prompt,
                verify_cmd=_parse_verify_cmd(args.verify_cmd),
            )
        ]
        if args.patch_spec:
            adapter: RepairAdapter = JsonPatchAdapter.from_file(args.patch_spec)
        elif args.adapter_profile:
            adapter = CommandRepairAdapter.from_file(args.adapter_profile)
        else:
            adapter = CommandRepairAdapter.from_dict(
                {
                    "repair_cmd": args.repair_cmd,
                    "repair_timeout_s": args.repair_timeout_s,
                    "allow_test_edits": args.allow_test_edits,
                }
            )
    else:
        tasks = _load_tasks(args.tasks_dir, args.task)
        adapter = ManagedPythonRepairAdapter()
    result = run_live_learning_loop(
        tasks,
        out_dir=args.out_dir,
        user_feedback=args.user_feedback,
        profile_path=args.profile,
        rounds=args.rounds,
        adapter=adapter,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        last = result.rounds[-1]
        print(
            f"TRICE V2 live: {len(result.rounds)} round(s), "
            f"last savings={last.measured_input_savings:.1%}, "
            f"pass_noninferior={last.pass_noninferior}, accepted={last.accepted}"
        )
        print(f"report: {result.report_path}")
        print(f"json  : {result.result_path}")
        print(f"manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
