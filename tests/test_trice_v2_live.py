import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.trice.adapters import CommandRepairAdapter, JsonPatchAdapter
from benchmark.trice.bundle import export_evidence_bundle, verify_evidence_bundle
from benchmark.trice.evidence import canonical_json, verify_manifest
from benchmark.trice.live import LiveTask, run_live_learning_loop
from benchmark.trice.receipt import validate_run_receipt_file
from benchmark.trice.schemas import load_schema, schema_path, validate_adapter_profile_file, validate_patch_spec_file, validate_suite_manifest_file
from benchmark.trice.stats import bootstrap_mean_ci, claim_gate_from_rounds, clustered_bootstrap_mean_ci, wilson_ci
from benchmark.trice.suite import run_suite_manifest, verify_suite_evidence
from benchmark.trice.user import UserPreferenceProfile


REPO = Path(__file__).resolve().parents[1]


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 2, "a": [3, {"z": 1}]}) == canonical_json({"a": [3, {"z": 1}], "b": 2})


def test_public_tracerazor_trice_import_surface():
    import tracerazor.trice as trice

    assert trice.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert callable(trice.run_live_learning_loop)
    assert callable(trice.claim_gate_from_rounds)
    assert callable(trice.CommandRepairAdapter.from_dict)
    assert callable(trice.JsonPatchAdapter.from_dict)
    assert callable(trice.export_evidence_bundle)
    assert callable(trice.verify_evidence_bundle)
    assert trice.load_schema("patch")["title"] == "TRICE deterministic patch spec"


def test_schema_helpers_validate_example_patch():
    path = schema_path("patch")
    assert path.name == "trice_patch_spec.schema.json"
    assert load_schema("manifest")["title"] == "TRICE evidence manifest"
    assert load_schema("suite")["title"] == "TRICE live suite manifest"
    assert load_schema("bundle")["title"] == "TRICE evidence bundle manifest"
    assert load_schema("adapter")["title"] == "TRICE adapter profile"
    assert load_schema("receipt")["title"] == "TRICE run receipt"
    verdict = validate_patch_spec_file(REPO / "examples" / "trice_patch_fix_offbyone.json")
    assert verdict["ok"] is True
    assert verdict["edit_count"] == 1
    suite_verdict = validate_suite_manifest_file(REPO / "examples" / "trice_suite_fix_offbyone.json")
    assert suite_verdict["ok"] is True
    assert suite_verdict["task_count"] == 1
    assert suite_verdict["run_count"] == 3
    adapter_verdict = validate_adapter_profile_file(REPO / "examples" / "trice_adapter_profile_echo.json")
    assert adapter_verdict["ok"] is True
    assert adapter_verdict["type"] == "command"
    bundled_adapter = validate_adapter_profile_file(REPO / "examples" / "trice_adapter_profile_bundled_tasks.json")
    assert bundled_adapter["ok"] is True
    assert "trice_repair_bundled_tasks.py" in bundled_adapter["command"][1]
    bundled_suite = validate_suite_manifest_file(REPO / "examples" / "trice_suite_bundled_live.json")
    assert bundled_suite["ok"] is True
    assert bundled_suite["task_count"] == 6


def test_deterministic_stats_are_repeatable_and_gate_local_claim():
    values = [0.773, 0.802, 0.811, 0.749, 0.776, 0.808]
    ci1 = bootstrap_mean_ci(values)
    ci2 = bootstrap_mean_ci(list(reversed(values)))
    assert ci1.mean == ci2.mean
    assert ci1.low > 0.70
    assert ci1.high < 0.83
    pass_ci = wilson_ci(6, 6)
    assert pass_ci.mean == 1.0
    rounds = [
        {
            "measured_input_savings": v,
            "baseline": {"passed": True},
            "optimized": {"passed": True},
            "accepted": True,
        }
        for v in values
    ]
    gate = claim_gate_from_rounds(rounds, target_savings=0.60)
    assert gate.smoke_gate_passed is True
    assert gate.broad_claim_allowed is False
    assert gate.savings_ci.low >= 0.60
    clustered = clustered_bootstrap_mean_ci({"repo-a": [0.7, 0.8], "repo-b": [0.6]})
    assert clustered.mean == 0.7
    assert 0.6 <= clustered.low <= clustered.high <= 0.8


def test_user_profile_learns_live_aggressive_target_from_feedback(tmp_path):
    profile = UserPreferenceProfile.load(tmp_path / "profile.json")
    profile.ingest_feedback("real runs, not replay runs, learn from the user, hit 60% savings")
    assert profile.require_live_rollout is True
    assert profile.target_savings == 0.60
    assert profile.budget_ratio <= 0.40
    assert any("live rollout" in lesson for lesson in profile.lessons)


def test_json_patch_adapter_refuses_test_edits_and_path_escape(tmp_path):
    workspace = tmp_path / "repo"
    (workspace / "tests").mkdir(parents=True)
    (workspace / "tests" / "test_x.py").write_text("x = 1\n", encoding="utf-8")
    adapter = JsonPatchAdapter.from_dict(
        {"edits": [{"op": "replace", "path": "tests/test_x.py", "old": "1", "new": "2"}]}
    )
    with pytest.raises(ValueError, match="forbidden path"):
        adapter.apply_fix(object(), workspace)
    escape = JsonPatchAdapter.from_dict(
        {"edits": [{"op": "write", "path": "../escape.py", "content": "bad"}]}
    )
    with pytest.raises(ValueError, match="inside workspace"):
        escape.apply_fix(object(), workspace)


def test_command_repair_adapter_runs_real_command_and_blocks_test_edits(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    repair = tmp_path / "repair.py"
    repair.write_text(
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "p = Path('app.py')\n"
        "p.write_text(p.read_text(encoding='utf-8').replace('1', '2'), encoding='utf-8')\n"
        "Path('.trice').mkdir(exist_ok=True)\n"
        "Path('.trice/agent_receipt.json').write_text(json.dumps({\n"
        "  'model': 'unit-model',\n"
        "  'token_accounting': {\n"
        "    'input_tokens': int(os.environ['TRICE_INPUT_TOKENS']),\n"
        "    'baseline_input_tokens': int(os.environ['TRICE_BASELINE_INPUT_TOKENS']),\n"
        "  },\n"
        "  'context_mode': os.environ['TRICE_CONTEXT_MODE'],\n"
        "}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    adapter = CommandRepairAdapter(command=(sys.executable, str(repair)), timeout_s=30)
    changed = adapter.apply_fix(
        type(
            "Task",
            (),
            {
                "task_id": "cmd",
                "prompt": "fix value",
                "trice_context": {
                    "schema_version": "trice-context-envelope/v1",
                    "condition": "trice-v2",
                    "context_mode": "trice_policy",
                    "input_tokens": 456,
                    "baseline_input_tokens": 1234,
                    "policy_tokens": 456,
                    "budget_tokens": 494,
                    "budget_ratio": 0.4,
                    "realized_budget_ratio": 0.36953,
                    "projected_input_savings_pct": 63.05,
                    "policy_action_counts": {"keep": 2, "extract": 1},
                },
            },
        )(),
        workspace,
    )
    assert changed == ["app.py"]
    assert (workspace / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert adapter.last_receipt
    assert adapter.last_receipt["agent_reported"]["token_accounting"]["input_tokens"] == 456
    assert adapter.last_receipt["agent_reported"]["token_accounting"]["baseline_input_tokens"] == 1234
    assert adapter.last_receipt["trice_context"]["input_tokens"] == 456
    assert "TRICE_INPUT_TOKENS" in adapter.last_receipt["reserved_env_keys"]
    assert len(adapter.last_receipt["workspace_before_sha256"]) == 64
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(adapter.last_receipt, indent=2, sort_keys=True), encoding="utf-8")
    receipt_verdict = validate_run_receipt_file(receipt_path)
    assert receipt_verdict["agent_reported_input_tokens"] == 456
    assert receipt_verdict["trice_context_mode"] == "trice_policy"
    assert receipt_verdict["trice_input_tokens"] == 456

    bad_workspace = tmp_path / "bad-repo"
    bad_workspace.mkdir()
    bad = tmp_path / "bad_repair.py"
    bad.write_text(
        "from pathlib import Path\n"
        "Path('tests').mkdir(exist_ok=True)\n"
        "Path('tests/test_x.py').write_text('def test_x():\\n    assert True\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden path"):
        CommandRepairAdapter(command=(sys.executable, str(bad)), timeout_s=30).apply_fix(object(), bad_workspace)


def test_trice_v2_live_rollout_edits_real_workspace_and_updates_profile(tmp_path):
    task = LiveTask.from_dir(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one")
    result = run_live_learning_loop(
        [task],
        out_dir=tmp_path,
        user_feedback="real runs, not replay runs, learn from the user, target 60% token savings",
        rounds=1,
    )

    assert result.rounds
    live_round = result.rounds[0]
    assert live_round.baseline.passed is True
    assert live_round.optimized.passed is True
    assert live_round.pass_noninferior is True
    assert live_round.measured_input_savings >= 0.60
    assert live_round.accepted is True

    workspace = Path(live_round.optimized.workspace)
    assert workspace.is_dir()
    assert "size - 1" not in (workspace / "chunker.py").read_text(encoding="utf-8")
    assert (workspace / "chunker.py").read_text(encoding="utf-8").count("size") >= 2
    assert live_round.optimized.policy_path
    assert Path(live_round.optimized.policy_path).is_file()
    assert Path(result.report_path or "").is_file()
    assert Path(result.result_path or "").is_file()
    assert Path(result.manifest_path or "").is_file()
    verdict = verify_manifest(result.manifest_path)
    assert verdict["ok"] is True
    receipt = Path(live_round.optimized.receipt_path or "")
    assert receipt.is_file()
    receipt_verdict = validate_run_receipt_file(receipt)
    assert receipt_verdict["adapter_type"] == "managed_python"
    assert receipt_verdict["trice_context_mode"] == "trice_policy"
    assert receipt_verdict["trice_input_tokens"] == live_round.optimized.input_tokens
    baseline_receipt = validate_run_receipt_file(live_round.baseline.receipt_path)
    assert baseline_receipt["trice_context_mode"] == "full_context"
    assert baseline_receipt["trice_input_tokens"] == live_round.baseline.input_tokens
    assert any("accepted" in lesson for lesson in result.profile["lessons"])


def test_generic_repo_json_patch_rollout(tmp_path):
    seed = REPO / "benchmark" / "live" / "tasks" / "fix-offby-one" / "seed"
    task = LiveTask.from_repo(
        seed,
        task_id="generic-fix-offby-one",
        prompt="Fix chunker.py without editing tests.",
        verify_cmd=["python", "-m", "pytest", "-q", "--tb=short"],
    )
    adapter = JsonPatchAdapter.from_dict(
        {
            "name": "generic-offby-one",
            "edits": [{"op": "replace", "path": "chunker.py", "old": "size - 1", "new": "size"}],
        }
    )
    result = run_live_learning_loop(
        [task],
        out_dir=tmp_path / "generic",
        user_feedback="real runs, not replay; target 60% savings",
        rounds=1,
        adapter=adapter,
    )
    live_round = result.rounds[0]
    assert live_round.optimized.passed is True
    assert live_round.optimized.modified_files == ["chunker.py"]
    assert validate_run_receipt_file(live_round.optimized.receipt_path)["adapter_type"] == "json_patch"
    assert live_round.measured_input_savings >= 0.60
    assert result.claim_gate["smoke_gate_passed"] is True
    assert verify_manifest(result.manifest_path)["ok"] is True


def test_manifest_driven_suite_runs_real_repo_and_deep_verifies(tmp_path):
    source_repo = tmp_path / "source-repo"
    shutil.copytree(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one" / "seed", source_repo)
    subprocess.run(["git", "init"], cwd=source_repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "trice@example.test"], cwd=source_repo, check=True)
    subprocess.run(["git", "config", "user.name", "TRICE Test"], cwd=source_repo, check=True)
    subprocess.run(["git", "add", "."], cwd=source_repo, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=source_repo, check=True, capture_output=True, text=True)
    git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_repo, check=True, capture_output=True, text=True).stdout.strip()
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    patch_path = patch_dir / "fix.json"
    shutil.copy2(REPO / "examples" / "trice_patch_fix_offbyone.json", patch_path)
    manifest_path = tmp_path / "suite.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "trice-suite/v1",
                "name": "tmp-real-suite",
                "user_feedback": "real runs, not replay; target 60% savings",
                "target_savings": 0.60,
                "rounds": 1,
                "replicates": 2,
                "tasks": [
                    {
                        "task_id": "tmp-fix-offby-one",
                        "git": {"url": source_repo.resolve().as_uri(), "rev": git_rev},
                        "patch_spec": "patches/fix.json",
                        "prompt": "Fix chunker.py without editing tests.",
                        "verify_cmd": ["python", "-m", "pytest", "-q", "--tb=short"],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "suite-out"
    result = run_suite_manifest(manifest_path, out_dir=out_dir, rounds=1)

    assert len(result.tasks) == 2
    assert result.tasks[0].mean_savings >= 0.60
    assert result.tasks[0].accepted_rounds == 1
    assert result.tasks[0].source["source_type"] == "git"
    assert result.tasks[0].source["git"]["resolved_commit"] == git_rev
    assert result.tasks[0].source["repo_tree"]["algorithm"] == "trice-tree-sha256/v1"
    assert result.tasks[0].source["repo_tree"]["file_count"] >= 2
    assert len(result.tasks[0].source["repo_tree"]["digest"]) == 64
    assert len(result.tasks[0].source["patch_sha256"]) == 64
    assert result.tasks[1].replicate_index == 2
    assert result.claim_gate["replicate_count"] == 2
    assert result.claim_gate["task_cluster_count"] == 1
    assert result.claim_gate["clustered_savings_ci"]["low"] >= 0.60
    assert result.claim_gate["smoke_gate_passed"] is True
    assert result.claim_gate["s_tier_gate"]["passed"] is False
    assert "task_clusters" in result.claim_gate["s_tier_gate"]["missing_requirements"]
    assert "adapter_profiles" in result.claim_gate["s_tier_gate"]["missing_requirements"]
    assert (out_dir / "trice_suite_sources.json").is_file()
    verdict = verify_suite_evidence(out_dir / str(result.manifest_path))
    assert verdict["ok"] is True
    assert len(verdict["children"]) == 2
    assert all(child["ok"] for child in verdict["children"])

    bundle = export_evidence_bundle(out_dir / str(result.manifest_path), out_dir / "suite.trice.zip")
    bundle_verdict = verify_evidence_bundle(bundle)
    assert bundle_verdict["ok"] is True
    assert bundle_verdict["entry_count"] >= 10

    tampered = out_dir / "tampered.trice.zip"
    shutil.copy2(bundle, tampered)
    with tampered.open("r+b") as fh:
        fh.seek(-24, 2)
        byte = fh.read(1)
        fh.seek(-24, 2)
        fh.write(bytes([byte[0] ^ 1]))
    assert verify_evidence_bundle(tampered)["ok"] is False


def test_manifest_suite_can_use_command_repair_adapter(tmp_path):
    source_repo = tmp_path / "source-repo"
    shutil.copytree(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one" / "seed", source_repo)
    repair = tmp_path / "repair_offbyone.py"
    repair.write_text(
        "from pathlib import Path\n"
        "p = Path('chunker.py')\n"
        "p.write_text(p.read_text(encoding='utf-8').replace('size - 1', 'size'), encoding='utf-8')\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "suite-command.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "trice-suite/v1",
                "name": "tmp-command-suite",
                "user_feedback": "real runs, not replay; target 60% savings",
                "target_savings": 0.60,
                "rounds": 1,
                "replicates": 1,
                "tasks": [
                    {
                        "task_id": "tmp-command-fix",
                        "repo": str(source_repo),
                        "repair_cmd": [sys.executable, str(repair)],
                        "repair_timeout_s": 30,
                        "prompt": "Fix chunker.py without editing tests.",
                        "verify_cmd": ["python", "-m", "pytest", "-q", "--tb=short"],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    assert validate_suite_manifest_file(manifest_path)["ok"] is True
    out_dir = tmp_path / "suite-command-out"
    result = run_suite_manifest(manifest_path, out_dir=out_dir, rounds=1)

    assert len(result.tasks) == 1
    task = result.tasks[0]
    assert task.patch_spec is None
    assert task.repair_cmd == [sys.executable, str(repair)]
    assert task.adapter_profile is None
    assert task.source["adapter_type"] == "command"
    assert "patch_sha256" not in task.source
    assert task.source["repair_timeout_s"] == 30
    assert task.mean_savings >= 0.60
    assert task.accepted_rounds == 1
    assert verify_suite_evidence(out_dir / str(result.manifest_path))["ok"] is True


def test_manifest_suite_can_use_adapter_profile(tmp_path):
    source_repo = tmp_path / "source-repo"
    shutil.copytree(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one" / "seed", source_repo)
    repair = tmp_path / "repair_profile.py"
    repair.write_text(
        "from pathlib import Path\n"
        "p = Path('chunker.py')\n"
        "p.write_text(p.read_text(encoding='utf-8').replace('size - 1', 'size'), encoding='utf-8')\n",
        encoding="utf-8",
    )
    profile = tmp_path / "adapter-profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "trice-adapter-profile/v1",
                "name": "tmp-profile-command",
                "type": "command",
                "command": [sys.executable, str(repair)],
                "timeout_s": 30,
                "allow_test_edits": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "suite-profile.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "trice-suite/v1",
                "name": "tmp-profile-suite",
                "user_feedback": "real runs, not replay; target 60% savings",
                "rounds": 1,
                "replicates": 1,
                "s_tier_gate": {
                    "min_task_clusters": 1,
                    "min_replicates_per_task": 1,
                    "require_locked_git_sources": False,
                    "require_remote_git_sources": False,
                    "require_adapter_profiles": True,
                    "min_mean_savings": 0.60,
                    "min_clustered_savings_ci_low": 0.60,
                },
                "tasks": [
                    {
                        "task_id": "tmp-profile-fix",
                        "repo": str(source_repo),
                        "adapter_profile": str(profile),
                        "prompt": "Fix chunker.py without editing tests.",
                        "verify_cmd": ["python", "-m", "pytest", "-q", "--tb=short"],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_suite_manifest(manifest_path, out_dir=tmp_path / "suite-profile-out", rounds=1)

    task = result.tasks[0]
    assert task.adapter_profile == str(profile)
    assert task.source["adapter_type"] == "command_profile"
    assert len(task.source["adapter_profile_sha256"]) == 64
    assert task.mean_savings >= 0.60
    assert result.claim_gate["adapter_breakdown"]["command_profile"]["runs"] == 1
    assert result.claim_gate["failure_breakdown"]["pass_regression_runs"] == 0
    assert result.claim_gate["s_tier_gate"]["passed"] is True
    assert result.claim_gate["s_tier_gate"]["claim_level"] == "s_tier"


def test_live_rollout_manifest_repeats_for_identical_real_run(tmp_path):
    task = LiveTask.from_dir(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one")
    out_dir = tmp_path / "repeatable"

    def run_once() -> dict:
        result = run_live_learning_loop(
            [task],
            out_dir=out_dir,
            user_feedback="real runs, not replay; target 60% savings",
            rounds=1,
        )
        manifest = json.loads(Path(result.manifest_path or "").read_text(encoding="utf-8"))
        trace = json.loads(Path(result.rounds[0].optimized.trace_path).read_text(encoding="utf-8"))
        assert "wall_s" not in trace["metadata"]
        assert "in <duration>" in result.rounds[0].optimized.verify_output_excerpt
        assert verify_manifest(result.manifest_path)["ok"] is True
        return manifest

    first = run_once()
    shutil.rmtree(out_dir)
    second = run_once()

    assert second["result_sha256"] == first["result_sha256"]
    assert second["canonical_result_sha256"] == first["canonical_result_sha256"]
    assert second["artifacts"] == first["artifacts"]


def test_manifest_verification_rejects_invalid_hashed_run_receipt(tmp_path):
    task = LiveTask.from_dir(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one")
    result = run_live_learning_loop(
        [task],
        out_dir=tmp_path / "bad-receipt",
        user_feedback="real runs, not replay; target 60% savings",
        rounds=1,
    )
    receipt_path = Path(result.rounds[0].optimized.receipt_path or "")
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    data["changed_file_count"] = 999
    receipt_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_path = Path(result.manifest_path or "")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact["path"].endswith("trice-v2/run_receipt.json"):
            artifact["sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            artifact["bytes"] = receipt_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verdict = verify_manifest(manifest_path)
    assert verdict["ok"] is False
    assert any("invalid run receipt" in err for err in verdict["errors"])
