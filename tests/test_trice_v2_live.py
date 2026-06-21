import json
import shutil
from pathlib import Path

import pytest

from benchmark.trice.adapters import JsonPatchAdapter
from benchmark.trice.evidence import canonical_json, verify_manifest
from benchmark.trice.live import LiveTask, run_live_learning_loop
from benchmark.trice.schemas import load_schema, schema_path, validate_patch_spec_file, validate_suite_manifest_file
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
    assert callable(trice.JsonPatchAdapter.from_dict)
    assert trice.load_schema("patch")["title"] == "TRICE deterministic patch spec"


def test_schema_helpers_validate_example_patch():
    path = schema_path("patch")
    assert path.name == "trice_patch_spec.schema.json"
    assert load_schema("manifest")["title"] == "TRICE evidence manifest"
    assert load_schema("suite")["title"] == "TRICE live suite manifest"
    verdict = validate_patch_spec_file(REPO / "examples" / "trice_patch_fix_offbyone.json")
    assert verdict["ok"] is True
    assert verdict["edit_count"] == 1
    suite_verdict = validate_suite_manifest_file(REPO / "examples" / "trice_suite_fix_offbyone.json")
    assert suite_verdict["ok"] is True
    assert suite_verdict["task_count"] == 1
    assert suite_verdict["run_count"] == 3


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
    assert live_round.measured_input_savings >= 0.60
    assert result.claim_gate["smoke_gate_passed"] is True
    assert verify_manifest(result.manifest_path)["ok"] is True


def test_manifest_driven_suite_runs_real_repo_and_deep_verifies(tmp_path):
    seed = tmp_path / "seed"
    shutil.copytree(REPO / "benchmark" / "live" / "tasks" / "fix-offby-one" / "seed", seed)
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
                        "repo": "seed",
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
    assert result.tasks[1].replicate_index == 2
    assert result.claim_gate["replicate_count"] == 2
    assert result.claim_gate["task_cluster_count"] == 1
    assert result.claim_gate["clustered_savings_ci"]["low"] >= 0.60
    assert result.claim_gate["smoke_gate_passed"] is True
    verdict = verify_suite_evidence(out_dir / str(result.manifest_path))
    assert verdict["ok"] is True
    assert len(verdict["children"]) == 2
    assert all(child["ok"] for child in verdict["children"])


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
