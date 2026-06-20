from pathlib import Path

from benchmark.trice.live import LiveTask, run_live_learning_loop
from benchmark.trice.user import UserPreferenceProfile


REPO = Path(__file__).resolve().parents[1]


def test_user_profile_learns_live_aggressive_target_from_feedback(tmp_path):
    profile = UserPreferenceProfile.load(tmp_path / "profile.json")
    profile.ingest_feedback("real runs, not replay runs, learn from the user, hit 60% savings")
    assert profile.require_live_rollout is True
    assert profile.target_savings == 0.60
    assert profile.budget_ratio <= 0.40
    assert any("live rollout" in lesson for lesson in profile.lessons)


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
    assert any("accepted" in lesson for lesson in result.profile["lessons"])
