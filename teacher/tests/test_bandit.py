"""Tests for LinUCBBandit: convergence, serialization, context encoding."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from teacher.bandit import LinUCBBandit, _DIM, _ctx, _dot, _inv, _solve


def test_context_dimension():
    x = _ctx(0.5, 30.0, 5, "srr")
    assert len(x) == _DIM, f"expected {_DIM}, got {len(x)}"
    assert x[0] == 1.0, "first element must be bias=1.0"
    assert 0.0 <= x[2] <= 1.0, "waste_pct feature must be in [0,1]"


def test_context_kind_onehot():
    for kind in ("srr", "ldi", "tca", "rda", "cce", "vdi", "shl"):
        x = _ctx(0.5, 20.0, 3, kind)
        onehot = x[4:]
        assert sum(onehot) == 1.0, f"onehot sum != 1 for kind={kind}"
    # Unknown kind -> all-zero onehot
    x = _ctx(0.5, 20.0, 3, "unknown")
    assert sum(x[4:]) == 0.0, "unknown kind should produce all-zero onehot"


def test_solve_identity():
    n = _DIM
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    b = [float(i + 1) for i in range(n)]
    x = _solve(I, b)
    for expected, got in zip(b, x):
        assert abs(expected - got) < 1e-9, f"solve(I, b) != b at position: {expected} vs {got}"


def test_inv_identity():
    n = 3
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    Iinv = _inv(I)
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            assert abs(Iinv[i][j] - expected) < 1e-9, f"inv(I) != I at [{i}][{j}]"


def _fake_iv(key: str, kind_val: str):
    class FakeIv:
        waste_pattern = type("WP", (), {"value": kind_val})()
    iv = FakeIv()
    iv.key = key
    return iv


def _fake_diag():
    class FakeDiag:
        patterns = []
        total_tokens = 1000
    return FakeDiag()


def test_bandit_initial_ucb_is_positive():
    """Untrained arms with the identity prior should return a positive UCB."""
    b = LinUCBBandit(alpha=1.0)
    iv = _fake_iv("NO_HEDGING", "shl")
    score = b._arm(iv.key).ucb_score(b._encode(iv, _fake_diag()), alpha=1.0)
    assert score > 0, f"UCB should be positive for untrained arm, got {score}"


def test_bandit_convergence():
    """After many rounds, bandit should prefer the consistently high-reward arm."""
    b = LinUCBBandit(alpha=0.3)
    iv_good = _fake_iv("NO_HEDGING", "shl")
    iv_bad = _fake_iv("step_cap", "rda")
    diag = _fake_diag()

    for _ in range(40):
        b.update("NO_HEDGING", iv_good, diag, 0.30)
        b.update("step_cap", iv_bad, diag, -0.05)

    chosen = b.select([iv_bad, iv_good], diag)
    assert chosen.key == "NO_HEDGING", (
        f"Bandit should prefer high-reward arm after training, chose {chosen.key}"
    )


def test_bandit_serialization():
    """Round-trip save/load preserves arm state."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        b1 = LinUCBBandit(alpha=1.0, path=path)
        iv = _fake_iv("loop_breaker", "ldi")
        b1.update("loop_breaker", iv, _fake_diag(), 0.20)
        b1.save()

        b2 = LinUCBBandit(alpha=1.0, path=path)
        assert "loop_breaker" in b2.arms, "arm not persisted"
        assert b2.arms["loop_breaker"].t == 1, f"expected 1 trial, got {b2.arms['loop_breaker'].t}"
    finally:
        os.unlink(path)


def test_bandit_summary_empty():
    b = LinUCBBandit()
    assert "no data" in b.summary()


def test_bandit_summary_nonempty():
    b = LinUCBBandit()
    iv = _fake_iv("NO_HEDGING", "shl")
    b.update("NO_HEDGING", iv, _fake_diag(), 0.15)
    s = b.summary()
    assert "NO_HEDGING" in s
    assert "1" in s


def test_teacher_bandit_wiring():
    """Teacher.improve() with a bandit runs without error and records arm data."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from teacher.runner import Task
    from teacher.schemas import AgentConfig
    from teacher.teacher import Teacher, Mode

    tasks = [
        Task("t1", "goal_one", ["search_db"]),
        Task("t2", "goal_two", ["search_db", "send_email"]),
    ]
    bandit = LinUCBBandit(alpha=1.0)
    t = Teacher(AgentConfig(), mode=Mode.CURRICULUM, bandit=bandit)
    result = t.improve(tasks=tasks, max_rounds=3)

    assert result is not None, "improve() returned None"
    assert len(result.history) > 0, "no intervention history recorded"
    # Bandit should have recorded at least one update
    assert len(bandit.arms) > 0, "bandit received no updates"


if __name__ == "__main__":
    import traceback
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
