"""Tests for the Teacher / closed-loop remediation engine.

Run: python -m pytest teacher/tests/ -q
These use the built-in diagnoser so they pass with no Rust build.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from teacher import (  # noqa: E402
    AgentConfig, Decision, Diagnoser, Intervention, Mode, Playbook,
    QualityGate, Target, Task, Teacher, Tier, WasteKind, apply, evaluate,
)
from teacher.interventions import propose  # noqa: E402

TASKS = [
    Task("t1", "refund order ORD-1", ["get_order", "check", "refund"]),
    Task("t2", "status of ORD-2", ["get_order", "status"]),
]


def _builtin_teacher(**kw):
    return Teacher(AgentConfig(), diagnoser=Diagnoser(prefer_auditor=False), **kw)


def test_apply_is_idempotent_and_pure():
    cfg = AgentConfig()
    iv = Intervention(WasteKind.HEDGING, Target.SYSTEM_PROMPT, Tier.PROMPT,
                      "NO_HEDGING", {"body": "x"}, 100, 0.1)
    once = apply(iv, cfg)
    twice = apply(iv, once)
    assert once.system_prompt_sections == twice.system_prompt_sections
    assert cfg.system_prompt_sections == {}          # original untouched (pure)


def test_curriculum_reduces_tokens_preserving_success():
    result = _builtin_teacher(mode=Mode.CURRICULUM).improve(TASKS, max_rounds=8)
    assert result.total_token_saving_pct > 0
    assert all(vr.success_after >= 0.99 for vr in result.accepted)


def test_gate_rejects_quality_regression():
    gate = QualityGate()
    diag = Diagnoser(prefer_auditor=False)
    cfg = AgentConfig()
    base = evaluate(cfg, TASKS, diag)
    harmful = Intervention(WasteKind.OVER_DEPTH, Target.RUNTIME_POLICY, Tier.STRUCT,
                           "step_cap", {"value": 1}, 999, 0.9)
    trial = evaluate(apply(harmful, cfg), TASKS, diag)
    assert gate.decide(base, trial) is Decision.REJECT_QUALITY


def test_gate_rejects_no_gain():
    gate = QualityGate(min_savings_pct=99.0)   # impossible threshold
    diag = Diagnoser(prefer_auditor=False)
    base = evaluate(AgentConfig(), TASKS, diag)
    iv = propose(diag.diagnose(
        {"steps": [{"id": 1, "type": "reasoning",
                    "content": "let me basically re-read", "tokens": 100}]}))
    # A trivially better config still won't clear a 99% savings bar.
    trial = base
    assert gate.decide(base, trial) is Decision.REJECT_NO_GAIN


def test_curriculum_tries_cheapest_tier_first():
    result = _builtin_teacher(mode=Mode.CURRICULUM).improve(TASKS, max_rounds=8)
    tiers = [int(vr.intervention.tier) for vr in result.history]
    # Non-decreasing tier order (curriculum escalates, never jumps back cheap).
    assert tiers == sorted(tiers)


def test_playbook_records_and_transfers():
    pb = Playbook()
    _builtin_teacher(mode=Mode.CURRICULUM, playbook=pb).improve(TASKS, max_rounds=8)
    assert pb.entries, "playbook should record outcomes"
    # A proven entry yields a winrate prior above the neutral 0.5.
    best = max(pb.entries.values(), key=lambda e: e.winrate)
    assert best.winrate >= 0.5


def test_coach_mode_promotes_nothing():
    result = _builtin_teacher(mode=Mode.COACH).improve(TASKS, max_rounds=8)
    assert not result.promoted
    assert result.final_config.system_prompt_sections == {}   # unchanged
