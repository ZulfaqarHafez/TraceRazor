"""Tests for the two new layers:
  L1 -- rich real-auditor Diagnosis parsing (+ auditor-fix mapping)
  L2 -- LangGraph adapter + COACH mode

The auditor tests self-skip when no Rust binary / native module is available.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from teacher import (  # noqa: E402
    AgentConfig, Diagnoser, LangGraphAdapter, Mode, RunRecorder, Teacher,
    Target, Tier, from_auditor_fixes,
)
from teacher.schemas import Diagnosis, WasteKind  # noqa: E402


def _auditor_available() -> bool:
    return Diagnoser(prefer_auditor=True).backend in ("native", "subprocess")


# --------------------------------------------------------------------------- #
# L1 -- real-auditor Diagnosis
# --------------------------------------------------------------------------- #
def test_auditor_backend_selected_when_present():
    d = Diagnoser(prefer_auditor=True)
    assert d.backend in ("native", "subprocess", "builtin")
    if d.binary or d._audit:
        assert d.backend in ("native", "subprocess")


def test_auditor_rich_parse_extracts_patterns_and_fixes():
    if not _auditor_available():
        print("skip: no auditor backend"); return
    import json
    trace = json.load(open("traces/support-agent-run-2847.json"))
    d = Diagnoser(prefer_auditor=True).diagnose(trace)
    assert d.source == "auditor"
    assert d.tas_score > 0 and d.total_tokens > 0
    assert d.patterns, "expected waste patterns from the real auditor"
    # severities are clamped 0..1 and step ids are ints
    for p in d.patterns:
        assert 0.0 < p.severity <= 1.0
        assert all(isinstance(i, int) for i in p.step_ids)
    assert d.auditor_fixes, "auditor should emit fixes for this trace"
    assert "reduction_pct" in d.savings


def test_from_auditor_fixes_maps_to_applicable_interventions():
    diag = Diagnosis(
        "t", "a", "langgraph", tas_score=60.0, total_tokens=1000, patterns=[],
        auditor_fixes=[
            {"fix_type": "termination_guard", "target": "system_prompt",
             "patch": "stop looping", "estimated_token_savings": 420},
            {"fix_type": "tool_schema", "target": "check_refund_eligibility",
             "patch": "missing required parameter: order_id",
             "estimated_token_savings": 580},
        ])
    ivs = from_auditor_fixes(diag)
    by_key = {iv.key: iv for iv in ivs}
    assert "loop_breaker" in by_key
    assert by_key["loop_breaker"].target is Target.RUNTIME_POLICY
    assert by_key["loop_breaker"].predicted_savings == 420
    assert "check_refund_eligibility" in by_key
    tool_iv = by_key["check_refund_eligibility"]
    assert tool_iv.target is Target.TOOL_DEF and tool_iv.tier is Tier.TOOL
    assert "order_id" in tool_iv.payload["params"]   # param extracted from patch


# --------------------------------------------------------------------------- #
# L2 -- LangGraph adapter + coach
# --------------------------------------------------------------------------- #
def _adapter_with_runs() -> LangGraphAdapter:
    a = LangGraphAdapter(agent_name="support", framework="langgraph")
    for order in ("ORD-1", "ORD-2"):
        r = a.new_run()
        r.llm("Certainly! Let me think. I think possibly the user wants a refund.")
        r.llm(f"Basically let me re-read: refund {order}.")
        r.tool("get_order", {"order_id": order}, output="ok")
        r.tool("get_order", {"order_id": order}, output="ok")   # duplicate
        r.final("done")
        a.add_run(r)
    return a


def test_run_recorder_builds_auditor_schema():
    r = RunRecorder("x").llm("hi").tool("t", {"a": 1}, output="o").final("done")
    trace = r.end()
    assert {"trace_id", "agent_name", "framework", "task_value_score", "steps"} <= trace.keys()
    assert trace["steps"][0]["type"] == "reasoning"
    assert trace["steps"][1]["tool_name"] == "t"


def test_adapter_collect_and_reset():
    a = _adapter_with_runs()
    assert len(a.collect_traces()) == 2
    a.reset()
    assert a.collect_traces() == []


def test_coach_ranks_and_proposes_without_promoting():
    a = _adapter_with_runs()
    teacher = Teacher(AgentConfig(), mode=Mode.COACH,
                      diagnoser=Diagnoser(prefer_auditor=False))
    report = teacher.coach(a.collect_traces())
    assert report.n_traces == 2
    assert report.recommendations
    # recommendations sorted by (savings * prior) descending
    scores = [r.total_predicted_savings * r.prior_winrate for r in report.recommendations]
    assert scores == sorted(scores, reverse=True)
    # promotes nothing: base config untouched, proposal is separate
    assert teacher.base_config.system_prompt_sections == {}
    # STRUCT (step_cap) is never auto-applied into the proposed diff
    assert "step_cap" not in report.proposed_config.policies


def test_coach_proposed_diff_nonempty_with_waste():
    a = _adapter_with_runs()
    teacher = Teacher(AgentConfig(), mode=Mode.COACH,
                      diagnoser=Diagnoser(prefer_auditor=False))
    report = teacher.coach(a.collect_traces())
    proposed = report.proposed_config
    assert proposed.system_prompt_sections or proposed.policies
