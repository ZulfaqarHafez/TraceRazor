"""Tests for real online verification (teacher.online + teacher.stats).

Uses the bundled stdlib OpenAI-compatible server, so these make genuine HTTP
requests and parse genuine chat-completion responses -- no API key, no external
network. Self-skips if `requests` is unavailable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import requests  # noqa: F401
    _HAVE_REQUESTS = True
except Exception:
    _HAVE_REQUESTS = False

from teacher import AgentConfig, Diagnoser  # noqa: E402
from teacher.schemas import Decision, EvalResult  # noqa: E402
from teacher.stats import StatGate  # noqa: E402


def _tools():
    from teacher.online import ToolSpec
    schema = {"type": "object", "properties": {"order_id": {"type": "string"}},
              "required": ["order_id"]}
    return {n: ToolSpec(n, n, schema, (lambda order_id="", **_: f"{n}:{order_id}"))
            for n in ("get_order", "check_eligibility", "refund", "get_status")}


def _with_agent(fn):
    """Run fn(agent) against a fresh mock server, guaranteeing shutdown."""
    from teacher._mockserver import serve_in_thread
    from teacher.online import LLMClient, OnlineAgent
    base_url, server = serve_in_thread()
    try:
        agent = OnlineAgent(LLMClient(base_url=base_url, model="mock"), _tools())
        return fn(agent)
    finally:
        server.shutdown()


# --------------------------------------------------------------------------- #
# Stats gate (pure stdlib, always runs)
# --------------------------------------------------------------------------- #
def test_statgate_accepts_real_paired_savings():
    base = EvalResult(tokens=[100, 200, 100, 200, 100, 200], success=[True] * 6)
    trial = EvalResult(tokens=[80, 160, 80, 160, 80, 160], success=[True] * 6)
    assert StatGate(min_savings_pct=3.0).decide(base, trial) is Decision.ACCEPT


def test_statgate_rejects_quality_regression():
    base = EvalResult(tokens=[100] * 6, success=[True] * 6)
    trial = EvalResult(tokens=[40] * 6, success=[False] * 6)   # huge token win, broken
    ev = StatGate().evaluate(base, trial)
    assert ev.decision is Decision.REJECT_QUALITY
    assert ev.success_delta_lo90 < -0.05


def test_statgate_rejects_no_gain():
    base = EvalResult(tokens=[100] * 6, success=[True] * 6)
    trial = EvalResult(tokens=[99] * 6, success=[True] * 6)    # negligible
    assert StatGate(min_savings_pct=3.0).decide(base, trial) is Decision.REJECT_NO_GAIN


# --------------------------------------------------------------------------- #
# Online agent loop (real HTTP to mock server)
# --------------------------------------------------------------------------- #
def test_online_agent_completes_task_with_real_usage():
    if not _HAVE_REQUESTS:
        print("skip: no requests"); return
    from teacher.online import OnlineTask

    def check(agent):
        task = OnlineTask("ORD-1", "refund", ["get_order", "check_eligibility", "refund"])
        out = agent.run(AgentConfig(), task)
        assert out.success is True
        assert out.tokens > 0 and out.n_calls >= 3
        assert set(["get_order", "check_eligibility", "refund"]).issubset(set(out.executed_tools))
    _with_agent(check)


def test_loop_breaker_reduces_tokens_online():
    if not _HAVE_REQUESTS:
        print("skip: no requests"); return
    from teacher.online import OnlineTask, evaluate_online

    def check(agent):
        holdout = [OnlineTask("ORD-1", "refund", ["get_order", "check_eligibility", "refund"])]
        base = evaluate_online(agent, AgentConfig(), holdout, repeats=1)
        cfg = AgentConfig(policies={"loop_breaker": {"max_repeats": 1}})
        trial = evaluate_online(agent, cfg, holdout, repeats=1)
        assert trial.mean_tokens < base.mean_tokens       # duplicate call suppressed
        assert trial.success_rate == 1.0
    _with_agent(check)


def test_step_cap_breaks_success_online():
    if not _HAVE_REQUESTS:
        print("skip: no requests"); return
    from teacher.online import OnlineTask, evaluate_online

    def check(agent):
        holdout = [OnlineTask("ORD-1", "refund", ["get_order", "check_eligibility", "refund"])]
        cfg = AgentConfig(policies={"step_cap": 2})
        res = evaluate_online(agent, cfg, holdout, repeats=1)
        assert res.success_rate < 1.0                     # cannot finish 3 tools in 2 turns
    _with_agent(check)


def test_verify_online_accepts_safe_rejects_unsafe():
    if not _HAVE_REQUESTS:
        print("skip: no requests"); return
    from teacher.online import OnlineTask, verify_online

    def check(agent):
        holdout = [
            OnlineTask("ORD-1", "refund", ["get_order", "check_eligibility", "refund"]),
            OnlineTask("ORD-2", "status", ["get_order", "get_status"]),
        ]
        res = verify_online(AgentConfig(), holdout, agent,
                            Diagnoser(prefer_auditor=True), gate=StatGate(), repeats=3)
        assert res.baseline.success_rate >= 0.99
        assert res.accepted, "expected at least one accepted intervention"
        # every accepted intervention preserved success
        verdicts = {iv.key: ev.decision for iv, ev in res.rounds}
        if "step_cap" in verdicts:
            assert verdicts["step_cap"] is Decision.REJECT_QUALITY
    _with_agent(check)
