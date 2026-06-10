"""Ship-plan Phase 1 acceptance: verdict precision on the real corpus.

Two corpus-wide invariants from the reviewer adjudications, plus the pinned
step-level verdicts on the two hand-adjudicated traces.
"""
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _binary():
    env = os.environ.get("TRACERAZOR_BIN")
    if env and Path(env).is_file():
        return env
    for cand in ("release", "debug"):
        p = REPO / "target" / cand / "tracerazor"
        if p.is_file():
            return str(p)
    return None


def _all_traces():
    base = REPO / "traces" / "external"
    return sorted(base.rglob("*.json"))


def _audit(path):
    binary = _binary()
    env = dict(os.environ, HOME=tempfile.mkdtemp())
    env.pop("OPENAI_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    out = subprocess.run(
        [binary, "audit", str(path), "--format", "json", "--hermetic",
         "--min-steps", "2"],
        capture_output=True, text=True, env=env,
    )
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


MUTATING_NAME = re.compile(
    r"book|create|update|delete|remove|exchange|send|post|write|edit|insert"
    r"|cancel|modify|transfer|pay|refund|commit|push|upload|submit"
)

# Syntax shapes that must never appear as "ungrounded" evidence items:
# markdown emphasis, shell variables, regex classes, awk blocks, globs.
SYNTAX_ARTIFACT = re.compile(r"^\s*[\$\[\{~]|\*\*|^`|\*|\{print")


@pytest.fixture(scope="module")
def corpus_reports():
    if _binary() is None:
        pytest.skip("tracerazor binary not built")
    reports = {}
    for f in _all_traces():
        r = _audit(f)
        if r is not None:
            reports[f.name] = (r, json.loads(f.read_text()))
    assert len(reports) >= 30, "expected the full real corpus to audit"
    return reports


def test_no_successful_mutating_call_is_deleted(corpus_reports):
    violations = []
    for name, (report, trace) in corpus_reports.items():
        ok_mutating = {
            s["id"] for s in trace["steps"]
            if s.get("tool_name")
            and MUTATING_NAME.search(s["tool_name"].lower())
            and s.get("tool_success") is not False
        }
        for d in report.get("diff", []):
            if d["step_id"] in ok_mutating and d["action"] == "delete":
                violations.append((name, d["step_id"]))
    assert not violations, f"successful mutating calls deleted: {violations}"


def test_no_syntax_artifacts_in_ungrounded(corpus_reports):
    artifacts = []
    for name, (report, _) in corpus_reports.items():
        for u in (report.get("agf") or {}).get("ungrounded", []):
            if SYNTAX_ARTIFACT.search(u["literal"]):
                artifacts.append((name, u["literal"]))
    assert not artifacts, f"syntax flagged as ungrounded: {artifacts}"


def test_adjudicated_airline_task0_verdicts(corpus_reports):
    report, _ = corpus_reports["gpt-4o_airline_task0.json"]
    actions = {d["step_id"]: d["action"] for d in report["diff"]}
    # The failed booking is the dead weight; the responsive re-searches, the
    # corrected re-confirmation, the successful retry, and the final
    # confirmation all stay (eval-engineer adjudication, ship-plan 1.1/1.2).
    assert actions[10] == "delete", actions
    for kept in (6, 7, 14, 15):
        assert actions[kept] != "delete", f"step {kept}: {actions[kept]}"


def test_adjudicated_marshmallow_verification_kept(corpus_reports):
    report, _ = corpus_reports["marshmallow_cursors.json"]
    actions = {d["step_id"]: d["action"] for d in report["diff"]}
    # The post-edit verification re-run must survive (ship-plan 1.1/1.3).
    assert actions[20] != "delete", actions[20]


def test_fixes_carry_risk_classes(corpus_reports):
    risks = set()
    for _, (report, _) in corpus_reports.items():
        for f in report.get("fixes", []):
            assert f.get("risk") in ("safe", "needs_review", "dangerous"), f
            risks.add(f["risk"])
    assert "safe" in risks
