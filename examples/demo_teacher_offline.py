"""End-to-end Teacher demo -- offline, no API keys.

Shows the full closed loop:

  1. A deliberately wasteful agent (hedging, reformulation, a tool loop,
     over-deep reasoning) is diagnosed by the TraceRazor auditor (the real
     Rust binary if built, else a transparent built-in heuristic).
  2. The Teacher runs a CURRICULUM: it proposes typed interventions, applies
     each to a candidate config, RE-RUNS the agent on a holdout task set, and
     promotes a change only if the quality gate passes (tokens down AND task
     success preserved).
  3. An over-aggressive STRUCT intervention (a step cap that starves real tool
     calls) is REJECTED by the gate -- proving quality preservation.
  4. Lessons are written to a Playbook that transfers across agents.

Run:
    python examples/demo_teacher_offline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teacher import (  # noqa: E402
    AgentConfig,
    Diagnoser,
    Mode,
    Playbook,
    QualityGate,
    Task,
    Teacher,
    render,
    run_task,
)


def main() -> None:
    # A holdout of multi-tool tasks. Each needs ALL its tool calls to succeed,
    # so an over-aggressive step cap will break success and must be rejected.
    tasks = [
        Task("refund-1", "refund order ORD-9182", ["get_order", "check_eligibility", "refund"]),
        Task("status-1", "status of order ORD-5500", ["get_order", "get_status"]),
        Task("cancel-1", "cancel order ORD-7301", ["get_order", "check_eligibility", "cancel"]),
    ]

    diagnoser = Diagnoser(prefer_auditor=True)
    backend = "real Rust auditor" if diagnoser.binary else "built-in heuristic"
    print(f"[diagnostic backend: {backend}]\n")

    # Show the baseline waste the auditor sees.
    baseline_trace = run_task(AgentConfig(), tasks[0])
    diag = diagnoser.diagnose(baseline_trace["trace"])
    print(f"Baseline trace: {baseline_trace['tokens']} tokens, "
          f"TAS {diag.tas_score} ({diag.source})")
    print("Detected waste patterns:",
          ", ".join(f"{p.kind.value}(sev {p.severity:.1f})" for p in diag.patterns) or "none")

    playbook = Playbook(path=str(Path(__file__).parent / "teacher_playbook.json"))
    teacher = Teacher(
        AgentConfig(), framework="langgraph", mode=Mode.CURRICULUM,
        gate=QualityGate(min_savings_pct=3.0, success_delta=0.02),
        playbook=playbook, diagnoser=diagnoser,
    )

    result = teacher.improve(tasks, max_rounds=8)
    print("\n" + render(result))

    # Demonstrate cross-agent memory transfer: a second agent with the same
    # waste pattern is primed by the playbook.
    print("\nPLAYBOOK (persisted, transfers across agents)")
    print(playbook.summary())
    playbook.save()

    # QUALITY-GATE STRESS TEST: directly try an over-aggressive STRUCT fix (a
    # hard step cap that starves required tool calls) on the promoted config.
    # The gate must REJECT it because it breaks task success -- this is the
    # property that makes the savings claim safe and falsifiable.
    from teacher import Intervention, Target, Tier, WasteKind, apply, evaluate

    promoted = result.final_config
    base_eval = evaluate(promoted, tasks, diagnoser)
    harmful = Intervention(
        WasteKind.OVER_DEPTH, Target.RUNTIME_POLICY, Tier.STRUCT, "step_cap",
        {"value": 2}, predicted_savings=999, predicted_risk=0.8,
        rationale="Aggressive hard step cap.")
    harmful_eval = evaluate(apply(harmful, promoted), tasks, diagnoser)
    verdict = teacher.gate.decide(base_eval, harmful_eval)
    print("\nQUALITY-GATE STRESS TEST")
    print(f"  step_cap=2 would change tokens {base_eval.mean_tokens:.0f} -> "
          f"{harmful_eval.mean_tokens:.0f} but success "
          f"{base_eval.success_rate*100:.0f}% -> {harmful_eval.success_rate*100:.0f}%")
    print(f"  gate verdict: {verdict.value}  (token win ignored -- it breaks the task)")

    # Acceptance criteria for the demo.
    assert result.total_token_saving_pct > 0, "expected net token savings"
    assert all(vr.success_after >= 0.99 for vr in result.accepted), \
        "no accepted change may regress task success"
    assert verdict.value == "REJECT_QUALITY", \
        "the over-aggressive step cap should have been rejected by the gate"
    print("\n[OK] tokens reduced, task success preserved, unsafe fix rejected.")


if __name__ == "__main__":
    main()
