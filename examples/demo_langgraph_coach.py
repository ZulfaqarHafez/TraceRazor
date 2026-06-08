"""LangGraph adapter + COACH demo -- offline, no API keys, no langchain needed.

Simulates two recorded LangGraph runs of a wasteful support agent (hedging,
context reformulation, a duplicated tool call), feeds them through the
``LangGraphAdapter``, diagnoses each with the TraceRazor auditor (real Rust
binary if built, else built-in heuristic), and prints a COACH report: ranked,
human-approvable efficiency interventions + a proposed system-prompt / policy
diff. Promotes nothing -- this is the human-in-the-loop mode.

Run:
    python examples/demo_langgraph_coach.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teacher import AgentConfig, Diagnoser, LangGraphAdapter, Mode, Teacher  # noqa: E402


def record_wasteful_run(adapter: LangGraphAdapter, order: str) -> None:
    """Pretend a LangGraph graph executed; record its steps as they happen."""
    r = adapter.new_run(task_value_score=1.0)
    r.context(f"User: I want a refund for order {order}")
    r.llm("Certainly! I'd be happy to help. Let me carefully think. I think that, "
          "generally speaking, the customer wants a refund and I should look it up.",
          prompt="system+history")
    # Reformulation: restating context it already has.
    r.llm("Basically, to be honest, let me re-read the request again: the user wants "
          f"a refund for order {order}. Essentially I will now proceed.", prompt="x")
    r.tool("get_order", {"order_id": order}, output="blue jacket $45 delivered")
    # Redundant duplicate of the same call.
    r.tool("get_order", {"order_id": order}, output="blue jacket $45 delivered")
    r.tool("check_eligibility", {"order_id": order}, output="eligible")
    r.tool("refund", {"order_id": order}, output="refunded $45")
    r.final("Your refund of $45 has been processed.")
    adapter.add_run(r)


def main() -> None:
    diagnoser = Diagnoser(prefer_auditor=True)
    print(f"[diagnostic backend: {diagnoser.backend}]\n")

    adapter = LangGraphAdapter(agent_name="support-agent", framework="langgraph")
    record_wasteful_run(adapter, "ORD-9182")
    record_wasteful_run(adapter, "ORD-5500")

    traces = adapter.collect_traces()
    print(f"Collected {len(traces)} LangGraph runs "
          f"({sum(len(t['steps']) for t in traces)} steps total)\n")

    teacher = Teacher(AgentConfig(), framework="langgraph", mode=Mode.COACH,
                      diagnoser=diagnoser)
    report = teacher.coach(traces)
    print(report.render())

    assert report.n_traces == 2
    assert report.recommendations, "coach should surface recommendations"
    assert report.proposed_config.system_prompt_sections or \
        report.proposed_config.policies, "coach should propose a non-empty diff"
    print("\n[OK] LangGraph runs ingested -> ranked coach recommendations + diff.")


if __name__ == "__main__":
    main()
