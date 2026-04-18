"""Offline demo — no API key required.

Simulates a 3-step airline booking agent using mock_llm.
Shows: adaptive K, mutation boundary collapse, ConsensusReport.

Run:
    python demo_offline.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tracerazor import AdaptiveKNode, ConsensusReport, mock_llm


# ── tool implementations (deterministic stubs) ───────────────────────────────

def search_direct_flight(origin: str, destination: str, date: str) -> str:
    return (
        f"[{{'flight_id': 'AA100', 'from': '{origin}', 'to': '{destination}', "
        f"'date': '{date}', 'price': 450, 'seats_available': 12}}]"
    )


def book_reservation(flight_id: str, passenger_name: str) -> str:
    return f"{{'confirmation': 'CONF-{flight_id}-9182', 'status': 'confirmed'}}"


# ── scripted LLM responses (all K branches agree → full consensus) ───────────
# mock_llm returns same response to all K branches at each step

RESPONSES = [
    # Step 0: search for a flight
    {
        "tool_name": "search_direct_flight",
        "arguments": {"origin": "JFK", "destination": "LAX", "date": "2024-06-15"},
        "input_tokens": 320,
        "output_tokens": 45,
        "cached_tokens": 0,
    },
    # Step 1: book the chosen flight (mutating)
    {
        "tool_name": "book_reservation",
        "arguments": {"flight_id": "AA100", "passenger_name": "Alice Smith"},
        "input_tokens": 480,
        "output_tokens": 38,
        "cached_tokens": 320,  # prefix cached from step 0
    },
    # Step 2: final answer
    {
        "final_answer": (
            "I've booked flight AA100 from JFK to LAX on June 15. "
            "Your confirmation number is CONF-AA100-9182."
        ),
        "input_tokens": 520,
        "output_tokens": 52,
        "cached_tokens": 480,
    },
]

# ── divergence scenario: step 0 has 2 different tool choices ─────────────────
# Use a custom LLM that disagrees on the first step to demonstrate PARTIAL outcome.

_STEP_RESPONSES = [
    # step 0: branches 0-3 agree, branch 4 proposes different destination
    [
        {"tool_name": "search_direct_flight", "arguments": {"origin": "JFK", "destination": "LAX", "date": "2024-06-15"}, "input_tokens": 300, "output_tokens": 40},
        {"tool_name": "search_direct_flight", "arguments": {"origin": "JFK", "destination": "LAX", "date": "2024-06-15"}, "input_tokens": 300, "output_tokens": 40},
        {"tool_name": "search_direct_flight", "arguments": {"origin": "JFK", "destination": "LAX", "date": "2024-06-15"}, "input_tokens": 300, "output_tokens": 40},
        {"tool_name": "search_direct_flight", "arguments": {"origin": "JFK", "destination": "SFO", "date": "2024-06-15"}, "input_tokens": 300, "output_tokens": 40},
    ],
    # step 1 (k=5 reset): all agree on booking
    [{"tool_name": "book_reservation", "arguments": {"flight_id": "AA100", "passenger_name": "Alice"}, "input_tokens": 450, "output_tokens": 35, "cached_tokens": 300}] * 5,
    # step 2: final answer
    [{"final_answer": "Booking confirmed: CONF-AA100-9182", "input_tokens": 500, "output_tokens": 45, "cached_tokens": 450}] * 5,
]


async def _divergence_llm(messages, tools):
    step = sum(1 for m in messages if m.get("role") == "tool")
    k_in_step = getattr(_divergence_llm, "_call_count_at_step", {}).get(step, 0)
    counts = getattr(_divergence_llm, "_call_count_at_step", {})
    counts[step] = k_in_step + 1
    _divergence_llm._call_count_at_step = counts
    row = _STEP_RESPONSES[min(step, len(_STEP_RESPONSES) - 1)]
    return dict(row[min(k_in_step, len(row) - 1)])


# ── main ──────────────────────────────────────────────────────────────────────

async def run_full_consensus_demo():
    print("=" * 60)
    print("DEMO 1: Full-consensus airline booking (k_max=5)")
    print("=" * 60)

    llm = mock_llm(RESPONSES)
    node = AdaptiveKNode(
        llm=llm,
        tools={
            "search_direct_flight": search_direct_flight,
            "book_reservation": book_reservation,
        },
        k_max=5,
        k_min=2,
        mutation_metadata={
            "book_reservation": True,
            "search_direct_flight": False,
        },
    )

    state = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful airline booking assistant.",
            },
            {
                "role": "user",
                "content": "Book me a flight from JFK to LAX on June 15 for Alice Smith.",
            },
        ]
    }

    result = await node(state)
    report: ConsensusReport = result["consensus_report"]

    print(f"\nResult: {report.trajectory[-1]['content']}\n")
    print(f"Summary: {report.summary()}\n")

    print("Per-step breakdown:")
    print(f"{'Step':>4}  {'Tool/Answer':<35}  {'Consensus':>9}  {'K':>2}  {'Outcome':<10}  {'Tokens':>6}")
    print("-" * 80)
    for s in report.steps:
        label = s.executed_tool or ("-> " + (s.final_answer or "")[:30])
        mut = " [M]" if s.is_mutating else ""
        print(
            f"{s.step_number:>4}  {label + mut:<35}  {s.consensus_rate:>9.1%}  "
            f"{s.k_used:>2}  {s.outcome:<10}  {s.tokens.total:>6,}"
        )

    print(f"\nToken cost: {report.tokens.total:,} total  "
          f"({report.tokens.cached:,} cached / {report.tokens.fresh:,} fresh)")


async def run_divergence_demo():
    print("\n" + "=" * 60)
    print("DEMO 2: Partial-consensus demo (one branch diverges at step 0)")
    print("=" * 60)

    node = AdaptiveKNode(
        llm=_divergence_llm,
        tools={
            "search_direct_flight": search_direct_flight,
            "book_reservation": book_reservation,
        },
        k_max=4,
        k_min=2,
        mutation_metadata={
            "book_reservation": True,
            "search_direct_flight": False,
        },
    )

    state = {
        "messages": [{"role": "user", "content": "Book LAX for Alice"}]
    }
    result = await node(state)
    report = result["consensus_report"]

    print(f"\nSummary: {report.summary()}")
    print(f"Divergent steps: {report.divergences}")
    if report.steps[0].divergent_alternatives:
        print(f"Step 0 alternative: {report.steps[0].divergent_alternatives[0]}")


if __name__ == "__main__":
    asyncio.run(run_full_consensus_demo())
    asyncio.run(run_divergence_demo())
