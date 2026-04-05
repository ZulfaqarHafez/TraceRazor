"""
Example: TraceRazor + LangGraph customer-support agent.

Demonstrates how to attach the TraceRazorCallback to a LangGraph agent
and retrieve the efficiency report after execution.

Run with:
    OPENAI_API_KEY=<your-key> python customer_support.py
"""

import os
from dotenv import load_dotenv

# Load .env from the repo root (two levels up from this file).
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../../.env"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Import the TraceRazor callback.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracerazor_langgraph import TraceRazorCallback


# ── Fake tools (replace with real ones) ──────────────────────────────────────

@tool
def get_order_details(order_id: str) -> str:
    """Retrieve order details for a given order ID."""
    return f"Order {order_id}: blue jacket, $45.00, placed 2024-01-15, delivered 2024-01-20"


@tool
def check_refund_eligibility(order_id: str) -> str:
    """Check whether an order is eligible for a refund."""
    return f"Order {order_id} is eligible for refund (within 30-day return window)"


@tool
def process_refund(order_id: str, amount: float) -> str:
    """Process a refund for the given order."""
    return f"Refund REF-{hash(order_id) % 9999:04d} processed. ${amount:.2f} credited in 3-5 days."


# ── Agent setup ───────────────────────────────────────────────────────────────

def run_support_agent(user_message: str) -> dict:
    """
    Run the customer-support agent with TraceRazor monitoring.
    Returns a dict with the agent response and efficiency report.
    """
    # Initialise the TraceRazor callback.
    callback = TraceRazorCallback(
        agent_name="customer-support-v3",
        framework="langgraph",
        threshold=70,
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_order_details, check_refund_eligibility, process_refund]
    agent = create_react_agent(model, tools)

    # Run the agent with the TraceRazor callback attached.
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content="You are a helpful customer support agent. Be concise and efficient."),
                HumanMessage(content=user_message),
            ]
        },
        config={"callbacks": [callback]},
    )

    # Analyse the trace after the run completes.
    report = callback.analyse()

    print("\n" + "=" * 60)
    print(report.markdown())
    print("=" * 60 + "\n")

    # Optionally assert efficiency in CI/CD.
    # callback.assert_passes()  # raises AssertionError if TAS < 70

    return {
        "response": result["messages"][-1].content,
        "tas_score": report.tas_score,
        "grade": report.grade,
        "tokens_saved": report.savings.get("tokens_saved", 0),
        "monthly_savings": report.savings.get("monthly_savings_usd", 0),
    }


if __name__ == "__main__":
    result = run_support_agent("I want a refund for order ORD-9182")
    print(f"\nAgent response: {result['response']}")
    print(f"TAS: {result['tas_score']:.1f} [{result['grade']}]")
    print(f"Est. monthly savings at 50K runs: ${result['monthly_savings']:.2f}")
