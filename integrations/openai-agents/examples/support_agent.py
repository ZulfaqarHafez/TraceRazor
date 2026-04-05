"""
Example: TraceRazor + OpenAI Agents SDK.

pip install tracerazor-openai-agents[agents]
export OPENAI_API_KEY=sk-...
export TRACERAZOR_BIN=/path/to/TraceRazor/target/release/tracerazor
"""

import asyncio
from agents import Agent, Runner, function_tool
from tracerazor_openai_agents import TraceRazorHooks


@function_tool
def get_order_details(order_id: str) -> str:
    """Look up order information."""
    return f"Order {order_id}: blue jacket, $45.00, delivered 2024-01-20"


@function_tool
def process_refund(order_id: str, reason: str) -> str:
    """Process a refund for an order."""
    return f"Refund for {order_id} approved. $45.00 will be credited within 3-5 days."


support_agent = Agent(
    name="SupportAgent",
    instructions="You are a helpful customer support agent. Use tools to resolve customer issues.",
    tools=[get_order_details, process_refund],
)


async def main():
    hooks = TraceRazorHooks(
        agent_name="support-agent",
        threshold=70,
    )

    result = await Runner.run(
        support_agent,
        "I want a refund for order ORD-9182",
        hooks=hooks,
    )

    print("Agent output:", result.final_output)
    print()

    # Analyse after the run.
    report = hooks.analyse()
    print(report.markdown())

    # CI/CD gate:
    hooks.assert_passes()


if __name__ == "__main__":
    asyncio.run(main())
