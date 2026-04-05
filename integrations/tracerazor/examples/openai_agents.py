"""
Example: TraceRazor + OpenAI (plain openai SDK, any model).

Works the same way for Anthropic, Gemini, or any model —
just swap the API call and extract tokens from the response object.
"""

import os
from openai import OpenAI
from tracerazor_sdk import Tracer

client = OpenAI()


def run_agent(user_message: str):
    tracer = Tracer(agent_name="openai-support-agent", framework="openai")

    # ── Step 1: initial reasoning ─────────────────────────────────────────────
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful customer support agent."},
            {"role": "user", "content": user_message},
        ],
    )
    thought = resp.choices[0].message.content
    tracer.reasoning(
        content=thought,
        tokens=resp.usage.total_tokens,
        input_context=user_message,
    )

    # ── Step 2: tool call (simulated) ─────────────────────────────────────────
    order_details = "Order ORD-9182: blue jacket, $45.00, delivered 2024-01-20"
    tracer.tool(
        name="get_order_details",
        params={"order_id": "ORD-9182"},
        output=order_details,
        success=True,
        tokens=120,
    )

    # ── Step 3: final reasoning ───────────────────────────────────────────────
    resp2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful customer support agent."},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": thought},
            {"role": "user", "content": f"Order details: {order_details}. Now process the refund."},
        ],
    )
    final = resp2.choices[0].message.content
    tracer.reasoning(
        content=final,
        tokens=resp2.usage.total_tokens,
        input_context=f"{user_message}\n{order_details}",
    )

    # ── Analyse ───────────────────────────────────────────────────────────────
    report = tracer.analyse()
    print(report.markdown())
    return report


if __name__ == "__main__":
    report = run_agent("I want a refund for order ORD-9182")
    print(f"\nTAS: {report.tas_score:.1f} [{report.grade}]")
