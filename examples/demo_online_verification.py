"""REAL online verification demo.

Runs an actual tool-calling agent over HTTP against an OpenAI-compatible
endpoint, measures real token usage + task success, then for each proposed
intervention RE-RUNS the agent online and gates on a statistical
non-inferiority test. Promotes only changes that provably cut tokens without
regressing success.

By default it targets a bundled stdlib OpenAI-compatible server so it runs with
NO API key and NO network egress. To verify against a real provider instead:

    export TRACERAZOR_LLM_BASE_URL=https://api.openai.com/v1
    export TRACERAZOR_LLM_API_KEY=sk-...
    export TRACERAZOR_LLM_MODEL=gpt-4o-mini
    python examples/demo_online_verification.py --live

The agent loop, token accounting, and gate are identical in both modes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teacher import AgentConfig, Diagnoser  # noqa: E402
from teacher.online import (  # noqa: E402
    LLMClient, OnlineAgent, OnlineTask, ToolSpec, verify_online,
)
from teacher.stats import StatGate  # noqa: E402


def build_tools() -> dict:
    def get_order(order_id="", **_): return f"order {order_id}: blue jacket $45 delivered"
    def check_eligibility(order_id="", **_): return "eligible for refund"
    def refund(order_id="", **_): return f"refunded $45 for {order_id}"
    def get_status(order_id="", **_): return "in transit"
    schema = {"type": "object", "properties": {"order_id": {"type": "string"}},
              "required": ["order_id"]}
    specs = [
        ToolSpec("get_order", "Fetch order details", schema, get_order),
        ToolSpec("check_eligibility", "Check refund eligibility", schema, check_eligibility),
        ToolSpec("refund", "Issue a refund", schema, refund),
        ToolSpec("get_status", "Get shipping status", schema, get_status),
    ]
    return {s.name: s for s in specs}


def main() -> None:
    live = "--live" in sys.argv
    server = None
    if live:
        client = LLMClient.from_env()
        print(f"[LIVE] endpoint={client.base_url} model={client.model} "
              f"key={'set' if client.api_key else 'MISSING'}")
        if not client.api_key:
            print("No API key in env; aborting --live.")
            return
    else:
        from teacher._mockserver import serve_in_thread
        base_url, server = serve_in_thread()
        client = LLMClient(base_url=base_url, api_key="", model="mock-gpt")
        print(f"[OFFLINE] stdlib OpenAI-compatible server at {base_url}")

    agent = OnlineAgent(client, build_tools(), max_turns=12)
    holdout = [
        OnlineTask("ORD-9182", "Refund this order.", ["get_order", "check_eligibility", "refund"]),
        OnlineTask("ORD-5500", "What's the status of this order?", ["get_order", "get_status"]),
        OnlineTask("ORD-7301", "Refund this order.", ["get_order", "check_eligibility", "refund"]),
    ]

    try:
        diagnoser = Diagnoser(prefer_auditor=True)
        print(f"[diagnostic backend: {diagnoser.backend}]\n")
        result = verify_online(
            AgentConfig(), holdout, agent, diagnoser,
            gate=StatGate(min_savings_pct=3.0, success_delta=0.05), repeats=3)
        print(result.render())

        # Acceptance checks for the demo.
        assert result.baseline and result.baseline.success_rate >= 0.99
        assert result.accepted, "expected at least one verified, accepted intervention"
        final_tokens = sum(
            agent.run(result.final_config, t, seed=0).tokens for t in holdout)
        base_tokens = sum(
            agent.run(result.base_config, t, seed=0).tokens for t in holdout)
        saved = 100 * (base_tokens - final_tokens) / base_tokens
        print(f"\n  verified net token reduction: {saved:.1f}% "
              f"({base_tokens} -> {final_tokens} tokens), success preserved")
        assert saved > 0
        print("\n[OK] real HTTP agent loop, real usage tokens, statistical gate.")
    finally:
        if server is not None:
            server.shutdown()


if __name__ == "__main__":
    main()
