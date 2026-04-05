"""
Example: TraceRazor HTTP mode.

Use this when you don't want the tracerazor binary on every machine.
Start the server once (e.g. Docker), then POST from anywhere.

Start server:
    docker compose up
    # or: ./target/release/tracerazor-server

Install SDK:
    pip install tracerazor[http]
"""

from tracerazor_sdk import Tracer

# Pass server= to use HTTP mode. No binary needed.
tracer = Tracer(
    agent_name="my-agent",
    framework="custom",
    server="http://localhost:8080",
    threshold=70,
)

# Record steps the same way as CLI mode.
tracer.reasoning("Analysing the user request", tokens=500, input_context="User asked for a refund")
tracer.tool("get_order", params={"order_id": "ORD-001"}, output="Order found", success=True, tokens=100)
tracer.reasoning("Processing the refund", tokens=300)

# Analyse — sends to server, returns report.
report = tracer.analyse()
print(report.summary())
print(f"Anomalies: {report.anomalies}")  # populated after 5+ runs for this agent
