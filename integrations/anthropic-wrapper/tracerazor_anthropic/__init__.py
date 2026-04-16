"""
tracerazor-anthropic — drop-in replacement for anthropic.Anthropic with
automatic TraceRazor trace capture.

Every messages.create() call is recorded as a reasoning step. After your
workflow finishes, call client.audit() to get a full TAS report.

    from tracerazor_anthropic import Anthropic

    client = Anthropic(agent_name="support-bot")

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": "Refund order ORD-9182"}],
    )

    report = client.audit()
    print(report.summary())
"""
from .client import Anthropic

__all__ = ["Anthropic"]
__version__ = "0.2.0"