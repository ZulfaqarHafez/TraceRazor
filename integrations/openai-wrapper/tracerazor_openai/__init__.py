"""
tracerazor-openai — drop-in replacement for openai.OpenAI with automatic
TraceRazor trace capture.

Every chat.completions.create() call is recorded as a reasoning step. After
your workflow finishes, call client.audit() to get a full TAS report.

    from tracerazor_openai import OpenAI

    client = OpenAI(agent_name="support-bot")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Refund order ORD-9182"}],
    )

    report = client.audit()
    print(report.summary())
"""
from .client import OpenAI

__all__ = ["OpenAI"]
__version__ = "0.2.0"