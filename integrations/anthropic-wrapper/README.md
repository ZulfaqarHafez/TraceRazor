# tracerazor-anthropic

Drop-in replacement for `anthropic.Anthropic` with automatic TraceRazor token-efficiency auditing. Change one import and every `messages.create()` call is captured as a reasoning step.

## Install

```bash
pip install tracerazor-anthropic
```

You'll also need the `tracerazor` binary on your PATH (or a running `tracerazor-server`).

## Use

```python
from tracerazor_anthropic import Anthropic

client = Anthropic(agent_name="support-bot")

resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=512,
    messages=[{"role": "user", "content": "Refund order ORD-9182"}],
)

# ... more calls ...

report = client.audit()
print(report.summary())
```

Constructor accepts every kwarg the plain `anthropic.Anthropic` does plus:

| Kwarg | Default | Meaning |
|---|---|---|
| `agent_name` | `"claude-agent"` | Name used in the TraceRazor report |
| `tracer` | `None` | Append to an existing `Tracer` instead of creating one |
| `server` | `None` | URL of a running `tracerazor-server` (default: local binary) |

## Not yet wrapped

- Streaming (`messages.stream`) — fall through to the underlying client via `client._inner` for now.
- Async client (`anthropic.AsyncAnthropic`).

Both are planned.

## License

Apache-2.0