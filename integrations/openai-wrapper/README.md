# tracerazor-openai

Drop-in replacement for `openai.OpenAI` with automatic TraceRazor token-efficiency auditing. Zero friction — change one import line and every `chat.completions.create()` call is captured as a reasoning step.

## Install

```bash
pip install tracerazor-openai
```

You'll also need the `tracerazor` binary on your PATH (or a running `tracerazor-server`).

## Use

```python
from tracerazor_openai import OpenAI

client = OpenAI(agent_name="support-bot")

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Refund order ORD-9182"}],
)

# ... more calls ...

report = client.audit()
print(report.summary())
```

Constructor accepts every kwarg the plain `openai.OpenAI` does plus:

| Kwarg | Default | Meaning |
|---|---|---|
| `agent_name` | `"openai-agent"` | Name used in the TraceRazor report |
| `tracer` | `None` | Append to an existing `Tracer` instead of creating one |
| `server` | `None` | URL of a running `tracerazor-server` (default: local binary) |

## How it works

The wrapper uses composition — every attribute of the underlying `openai.OpenAI` client is proxied via `__getattr__`. Only `chat.completions.create()` is intercepted to record the call as a reasoning step. If instrumentation fails for any reason (parsing, network, etc.) the original response is still returned unchanged.

## License

Apache-2.0