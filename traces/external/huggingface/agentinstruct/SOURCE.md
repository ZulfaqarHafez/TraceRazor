# Provenance: Hugging Face AgentInstruct real-data corpus

These traces are converted from the Hugging Face dataset
[`zai-org/AgentInstruct`](https://huggingface.co/datasets/zai-org/AgentInstruct)
— the renamed `THUDM/AgentInstruct` corpus of **1,866 real ReAct agent
trajectories** (28.5K downloads, 235 likes), introduced in *AgentTuning:
Enabling Generalized Agent Abilities for LLMs* (arXiv:2310.12823). License:
Apache-2.0.

## What is here

`agentinstruct-*.json` are TraceRazor raw traces produced by
`tools/convert_agentinstruct.py` from a vendored real sample of the dataset
(`benchmark/data/_agentinstruct_hf_sample.py`). The sample covers two of the
dataset's six task splits:

| Split | Agent | Action format | Traces |
|---|---|---|---|
| `os` | Linux/bash operator | `Think: … / Act: bash \`\`\`…\`\`\`` | os_0 … os_6 |
| `db` | MySQL operator | `… / Action: Operation \`\`\`sql…\`\`\`` | db_0, db_1 |

The `os` trajectories are multi-step (5–9 steps) and analysable; the `db`
trajectories are short (3 steps) and intentionally land below TraceRazor's
5-step floor, exercising the skip path.

## How the rows were obtained

The rows were read from the Hugging Face Hub dataset viewer for
`zai-org/AgentInstruct` (config `default`, splits `os` and `db`). The agent
turns (the `Think:`/`Act:` reasoning that TraceRazor scores) are verbatim.
Terminal colour/control bytes in the *observation* turns were normalised to
their visible text — a standard terminal rendering — while preserving the
`[truncated because the output is too long]` markers and visible payloads so
token magnitudes stay realistic.

## Reproducing / refreshing

Anywhere the Hub is reachable, fetch live rows and regenerate:

```python
from benchmark.hf_loader import HFAgentInstructLoader
from tools.convert_agentinstruct import convert_row

rows = HFAgentInstructLoader(source="live", split="os", max_rows=50).load()
traces = [convert_row(r) for r in rows]
```

Or from the vendored sample (offline, hermetic — what CI uses):

```bash
python -m tools.convert_agentinstruct --bundled \
    --out-dir traces/external/huggingface/agentinstruct
```

Then audit and collect statistics:

```bash
cargo build --release -p tracerazor
python -m benchmark.hf_audit_stats   # writes STATS.json + docs/huggingface_agentinstruct_audit.md
```
