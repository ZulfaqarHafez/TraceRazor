# Provenance: Hugging Face AgentInstruct real-data corpus

These traces are converted from the Hugging Face dataset
[`zai-org/AgentInstruct`](https://huggingface.co/datasets/zai-org/AgentInstruct)
— the renamed `THUDM/AgentInstruct` corpus of **1,866 real ReAct agent
trajectories** (28.5K downloads, 235 likes), introduced in *AgentTuning:
Enabling Generalized Agent Abilities for LLMs* (arXiv:2310.12823).

**License status:** unconfirmed as of 2026-07-12. The Hugging Face dataset
card/API does not declare a license, and the linked upstream AgentTuning
repository does not publish a repository license. Do not assume Apache-2.0 or
redistribute this sample as a TraceRazor product asset without resolving
permission. The sample and converted traces are excluded from distribution
artifacts; they remain source-tree research inputs with provenance retained.

## What is here

`agentinstruct-*.json` are TraceRazor raw traces produced by
`tools/convert_agentinstruct.py` from a vendored real sample of the dataset
(`benchmark/data/_agentinstruct_hf_sample.py`). The sample covers two of the
dataset's six task splits:

| Split | Agent | Action format | Traces |
|---|---|---|---|
| `os` | Linux/bash operator | `Think: … / Act: bash \`\`\`…\`\`\`` | os_0…os_7, os_11, os_16, os_18 |
| `db` | MySQL operator | `… / Action: Operation \`\`\`sql…\`\`\`` | db_0, db_1 |

**Few-shot scaffolding is excluded.** Upstream rows embed the dataset's fixed
one-shot demonstration (the "count files in /etc" example, identical in every
`os` row) and the `db` split's "Ok." acknowledgement before the real
trajectory, marked with `loss: false` on the gpt turns (the real agent turns
carry `loss: true`). The converter audits only the real-task turns: auditing
the scaffolding would pseudo-replicate the same canned steps into every trace
and mis-anchor the goal metrics.

With scaffolding excluded, most real `os` trajectories are 3–4 steps: 4 traces
(os_0, os_5, os_6, os_11) meet TraceRazor's 5-step analysis floor and 9 land
below it. The sub-floor majority is kept deliberately — it exercises the skip
path and *measures the floor's coverage cost on real data* (≈69% of this
sample).

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
