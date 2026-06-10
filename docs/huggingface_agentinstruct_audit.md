# TraceRazor on Real Hugging Face Agent Trajectories

Audit statistics for the product run over real ReAct agent trajectories
from the Hugging Face dataset [`zai-org/AgentInstruct`](https://huggingface.co/datasets/zai-org/AgentInstruct) (formerly `THUDM/AgentInstruct`; arXiv:2310.12823). The corpus is the
vendored real sample converted by `tools/convert_agentinstruct.py`; see
`traces/external/huggingface/agentinstruct/SOURCE.md` for provenance and
the live-fetch path. Reproduce with `python -m benchmark.hf_audit_stats`.

## Corpus

- Traces: **13** (4 analysable, 9 skipped <5 steps)
- Mean TAS: **78.6** (median 78.7)
- Grade distribution: {'Good': 4}
- Mean MVTG (structural waste): **0.364**
- Fix patches emitted: **10**

## Mean normalised metric scores (1.0 = no waste detected)

| Metric | Mean (normalised) |
|---|---:|
| SRR | 0.65 |
| LDI | 0.833 |
| TCA | 0.9 |
| RDA | 0.45 |
| ISR | 0.958 |
| TUR | 0.888 |
| CCE | 0.956 |
| DBO | 0.875 |
| VDI | 0.902 |
| SHL | 0.967 |
| CCR | 0.857 |
| GAR | 0.348 |
| CSD | 0.487 |
| OBS | 0.93 |

## Per-trace

| Trace | Steps | Tokens | TAS | Grade | SRR | LDI | GAR | OBS | Fixes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| agentinstruct-os_0 | 6 | 414 | 82.2 | Good | 0.833 | 1.0 | 0.284 | 0.925 | 3 |
| agentinstruct-os_11 | 5 | 638 | 81.1 | Good | 0.6 | 1.0 | 0.467 | 0.956 | 1 |
| agentinstruct-os_5 | 6 | 691 | 74.8 | Good | 0.667 | 1.0 | 0.308 | 0.945 | 4 |
| agentinstruct-os_6 | 6 | 305 | 76.3 | Good | 0.5 | 0.333 | 0.331 | 0.895 | 2 |
