# TraceRazor on Real Hugging Face Agent Trajectories

Audit statistics for the product run over real ReAct agent trajectories
from the Hugging Face dataset [`zai-org/AgentInstruct`](https://huggingface.co/datasets/zai-org/AgentInstruct) (formerly `THUDM/AgentInstruct`; arXiv:2310.12823). The corpus is the
vendored real sample converted by `tools/convert_agentinstruct.py`; see
`traces/external/huggingface/agentinstruct/SOURCE.md` for provenance and
the live-fetch path. Reproduce with `python -m benchmark.hf_audit_stats`.

## Corpus

- Traces: **9** (7 analysable, 2 skipped <5 steps)
- Mean TAS: **80.6** (median 82.0)
- Grade distribution: {'Good': 7}
- Mean MVTG (structural waste): **0.278**
- Fix patches emitted: **17**

## Mean normalised metric scores (1.0 = no waste detected)

| Metric | Mean (normalised) |
|---|---:|
| SRR | 0.674 |
| LDI | 0.937 |
| TCA | 0.959 |
| RDA | 0.726 |
| ISR | 0.984 |
| TUR | 0.945 |
| CCE | 0.974 |
| DBO | 0.588 |
| VDI | 0.867 |
| SHL | 1.0 |
| CCR | 0.913 |
| GAR | 0.202 |
| CSD | 0.48 |
| OBS | 0.913 |

## Per-trace

| Trace | Steps | Tokens | TAS | Grade | SRR | LDI | GAR | OBS | Fixes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| agentinstruct-os_0 | 9 | 534 | 80.5 | Good | 0.778 | 1.0 | 0.315 | 0.916 | 3 |
| agentinstruct-os_1 | 6 | 294 | 82.0 | Good | 0.667 | 1.0 | 0.0 | 0.908 | 3 |
| agentinstruct-os_2 | 6 | 429 | 82.4 | Good | 0.667 | 1.0 | 0.0 | 0.937 | 3 |
| agentinstruct-os_3 | 6 | 356 | 84.7 | Good | 0.667 | 1.0 | 0.396 | 0.879 | 1 |
| agentinstruct-os_4 | 7 | 468 | 84.1 | Good | 0.714 | 1.0 | 0.25 | 0.921 | 1 |
| agentinstruct-os_5 | 9 | 811 | 77.2 | Good | 0.667 | 1.0 | 0.328 | 0.936 | 4 |
| agentinstruct-os_6 | 9 | 425 | 73.6 | Good | 0.556 | 0.556 | 0.126 | 0.892 | 2 |
