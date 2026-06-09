# TraceRazor on Real Hugging Face Agent Trajectories

Audit statistics for the product run over real ReAct agent trajectories
from the Hugging Face dataset [`zai-org/AgentInstruct`](https://huggingface.co/datasets/zai-org/AgentInstruct) (formerly `THUDM/AgentInstruct`; arXiv:2310.12823). The corpus is the
vendored real sample converted by `tools/convert_agentinstruct.py`; see
`traces/external/huggingface/agentinstruct/SOURCE.md` for provenance and
the live-fetch path. Reproduce with `python -m benchmark.hf_audit_stats`.

## Corpus

- Traces: **9** (7 analysable, 2 skipped <5 steps)
- Mean TAS: **80.6** (median 81.4)
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
| GAR | 0.247 |
| CSD | 0.415 |
| OBS | 0.913 |

## Per-trace

| Trace | Steps | Tokens | TAS | Grade | SRR | LDI | GAR | OBS | Fixes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| agentinstruct-os_0 | 9 | 534 | 80.4 | Good | 0.778 | 1.0 | 0.196 | 0.916 | 3 |
| agentinstruct-os_1 | 6 | 294 | 81.5 | Good | 0.667 | 1.0 | 0.254 | 0.908 | 3 |
| agentinstruct-os_2 | 6 | 429 | 81.4 | Good | 0.667 | 1.0 | 0.182 | 0.937 | 3 |
| agentinstruct-os_3 | 6 | 356 | 85.0 | Good | 0.667 | 1.0 | 0.313 | 0.879 | 1 |
| agentinstruct-os_4 | 7 | 468 | 84.0 | Good | 0.714 | 1.0 | 0.242 | 0.921 | 1 |
| agentinstruct-os_5 | 9 | 811 | 77.2 | Good | 0.667 | 1.0 | 0.273 | 0.936 | 4 |
| agentinstruct-os_6 | 9 | 425 | 75.0 | Good | 0.556 | 0.556 | 0.268 | 0.892 | 2 |
