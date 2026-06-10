# TraceRazor on Real Hugging Face Agent Trajectories

> Provenance: numbers regenerated under the 9-signal composite — GAR, CSD,
> DBO, VDI and SHL are now weight-0 diagnostics (see
> `docs/metric_effectiveness.md`). They still compute and appear in the
> per-metric tables below; their normalised scores are unchanged.

Audit statistics for the product run over real ReAct agent trajectories
from the Hugging Face dataset [`zai-org/AgentInstruct`](https://huggingface.co/datasets/zai-org/AgentInstruct) (formerly `THUDM/AgentInstruct`; arXiv:2310.12823). The corpus is the
vendored real sample converted by `tools/convert_agentinstruct.py`; see
`traces/external/huggingface/agentinstruct/SOURCE.md` for provenance and
the live-fetch path. Reproduce with `python -m benchmark.hf_audit_stats`.

## Corpus

- Traces: **13** (4 analysable, 9 skipped <5 steps)
- Mean TAS: **79.8** (median 80.2)
- Grade distribution: {'Good': 4}
- Mean MVTG (structural waste): **0.364**
- Fix patches emitted: **11**
- Mean AGF (grounding-fidelity diagnostic): **0.951**

## Mean normalised metric scores (1.0 = no waste detected)

| Metric | Mean (normalised) |
|---|---:|
| SRR | 0.65 |
| LDI | 0.833 |
| TCA | 0.9 |
| RDA | 0.367 |
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
| agentinstruct-os_0 | 6 | 414 | 86.1 | Good | 0.833 | 1.0 | 0.284 | 0.925 | 3 |
| agentinstruct-os_11 | 5 | 638 | 83.6 | Good | 0.6 | 1.0 | 0.467 | 0.956 | 1 |
| agentinstruct-os_5 | 6 | 691 | 76.9 | Good | 0.667 | 1.0 | 0.308 | 0.945 | 4 |
| agentinstruct-os_6 | 6 | 305 | 72.6 | Good | 0.5 | 0.333 | 0.331 | 0.895 | 3 |

## Full corpus with the short-trace opt-in (`--min-steps 2`)

Most real ReAct task runs finish in 3-4 steps — below the default
5-step floor. With `--min-steps 2` the audit covers the entire corpus:

- Audited: **13/13**
- Mean TAS: **86.1**
- Fix patches: **20**

| Trace | Steps | TAS | Grade | Fixes |
|---|---:|---:|---|---:|
| agentinstruct-db_0 | 2 | 90.1 | Excellent | 0 |
| agentinstruct-db_1 | 2 | 92.9 | Excellent | 1 |
| agentinstruct-os_0 | 6 | 86.1 | Good | 3 |
| agentinstruct-os_1 | 3 | 93.7 | Excellent | 1 |
| agentinstruct-os_11 | 5 | 83.6 | Good | 1 |
| agentinstruct-os_16 | 4 | 77.3 | Good | 1 |
| agentinstruct-os_18 | 4 | 87.2 | Good | 1 |
| agentinstruct-os_2 | 3 | 93.9 | Excellent | 2 |
| agentinstruct-os_3 | 3 | 87.2 | Good | 1 |
| agentinstruct-os_4 | 4 | 94.6 | Excellent | 1 |
| agentinstruct-os_5 | 6 | 76.9 | Good | 4 |
| agentinstruct-os_6 | 6 | 72.6 | Good | 3 |
| agentinstruct-os_7 | 4 | 83.7 | Good | 1 |
