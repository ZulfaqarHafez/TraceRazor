# TRICE Research Ledger

Date: 2026-06-21

This ledger records the research base used to design TRICE, the TraceRazor
Information Control Engine. The method is not a copy of any one paper. It
synthesizes context compression, KV/cache efficiency, memory/retrieval,
software-agent harness design, and real-repo evaluation into one product rule:
preserve decisions under a measured token budget.

## Synthesis

TRICE treats context as a budgeted portfolio. Each trace segment is assigned a
state (`essential`, `rehydratable`, `expired`, `redundant`, `distractor`,
`unknown`) and one action (`keep`, `extract`, `summarize`, `mask_with_receipt`,
`anchor_prefix`, `lazy_recall`). A segment is optimized only when the policy can
preserve locked anchors and provide a receipt or rehydration pointer for risky
removals.

The implementation target is not "shortest prompt"; it is:

```text
minimize input tokens and cache/prefill cost
while preserving next-action fidelity, evidence recall, and end-to-end pass rate
```

## 120-Paper Working Ledger

| # | Area | Source | TRICE takeaway |
|---:|---|---|---|
| 1 | agent optimization | [A Self-Improving Coding Agent](https://hf.co/papers/2504.15228) | Closed-loop self-improvement must be gated by benchmark outcomes. |
| 2 | prompt/memory | [CodeAgents](https://hf.co/papers/2507.03254) | Structured compact representations can outperform verbose dialogue traces. |
| 3 | cost analysis | [Tokenomics](https://hf.co/papers/2601.14470) | Input tokens dominate software-agent cost; optimize carried context first. |
| 4 | software agents | [From LLMs to LLM-based Agents for Software Engineering](https://hf.co/papers/2408.02479) | Agent context, memory, tools, and evaluation must be treated as one system. |
| 5 | plan caching | [Cost-Efficient Serving of LLM Agents via Test-Time Plan Caching](https://hf.co/papers/2506.14852) | Reusable plans deserve cache-aware stable prefixes. |
| 6 | harness | [DrafterBench](https://hf.co/papers/2507.11527) | Technical workflows need structured-data and execution-aware evaluation. |
| 7 | trajectory reduction | [AgentDiet](https://hf.co/papers/2509.23586) | Trajectory reduction is a baseline; TRICE must add decision/evidence gates. |
| 8 | memory | [Jenius Agent](https://hf.co/papers/2601.01857) | Experience and layered memory can guide cheaper future decisions. |
| 9 | test-time scaling | [General AgentBench Test-Time Scaling](https://hf.co/papers/2602.18998) | More compute is not automatically better; budgeted frontiers matter. |
| 10 | skills | [SkillReducer](https://hf.co/papers/2603.29919) | Procedural instructions should be progressively disclosed and compressed. |
| 11 | context compression | [Sentinel Tokens](https://hf.co/papers/2310.08152) | Stable summary tokens motivate receipt/anchor representations. |
| 12 | agent tuning | [AgentTuning](https://hf.co/papers/2310.12823) | Agent ability and context efficiency are coupled. |
| 13 | repo agents | [CodeAgent](https://hf.co/papers/2401.07339) | Tool-integrated repo agents expose code-navigation context bloat. |
| 14 | serving | [RelayAttention](https://hf.co/papers/2402.14808) | Long system prompts create serving overhead even when text is repeated. |
| 15 | benchmark creation | [BENCHAGENTS](https://hf.co/papers/2410.22584) | Automated benchmark creation needs verification, not only generated tasks. |
| 16 | bug fixing | [Empirical Study on LLM Agents for Bug Fixing](https://hf.co/papers/2411.10213) | Fault localization and repair have different context needs. |
| 17 | repo deployment | [CSR-Bench](https://hf.co/papers/2502.06111) | Research repos stress setup logs, dependency errors, and retry loops. |
| 18 | project-level eval | [ProjectEval](https://hf.co/papers/2503.07010) | Project-level generation needs whole-repo context discipline. |
| 19 | trajectory judging | [AgentRewardBench](https://hf.co/papers/2504.08942) | Automatic trajectory scores can undercount success; keep objective oracles. |
| 20 | research agents | [ResearchCodeAgent](https://hf.co/papers/2504.20117) | Multi-agent research coding creates reusable plan/context fragments. |
| 21 | prompt compression | [Empirical Study on Prompt Compression](https://hf.co/papers/2505.00019) | Compression needs quality/rate tradeoff measurement. |
| 22 | code graphs | [Code Graph Model](https://hf.co/papers/2505.16901) | Dependency centrality should raise utility for code identifiers. |
| 23 | self-evolving workflows | [SEW](https://hf.co/papers/2505.18646) | The optimizer itself should learn across benchmark rounds. |
| 24 | agent issues | [Can Agents Fix Agent Issues?](https://hf.co/papers/2505.20749) | Agent-system repos are hard, realistic product benchmarks. |
| 25 | security tasks | [SEC-bench](https://hf.co/papers/2506.11791) | Security workflows require strict evidence preservation. |
| 26 | unified SWE agents | [USEagent](https://hf.co/papers/2506.14683) | Multiple SWE task modes demand adapter-level evaluation. |
| 27 | research extension | [RExBench](https://hf.co/papers/2506.22598) | Long-horizon research tasks stress memory and false-progress pruning. |
| 28 | KV cache | [KVFlow](https://hf.co/papers/2507.07400) | Agent workflow schedules can guide cache preservation. |
| 29 | context routing | [RCR-Router](https://hf.co/papers/2508.04903) | Route only role-relevant context subsets to each agent. |
| 30 | observation masking | [The Complexity Trap](https://hf.co/papers/2508.21433) | Simple masking can beat expensive summaries; receipts are first-class. |
| 31 | distillation | [MapCoder-Lite](https://hf.co/papers/2509.17489) | Expensive multi-agent traces can teach compact future behavior. |
| 32 | semantic cache | [SemShareKV](https://hf.co/papers/2509.24832) | Similar prompts can share cache state beyond exact text matches. |
| 33 | context optimization | [ACON](https://hf.co/papers/2510.00615) | Failure-driven compressor optimization belongs in the loop. |
| 34 | SWE benchmark survey | [Benchmarks and Solutions in LLM-Empowered Agentic Systems](https://hf.co/papers/2510.09721) | Report task families, not one blended score. |
| 35 | agent systems efficiency | [What Limits Agentic Systems Efficiency?](https://hf.co/papers/2510.16276) | Environment and API overheads can dominate LLM-only estimates. |
| 36 | environments | [Multi-Docker-Eval](https://hf.co/papers/2512.06915) | Environment setup is itself a real benchmark dimension. |
| 37 | reliability | [ReliabilityBench](https://hf.co/papers/2601.06112) | Noninferiority should include perturbations and failure tolerance. |
| 38 | state diffs | [Agent-Diff](https://hf.co/papers/2602.11224) | State-diff evaluation catches collateral task damage. |
| 39 | replicability | [ReplicatorBench](https://hf.co/papers/2602.11354) | Reproducibility and exact environment state matter. |
| 40 | inference optimization | [ISO-Bench](https://hf.co/papers/2602.19594) | Real optimization tasks test engineering feedback loops. |
| 41 | memory generation | [Trajectory-Informed Memory Generation](https://hf.co/papers/2603.10600) | Useful memory should be learned from trajectories, not handcrafted once. |
| 42 | collaboration cache | [RelayCaching](https://hf.co/papers/2603.13289) | Multi-agent collaboration benefits from decoding/cache reuse. |
| 43 | code execution | [Executing as You Generate](https://hf.co/papers/2604.00491) | Overlap and early interruption can reduce wasted tool cycles. |
| 44 | psychometrics | [Agent Psychometrics](https://hf.co/papers/2604.00594) | Predict task difficulty before setting token budgets. |
| 45 | wild prompt compression | [Prompt Compression in the Wild](https://hf.co/papers/2604.02985) | Latency break-even must include compressor overhead. |
| 46 | multi-agent KV | [TokenDance](https://hf.co/papers/2604.03143) | Collective KV sharing motivates shared context anchors. |
| 47 | productivity agents | [ClawsBench](https://hf.co/papers/2604.05172) | Simulated workspaces need safety and unsafe-action tracking. |
| 48 | agent RL | [Agent^2 RL-Bench](https://hf.co/papers/2604.10547) | Optimization loops require stable, verifiable tasks. |
| 49 | self-evolving tasks | [Frontier-Eng](https://hf.co/papers/2604.12290) | S-tier claims should include real engineering tasks. |
| 50 | coworker agents | [ClawMark](https://hf.co/papers/2604.23781) | Multi-day state changes make stale context dangerous. |
| 51 | live benchmarks | [Claw-Eval-Live](https://hf.co/papers/2604.28139) | Living benchmarks reduce contamination risk. |
| 52 | repo generation | [RepoZero](https://hf.co/papers/2605.07122) | Scratch repos stress planning context and artifact state. |
| 53 | long horizon | [WildClawBench](https://hf.co/papers/2605.10912) | Real CLI tasks catch harness assumptions. |
| 54 | termination | [AgentStop](https://hf.co/papers/2605.15206) | Early stop is useful only with success-preservation guards. |
| 55 | skills | [SkillGenBench](https://hf.co/papers/2605.18693) | Generated skills must be measured for token overhead and utility. |
| 56 | workflow optimization | [Temporal Semantic Caching](https://hf.co/papers/2605.20630) | Temporal reuse should be modeled separately from static text length. |
| 57 | harness effects | [Harness-Bench](https://hf.co/papers/2605.27922) | Execution-layer configuration changes agent capability and cost. |
| 58 | full lifecycle | [OR-Space](https://hf.co/papers/2605.28158) | Persistent workspaces need lifecycle-aware memory. |
| 59 | benchmark generation | [Benchmark Everything Everywhere All at Once](https://hf.co/papers/2606.06462) | Generated benchmarks require consistency checks. |
| 60 | repo exploration | [FastContext](https://hf.co/papers/2606.14066) | Separate exploration context from solving context. |
| 61 | cache-aware context | [TokenPilot](https://hf.co/papers/2606.17016) | Prefix stability and lifecycle eviction are core cost levers. |
| 62 | prompt engineering | [Large Language Models Are Human-Level Prompt Engineers](https://hf.co/papers/2211.01910) | Prompt optimization can be automated but needs held-out validation. |
| 63 | prompt patterns | [Prompt Pattern Catalog](https://hf.co/papers/2302.11382) | Prompt structure should be reusable and compact. |
| 64 | gist tokens | [Learning to Compress Prompts with Gist Tokens](https://hf.co/papers/2304.08467) | Recurrent instructions can become compact anchors. |
| 65 | compressed models | [Compress, Then Prompt](https://hf.co/papers/2305.11186) | Accuracy-efficiency tradeoffs must be jointly measured. |
| 66 | context adapters | [Adapting Language Models to Compress Contexts](https://hf.co/papers/2305.14788) | Recursive compression inspires hierarchical summaries. |
| 67 | translation memory | [Translation Memories for LLM Translators](https://hf.co/papers/2305.17367) | Memory retrieval must avoid resending full prior context. |
| 68 | prompt space | [Prompt Space](https://hf.co/papers/2306.03799) | Embedding-space prompt choice suggests policy features. |
| 69 | watermarking | [PromptCARE](https://hf.co/papers/2308.02816) | Receipts/hashes make compressed context auditable. |
| 70 | agent eval | [AgentBench](https://hf.co/papers/2308.03688) | Evaluate across environments, not just coding. |
| 71 | RL compression | [Discrete Prompt Compression with RL](https://hf.co/papers/2308.08758) | Compression can be learned from reward, but v1 should stay offline. |
| 72 | dynamic prompting | [Dynamic Prompting for Compressed LLMs](https://hf.co/papers/2310.00867) | Runtime prompt selection can recover capability after compression. |
| 73 | memory architecture | [L2MAC](https://hf.co/papers/2310.02003) | State externalization reduces long-context pressure. |
| 74 | agent network | [Dynamic LLM-Agent Network](https://hf.co/papers/2310.02170) | Agent collaboration topology changes context needs. |
| 75 | prompt compression | [LLMLingua](https://hf.co/papers/2310.05736) | Budget controllers are useful, but code safety needs locked anchors. |
| 76 | real repo benchmark | [SWE-bench](https://hf.co/papers/2310.06770) | Real GitHub issues are the main rollout target. |
| 77 | long prompt compression | [LongLLMLingua](https://hf.co/papers/2310.06839) | Position and key information placement matter. |
| 78 | KV loading | [CacheGen](https://hf.co/papers/2310.07240) | Avoid retransmitting or recomputing stable long context. |
| 79 | planning | [Tree-Planner](https://hf.co/papers/2310.08582) | Plans should be cached and compared against outcomes. |
| 80 | prompt survey | [Prompt Engineering Review](https://hf.co/papers/2310.14735) | Prompt rituals can add cost; measure each one. |
| 81 | RAG compression | [TCRA-LLM](https://hf.co/papers/2310.15556) | Retrieved text should be compressed by task relevance. |
| 82 | prompt cache | [Prompt Cache](https://hf.co/papers/2311.04934) | Stable prompt modules should remain byte-identical. |
| 83 | prompt engineer | [Prompt Engineering a Prompt Engineer](https://hf.co/papers/2311.05661) | Meta-prompting needs regression tests. |
| 84 | multi-agent cognition | [MAgIC](https://hf.co/papers/2311.08562) | Collaboration adds rationality and adaptability failure modes. |
| 85 | demonstrations | [AlignedCoT](https://hf.co/papers/2311.13538) | Demonstrations may be useful but expensive; route only when needed. |
| 86 | efficiency survey | [Efficiency Spectrum of LLMs](https://hf.co/papers/2312.00678) | Token, model, and system efficiency are separate axes. |
| 87 | semantic compression | [Extending Context Window via Semantic Compression](https://hf.co/papers/2312.09571) | Semantic compression must expose reconstruction risk. |
| 88 | believable agents | [Believable AI Agents](https://hf.co/papers/2312.17115) | Agent behavior metrics go beyond final answer correctness. |
| 89 | self-planning | [AUTOACT](https://hf.co/papers/2401.05268) | Agents can learn workflows, but budget should constrain exploration. |
| 90 | memory attention | [LoMA](https://hf.co/papers/2401.09486) | Memory compression should aim for lossless anchors where possible. |
| 91 | safety | [R-Judge](https://hf.co/papers/2401.10019) | Safety-critical context is never lossy-compressed in v1. |
| 92 | goal prompting | [Goal-Oriented Prompt Engineering Survey](https://hf.co/papers/2401.14043) | Goal alignment is a utility signal. |
| 93 | social prompting | [Wordflow](https://hf.co/papers/2401.14447) | Prompt collaboration can bloat context unless scoped. |
| 94 | code actions | [Executable Code Actions](https://hf.co/papers/2402.01030) | Executable state can replace verbose natural language memory. |
| 95 | KV quantization | [KIVI](https://hf.co/papers/2402.02750) | KV bytes belong in cost models. |
| 96 | calibration | [Intent-Based Prompt Calibration](https://hf.co/papers/2402.03099) | Boundary cases should test compression safety. |
| 97 | sublinear generation | [SubGen](https://hf.co/papers/2402.06082) | Long-context generation cost is reducible with selection/sampling. |
| 98 | prompt survey | [Systematic Survey of Prompt Engineering](https://hf.co/papers/2402.07927) | Taxonomies help classify prompt overhead. |
| 99 | gist memory | [ReadAgent](https://hf.co/papers/2402.09727) | Long context can be replaced with episodic gist memory. |
| 100 | agent graphs | [Language Agents as Optimizable Graphs](https://hf.co/papers/2402.16823) | Treat agent traces as optimizable graphs, not flat transcripts. |
| 101 | KV precision | [No Token Left Behind](https://hf.co/papers/2402.18096) | Rare high-value tokens need special protection. |
| 102 | NL compression | [Learning to Compress Prompt in Natural Language Formats](https://hf.co/papers/2402.18700) | Natural-language compression can be portable but risky. |
| 103 | code-as-policy | [RL-GPT](https://hf.co/papers/2402.19299) | Programmatic state can compress repeated action logic. |
| 104 | KV recipe | [GEAR](https://hf.co/papers/2403.05527) | Near-lossless KV compression motivates conservative loss budgets. |
| 105 | agent data | [Agent-FLAN](https://hf.co/papers/2403.12881) | Agent training data should include efficient trajectories. |
| 106 | faithful compression | [LLMLingua-2](https://hf.co/papers/2403.12968) | Extractive compression is safer than abstractive summaries for code. |
| 107 | GitHub issues | [MAGIS](https://hf.co/papers/2403.17927) | GitHub issue agents need multi-step repo navigation. |
| 108 | efficient prompting | [Efficient Prompting Methods Survey](https://hf.co/papers/2404.01077) | Prompt efficiency deserves its own benchmark axis. |
| 109 | layer budgets | [SqueezeAttention](https://hf.co/papers/2404.04793) | Budget allocation can vary by abstraction level. |
| 110 | soft prompt compression | [SoftPromptComp](https://hf.co/papers/2404.04997) | Learned soft prompts are future work; v1 stays text/audit based. |
| 111 | memory survey | [Memory Mechanism Survey](https://hf.co/papers/2404.13501) | Memory write, storage, retrieval, and evaluation are separate decisions. |
| 112 | KV runahead | [KV-Runahead](https://hf.co/papers/2405.05329) | Prefill/decode scheduling affects cost beyond token count. |
| 113 | salient KV | [ZipCache](https://hf.co/papers/2405.14256) | Salient token detection informs identifier protection. |
| 114 | depth compression | [MiniCache](https://hf.co/papers/2405.14366) | Cache compression can happen by layer/depth, not only text. |
| 115 | agent-computer interface | [SWE-agent](https://hf.co/papers/2405.15793) | Harness design changes both success and token use. |
| 116 | frozen-model compression | [SelfCP](https://hf.co/papers/2405.17052) | Frozen compressors motivate no-finetune v1 methods. |
| 117 | serving | [Parrot](https://hf.co/papers/2405.19888) | Semantic variables can externalize repeated context. |
| 118 | repo understanding | [RepoUnderstander](https://hf.co/papers/2406.01422) | Repo graph context should be recalled by relevance. |
| 119 | dataset agents | [DCA-Bench](https://hf.co/papers/2406.07275) | Dataset curation traces stress repetitive inspection. |
| 120 | token importance | [Value Also Matters](https://hf.co/papers/2406.12335) | Attention alone is insufficient; value/content signals matter. |

## Product Decision From The Ledger

TRICE v1 should stay deterministic and measurable: no finetuning, no hidden
LLM judge in the default path, no 60% claim without reruns. The first product
step is a context policy and replay layer that makes every compression decision
visible, reversible, and tied to evidence recall.
