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

## 2026-06-21 Readiness-Gate Refresh

The newest product layer is the no-execution readiness gate. Its purpose is to
stop TraceRazor from spending tokens on a suite that cannot support the claim it
is about to make. This came from four recurring research findings:

| Theme | Sources | TRICE product rule |
|---|---|---|
| Real tasks beat transcript-only evidence | [SWE-bench](https://arxiv.org/abs/2310.06770), [SWE-agent](https://arxiv.org/abs/2405.15793), [SWT-Bench](https://arxiv.org/abs/2406.12952) | Claim suites must use executable repositories with objective verifier commands. |
| Agent benchmarks need repeated, cost-aware trials | [AI Agents That Matter](https://arxiv.org/abs/2407.01502), [tau-bench](https://arxiv.org/abs/2406.12045), [AgentBench](https://arxiv.org/abs/2308.03688) | Preflight must estimate planned runs and verifier invocations before execution. |
| Tool and web agents expose environment drift | [ToolLLM](https://arxiv.org/abs/2307.16789), [WebArena](https://arxiv.org/abs/2307.13854), [Voyager](https://arxiv.org/abs/2305.16291) | Held-out suites must pin remote Git commits and fixed adapter profiles. |
| Compression is only useful when decisions survive | [LLMLingua](https://arxiv.org/abs/2310.05736), [LongLLMLingua](https://arxiv.org/abs/2310.06839), [Prompt Cache](https://arxiv.org/abs/2311.04934), [PagedAttention](https://arxiv.org/abs/2309.06180) | Readiness is not a savings claim; savings come only from live suite results plus verified claim cards. |

The readiness gate therefore emits `trice-suite-readiness/v1` with
`smoke_ready`, `pilot_ready`, or `claim_ready`. A result may be claim-ready to
run, but it is not S-tier evidence until the live suite, evidence bundle, and
claim card all verify.

The artifact-review layer emits `trice-artifact-card/v1`. It is inspired by
artifact-review norms: an evidence package should be available, functional,
reusable, and honest about what it does not prove. The card verifies README,
paper source/PDF, paper manifest, evidence bundle, readiness card, protocol
lock, claim card, library contract, and public schemas together so a reviewer
can reproduce the trust boundary from one command.

The protocol-lock layer emits `trice-protocol-lock/v1`. It is a deterministic
pre-outcome contract, not another result card. It freezes the primary metric,
pass-preservation guardrail, clustered confidence rule, held-out source rule,
adapter-profile rule, receipt-validation rule, claim-card rule, and artifact-card
rule before a live run can be interpreted. The research reason is simple:
agent benchmarks are sensitive to harness choices, environment drift, and
post-hoc metric selection. A protocol lock makes those choices hash-bound to the
suite manifest before any claim run is executed.

The design-card layer emits `trice-design-card/v1`. It is a statistical-design
review over the locked protocol and observed suite results. It estimates
task-cluster variance, projects the claim-run lower confidence bound, and still
refuses claim readiness when external-validity requirements are missing. This
separates three questions that are often blurred in agent reports: did the
current run show a signal, would the planned sample size plausibly clear the
target, and is the protocol externally valid enough to make that claim?

The research-card layer emits `trice-research-card/v1`. It turns this ledger
from supporting prose into a deterministic product input: rows are parsed,
categorized, hash-bound, checked for source coverage, and rendered as JSON,
Markdown, SVG, and LaTeX. This is not an outcome claim. It proves that the
research basis is broad enough, reviewable enough, and stable enough to bind
the README, paper, release evidence, and integrity proof graph.

## 100+ Paper And Source Working Ledger

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
| 121 | supply-chain provenance | [SLSA specification](https://slsa.dev/spec/) | Release trust should bind artifact provenance, not only local tests. |
| 122 | project health checks | [OpenSSF Scorecard](https://scorecard.dev/) | Public trust can be represented as automated checks with explicit failures. |
| 123 | package credentials | [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) | OIDC publishing reduces long-lived token risk and belongs in release gates. |
| 124 | package attestations | [PyPI Digital Attestations](https://docs.pypi.org/attestations/) | Index-hosted attestations make Python release artifacts more verifiable. |
| 125 | signing/provenance | [Sigstore documentation](https://docs.sigstore.dev/) | Keyless signing and transparency logs motivate public release receipts. |
| 126 | SBOM contract | [CycloneDX specification](https://cyclonedx.org/specification/overview/) | SBOMs and checksums should be tracked as release-card requirements. |
| 127 | artifact badging | [ACM Artifact Review and Badging](https://www.acm.org/publications/policies/artifact-review-badging) | Remote smoke evidence should be available, functional, and reusable, but claims must stay scoped. |
| 128 | benchmark submissions | [MLCommons Inference Policies](https://docs.mlcommons.org/inference/policies/) | Public benchmark evidence needs locked rules, fixed inputs, and reproducible submitted artifacts. |
| 129 | packaging sample repo | [PyPA sampleproject](https://github.com/pypa/sampleproject) | A tiny public Python repo is useful for exercising remote Git clone, revision lock, source-only patch, and verifier evidence paths. |
| 130 | reproducible smoke | [TRICE remote smoke suite](../examples/trice_remote_smoke_suite.json) | Small remote-git smoke should prove infrastructure, not upgrade the product to S-tier. |
| 131 | positional evidence loss | [Lost in the Middle](https://arxiv.org/abs/2307.03172) | Context control must prove that early and middle evidence still survives compression, not only that the prompt is shorter. |
| 132 | retrieval context recall | [RAGAS](https://arxiv.org/abs/2309.15217) | Recall-style evaluation motivates TRICE's deterministic evidence-recall floor, but TRICE binds it to receipts and live verifier outcomes instead of LLM-judge scoring. |
| 133 | SemVer public API | [Semantic Versioning 2.0.0](https://semver.org/) | Compatibility claims are not meaningful until the public API is declared. |
| 134 | Python package versioning | [Python version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/) | Public registry versions become downstream dependency facts, so release cards should bind local package version and public surface. |
| 135 | JSON contract schemas | [JSON Schema 2020-12](https://json-schema.org/draft/2020-12) | Public receipts, cards, bundles, and manifests should have machine-checkable schemas. |
| 136 | reproducible release inputs | [Reproducible Builds](https://reproducible-builds.org/) | Examples, schemas, and docs should be hashed inputs rather than mutable prose around a binary. |
| 137 | attestation envelope | [in-toto attestations](https://in-toto.io/) | Release evidence should use a portable statement envelope with named subjects and predicate type. |
| 138 | local release verification | [TRICE release evidence packet](trice_release_evidence.md) | Platform wheels, standalone binaries, legal-boundary files, SBOMs, checksums, provenance, proof cards, paper, and evidence bundles should verify together before publication. |
| 139 | GitHub release attestations | [GitHub Artifact Attestations](https://docs.github.com/actions/security-guides/using-artifact-attestations-to-establish-provenance-for-builds) | GitHub-hosted provenance should cover release assets generated by Actions, not only local files committed to the repo. |
| 140 | Scorecard workflow | [OpenSSF Scorecard Action](https://github.com/ossf/scorecard-action) | Public project health should have an independently hosted signal that can be checked by doctor. |
| 141 | proof graph integrity | [SLSA Verification Summary Attestation](https://slsa.dev/spec/v1.0/verification_summary) | Provenance and evidence are only useful when a top-level verifier summarizes which predicates and subjects were actually checked. |
| 142 | Cargo staged publishing | [Cargo publishing reference](https://doc.rust-lang.org/cargo/reference/publishing.html) | Rust distribution trust needs a staged publish plan because dependent workspace crates can only verify against crates.io after upstream crates are indexed. |
| 143 | packaging | [Python Packaging User Guide](https://packaging.python.org/tutorials/packaging-projects/) | Build artifacts must be followed by clean install/import/console verification. |
| 144 | packaging | [pip install Reference](https://pip.pypa.io/en/stable/cli/pip_install/) | Installability should be measured with the same command path users run. |
| 145 | packaging | [Wheel Format PEP 427](https://peps.python.org/pep-0427/) | Wheels are binary distribution contracts; packaged data and scripts must be verified after install. |
| 146 | benchmark methodology | [HELM](https://arxiv.org/abs/2211.09110) | Evaluation should separate scenarios, metrics, models, and transparency artifacts. |
| 147 | code generation | [HumanEval](https://arxiv.org/abs/2107.03374) | Executable tests make coding-agent claims more objective than text-only grading. |
| 148 | code generation | [MBPP](https://arxiv.org/abs/2108.07732) | Small programming tasks are useful smoke tests but cannot substitute for repo-level repair. |
| 149 | live code benchmark | [LiveCodeBench](https://arxiv.org/abs/2403.07974) | Time-split evaluation reduces contamination risk and motivates held-out remote tasks. |
| 150 | real-repo benchmark | [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) | Human-validated subsets improve trust in software-repair outcome labels. |
| 151 | machine-learning agents | [MLE-bench](https://arxiv.org/abs/2410.07095) | Agent benchmarks should include realistic setup, execution, and scoring pipelines. |
| 152 | general assistants | [GAIA](https://arxiv.org/abs/2311.12983) | Tool-using agents need verifiable answers and multi-step evidence, not just final text. |
| 153 | multi-agent systems | [AutoGen](https://arxiv.org/abs/2308.08155) | Multi-agent workflows need explicit message routing and termination controls to avoid context growth. |
| 154 | multi-agent software | [MetaGPT](https://arxiv.org/abs/2308.00352) | Role decomposition can help software tasks, but every role adds context and coordination cost. |
| 155 | prompt programming | [DSPy](https://arxiv.org/abs/2310.03714) | Optimized prompt/program pipelines need held-out validation and reproducible compilation inputs. |
| 156 | tool agents | [API-Bank](https://arxiv.org/abs/2304.08244) | Tool calls should be evaluated with execution state and argument correctness. |
| 157 | tool learning | [ToolBench](https://arxiv.org/abs/2307.16789) | Tool-use traces stress retrieval, API selection, and compact tool-schema presentation. |
| 158 | long-context evaluation | [LongBench](https://arxiv.org/abs/2308.14508) | Long-context capability must be measured by task family instead of assumed from context window size. |
| 159 | evaluation reproducibility | [Towards Reproducible LLM Evaluation](https://arxiv.org/abs/2410.03492) | Evaluation claims need frozen prompts, decoding settings, data versions, and reporting metadata. |
| 160 | statistical rigor | [Paired Bootstrap Protocols for LLM Evaluation](https://arxiv.org/abs/2511.19794) | Small improvements should be judged with paired or clustered uncertainty, not raw means alone. |
| 161 | artifact review | [ACM Reproducibility Badging](https://www.acm.org/publications/policies/artifact-review-and-badging-current) | Paper artifacts should make availability, functionality, reproducibility, and reuse separately inspectable. |
| 162 | secure development | [NIST SSDF](https://csrc.nist.gov/Projects/ssdf) | Public trust needs secure-development controls beyond package build success. |
| 163 | package attestations | [PEP 740](https://peps.python.org/pep-0740/) | Index-hosted attestations turn distribution metadata into a verifiable user-facing signal. |
| 164 | deterministic research basis | [TRICE research card](trice_research_card.md) | The literature base should be hash-bound, categorized, and regenerated with the paper. |
| 165 | proof graph release | [TRICE integrity card](trice_integrity_card.md) | Research, contract, release, installability, evidence, and paper artifacts should verify as one graph. |

## Product Decision From The Ledger

TRICE v1 should stay deterministic and measurable: no finetuning, no hidden
LLM judge in the default path, no 60% claim without reruns. The first product
step is a context policy and replay layer that makes every compression decision
visible, reversible, and tied to evidence recall.

The release layer should be deterministic in a different way: it snapshots the
public doctor result, binds proof cards and package metadata, and refuses public
release readiness until registries, tags, CI, and distribution provenance are
green. This makes release trust inspectable without pretending current public
state is already solved.

The remote-smoke layer adds the first public Git clone evidence shard. It
checks that `tracerazor-trice suite scaffold`, locked revisions, source-only
patch specs, suite verification, claim-card verification, and `.trice.zip`
bundle verification all work on a public repository. It intentionally keeps the
claim level at smoke because a one-repo fixture cannot estimate generalization
or held-out variance.

The evidence-recall layer is now explicit in the product contract. Essential or
locked evidence segments must either remain visible or carry a receipt and
rehydration pointer; claim cards report the minimum recall and failure count.
This turns "compressed but maybe forgot the important bit" into a verifier
visible failure mode.

The public-contract layer makes TraceRazor a more credible library rather than
only a benchmark bundle. `tracerazor-trice contract` binds SemVer, import
exports, CLI commands, schemas, examples, docs, and package metadata into a
single deterministic card. Artifact and release cards now require that contract
to verify before the local trust story is considered review-ready.

The release-evidence layer turns the supply-chain plan into a verifier packet.
`tracerazor-trice release-evidence` binds the wheel, source distribution, Rust
CLI binary, installability card, proof cards, paper artifacts, broad and remote
evidence bundles, SHA-256 checksum sidecar, CycloneDX-style Python and Cargo
SBOM sidecars, and an in-toto/SLSA-shaped provenance sidecar. The packet is
intentionally separate from S-tier outcome evidence: it proves release artifacts
are inspectable, not that the held-out 50 x 3 remote-git suite has passed.

The installability layer follows the Python Packaging User Guide, pip install
semantics, and the wheel format contract: building a distribution is not enough
if packaged data, console scripts, or bundled binaries are missing after
installation. `tracerazor-trice install` therefore creates a clean virtual
environment, installs the built wheel with `pip install --no-deps`, imports the
packaged schema/API surface, runs the installed `tracerazor-trice` console, and
separately records whether the Rust `tracerazor` command can find a bundled
binary. The product rule is honest scope: generic wheels may be Python/TRICE
install-ready while full no-Rust-toolchain CLI claims require platform-wheel
install cards.

The hosted-provenance layer is the next public-trust step. The release workflow
now generates the release-evidence sidecars during the GitHub release run and
uses GitHub artifact attestations over wheels, sdists, binaries, and the
release-evidence files. A separate OpenSSF Scorecard workflow publishes a
security-health signal; `tracerazor-trice doctor` treats a missing Scorecard or
score below 7.0 as a public-release blocker. This keeps local proof, registry
attestations, hosted release attestations, and project-health signals separate
enough to fail independently.

The crates-publish layer turns the remaining Rust install gap into a
deterministic preflight instead of a README promise. `tracerazor-trice crates`
binds the Cargo workspace manifests, local dependency DAG, crates.io status
snapshot, and README cargo-install honesty. It can report
`publish_plan_locked` while still refusing `cargo_install_claim_allowed`, which
is the correct state until the final `tracerazor` CLI crate is visible on
crates.io and a clean `cargo install tracerazor` succeeds.

The integrity layer binds the proof graph itself. `tracerazor-trice integrity`
checks offline doctor output, contract, artifact, reproduction, release,
release-evidence, crates publish card, installability card, research card,
paper-manifest, schema, and workflow-hook verifiers from one card. The product
reason is that a repo can have many individually impressive artifacts and still
be untrustworthy if they drift apart. Integrity cards make that drift a single
deterministic CI failure.

The research-card layer makes the literature base part of that same graph.
`tracerazor-trice research` parses this ledger, enforces minimum source and
category coverage, binds every row by SHA-256, and renders a README/PDF-ready
coverage visualization. This gives the product a paper-worthy source boundary:
reviewers can inspect the sources and re-run the card, while the card still
refuses to act as a 60% savings or S-tier outcome claim.
