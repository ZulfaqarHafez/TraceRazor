# TraceRazor v2 — Production-Grade Improvement Plan

> Output of a 5-department / 10-track planning exercise. Each department owns one
> gap surfaced by the v1 evaluation and one half of the v2 product thesis:
> **a closed-loop efficiency compiler that diagnoses token waste, alters the
> agent to remove it, verifies the change preserves task success, and is guided
> by a teacher/orchestrator that learns what works.**
>
> A runnable prototype of the centerpiece (Teacher + closed-loop remediation +
> quality gate) ships in [`teacher/`](../teacher) with an offline demo at
> [`examples/demo_teacher_offline.py`](../examples/demo_teacher_offline.py).

---

## Why v2 (the v1 diagnosis in one paragraph)

The v1 audit (PCA over 24 real traces) found the 13 metrics carry only **~5
independent signals** (participation ratio 4.99): a collinear efficiency core,
a mirror pair (SRR↔ISR = −0.73), triple-counted verbosity (VDI/SHL/CCR + the
AVS re-sum), and a near-dead tail (VDI CV 0.03, SHL passes 23/24). The TAS score
is an **uncalibrated ordinal** with Lighthouse-borrowed bands; the "30–60%
waste" headline is **circular** (computed from the same flags it claims to
validate); the auditor is **O(n²)** (693 ms @ 1000 steps vs a "sub-5 ms" claim);
and the fix patches are **static strings, never re-run or verified**. The
engineering and honesty culture are strong — v2 keeps both and fixes the rest.

---

## Department map

| Dept | Owns | Headline deliverable |
|---|---|---|
| **P1 — Metric Science & Calibration** | the entropy problem | 13→6 orthogonal signals + calibrated cardinal score |
| **P2 — Closed-Loop Remediation Engine** | *altering the agent* | typed/reversible interventions + quality-preservation gate |
| **P3 — Teacher / Orchestrator** | the *guiding teacher* | perceive→diagnose→plan→act→verify→remember loop + Playbook |
| **P4 — Platform & Hardening** | production infra | kill O(n²), PyO3, deny-by-default CORS, supply-chain gates |
| **P5 — Validation, Trust & GTM** | honest proof + adoption | reproducible bench + "no number without a regenerator" |

---

## P1 — Metric Science & Calibration

**Collapse 13 metrics → 6 near-orthogonal signals**, each owning one empirical
latent and computed by a single canonical operator so it cannot drift back into
collinearity:

| Survivor | Latent | Absorbs |
|---|---|---|
| **RDX** Redundancy | "repeats covered ground" | SRR, ISR, LDI, CCE |
| **TEF** Tool Efficiency | "calls that don't land" | TCA, DBO |
| **DPF** Depth Fit | "more steps than needed" | RDA, (TUR) |
| **COH** Coherence | "wanders off goal" | GAR, CSD |
| **PRS** Presentation | "linguistic filler" | VDI, SHL, CCR + AVS |
| **TVS** Task Value | quality gate | task_value_score |

- **Uniform waste-direction sign** (every signal ∈ [0,1], higher = worse) — fixes
  the mixed-sign confusion that made correlations hard to reason about.
- **TUR is deleted** as a metric (it's a flag-aggregate of the others) and reborn
  as the *prediction target*.
- **Redundancy guardrail** (CI test): reject any new metric with CV < 0.10, or
  |Spearman r| > 0.80 vs a survivor, or < 0.02 incremental R² — this would have
  caught TUR, VDI, and the AVS re-sum.
- **Calibrated composite**: replace the 1.20-weight ordinal sum + Lighthouse
  bands with an NNLS + isotonic scorer fit to a labelled corpus, emitting a
  **cardinal `removable_token_fraction` with a bootstrap 90% CI** and per-signal
  attribution. Bands become empirical percentiles.

## P2 — Closed-Loop Remediation Engine  *(prototyped — see `teacher/`)*

Replaces static string patches with **typed, idempotent, reversible
`Intervention` objects** across five targets (`system_prompt`, `tool_def`,
`runtime_policy`, `memory_policy`, `decoding`), each with an
`apply(AgentConfig) → AgentConfig` contract.

- **Taxonomy**: every waste pattern maps to a structured edit *and* (where
  possible) a runtime-policy nudge driven by the existing `tracerazor-proxy`
  layers — not just prose.
- **Guardrailed LLM synthesis**: prompt rewrites must pass invariant-preservation
  + semantic-equivalence + judge + diff-budget checks before they can exist.
- **The Quality-Preservation Gate** (the missing piece): *accept iff token
  savings are significant **and** task success is non-inferior within margin δ.*
  A token win that breaks the agent is rolled back. This makes savings
  **falsifiable, not projected** (v1 only `simulate()`d).
- Shadow / A-B / offline-replay runners; emits a `VerifiedInterventionBundle`
  with evidence to the Teacher.

> **Prototype status:** `teacher/interventions.py` (taxonomy + apply),
> `teacher/gate.py` (the non-inferiority gate), `teacher/runner.py` (offline
> re-run harness). The demo shows ~49% token reduction with 100% success
> preserved and the gate rejecting an over-aggressive step cap (47% token win,
> 0% success).

## P3 — Teacher / Orchestrator  *(prototyped — see `teacher/teacher.py`)*

An agent that sits above the target agent and runs a curriculum:

```
perceive ─▶ diagnose ─▶ plan (curriculum) ─▶ act ─▶ verify (gate) ─▶ remember
   ▲                                                                      │
   └──────────────── loop until TAS plateaus / budget ────────────────────┘
```

- **Curriculum**: try interventions cheapest/safest first —
  `INLINE → PROMPT → TOOL → STRUCT` — escalating only on plateau. A contextual
  bandit (LinUCB in the full design; playbook-primed scoring in the prototype)
  learns which fix pays off for which pattern.
- **Three modes**: **Autonomous** (auto-patch within the gate), **Coach**
  (human-readable report + PR-style diff, promotes nothing), **Curriculum**
  (re-run a task set until TAS plateaus).
- **Playbook memory** keyed on `WastePattern.signature` (not agent id) so a
  lesson learned on one agent is ranked first on the next — a cross-agent data
  flywheel.
- **Inline runtime**: tier-0 actions are exactly the proxy crate's
  verbosity/budget/scope/semantic-guard layers; shadow-mode-first with gradual
  rollout + kill-switch.

## P4 — Platform & Hardening

- **Kill the O(n²)**: a shared windowed-similarity primitive (bound ISR like SRR
  already is) + a **MinHash-LSH** near-duplicate index → O(n·K); `rayon` the
  per-agent re-`analyse`. Target < 15 ms @ 1000 steps, enforced by a **criterion
  CI perf gate** so the README curve can't silently regress.
- **PyO3 bindings** (maturin/abi3) so Python calls the Rust core natively instead
  of shelling to the CLI.
- **Security**: deny-by-default CORS (opt-in `TRACERAZOR_DEV_INSECURE`),
  API-key auth, rate limiting + body cap, `cargo audit` / `pip-audit` /
  `cargo deny` + SBOM CI gates.
- **Pluggable store** (SQLite → Postgres) for multi-tenant horizontal scale.
- **One version source of truth** (`workspace.package.version`); classifier
  `Production/Stable` → `Beta`; an **alpha → beta → GA** readiness ladder.

## P5 — Validation, Trust & GTM

- **The claim that matters**: "after remediation the agent uses materially fewer
  tokens **and** task success does not regress" — a paired (token-Δ, success-Δ)
  non-inferiority test on real τ-bench / SWE-agent trajectories with CIs.
- **"No number without a regenerator"**: every published figure has a
  `make`-target regenerator and a **CI equality-check** (committed ≠ recomputed →
  build fails). This permanently fixes the fabricated README sample (real run is
  TAS 76 / 11 steps / 14,280 tok, not 64 / 9 / 18,420), the stale `RESULTS.md`,
  and restates "sub-5 ms" as an honest length-vs-latency curve.
- **Break the circularity**: an independent labelled corpus (human κ ≥ 0.67 +
  strong-LLM judge) marking removable tokens, never shown the heuristic flags,
  validates detection precision/recall — replacing the circular "30–60%".
- **Trust contract** for auto-altering agents: append-only audit log, Coach vs
  Auto modes, guaranteed rollback, a quality-gate SLA, savings reported as
  **verified, not estimated**. GTM ladder: OSS core → CI efficiency-gate →
  hosted teacher, priced on verified savings.

---

## Sequencing (≈10 weeks)

1. **Honesty & safety (wk 1–2):** version collapse, CORS deny-by-default,
   regenerate README sample / RESULTS.md + CI equality-check, ISR windowing +
   perf gate. *Ship 0.2.0-beta.*
2. **Closed loop offline (wk 3–5):** harden the prototype here into the product
   engine — typed interventions in Rust serde, `tau2_evaluator` quality gate,
   labelled-corpus v0, MinHash-LSH, supply-chain gates.
3. **Calibration & SDK (wk 6–8):** fit the cardinal scorer, PyO3 wheels,
   `/api/v1` + OpenAPI, streaming auditor, teacher Coach mode GA.
4. **Multi-tenant GA (wk 9–10):** Postgres store, shadow/A-B online remediation,
   signed releases, GA sign-off → 1.0.0.

---

## Try the prototype

```bash
python examples/demo_teacher_offline.py     # closed-loop remediation + quality gate (mock agent)
python examples/demo_langgraph_coach.py     # LangGraph ingest -> COACH recommendations
python examples/demo_online_verification.py # REAL online verification (HTTP agent + stat gate)
python teacher/tests/run.py                 # 21 tests (no pytest needed)
```

Everything runs offline with no API keys.

### Real online verification (Layer 3)

`teacher.online` runs an **actual tool-calling agent over HTTP** against an
OpenAI-compatible endpoint, measures **real token usage** (from the API `usage`
block) and real task success, applies a candidate config, **re-runs online**,
and gates on a **statistical non-inferiority test** (`teacher.stats.StatGate`:
paired bootstrap CI on token savings + a one-sided non-inferiority bound on the
success proportion). It promotes only changes that *provably* cut tokens
without regressing success.

- **Runtime interventions enforced live:** `loop_breaker` suppresses a repeated
  identical tool call; `step_cap` hard-stops the loop. Prompt-section
  interventions ride in the real system prompt, so a real model emits fewer
  tokens.
- **Drop-in for any provider:** set `TRACERAZOR_LLM_BASE_URL` +
  `TRACERAZOR_LLM_API_KEY` (+ `TRACERAZOR_LLM_MODEL`) and run with `--live`
  against OpenAI / Azure / Groq / vLLM / Ollama / LiteLLM — the agent loop,
  token accounting, and gate are identical.
- **Verified here without a key** against a bundled stdlib OpenAI-compatible
  server (`teacher/_mockserver.py`) whose token output genuinely responds to the
  installed interventions. Sample run: 3 interventions verified-and-accepted
  (loop_breaker, EFFICIENCY_RULES, STEP_BUDGET), unsupported ones rejected as
  no-gain, and an aggressive `step_cap` rejected for breaking success
  (100%→0%) — **49.8% net verified token reduction, success preserved**.
- **Wired into the curriculum:** `Teacher.improve` takes a pluggable `runner`.
  Pass `teacher.online.OnlineRunner(agent, holdout, ...)` (with
  `gate=StatGate()`) to drive the whole perceive→diagnose→plan→verify→remember
  loop **live**, re-diagnosing from real traces each round; omit it to use the
  deterministic `OfflineRunner` (default). Same Teacher, same Playbook, either
  backend.

```python
from teacher import Teacher, Mode, StatGate, AgentConfig
from teacher.online import OnlineRunner, OnlineAgent, LLMClient
agent  = OnlineAgent(LLMClient.from_env(), tools)
runner = OnlineRunner(agent, holdout, diagnoser, repeats=3)
result = Teacher(AgentConfig(), mode=Mode.CURRICULUM, gate=StatGate()).improve(runner=runner)
```

> Note: `api.openai.com` / `api.anthropic.com` are network-reachable from this
> environment, but no credential is present, so the **live** path is exercised
> against the local server; it targets a real provider unchanged once a key is
> set.

### Diagnostic backends (Layer 1)

`teacher.Diagnoser` selects a backend automatically and parses the auditor's
**full** report — all 13 metric blobs (with `pass`/`target` and detail like
`srr.redundant_steps`, `tca.misfires`, `cce.bloated_steps`), the step-level
`diff`, `savings`, and the ready-made `fixes[]` — into `WastePattern`s with real
severity + step-id + token attribution, and maps the auditor's own fixes into
applicable interventions.

| Backend | When | How |
|---|---|---|
| **native** | `import tracerazor_native` succeeds | PyO3 binding (`crates/tracerazor-py`), in-process — no subprocess |
| **subprocess** | `tracerazor` binary present | shells `tracerazor audit --format json` |
| **builtin** | neither | transparent pure-Python heuristic (CI/offline) |

> The PyO3 crate (`crates/tracerazor-py`) is **excluded from the Cargo
> workspace** so `cargo build --workspace`/CI never depend on `pyo3`. It was
> authored against the real core API but is **not compiled in this sandbox** (no
> crates.io access); build it where crates.io is reachable with
> `maturin develop -m crates/tracerazor-py/Cargo.toml`. The subprocess backend
> is the verified default until then.

### Framework adapters (Layer 2)

`teacher.LangGraphAdapter` ingests real LangGraph/LangChain runs into
auditor-schema traces — via a dependency-free `RunRecorder` (works with no
langchain installed) or `from_tracerazor_callback(cb)` to reuse the official
`tracerazor` LangGraph integration. Because captured live traces can't be
re-run offline, `Teacher.coach(traces)` ranks interventions by the auditor's
own predicted savings + playbook priors and emits a proposed config diff for
human approval — promoting nothing.
