# The Measured Case Study — live agent runs

**Status: MEASURED.** On 2026-06-10 the full protocol ran end-to-end against
a live agent — real LLM calls, real tool use, real pass/fail checks — for a
total API spend of about **$1.30**. This page reports what was measured,
including the round that went *against* the product, because a case study
that can't lose isn't evidence.

## The question

The audit's "savings estimate" is a sum of per-fix heuristic projections. A
projection is not evidence. The only number that matters to a buyer is:

> tokens before vs tokens after applying TraceRazor's fixes, **at unchanged
> task pass rate**, with confidence intervals, on real tasks.

Jointly reporting cost and task outcome — never one without the other — is
the methodology argued for by Kapoor et al., *AI Agents That Matter*
([arXiv:2407.01502](https://arxiv.org/abs/2407.01502)). The harness
enforces it: any task whose pass flag flips between conditions is called
out as **FLIPPED, not a saving**, and the run exits non-zero.

## Setup

> **Scorer version note.** The TAS columns below were measured under the
> v0.4 fourteen-metric composite. v0.5 demoted five metrics to diagnostics
> (see [`docs/metric_effectiveness.md`](metric_effectiveness.md)), which
> shifts absolute TAS values; the token numbers, pass rates and CIs — the
> study's actual claims — are scorer-independent. Re-running
> `python -m benchmark.case_study --pairs-dir benchmark/live/traces`
> against the current binary reproduces the token results exactly.

| | |
|---|---|
| Subject agent | Claude Code 2.1.170, headless (`claude -p`), model `claude-haiku-4-5` |
| Tasks | 6 small Python engineering tasks × 2 replicates = 12 pairs ([`benchmark/live/tasks/`](../benchmark/live/tasks/)) |
| Outcome check | `python3 -m pytest -q` in the sandbox after the agent finishes (objective, binary) |
| Tool envelope | pytest-only Bash + Read/Edit/Write/Glob/Grep, `--max-turns 25` — identical across conditions |
| Trace capture | session transcript → [`benchmark/convert_claude_code.py`](../benchmark/convert_claude_code.py) |
| Measurement | `python -m benchmark.case_study`: `tracerazor bench` per pair, seeded 10,000-resample bootstrap 95% CIs |

The tasks are the kind of work coding agents actually do — fix an
off-by-one, implement to a failing spec, rename an API across files,
deduplicate drifted helpers, repair package imports. Every task starts
RED (failing tests) and the agent must end GREEN. Replicates capture
run-to-run variance, which for coding agents is substantial (Bai et al.,
[arXiv:2604.22750](https://arxiv.org/abs/2604.22750)); τ-bench's pass^k
makes the same point for tool agents
([arXiv:2406.12045](https://arxiv.org/abs/2406.12045)).

**Token accounting is marginal**: per turn,
`input + cache_creation + output` — the new tokens the turn introduced —
with two exclusions recorded in every trace's metadata: cache *reads*
(the re-fed prefix) and the first turn's prefix encoding. The latter is a
~20k-token infrastructure cost that appears as a cache *write* or a cache
*read* depending on cache warmth at launch — we observed ±22k-token swings
on otherwise-identical runs — and it is constant across conditions by
construction, so counting it would only add noise.

### Protocol, per pair

1. **Before:** run the agent on the task, stock configuration. Convert the
   transcript; record the pytest outcome in `task_value_score`.
2. **Audit and patch:** `tracerazor audit before.json --format json
   --hermetic`, then `tracerazor apply report.json --to prompt.txt`
   (safe patches only — the product's default). The applied subset is
   written to `<pair>.fixes.json` so estimate accuracy is scored against
   the fixes that actually ran.
3. **After:** re-run the *same task, same model, same tool envelope*, with
   the patched text passed via `--append-system-prompt`. The patch is the
   only delta.
4. **Measure:** `python -m benchmark.case_study --pairs-dir …`.

Orchestration lives in [`benchmark/live/`](../benchmark/live/)
(`run_live.py`, `audit_and_apply.py`, `reconvert.py`) and is fully
reproducible from this repository plus a Claude Code install.

## What the audits found (before traces)

TAS ranged 80.7–85.7 (mean 83.5) across the 12 stock runs. Recurring
findings: tool misfires in 6 of 12 audits (some genuine wrong-parameter
retries, some *expected-failure* pytest probes — see Limitations),
over-reasoning in 5, step redundancy in 1. Every audit emitted exactly one
auto-applicable safe fix: `goal_anchor`, triggered by trajectory path
entropy. Estimated savings: ~0.5k–2.4k tokens/run (median ≈ 1.8k).

## Round 1: the product's patch, as it was — a negative result

| Task | Tokens before | Tokens after | Saved | Saved % | TAS Δ | Pass held |
|---|---:|---:|---:|---:|---:|:---:|
| csv-filter | 5538 | 5505 | 33 | 0.6% | -1.9 | ✅ |
| csv-filter.r2 | 4968 | 5446 | -478 | -9.6% | -5.0 | ✅ |
| dedupe-helpers | 5781 | 5032 | 749 | 13.0% | +0.7 | ✅ |
| dedupe-helpers.r2 | 4627 | 5262 | -635 | -13.7% | -0.3 | ✅ |
| fix-imports | 3280 | 3939 | -659 | -20.1% | -2.8 | ✅ |
| fix-imports.r2 | 4566 | 4533 | 33 | 0.7% | +0.6 | ✅ |
| fix-offby-one | 3793 | 4123 | -330 | -8.7% | -2.9 | ✅ |
| fix-offby-one.r2 | 3801 | 3476 | 325 | 8.6% | +1.1 | ✅ |
| implement-median | 4861 | 5154 | -293 | -6.0% | -4.8 | ✅ |
| implement-median.r2 | 3648 | 3764 | -116 | -3.2% | +0.7 | ✅ |
| rename-api | 4223 | 5211 | -988 | -23.4% | -2.1 | ✅ |
| rename-api.r2 | 5432 | 5704 | -272 | -5.0% | -1.1 | ✅ |

**Aggregate over 12 pairs:** mean token reduction **−5.6%** (95% bootstrap
CI [−11.4%, +0.2%]); mean TAS delta −1.5 (95% CI [−2.7, −0.4]). Pass rate
12/12 → 12/12. **Estimate accuracy: −102%** (95% CI [−205%, 0%]) — the
audit predicted savings; the measurement found a cost.

The patch made things *worse*. The mechanism is visible in the patch text
itself: the old `goal_anchor` wording instructed the agent to **"restate
the stated task objective in one sentence before each reasoning step."**
On runs that were already on-track (GAR passing on all 12), there was no
off-path exploration to recover — the anchor had nothing to save and
charged a restating ritual on every turn anyway.

This is the case-study harness doing precisely what it exists to do:
catching a mispriced fix. That the fix it caught was *ours* is the
strongest validation of the method we could have published.

## The product change

[`fixes.rs`](../crates/tracerazor-core/src/fixes.rs) now emits an anchor
that can't add a standing cost: keep the objective as working context,
skip non-advancing actions, and **"do not restate the objective or
summarise progress unless explicitly asked."** The detection logic and the
conservative 25%-recovery estimate are unchanged; only the remediation
text changed. (Trajectory-reduction work reports the same shape of win —
trim what doesn't advance, never add ritual: AgentDiet,
[arXiv:2509.23586](https://arxiv.org/abs/2509.23586); DEPO,
[arXiv:2511.15392](https://arxiv.org/abs/2511.15392).)

## Round 2: the revised patch

Same 12 before-traces, same protocol, fresh audits with the revised
`goal_anchor`, 12 fresh after-runs:

| Task | Tokens before | Tokens after | Saved | Saved % | TAS Δ | Pass held |
|---|---:|---:|---:|---:|---:|:---:|
| csv-filter | 5538 | 5829 | -291 | -5.3% | -3.8 | ✅ |
| csv-filter.r2 | 4968 | 4704 | 264 | 5.3% | -1.7 | ✅ |
| dedupe-helpers | 5781 | 4373 | 1408 | 24.4% | +3.2 | ✅ |
| dedupe-helpers.r2 | 4627 | 4804 | -177 | -3.8% | +1.4 | ✅ |
| fix-imports | 3280 | 4281 | -1001 | -30.5% | -1.5 | ✅ |
| fix-imports.r2 | 4566 | 3633 | 933 | 20.4% | +4.3 | ✅ |
| fix-offby-one | 3793 | 4172 | -379 | -10.0% | -0.1 | ✅ |
| fix-offby-one.r2 | 3801 | 3467 | 334 | 8.8% | +1.0 | ✅ |
| implement-median | 4861 | 4011 | 850 | 17.5% | -0.7 | ✅ |
| implement-median.r2 | 3648 | 4251 | -603 | -16.5% | +0.5 | ✅ |
| rename-api | 4223 | 4983 | -760 | -18.0% | -1.9 | ✅ |
| rename-api.r2 | 5432 | 4545 | 887 | 16.3% | -1.2 | ✅ |

**Aggregate over 12 pairs:** mean token reduction **+0.7%** (95% bootstrap
CI [−8.9%, +9.9%]); mean TAS delta −0.0 (95% CI [−1.2, +1.2]). Pass rate
12/12 → 12/12.

### Reading the result honestly

- **The standing cost is gone.** Round 1's −5.6% mean (CI excluding
  meaningful savings) became +0.7% with a CI centred on zero. The revision
  did exactly what the diagnosis predicted: removed the per-turn ritual.
- **Zero is the correct value for this workload.** GAR passed on all 12
  before-traces — these agents weren't drifting, so a drift remediation
  has nothing to recover. The right behaviour for a patch on an on-track
  agent is *no effect*, and that is what round 2 measured. Whether the
  anchor produces positive savings on genuinely drifting agents is the
  next experiment, and it needs a workload where GAR actually fails.
- **Run-to-run variance dominates single pairs.** Per-pair deltas span
  −30% to +24% between *identical* configurations — the same high
  variance Bai et al. report for coding agents. Never quote a single
  before/after pair as evidence; the aggregate CI is the unit of claim.

## Limitations — read these before quoting the headline number

- **Scale.** 6 tasks × 2 replicates, one model (Haiku 4.5), one agent
  harness, one domain (small Python repairs). The CIs are honest about
  small-N width; they are not license to extrapolate across models or
  domains.
- **Expected-failure probes read as misfires.** A coding agent runs the
  failing test suite *on purpose* to see the errors; the converter
  faithfully marks the non-zero exit as a failed call and TCA counts it
  against the agent. The penalty is identical across conditions, so the
  comparison stands, but absolute TCA on coding traces reads low.
- **Metric priors.** CSD (semantic drift) failed on all 24 traces:
  read→edit→test steps are legitimately semantically distant, which the
  conversational-agent prior misreads as drift. TPE path entropy triggered
  `goal_anchor` on every trace for the same reason. Detection on coding
  traces is conservative-biased; remediation must therefore be cost-safe
  even when triggered spuriously — which is exactly what round 1 enforced.
- **One fix type exercised.** Only `goal_anchor` fired on this workload;
  the other safe patches (verbosity, hedging, reformulation) were not
  measured here.
- **Accounting convention.** Marginal accounting excludes cache reads;
  under gross accounting the deltas would be larger in absolute terms but
  noisier (cache warmth varies per run). The convention is stamped in
  every trace's metadata.

## Reproduce it

```bash
cargo build --release -p tracerazor
python3 -m benchmark.live.run_live --condition before
python3 -m benchmark.live.audit_and_apply
for p in benchmark/live/results/*.prompt.txt; do
  pair=$(basename "$p" .prompt.txt); base=${pair%.r2}
  sfx=""; [ "$pair" != "$base" ] && sfx=".r2"
  python3 -m benchmark.live.run_live --condition after --task "$base" \
    --name-suffix "$sfx" --append-system-prompt-file "$p"
done
python -m benchmark.case_study --pairs-dir benchmark/live/traces
```

Requires a Claude Code install with credentials; total cost for the full
study (24 live runs + pilot) was ≈ $1.30 on Haiku 4.5. The captured traces
and run logs (cost, turns, session ids) are committed under
[`benchmark/live/traces/`](../benchmark/live/traces/) and
[`benchmark/live/traces-round1/`](../benchmark/live/traces-round1/), so
the measurement step is re-runnable without any API spend.

## Harness validation (synthetic — kept for CI)

The measurement pipeline itself is validated in CI
(`tests/test_case_study.py`) against constructed traces with known deltas,
including a pair whose pass flag flips — which must fail the run. That
plumbing check is what previously occupied this page; it has been replaced
by the measured tables above.
