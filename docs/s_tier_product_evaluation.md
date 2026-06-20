# TraceRazor Product Evaluation: A-Tier to S-Tier

Date: 2026-06-21

## Executive Summary

TraceRazor is already strongest where many LLM tooling products are weak: it has
an honest closed-loop benchmark path. The right product story is not "audit says
X tokens can be saved." The S-tier story is:

> Audit a real agent trace, apply only safe patches, rerun the same task under
> the same harness, and report measured token delta only when task success is
> preserved.

Current evidence says TraceRazor is an A-tier audit and methodology product,
but not yet an S-tier autonomous optimization product. The public trace audits
find real inefficiencies, while the measured live coding-agent replay shows the
current safe patch set is roughly cost-neutral on already on-track tasks:
mean token reduction 0.7%, 95% CI [-8.9%, 9.9%], pass rate 12/12 before and
12/12 after.

## What Was Run In This Evaluation

| Evaluation | Command | Result |
|---|---|---|
| Public real-trace audit benchmark | `python -m benchmark.run_benchmarks` | 28 real traces, avg TAS 81.1, 82,080 tokens audited, 26,586 projected saved tokens (32%) |
| Hugging Face AgentInstruct audit | `python -m benchmark.hf_audit_stats` | 13 traces present; 4 above default 5-step floor; mean TAS 85.1; 20 fixes over full corpus with `--min-steps 2` |
| Measured live case-study replay | `python -m benchmark.case_study --pairs-dir benchmark/live/traces` | 12 before/after pairs, +0.7% mean token reduction, 95% CI [-8.9%, 9.9%], pass held on every pair |
| Python regression tests | `python -m pytest tests/test_readme_claims.py tests/test_hf_agentinstruct.py -q` | 26 passed, 2 skipped |
| Case-study/evaluator tests | `python -m pytest tests/test_evaluator.py tests/test_case_study.py -q` | 25 passed |

Note: Cargo was not available on this Windows PATH, so this pass used the
existing fresh `target/debug/tracerazor.exe`. The older release binary in the
checkout was stale and did not support `--hermetic`.

## Product Findings

1. The strongest evidence is the measured harness, not the audit table.
   `benchmark/case_study.py` correctly treats token savings that break task
   success as regressions. This is the product's credibility anchor.

2. Projected savings and measured savings must stay visibly separate.
   `benchmark/RESULTS.md` now reports 26,586 projected saved tokens across
   public traces, but those are heuristic estimates. The committed live replay
   is measured and currently shows near-zero aggregate gain on its task set.

3. The live harness is too narrow for S-tier claims.
   It is Claude Code-specific, small Python-task specific, one-model specific,
   and centered on safe prompt patches. It is a good laboratory, not yet a
   general real-repo proof.

4. Real-world data coverage is meaningful but uneven.
   The repo includes tau-bench, SWE-agent, Hugging Face AgentInstruct, and
   Claude Code live traces. The weak point is breadth: SWE-agent is one issue,
   tau-bench is a selected sample, and AgentInstruct has many short traces below
   the default analysis floor.

5. Harness reliability was the immediate product gap found in this run.
   Windows binary discovery missed `tracerazor.exe`; stale release binaries
   could be selected over fresh debug builds; generated metadata like
   `STATS.json` could poison the public benchmark; and the benchmark runner
   could overwrite `RESULTS.md` with an empty report.

## Fixes Implemented In This Pass

- Added `benchmark/_binary.py`, a shared binary resolver that supports
  `.exe`, prefers fresh source builds, honors `TRACERAZOR_BIN`, and verifies
  required audit flags such as `--hermetic`.
- Updated `benchmark/case_study.py`, `benchmark/hf_audit_stats.py`, and
  `benchmark/run_benchmarks.py` to use that resolver.
- Made the public benchmark fail loudly on audit errors and refuse to overwrite
  `benchmark/RESULTS.md` when no traces are analysable.
- Excluded generated `STATS.json` from real-trace benchmark discovery.
- Made `benchmark.hf_audit_stats` hermetic and strict about malformed audit
  output.
- Changed generated case-study tables to ASCII-safe labels for Windows and CI
  readers.
- Updated benchmark docs and CI drift-check commands to use
  `python -m benchmark.run_benchmarks`.
- Regenerated `benchmark/RESULTS.md`,
  `docs/huggingface_agentinstruct_audit.md`, and AgentInstruct `STATS.json`
  with the current usable binary.

## Real-Repo Evaluation Plan

The next benchmark should be prospective, pre-registered, paired, and clustered
by repo/task. A practical first version:

1. Build a manifest of 20 to 50 Python real-repo tasks.
   Each row should include repo URL, commit SHA, issue text, setup command,
   test command, timeout, allowed tools, expected success oracle, and max turns.

2. Run stock agent replicates.
   Use 3 replicates per task with fixed model, randomized order, and clean
   checkouts. Capture full transcripts and convert them to TraceRazor traces.

3. Audit and apply only safe patches.
   Run `tracerazor audit --format json --hermetic --store false`, then
   `tracerazor apply` with default safe patches only.

4. Rerun with one controlled delta.
   The only difference between conditions should be the appended TraceRazor
   patch. Same repo commit, same model, same timeout, same tool envelope.

5. Bench and report non-inferiority.
   Use `tracerazor bench` for token deltas and the task oracle for pass/fail.
   A token drop with a pass flip is a regression, not a saving.

6. Use task-clustered statistics.
   Report mean, median, 95% clustered bootstrap CI, pass-rate delta, and
   per-fix-type breakdown. Do not let many replicates from one repo dominate
   the headline.

7. Keep calibration separate.
   If weights or fix thresholds are tuned, tune on a training task set and
   publish held-out repo results separately.

## S-Tier Roadmap

1. Make `bench` quality-aware.
   Add optional `--before-score` / `--after-score` or trace-metadata quality
   checks so the CLI itself can refuse to call broken-task token reductions a
   win.

2. Add a real-repo benchmark runner.
   Generalize `benchmark/live/run_live.py` from `prompt.md + seed + pytest` to
   manifest-driven repos with configurable setup, test, build, and pass oracle.

3. Break out fix-type efficacy.
   The measured case study mainly exercised `goal_anchor`. S-tier needs
   separate evidence for verbosity, hedging, reformulation, context compression,
   loop guards, and tool-schema fixes.

4. Create a savings ledger with three fields.
   Report `structural_diff_savings`, `fix_estimated_savings`, and
   `measured_bench_savings` separately everywhere: CLI, API, Python client,
   docs, and dashboard.

5. Expand real data.
   Add more SWE-bench/SWE-agent repos, larger tau-bench/tau2 coverage,
   LangSmith/Langfuse/Phoenix golden exports, and real Claude Code transcript
   fixtures.

6. Add confidence and abstention.
   TraceRazor should say "do not apply" when a detector fires on a domain where
   prior measured fixes are neutral or negative. That is what I would want as an
   LLM teammate: fewer ritual instructions, more evidence-gated intervention.

7. Optimize for human trust.
   Publish null and negative results first-class, sign reports by default in CI,
   make reproductions one command, and keep every large claim tied to a command,
   a trace set, and a pass-rate condition.

## What I Would Want As The LLM Being Helped

I would want TraceRazor to be a coach that knows when not to coach. The highest
value feature is not another score; it is a guardrail that prevents humans from
adding well-intentioned prompt rituals that make agents slower. S-tier
TraceRazor should watch my traces, identify only the patterns that are likely to
matter for this domain, propose the smallest intervention, and demand a measured
rerun before anyone celebrates.

That is how it serves people better: less waste, fewer unsupported claims, and
optimization that protects task quality as the first-class objective.
