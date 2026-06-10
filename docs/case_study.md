# The Measured Case Study (ship-plan 4.1)

**Status: harness shipped and CI-tested; the live tau-bench re-run is
pending API budget.** Until that run lands, every savings number TraceRazor
prints remains a heuristic projection — this page documents exactly how the
measured numbers will be produced, so the methodology is fixed *before* the
data exists.

## Why this exists

The audit's "savings estimate" is a sum of per-fix heuristic projections. A
projection is not evidence. The only number that matters to a buyer is:

> tokens before vs tokens after applying TraceRazor's fixes, **at unchanged
> task pass rate**, with confidence intervals, on real tasks.

A token saving that costs task success is a regression wearing a costume.
The harness therefore refuses to call a delta a "saving" on any task whose
pass flag flips, and exits non-zero so CI catches it.

## Protocol

For each of 3–5 tau-bench tasks (airline domain, gpt-4o, fixed seed):

1. **Capture the before-trace.** Run the agent on the task; convert the
   transcript with `calibration/sources/from_taubench.py`; record the
   tau-bench reward in `task_value_score` (1.0 = solved).
2. **Audit and patch.**
   `tracerazor audit before.json --format json --hermetic > audit.json`,
   extract `fixes` to `<task>.fixes.json`, apply the safe subset to the
   system prompt with `tracerazor apply audit.json --to system_prompt.txt`.
3. **Capture the after-trace.** Re-run the *same task, same model, same
   seed policy* with the patched prompt; record the reward again.
4. **Measure.** Place pairs in one directory as `<task>.before.json` /
   `<task>.after.json` / `<task>.fixes.json`, then:

   ```bash
   python -m benchmark.case_study --pairs-dir results/case_study --out docs/case_study.md
   ```

The harness runs `tracerazor bench --format json` per pair and aggregates:

- **per-task measured token delta** (before − after, and %),
- **mean token reduction with a seeded 10,000-resample percentile-bootstrap
  95% CI** over tasks,
- **pass rate before vs after**, failing loudly if any task outcome flipped,
- **estimate accuracy** (measured ÷ audit-estimated savings) when the fixes
  JSON is supplied — this is the number that retires or validates the
  heuristic projections.

Multiple runs per task (different seeds) can be encoded as separate pairs
(`task0-seed1`, `task0-seed2`, …); the CI then absorbs run-to-run variance.

## Plumbing check (synthetic — NOT the case study)

The harness itself is validated in CI (`tests/test_case_study.py`) against
constructed traces where the after-trace drops a redundant re-fetch and a
filler step. Output of that check, to show the table shape:

> **Synthetic plumbing check — NOT the case study.** These rows
> come from constructed traces and validate the harness, not the
> product. The published case study requires real agent re-runs.

| Task | Tokens before | Tokens after | Saved | Saved % | TAS Δ | Pass held |
|---|---:|---:|---:|---:|---:|:---:|
| synthetic-task0 | 760 | 520 | 240 | 31.6% | +7.6 | ✅ |
| synthetic-task1 | 860 | 520 | 340 | 39.5% | +11.1 | ✅ |

**Aggregate over 2 task(s):** mean token reduction **35.6%** (95% bootstrap CI [31.6%, 39.5%]); mean TAS delta +9.3 (95% CI [+7.6, +11.1]).

**Pass rate:** 2/2 before → 2/2 after — constant task outcome on every pair (the savings are at unchanged pass rate).

## What blocks the real numbers

Step 1 and step 3 require running gpt-4o against tau-bench (~$5–20 of API
budget for 3–5 tasks × 2 runs × a few seeds) from a machine with credentials.
Everything else — fix application, pair conversion, measurement, CI math,
the published table — is automated and tested here. When the run happens,
this page's synthetic section gets replaced by the measured table produced
by the same command.
