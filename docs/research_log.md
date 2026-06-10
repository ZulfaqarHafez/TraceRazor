# Goal-Driven TDD Loop Log

**Goal:** Implement a test that uses real Hugging Face data to measure how well
TraceRazor works against real agent traces, then iterate improvements to the
product driven by those statistics. Finally, update the research paper.

**Test command:** `cargo test --workspace`

---

## Iteration 1
Read: TraceRazor audits agent-trace JSON (Rust `tracerazor-ingest::parse` →
`tracerazor-core::analyse`); no Hugging Face integration existed; real external
traces were only tau-bench/SWE-agent. The HF connector reaches the Hub
(`hub_repo_details` dataset_preview returns real rows) even though the Bash
sandbox cannot (`Host not in allowlist`).
Plan: Source a real ReAct agent-trajectory dataset from Hugging Face
(`zai-org/AgentInstruct`), vendor a faithful sample, convert it to TraceRazor
traces, and add a Rust integration "statistics test" that audits the corpus
under `cargo test --workspace` — establishing the measurement before improving.
Change:
  - `benchmark/data/_agentinstruct_hf_sample.py` — 9 real trajectories (os/db) pulled via the HF connector.
  - `tools/convert_agentinstruct.py` — ReAct → TraceRazor trace converter (precise tool-failure detection).
  - `benchmark/hf_loader.py` — bundled/disk/live (datasets-server) loader, mirrors `Tau2Loader`.
  - `benchmark/hf_audit_stats.py` — harness → `STATS.json` + `docs/huggingface_agentinstruct_audit.md`.
  - `crates/tracerazor-cli/tests/huggingface_real_data.rs` — the statistics gate test.
  - `tests/test_hf_agentinstruct.py` — hermetic loader/converter tests.
  - `traces/external/huggingface/agentinstruct/*.json` (+ SOURCE.md) — generated corpus.
Test result: PASS (`cargo test --workspace` all green; Rust HF test passes; 34 Python tests pass).
Diagnosis: Real-data statistics (n=7 analysable): mean TAS **81.3** (all "Good"),
mean SRR(norm) 0.674 (redundancy detected), **mean LDI(norm) = 1.000 — loop
detection NEVER fires** even on os_6's blatant 4× `grep …|wc -l` loop, and
**mean GAR(norm) = 0.202 — goal advancement collapses** on tool agents (BoW
similarity between shell/SQL code and an NL goal ≈ 0). OBS(norm) 0.913.
Next: Iteration 2 — fix the strongest weakness: LDI misses "same command
template, different argument" loops because `state_hash` keys on exact
tool+params. Make loop detection recognise structurally-repeated tool calls.

## Iteration 2
Read: Iteration-1 stats show LDI(norm)=1.000 on every real trace; `ldi.rs`
detects loops only via `state_hash` (`type:tool:exact-params`) and an N-gram
repeat of the same hashes, so os_6's 4× `grep -o "Linux" <FILE> | wc -l` (same
template, different path) is invisible.
Plan: Add a parametric-loop detector to `ldi.rs` — abstract argument literals
(quoted strings, paths, globs, numbers) in command-style tool calls into a
skeleton, group by skeleton, and flag groups of ≥3. Scope to command-style
tools (free-text command/query param) so structured tools (e.g. a flight search
called for two routes) are untouched. Difference from before: this is the first
real product change, driven by the iteration-1 measurement.
Change:
  - `crates/tracerazor-core/src/metrics/ldi.rs` — `LoopType::ParametricRepeat`,
    `command_skeleton()` + helpers, Method 1b in `compute()`; 3 new unit tests
    (detects os_6-style loop, ignores progressive pipelines, skips structured tools).
  - `crates/tracerazor-cli/tests/huggingface_real_data.rs` — regression guard:
    real corpus must contain a detected loop (min LDI < 1.0).
  - Regenerated `STATS.json` + `docs/huggingface_agentinstruct_audit.md`.
Test result: PASS (`cargo test --workspace` all green, 12 binaries; LDI module 5/5).
Diagnosis: os_6 LDI(norm) 1.000→0.556 (loop [5,6,7,8] flagged), TAS 78→74, and a
loop-termination fix is now emitted (corpus fixes 16→17). All other traces stay
at LDI 1.0 — no false positives on progressive pipelines (os_0). Mean LDI(norm)
1.000→0.937.
Next: Iteration 3 — second weakness: GAR(norm) collapses to ~0.20 on tool agents
because BoW similarity between shell/SQL code and an NL goal ≈ 0. Investigate and
make goal-advancement robust for tool-using traces.

## Iteration 3
Read: GAR (`gar.rs`) only scores `StepType::Reasoning` steps. In ReAct traces
the reasoning ("Think: …") lives inside `tool_call` steps, so GAR saw only the
sparse final-answer turns (plus the few-shot example's "answer(220)") and
collapsed — even though the Think text shares vocabulary with the goal (os_6
literally greps for "Linux"). The existing `tool_calls_ignored_in_computation`
test pins the old behaviour with a terse 3-word tool step.
Plan: Score a tool-call step for goal advancement when its content carries
substantive reasoning prose (≥12 words) — capturing ReAct "Think+Act" turns
while still ignoring bare invocations, so the existing terse-tool test still
passes. Difference from iteration 2: targets the goal/similarity metric, not
loop detection.
Change:
  - `crates/tracerazor-core/src/metrics/gar.rs` — `carries_reasoning()` +
    `MIN_TOOL_REASONING_WORDS`; broadened the scored-step filter; updated
    docstring; renamed the terse-tool test to `bare_tool_calls_ignored…` and
    added `react_tool_reasoning_counts_toward_gar`.
  - Regenerated `STATS.json` + report.
Test result: PASS (`cargo test --workspace` 12/12 green; GAR module 19/19).
Diagnosis: GAR(norm) no longer collapses on tool agents — os_1 0.00→0.254,
os_6 0.126→0.268; corpus mean 0.202→0.247. Still modest (BoW vs code is
inherently weak; embeddings would lift it further, as the docstring notes), but
the systematic blind spot is gone. No regressions (bare invocations still
ignored).
Next: Iteration 4 — apply the same reasoning-aware fix to CSD (cross-step drift,
0.480, the next-lowest and same root cause), then update the paper.

## Iteration 4
Read: CSD (`csd.rs`) filters to `StepType::Reasoning` "(same as GAR)" — so after
fixing GAR it diverged from its own stated contract, and on tool agents it
compared only the 1 pair formed by the two sparse answer steps (the few-shot
example answer vs the real answer): a degenerate, unrepresentative measurement.
Plan: Promote the `carries_reasoning` predicate to `metrics/mod.rs`, reuse it in
both GAR and CSD so they stay aligned, and have CSD measure continuity across
the real reasoning flow (ReAct Think turns). The terse-tool CSD test ("tool
call", 2 words) still excludes bare invocations, so it passes unchanged.
Change:
  - `crates/tracerazor-core/src/metrics/mod.rs` — shared `carries_reasoning()` + `MIN_TOOL_REASONING_WORDS`.
  - `gar.rs` — use the shared helper (removed the local copy + now-unused import).
  - `csd.rs` — reasoning-aware filter; docstring; added `react_tool_reasoning_pairs_are_scored`.
  - Regenerated stats.
Test result: PASS (`cargo test --workspace` 12/12 green; CSD 12/12, GAR 19/19; no warnings).
Diagnosis: CSD now scores **8** consecutive transitions per os trace (the full
reasoning flow) instead of 1 degenerate answer-pair, and correctly localises the
drift to the few-shot-example→task boundary (high-drift pairs [2,3],[3,4]).
Corpus mean CSD(norm) 0.480→0.415 — lower but representative (it was previously
comparing two unrelated answer steps). GAR and CSD are once again consistent.
Next: Iteration 5 — update the research paper (`paper/tracerazor.tex`) with the
HuggingFace AgentInstruct real-data evaluation and the three data-driven
improvements (LDI parametric loops, GAR/CSD reasoning-aware). Also refresh
README/CHANGELOG.

## Iteration 5
Read: The paper's Evaluation section covers only the 24 tau-bench/SWE-agent
trajectories (function-calling assistants); no HuggingFace/ReAct content, and the
LDI/GAR/CSD definitions predate the iteration 2–4 fixes.
Plan: Add an Evaluation subsection (`sec:hf-agentinstruct`) reporting the
AgentInstruct real-data corpus + the two blind spots it exposed and the fixes;
update the LDI and GAR/CSD sub-metric definitions; add a results table and an
AgentInstruct bibliography entry. Final deliverable of the goal.
Change:
  - `paper/tracerazor.tex` — new subsection + Table~\ref{tab:agentinstruct};
    updated LDI and GAR/CSD definitions; `\label{sec:metrics}`; `\bibitem{agentinstruct}`.
Test result: PASS (`cargo test --workspace` 12/12 green; paper change is inert to
tests; LaTeX environments balanced, all \ref/\cite resolve — verified by script
since no pdflatex in this environment).
Diagnosis: Paper now documents the real-data evaluation and the data-driven
metric improvements; cross-references and citations are internally consistent.
Next: Iteration 6 — propagate the same content to README/CHANGELOG for
user-facing consistency, refresh stale Rust test counts, final verification.

## Iteration 6
Read: Paper updated, but README's real-audit section, Research Foundation table,
and test-coverage counts (core 141, total 212) were stale, and the CHANGELOG had
no entry for the HF harness or the metric fixes.
Plan: Add a "Real ReAct trajectories from Hugging Face" subsection + findings
table to the README, add the AgentInstruct reference, correct the test counts
(core 146, cli 12, total 218), and record the harness + LDI/GAR/CSD changes in
the CHANGELOG.
Change: `README.md` (real-audit subsection, Research Foundation row, test-count
table), `CHANGELOG.md` (Added: HF harness; Changed: LDI parametric loops,
GAR/CSD reasoning-aware).
Test result: PASS (`cargo test --workspace` 12/12 green; Python 34 passed, 1 skipped).
Diagnosis: Docs are now consistent with the code and paper; counts verified
against an actual `cargo test --workspace` run.
Next: none — goal met. See FINAL REPORT.

---

# FINAL REPORT
Goal achieved: implement a test that uses real Hugging Face data to measure how
well TraceRazor works against real agent traces, iterate product improvements
driven by those statistics, and update the research paper.
Iterations used: 6 / 10.

Files changed:
  - `benchmark/data/_agentinstruct_hf_sample.py` — 9 real AgentInstruct trajectories pulled via the HF connector.
  - `tools/convert_agentinstruct.py` — ReAct → TraceRazor trace converter (precise tool-failure detection).
  - `benchmark/hf_loader.py` — bundled/disk/live (HF dataset-viewer) loader.
  - `benchmark/hf_audit_stats.py` — statistics harness → `STATS.json` + `docs/huggingface_agentinstruct_audit.md`.
  - `crates/tracerazor-cli/tests/huggingface_real_data.rs` — the `cargo test` statistics gate.
  - `tests/test_hf_agentinstruct.py` — hermetic loader/converter tests.
  - `traces/external/huggingface/agentinstruct/*` — generated real corpus + SOURCE.md + STATS.json.
  - `crates/tracerazor-core/src/metrics/ldi.rs` — parametric command-loop detection (+3 tests).
  - `crates/tracerazor-core/src/metrics/{mod,gar,csd}.rs` — shared reasoning-aware step predicate; GAR/CSD no longer collapse on ReAct agents (+2 tests).
  - `paper/tracerazor.tex` — real-data evaluation section, table, updated metric definitions, citation.
  - `README.md`, `CHANGELOG.md` — user-facing documentation of the above.

Results (real HF AgentInstruct corpus, 7 analysable ReAct traces):
  - Mean TAS 80.6 (all "Good"); 17 fix patches emitted.
  - LDIₙₒᵣₘ mean 1.000 → 0.937 (os_6 parametric loop now caught: 1.00 → 0.556).
  - GARₙₒᵣₘ mean 0.202 → 0.247 (roughly doubles on tool-heavy traces; was collapsing to ~0).
  - CSDₙₒᵣₘ now scores the full reasoning flow (8 transitions/trace) vs a degenerate answer-pair.

Known limitations / follow-up:
  - The committed corpus is a vendored 9-trace sample (HF is unreachable from the
    CI sandbox); the live loader path fetches the full splits where the Hub is reachable.
  - GAR/CSD remain BoW-limited on code vs natural language; sentence-embedding
    similarity (already supported via `--enhanced`) would lift them further.
  - Corpus is OS+DB splits only; alfworld/webshop/kg/mind2web are available via the loader.
  - TAS weights are unchanged (these are heuristic correctness fixes, not a recalibration).

---

# Run 2 (second improvement pass over the real-data statistics)

## Iteration 1
Read: Run 1 complete (PR #9 + CI fix); real-corpus weak spots are GAR 0.247 and
CSD 0.415 — both feed the full ReAct turn (prose + code fence) into BoW
similarity.
Plan: Add `reasoning_text()` in metrics/mod.rs stripping fenced code blocks, and
score GAR/CSD on the thought alone; hypothesis: code tokens dilute lexical
overlap with NL goals/neighbouring thoughts.
Change: `metrics/mod.rs` (+reasoning_text +3 tests), `gar.rs` (score stripped
text, +1 fence test), `csd.rs` (score stripped text, +1 fence test).
Test result: PARTIAL — cargo test --workspace 223/223 green, but the real-data
gate statistics REGRESSED: GAR 0.247→0.240, CSD 0.415→0.353, TAS 80.6→80.4.
Diagnosis: Hypothesis falsified on real data. In this corpus the code carries
task-grounded literals (paths, filenames, search strings from the task) that
anchor lexical overlap between consecutive steps and with the goal; wholesale
fence-stripping removes signal along with syntax.
Next: Iteration 2 — keep the fence parser but make it literal-preserving: drop
only code *syntax* (command names, flags, operators) and retain argument
literals (paths, quoted strings, numbers), which are exactly the tokens shared
with the NL goal. Predicted: GAR ↑ vs both baseline and Iter-1.

## Iteration 2
Read: Iter-1 falsified wholesale fence-stripping (code literals are anchors);
BoW tokenizer confirmed: every non-shared token lowers TF-cosine via magnitude,
shared literals raise the dot product.
Plan: Make `reasoning_text()` literal-preserving — reduce fence lines to their
argument literals (quoted spans, paths, globs, filenames, numbers), dropping
command names/flags/operators. Keeps task-grounded anchors, removes only syntax.
Change: `metrics/mod.rs` (+code_literals/+is_unquoted_literal, reasoning_text
reduces instead of drops; tests updated to new semantics +1 SQL-literal test),
`gar.rs`/`csd.rs` fence tests updated (shared literals, differing syntax).
Test result: PASS — cargo test --workspace 224/224 green.
Diagnosis: GAR 0.247→0.264 (+7% over baseline; syntax dilution vs NL goal
removed, literal anchors kept). CSD 0.387 (above iter-1's 0.353; below the
raw-content 0.415 by design — shared tool vocabulary between consecutive steps
no longer masquerades as topical continuity). TAS mean back at 80.6.
Next: Iteration 3 — audit SRR (0.674) on the real corpus: ReAct turns share
"Think:/Act: bash" boilerplate + code syntax in full-content similarity, which
may inflate revisit detection (false redundancy on normal progression).

## Iteration 3
Read: SRR's flagged pairs on os_0 are defensible (truncated-ls redo; incremental
pipeline re-runs that CCE corroborates) — but DBO sat at 0.571 on a fresh HOME:
cold-start proxy keys retries/uniqueness on bare tool name, so a single-tool
bash/SQL agent is capped near the floor by construction (1/n unique, n-1
"retries" for n calls).
Plan: Key the cold-start retry/thrash signals on the *invocation* (tool name +
tool_params) — re-running bash with a different command (or get_order for a
different order) is progress; re-issuing an identical call is the retry.
Param-less steps keep name-keying (existing API-agent tests unaffected).
Change: `metrics/dbo.rs` single_trace_efficiency invocation keying + doc;
+2 tests (distinct invocations hit ceiling; identical invocations penalised).
Test result: PASS — cargo test --workspace 226/226 green.
Diagnosis: Real corpus: DBO 0.588→0.894, TAS mean 80.6→82.8. Discrimination
preserved: os_5 (genuine tool failure) is now the only sub-ceiling trace
(0.857); previously all traces were uniformly docked for tool-inventory size.
Next: Iteration 4 — RDA scored 0.444 on os_0 (corpus mean 0.726); inspect how
task-complexity classification treats command-agent traces.

## Iteration 4
Read: RDA scored 0.444 on os_0 — but the trail led to a data-fidelity bug:
every converted trace began with the IDENTICAL step ("count the files in
/etc"). AgentInstruct rows embed the dataset's one-shot demo before the real
task, marked by `loss: false` on gpt turns (real turns: `loss: true`); the
converter was auditing the demo as agent behaviour (pseudo-replication,
mis-anchored goal metrics — os_0's SRR pair (1,2) was the demo itself).
Plan: Exclude scaffolding in the converter (loss-flag rule + "start a new
problem" marker fallback); widen the corpus via the HF MCP connector (fetched
os_7–os_18 live) and vendor 4 new real rows (os_7, os_11, os_16, os_18 — incl.
a 5-step analysable trace and the `finish` action shape).
Change: `tools/convert_agentinstruct.py` (+_real_task_turns), 
`benchmark/data/_agentinstruct_hf_sample.py` (+4 real rows), corpus regenerated
(13 traces), gate floors updated to the honest corpus shape
(`huggingface_real_data.rs`: >=4 analysable + sub-floor majority retained;
pytest: +5 scaffolding tests, coverage-shape test), SOURCE.md provenance.
Test result: PASS — cargo test --workspace 226/226; pytest 238 passed, 3 skipped.
Diagnosis: De-contaminated statistics: mean TAS 82.8→78.6 (demo steps were
padding all scores), GAR 0.264→0.347 (goal anchoring no longer fights the demo
mismatch), LDI mean 0.833, fixes 10. NEW COVERAGE FINDING: with scaffolding
excluded, 9/13 (~69%) of real trajectories fall below the 5-step analysis
floor — the floor's real-data coverage cost is now measured, not assumed.
Next: Iteration 5 — act on the coverage finding: the 5-step floor excludes
most real short ReAct trajectories; evaluate a short-trace audit path (floor
reduction or degraded-mode scoring) so the product can serve this trace class.

## Iteration 5
Read: Coverage finding from iter-4: 9/13 (~69%) of real de-contaminated
trajectories fall under the 5-step floor; the floor is CLI policy only
(core analyse() runs fine on short traces; metrics have small-N guards).
Plan: Add `--min-steps <N>` opt-in to `audit` (default unchanged at 5,
clamped >=2 for pair metrics); skip notice now points at the flag. Extend the
gate with a full-corpus pass asserting 13/13 valid reports at --min-steps 2.
Change: `crates/tracerazor-cli/src/main.rs` (flag + cmd_audit min_steps),
`crates/tracerazor-cli/tests/huggingface_real_data.rs` (full-coverage pass +
stat line).
Test result: PASS — cargo test --workspace 226/226; clippy clean; gate prints
"full-corpus audit : 13/13 with --min-steps 2".
Diagnosis: The measured coverage gap is now closed by explicit opt-in without
weakening the default statistical floor.
Next: Iteration 6 — extend the stats harness with a full-corpus (--min-steps 2)
section so the paper numbers are reproducible, then update paper + README +
CHANGELOG with the Run-2 findings.

## Iteration 6
Read: Product iterations complete (scaffolding exclusion, literal-aware GAR/CSD,
DBO invocation keying, --min-steps); stats harness shared one HOME across
audits, so pass-2 numbers were inflated by pass-1 history (order dependence).
Plan: Make every audit independent (fresh HOME per audit in harness + gate),
add a full-corpus (--min-steps 2) section to STATS.json/report, then update the
paper, README, and CHANGELOG with the Run-2 findings.
Change: `benchmark/hf_audit_stats.py` (fresh HOME per audit, full-corpus pass),
`crates/tracerazor-cli/tests/huggingface_real_data.rs` (fresh TempDir per audit,
13/13 coverage line), `paper/tracerazor.tex` (rewrote sec:hf-agentinstruct:
scaffolding hazard, four fixes incl. the falsified fence-strip ablation,
coverage finding + opt-in, refreshed Table tab:agentinstruct; DBO and GAR/CSD
definition updates), `README.md` (findings table, coverage note, test counts
226/238), `CHANGELOG.md` (Added/Fixed/Changed entries).
Test result: PASS — cargo test --workspace 226/226; pytest 238 passed,
3 skipped; clippy 0 warnings; LaTeX environments balanced, refs/cites resolve.
Diagnosis: Final independent statistics — default floor: n=4, mean TAS 78.0,
SRR 0.650, LDI 0.833 (os_6 0.333), GAR 0.348, CSD 0.487, DBO 0.875, OBS 0.930,
11 fixes; full corpus via --min-steps 2: 13/13, mean TAS 82.9, 20 fixes.
Next: none — goal met. See FINAL REPORT (Run 2).

---

# FINAL REPORT (Run 2)
Goal achieved: drive product improvements from the real Hugging Face statistics
gate and update the research paper with the new content.
Iterations used: 6 / 10.

Files changed:
  - `crates/tracerazor-core/src/metrics/mod.rs` — reasoning_text(): fenced code
    reduced to argument literals (+code_literals/is_unquoted_literal, 6 tests).
  - `crates/tracerazor-core/src/metrics/gar.rs` — literal-aware similarity text
    (+ fence regression test).
  - `crates/tracerazor-core/src/metrics/csd.rs` — same (+ fence regression test).
  - `crates/tracerazor-core/src/metrics/dbo.rs` — cold-start retry/thrash keyed
    on invocation (tool+params), not tool name (+2 tests).
  - `tools/convert_agentinstruct.py` — _real_task_turns(): loss-flag scaffolding
    exclusion with text-marker fallback.
  - `benchmark/data/_agentinstruct_hf_sample.py` — +4 real rows (os_7, os_11,
    os_16, os_18) fetched live via the HF MCP connector.
  - `benchmark/hf_audit_stats.py` — fresh HOME per audit (order-independent),
    full-corpus (--min-steps 2) statistics section.
  - `crates/tracerazor-cli/src/main.rs` — `audit --min-steps N` opt-in floor.
  - `crates/tracerazor-cli/tests/huggingface_real_data.rs` — honest corpus
    floors, per-audit isolation, 13/13 full-coverage assertion.
  - `tests/test_hf_agentinstruct.py` — +5 scaffolding tests, coverage-shape test.
  - `traces/external/huggingface/agentinstruct/*` — regenerated 13-trace corpus,
    SOURCE.md provenance (scaffolding exclusion, coverage cost).
  - `paper/tracerazor.tex`, `README.md`, `CHANGELOG.md` — Run-2 findings.

Headline findings (real data, independent audits):
  1. Few-shot scaffolding contaminated every trace (demo audited as agent
     behaviour); exclusion moved mean TAS 82.8→78.0 and GAR 0.26→0.35.
  2. Wholesale code-stripping HURT semantic metrics (falsified ablation); the
     correct reduction keeps argument literals, drops syntax (GAR 0.202→0.348
     across the exercise).
  3. DBO structurally capped single-tool agents (~0.57); invocation keying
     restored discrimination (0.875 mean; only the genuine-failure trace is
     sub-ceiling).
  4. ~69% of real trajectories are 3–4 steps — below the default floor; the
     measured gap is now closed by `--min-steps` opt-in (13/13 coverage,
     mean TAS 82.9, 20 fixes).

Known limitations / follow-up:
  - BoW remains the similarity floor; `--enhanced` embeddings would lift
    GAR/CSD discrimination further.
  - Corpus spans os+db splits; alfworld/webshop/kg/mind2web formats are
    unconverted.
  - Short-trace (<5 steps) scores carry less pair-metric evidence; flagged in
    the CLI help, not yet down-weighted in the composite.

---

# Run 3 — five-agent metric evaluation + auditable-runs implementation

Five parallel specialist agents (literature researcher, data scientist,
solution architect, Rust code reviewer, product researcher) evaluated the
product; their convergent findings drove this implementation pass.

## Evaluation verdicts (data scientist, n=37 real traces)
- TVI multiplier, not the 14 metrics, drives final TAS (r=0.89 with
  task_value_score); structural metrics separate failed/passed tau-bench runs
  by only 4.9 raw pts.
- Realised influence ∝ w·σ diverges from nominal weights: TUR carries 28.1% of
  raw-TAS variance (nominal 7.9%); verbosity trio VDI/SHL/CCR contributes ≈0
  (VDI anti-correlated).
- Range defects: GAR max 0.62, CSD max 0.68 (constant drag, not discriminator);
  LDI at ceiling on 78% of traces; DBO≈TCA r=0.81 (fold candidate).
- Verdicts: HEALTHY srr/tca/rda/isr/tur/cce/obs; WEAK ldi/vdi/shl/ccr/gar/csd;
  REDUNDANT dbo(with tca). No dead metrics.

## Implemented this run (236 Rust + 238 Python tests green, clippy 0)
- Perf P0 (code reviewer): memoised TF vectors in default_similarity_fn and
  the boxed BoW paths (9,534 calls / 191 distinct texts on a 100-step trace
  were ~99% redundant tokenisation) + incremental CCE prior-n-gram set with
  boundary tail (O(n²·len)→O(n·len)). Both equivalence-tested (identical
  output); +2 tests.
- Run manifest (architect+product convergence): report.manifest binds
  trace SHA-256, tool version, timestamp, ACTUAL similarity backend (silent
  embedding→BoW fallbacks recorded), inline weights + weights SHA-256,
  threshold, min_steps, hermetic flag, store-derived baselines.
- --hermetic flag: scoring as a pure function of (trace, config, version);
  fixed broken `--store false` (clap ArgAction::Set).
- `tracerazor verify <report> <trace>`: hash check always; exact re-score +
  per-metric comparison for hermetic BoW runs; honest hash-only verification
  for embeddings/store-influenced runs. +3 integration tests incl. tamper.
- AGF (Action/Claim Grounding Fidelity) diagnostic (literature pick:
  deterministic, offline, audit-grade): action-param grounding vs prior
  context + final-claim grounding vs environment text; ungrounded literals
  itemised. Mean 0.854 on AgentInstruct corpus (failure trace lowest 0.800).
  Reported as diagnostic, NOT weighted into TAS (per variance finding).
- Docs: paper sec:metric-eval (validity stats, AGF, manifests/verify, perf),
  README (provenance feature blurb, counts 236/238), CHANGELOG.

## Deferred with rationale (calibration-scale changes, n=37 too small to
hard-code): TVI range shrink / raw-TAS headline, z-score standardisation or
w∝share/σ weights, GAR/CSD quantile re-normalisation, DBO→TCA fold, prebuilt
wheels, evidence-bundle export, aggregate gating, store schema versioning.

---

# Run 4 — Ship-plan Phase 0 (trust hygiene)

All six Phase-0 items implemented and acceptance-tested:
0.1 versions → 0.4.0 everywhere (test-enforced); 0.2 README repaired with a
README-claims pytest that immediately caught one more phantom path
(traces/latest.json); 0.3 _find_binary one-level-up fix (+regression test);
0.4 exit-code contract 0/1/2 with opt-in gating (+integration test);
0.5 hygiene (research log moved here, PRD removed, benchmarks/ merged into
benchmark/, publish.sh fixed, example paths fixed); 0.6 RESULTS.md +
external_agent_audits.md regenerated hermetically (run_benchmarks.py now
order-independent; verified byte-identical across runs) + CI drift gate.
Suite: 237 Rust + 242 Python green, clippy 0.

---

# Run 5 — Ship-plan Phase 1 (verdict precision)

Implemented 1.1–1.6 with the two adjudicated traces pinned as tests:
- SRR responsiveness rules (new-input / fail→retry / verification-after-
  mutation) + most-similar-prior fix (1.2, 1.1, 1.6).
- LDI verification awareness: state-hash chains restart after mutations;
  parametric chains split at mutations (marshmallow step 20 now KEEP).
- TraceStep::is_mutating() + diff guard: successful mutating calls never
  deleted (corpus invariant test).
- Fix risk classes (safe/needs_review/dangerous) + --force gate; apply emits
  directives only (no meta-prose); tool_schema fix diagnosed from tool_error.
- AGF tokenizer rewrite (syntax/markdown/glob rejection, creation params
  excluded, same-step ic as evidence, prose apostrophes).
Adjudication outcome: airline_task0 6/6 delete verdicts correct (was 1/6),
AGF ungrounded 0 artifacts (was 0/10 correct); marshmallow verification kept.
Corpus effects (all regenerated): 24-trace mean TAS 71.3→73.5; airline SRR
35.9→15.5%; AgentInstruct AGF 0.951. Suites: 241 Rust + 247 Python, clippy 0.

---

# Run 6 — Ship-plan Phase 2 (installable + ingestible)

2.1 Platform wheels: scripts/build_platform_wheel.sh bundles the CLI into
tracerazor/bin/; new console script + client prefer it. Clean-room verified
locally (env -i, fresh venv, no repo, no Rust): console script and Python API
both work from the wheel alone. release.yml builds linux+macos wheels with the
same smoke test. 2.2 LangSmith: flat exports rebuild the tree via
parent_run_id (all runs kept; was first-only), tokens from run-level /
llm_output/usage_metadata locations; golden fixtures. 2.3 OTel: protojson
string ints, prompt/completion token keys, content from events + structured
messages + OpenLLMetry indexed attrs; golden fixtures incl. degraded case.
2.4 IngestQuality in every manifest + loud stderr warning >50% zero-token or
placeholder content. 2.5 Fleet mode: audit <dir> → one aggregate (37/38
analysable over traces/external, mean 76.8, worst-5 all airline); threshold
gates the mean. Suites: 245 Rust + 247 Python, clippy 0.
