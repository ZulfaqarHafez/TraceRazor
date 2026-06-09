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
