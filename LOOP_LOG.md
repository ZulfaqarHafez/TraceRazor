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
