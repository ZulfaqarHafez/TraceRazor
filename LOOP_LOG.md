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
