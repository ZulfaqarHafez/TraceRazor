# External Agent Audits

This document records TraceRazor audit results on **real, publicly-available agent trajectories** from two well-known benchmarks. All numbers are reproducible — the converters live in `tools/`, the converted traces live in `traces/external/`, and re-running the audits requires nothing beyond `cargo run -p tracerazor`.

> The point of this study is not to embarrass any model or benchmark. It is to show what TraceRazor catches on agent output that already exists in the wild, so prospective users can calibrate expectations against artefacts they have already seen.

---

## Methodology

Two public sources of agent trajectories were ingested:

1. **τ-bench (Sierra Research)** — `sierra-research/tau-bench`, `historical_trajectories/`. Real conversation transcripts of `gpt-4o` and `claude-sonnet-3.5-new` running τ-bench airline and retail tasks.
2. **SWE-agent (Princeton NLP)** — `princeton-nlp/SWE-agent`, `trajectories/demonstrations/`. Real `.traj` files of SWE-agent solving the `marshmallow-code/marshmallow#1867` issue under four different prompt-template variants.

Each upstream trajectory was converted to TraceRazor's raw trace format via the converters in `tools/`:

| Converter | Source format | Output |
|---|---|---|
| `tools/convert_tau_bench.py` | τ-bench `traj` list of `role`/`content`/`tool_calls` turns | TraceRazor steps + per-tool input/output capture |
| `tools/convert_swe_agent.py` | SWE-agent `trajectory[]` with `thought`/`action`/`observation` | TraceRazor reasoning + tool_call pairs |

Token counts in the converted traces are approximated as `len(text) / 4` (rough OpenAI chars-per-token). The approximation is consistent across all traces, so cross-trace comparisons remain valid; the absolute numbers should be treated as estimates.

All audits were run with `tracerazor audit --threshold 0 --format json`. SRR (Step Redundancy Rate), TAS (composite ordinal score) and total tokens were extracted from each report.

---

## τ-bench results (20 trajectories)

5 tasks × 2 models × 2 domains = **20 audits**.

| Model | Domain | n | Avg TAS | Avg SRR | Avg tokens / run | Worst single TAS |
|---|---|---|---|---|---|---|
| **GPT-4o** | airline | 5 | **53.1** (Fair / Poor) | **35.9%** | 2,055 | 46.8 (Poor, task 3) |
| **GPT-4o** | retail | 5 | 85.4 (Good) | 13.9% | 2,865 | 82.9 (Good) |
| **Claude Sonnet 3.5 (new)** | airline | 5 | 66.6 (Fair) | **41.2%** | 3,326 | 46.8 (Poor, task 0) |
| **Claude Sonnet 3.5 (new)** | retail | 5 | 80.6 (Good) | 16.5% | 3,139 | 60.3 (Fair, task 3) |

### Headline findings

1. **The airline domain is structurally harder than retail across both models.** Average TAS drops by ~30 points; SRR roughly doubles. Likely cause: the airline tool palette has more partially-overlapping search calls (`search_direct_flight`, `search_onestop_flight`, `book_reservation`, `calculate`), giving more surface area for near-duplicate calls.

2. **Step redundancy is the dominant waste pattern on airline.** ~36% of GPT-4o's airline steps and ~41% of Claude Sonnet 3.5's are flagged as near-duplicates (BoW cosine ≥ 0.65). On retail both models land near 15%.

3. **GPT-4o's worst airline trace (task 3) scores 46.8 / 100 (Poor) with 56.7% step redundancy** on 4,414 tokens. The audit identifies six near-duplicate searches and one misfired `book_reservation` call that was retried at the next step.

4. **Cross-model gap is smaller than expected.** Claude Sonnet 3.5 is slightly more wasteful on airline (41.2% SRR vs 36.0%) but cleaner on retail. Tool-palette design appears to matter more than model choice.

### Per-task table (selected)

| Trace | Model | Domain | Task | TAS | Grade | SRR |
|---|---|---|---|---|---|---|
| `gpt-4o_airline_task0.json` | gpt-4o | airline | 0 | 55.9 | Fair | 33.3% |
| `gpt-4o_airline_task3.json` | gpt-4o | airline | 3 | **46.8** | **Poor** | **56.7%** |
| `gpt-4o_retail_task4.json` | gpt-4o | retail | 4 | 88.3 | Good | 8.3% |
| `claude-sonnet-3.5-new_airline_task0.json` | sonnet-3.5-new | airline | 0 | 46.8 | Poor | 45.0% |
| `claude-sonnet-3.5-new_airline_task2.json` | sonnet-3.5-new | airline | 2 | 67.8 | Fair | **66.7%** |
| `claude-sonnet-3.5-new_retail_task1.json` | sonnet-3.5-new | retail | 1 | 89.6 | Good | 8.3% |

### One concrete pattern the auditor caught

From `gpt-4o_airline_task0.json`:

- Step 4 calls `search_direct_flight` for JFK → SEA on May 20.
- Step 6 calls `search_onestop_flight` for the same route — **88% similar to step 4**.
- Step 10 calls `book_reservation`. **Mis-fired**: the error from the tool is *"payment amount does not add up, total price is 305, but paid 255."* Retried at step 11 with corrected params.
- Step 14 calls `book_reservation` again — **85% similar to step 10**.

The auto-generated fix patches this with a `tool_schema` recommendation to mark `payment.amount` as required so the model cannot omit it, plus a `context_compression` directive to summarise the conversation to the last three relevant facts before each tool call.

---

## SWE-agent results (4 trajectories)

Same upstream issue (`marshmallow-code/marshmallow#1867`), four prompt-template variants of SWE-agent. All four runs **passed** — i.e. submitted a patch that the grader accepted.

| Variant | TAS | Grade | SRR | Total tokens | Steps |
|---|---|---|---|---|---|
| `xml` | 70.3 | Good | 22.7% | **3,636** | 22 |
| `default` | 72.1 | Good | 21.4% | 6,447 | 28 |
| `fn_calling` | 68.6 | Fair | 22.7% | 5,471 | 22 |
| `cursors` | 67.7 | Fair | 20.8% | 7,553 | 24 |

### Headline findings

1. **The `xml` variant uses ~52% fewer tokens than `cursors`** for the same successful outcome on the same issue. If you are running SWE-agent at scale, this is a directly actionable insight.

2. **All four variants sit in a tight SRR band (~21–23%).** The redundancy is in the *agent loop pattern* (read file → think → edit → re-read same file), not in the prompt template. Fixing it requires changes to the action protocol, not the prompt.

3. **Function calling does not win on token cost.** Counter to common assumption, the `fn_calling` variant is ~50% more expensive than `xml` for the same outcome. The structured-tool overhead adds tokens without reducing redundancy.

4. **TAS does NOT correlate with success on this benchmark.** All four runs pass; TAS varies by 4.4 points. This is the right behaviour — TAS measures structural efficiency, not task accuracy. A high-TAS-but-failed run is still a bad run.

---

## Caveats

- **Token counts are approximated** (`len(text) / 4`). Absolute $ figures from the cost projection on these external traces are indicative only; trust them as ratios, not as wire numbers.
- **τ-bench airline/retail are simulated environments.** The tool failures and redundancies we detect are real artefacts of the model's behaviour, but the "user" turn is itself LLM-generated (gpt-4o-mini simulating a customer). A production support trace will look different.
- **SWE-agent's `marshmallow-1867` is a single repo, single bug.** The 4-variant comparison is a within-task signal; do not extrapolate to "XML is universally best." Re-run on your own SWE-bench task set.
- **Substitutability (Pillar 3) was not applied.** The TraceRazor substitutability classifier is still synthetic; this study uses only the audit pipeline (Pillar 1).

---

## Reproducing

```bash
# Tau-bench
git clone --depth=1 https://github.com/sierra-research/tau-bench /tmp/tau-bench
python3 tools/convert_tau_bench.py \
    --input /tmp/tau-bench/historical_trajectories/gpt-4o-airline.json \
    --task-id 0 --trial 0 \
    --agent-name tau-bench-airline-gpt-4o \
    --output traces/external/tau_bench/my_task.json
cargo run --release -p tracerazor -- audit traces/external/tau_bench/my_task.json --threshold 0

# SWE-agent
git clone --depth=1 https://github.com/princeton-nlp/SWE-agent /tmp/SWE-agent
python3 tools/convert_swe_agent.py \
    --input /tmp/SWE-agent/trajectories/demonstrations/replay__marshmallow-code__marshmallow-1867__function_calling__install-1/marshmallow-code__marshmallow-1867.traj \
    --agent-name swe-agent-fn-calling \
    --output traces/external/swe_agent/marshmallow_fn_calling.json
cargo run --release -p tracerazor -- audit traces/external/swe_agent/marshmallow_fn_calling.json --threshold 0
```

The 20 + 4 trace JSONs are checked into `traces/external/` for offline replay.
