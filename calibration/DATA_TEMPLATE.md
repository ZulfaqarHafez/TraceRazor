# Data template: what to give me for production calibration

You chose to calibrate on real data. This is the spec for that data. Once you
drop files matching this in, calibration produces balanced weights across all 13
metrics and the paper's table becomes a production result instead of a controlled
one.

## What a sample is

One sample is a **before/after pair for the same task at equal quality**:

- **before**: the real, un-optimised run (more tokens).
- **after**: a leaner run of the *same task that still succeeded* (fewer tokens).

The recoverable fraction is computed for you as
`(before_tokens - after_tokens) / before_tokens`. The "equal quality" part is
what makes the label honest: both runs must have completed the task, so the token
difference is genuinely recoverable waste, not quality you gave up. Set
`task_value_score = 1.0` on both (or the same value), and only pair runs that
actually passed.

If you already have a measured recoverable fraction per run from your own
analysis, you can skip pairing and give a single trace plus that number instead.

## How much

Aim for **50 to 200 pairs**, spanning a range of waste levels (some near 0%
recoverable, some high) and ideally more than one agent/domain. The more the
metrics vary independently across samples, the more of the 13 metrics get real
weight (this is exactly what the synthetic data could not provide).

## Trace format

Any format the auditor reads: TraceRazor raw JSON, LangSmith export, or OTEL
JSON. The raw schema is simplest:

```json
{
  "trace_id": "run-001",
  "agent_name": "support-agent",
  "framework": "raw",
  "task_value_score": 1.0,
  "steps": [
    {"id": 1, "step_type": "reasoning", "content": "...", "tokens": 320},
    {"id": 2, "step_type": "tool_call", "content": "...", "tokens": 110,
     "tool_name": "get_order", "tool_success": true}
  ]
}
```

Minimum 5 steps per trace (shorter traces are skipped). `content` matters: the
redundancy, novelty, and drift metrics compare step text, so keep the real text
if you can. If it is sensitive, replace each distinct string with a stable
pseudonym (same input string maps to the same token) rather than blanking it,
otherwise the similarity metrics lose signal.

## How to hand it over

Pick whichever is easiest for your harness:

1. **A directory of paired files** named `<task>_before.json` / `<task>_after.json`:
   ```bash
   python -m calibration.adapt --pair-dir runs/ \
     --before-suffix _before.json --after-suffix _after.json --out manifest.json
   ```
2. **A CSV/JSONL of paths** (column names are yours to choose):
   ```bash
   python -m calibration.adapt --csv runs.csv \
     --before-col before_path --after-col after_path --out manifest.json
   ```
3. **A CSV/JSONL of traces with a measured fraction**:
   ```bash
   python -m calibration.adapt --jsonl runs.jsonl \
     --trace-col path --label-col recoverable_fraction --out manifest.json
   ```

See `calibration/template/` for a runnable two-pair example. Then:

```bash
python -m calibration.calibrate --dataset manifest.json \
  --out config/tas_weights.json --report config/calibration_report.md \
  --prior default --l2 0.1
```

`--prior default --l2 0.1` shrinks gently toward the current weights, which keeps
sane behaviour when a metric is under-represented in your sample. Drop the prior
flags for a pure fit once you have enough data.

## Using public trajectory datasets

If you would rather calibrate on public agent runs than your own, the right
sources are multi-config trajectory datasets where the same task is solved by
several models/scaffolds with a resolved/correct label, which gives before/after
pairs (verbose run vs lean run, both successful):

- `nebius/SWE-agent-trajectories` (80k runs, many models on shared SWE-bench
  instances, with correctness)
- `zai-org/CC-Bench-trajectories` (74 tasks, all models, full trajectories)
- `SWE-bench/SWE-smith-trajectories`, `open-thoughts/AgentTrove`

These live on Hugging Face. **This Claude Code environment's network policy must
allow Hugging Face for the download to work** (by default only PyPI, apt, and
GitHub are reachable; `huggingface.co` returns "Host not in allowlist"). Allow
these hosts on the environment, or download elsewhere and copy the file in:

```
huggingface.co
cdn-lfs.huggingface.co
cdn-lfs-us-1.huggingface.co
datasets-server.huggingface.co
```

Then dump the split to JSONL and run the connector (which converts the standard
OpenAI/ShareGPT messages format into traces and pairs same-instance resolved
runs):

```bash
pip install -e ".[calibrate]" && pip install datasets
python - <<'PY'
from datasets import load_dataset
load_dataset("nebius/SWE-agent-trajectories", split="train").to_json("traj.jsonl")
PY
python -m calibration.sources.from_messages --jsonl traj.jsonl \
  --out-dir calibration/converted --manifest manifest.json \
  --id-field instance_id --model-field model --resolved-field resolved --messages-field messages
python -m calibration.calibrate --dataset manifest.json \
  --out config/tas_weights.json --report config/calibration_report.md --prior default --l2 0.1
```

Field names (`--id-field` etc.) are adjustable to match the dataset's columns.
The connector and the full pipeline are tested end to end on the messages
format; the only thing they need is the data file present locally.

## Fastest way to unblock me

If exporting is involved, just paste **one** of your harness's records (one CSV
header + row, or one JSON line). I will confirm the exact adapter flags, or add a
mode for your shape, and tell you precisely what to dump.
