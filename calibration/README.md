# TAS Weight Calibration

TraceRazor's composite efficiency is a convex combination of 13 sub-metrics:

```
raw_efficiency = Σ_k (w_k · m_k) / Σ_k w_k        # m_k ∈ [0,1]
TAS            = 100 · raw_efficiency · (0.7 + 0.3 · task_value)
```

By default the weights `w_k` are **author-chosen heuristics**. This directory
replaces them with weights that are **fit on data**, so the score becomes
"calibrated on dataset X" with a reported fit quality rather than an arbitrary
choice. The calibration objective is the honest one for our use case:

> make `raw_efficiency` predict the **measured recoverable token waste** of a run
> (`efficiency = 1 - recoverable_fraction`).

## 1. Provide ground truth

For the exact data spec (what a before/after pair is, how much to provide, the
trace schema, and a runnable two-pair example under `calibration/template/`),
see [DATA_TEMPLATE.md](DATA_TEMPLATE.md). In short:

Create a dataset manifest. Two entry forms are supported:

```json
{
  "name": "industry-multiagent-2026q2",
  "entries": [
    {"trace": "runs/agent_01.json", "recoverable_fraction": 0.42},
    {"before": "runs/verbose_02.json", "after": "runs/lean_02.json"}
  ]
}
```

- `{"trace", "recoverable_fraction"}`, you supply the measured fraction
  directly (0-1).
- `{"before", "after"}`, the fraction is computed from measured token totals:
  `(before_tokens - after_tokens) / before_tokens`. The **before** run (the real,
  un-optimised trace) is the one that gets scored. This is the recommended form
  when you have measured before/after re-runs at constant task quality, e.g.
  from running your products against industry multi-agent solutions.

The key property: the target must come from **measurement**, not from
TraceRazor's own savings estimate (that would be circular).

### Import adapter (build the manifest from your harness export)

You usually do not hand-write the manifest. `adapt.py` converts the shapes a
harness typically emits into one, with configurable field names so it fits your
columns. Trace files may be raw, LangSmith, or OTEL JSON; the auditor detects
the format, and before/after fractions are computed from the measured token
totals of each run, so this works across formats.

```bash
# CSV or JSONL, before/after pairs:
python -m calibration.adapt --csv runs.csv \
  --before-col before_path --after-col after_path --out manifest.json

# CSV or JSONL, traces with a measured recoverable fraction:
python -m calibration.adapt --jsonl runs.jsonl \
  --trace-col path --label-col recoverable_fraction --out manifest.json

# A directory of files paired by name:
python -m calibration.adapt --pair-dir runs/ \
  --before-suffix _before.json --after-suffix _after.json --out manifest.json
```

Paths are resolved (relative to the source file's directory, or `--base-dir`)
and written as absolute paths. Use `--allow-missing` to skip rows whose files
are absent. If your export has a different shape, point the `*-col` flags at your
field names; if it still does not fit, send a sample and the adapter can grow a
mode for it.

## 2. Fit the weights

```bash
pip install -e ".[calibrate]"          # numpy + scipy
cargo build --release -p tracerazor    # the auditor the tool calls

python -m calibration.calibrate \
  --dataset path/to/manifest.json \
  --out config/tas_weights.calibrated.json \
  --report config/calibration_report.md \
  --cv 5 --l2 0.0
```

The tool audits each trace, reads the per-metric `metric_normalised` values the
engine actually uses, and fits non-negative weights summing to 1 that minimise
prediction error against the target. It reports **train R², cross-validated R²,
and the default-weights baseline** so you can see whether recalibration is
justified and how much it over-fits. Use `--l2 <λ>` to regularise toward uniform
weights when the dataset is small or narrow (prevents the fit from dumping all
mass on one or two metrics).

## 3. Use the calibrated weights

```bash
tracerazor audit run.json --weights config/tas_weights.calibrated.json
# or globally:
export TRACERAZOR_WEIGHTS=config/tas_weights.calibrated.json
tracerazor audit run.json
```

The default built-in weights are **left unchanged** in the engine until you
calibrate on your own data, shipping synthetic-fit weights as the default would
just trade one arbitrary choice for another.

## Worked examples (real data)

Two real, reproducible examples ship with the repo:

**1. In-repo, no network** — the SWE-agent edit-format variants (the same
SWE-bench task solved at different token cost, leanest = `xml`):

```bash
python -m calibration.calibrate \
  --dataset calibration/examples/swe_agent_pairs.json \
  --out config/tas_weights.json --report config/calibration_report.md \
  --cv 3 --prior default --l2 0.1
```

**2. Larger, from GitHub** — tau-bench real trajectories (233 within-agent
before/after pairs):

```bash
git clone --depth 1 https://github.com/sierra-research/tau-bench
python -m calibration.sources.from_taubench \
  --dir tau-bench/historical_trajectories --out taubench.jsonl --within-model
python -m calibration.sources.from_messages --jsonl taubench.jsonl \
  --out-dir converted --manifest manifest.json
python -m calibration.calibrate --dataset manifest.json \
  --out config/tas_weights.json --report config/calibration_report.md \
  --cv 5 --prior default --l2 0.1 --features
```

On the tau-bench pairs the calibrated cross-validated `R^2` is about `+0.08`
against recoverable waste (default-weights baseline negative), and the
`--features` flag shows the observation-accumulation signals that drove the
improvement (see the paper and `config/calibration_report.md`). Swap in your own
measured before/after pairs to calibrate for production.

## Why real before/after pairs (not synthetic)

Calibration needs a measured recoverable-waste label per trace, ideally from a
before/after re-run of the same task at equal quality. The examples above get
that from real runs: SWE-agent solves one task under several edit formats, and
tau-bench solves each task across models and repeated trials, so successful runs
of the same task at differing token cost form genuine pairs. Manufacturing a
label from TraceRazor's own savings estimate would be circular, which is why we
calibrate on measured token deltas from real runs.
