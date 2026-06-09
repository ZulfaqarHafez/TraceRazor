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

## Worked example (synthetic, reproducible)

`make_example_dataset.py` builds traces by injecting a *known* amount of
duplicate + verbose waste into clean base runs, so the recoverable fraction is
ground truth by construction (not derived from TraceRazor):

```bash
python -m calibration.make_example_dataset --out calibration/example_data --n 36
python -m calibration.calibrate --dataset calibration/example_data/manifest.json
```

The default generator builds 200 traces with six categories of injected waste
(duplicate reasoning, tool loops, failed tool calls, verbose filler, hedging,
restated context). On this controlled set the calibrated weights reach
cross-validated `R^2 = 0.64` against recoverable waste while the heuristic
defaults reach only `R^2 = 0.09`, and the fit concentrates mass on the metrics
that track the injected waste (SRR and CCE). See `config/calibration_report.md`
for the full table. This validates the procedure; swap in your real measured
dataset to calibrate for production.

## Why the example is synthetic, not public data

A fair question is why we calibrate on synthetic traces instead of real data
pulled from the internet. The reason is the target, not the features. Auditing
runs on real public traces works fine, and the repo already audits 24 of them
(tau-bench, SWE-agent). Calibration needs something those datasets do not carry:
a measured recoverable-waste label per trace, ideally from a before/after re-run
of the same task at equal quality. Public agent datasets are not published as
matched lean/verbose pairs. In this repo, for instance, only one SWE-agent task
ships in multiple edit-format variants, and the tau-bench traces are different
tasks across different models, so neither gives matched pairs at any useful
scale. Manufacturing a label from TraceRazor's own savings estimate would be
circular and would defeat the point of calibration.

So the honest path is the one wired up here: the synthetic example proves the
mechanism with ground truth that is known by construction, and you supply real
measured before/after pairs (for example from running your products against
industry multi-agent baselines) to calibrate for production.
