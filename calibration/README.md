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
> (`efficiency = 1 − recoverable_fraction`).

## 1. Provide ground truth

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

- `{"trace", "recoverable_fraction"}` — you supply the measured fraction
  directly (0–1).
- `{"before", "after"}` — the fraction is computed from measured token totals:
  `(before_tokens − after_tokens) / before_tokens`. The **before** run (the real,
  un-optimised trace) is the one that gets scored. This is the recommended form
  when you have measured before/after re-runs at constant task quality — e.g.
  from running your products against industry multi-agent solutions.

The key property: the target must come from **measurement**, not from
TraceRazor's own savings estimate (that would be circular).

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
calibrate on your own data — shipping synthetic-fit weights as the default would
just trade one arbitrary choice for another.

## Worked example (synthetic, reproducible)

`make_example_dataset.py` builds traces by injecting a *known* amount of
duplicate + verbose waste into clean base runs, so the recoverable fraction is
ground truth by construction (not derived from TraceRazor):

```bash
python -m calibration.make_example_dataset --out calibration/example_data --n 36
python -m calibration.calibrate --dataset calibration/example_data/manifest.json
```

On this synthetic set the calibrated weights concentrate on the metrics that
actually track the injected waste (SRR, VDI, LDI) and the cross-validated fit
beats the default weights by a wide margin (the defaults can even
*anti-correlate* with recoverable waste). See `config/calibration_report.md` for
the numbers. Swap in your real dataset to calibrate for production.
