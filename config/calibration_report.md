# TAS Weight Calibration Report — `synthetic-worked-example`

- Samples: **36**
- Target: efficiency = `1 - recoverable_token_fraction`
- L2 regularisation toward uniform: `0.0`

## Fit quality (calibrated weights)

| | R² | Pearson r |
|---|---:|---:|
| Train (in-sample) | 0.386 | 0.842 |
| 5-fold cross-validated | 0.315 | 0.821 |
| Default weights (baseline) | -0.326 | 0.836 |

Cross-validated numbers are the honest estimate of generalisation; the
train/CV gap indicates over-fit. Beat the default-weights baseline to
justify recalibration.

## Calibrated weights

| Metric | Calibrated | Default |
|---|---:|---:|
| SRR | 0.6691 | 0.1700 |
| LDI | 0.1012 | 0.1300 |
| TCA | 0.0000 | 0.1300 |
| RDA | 0.0000 | 0.1000 |
| ISR | 0.0000 | 0.1000 |
| TUR | 0.0000 | 0.1000 |
| CCE | 0.0000 | 0.1000 |
| DBO | 0.0000 | 0.0900 |
| VDI | 0.2298 | 0.0800 |
| SHL | 0.0000 | 0.0500 |
| CCR | 0.0000 | 0.0300 |
| GAR | 0.0000 | 0.0700 |
| CSD | 0.0000 | 0.0500 |

_Weights sum to 1.0 (convex combination)._
