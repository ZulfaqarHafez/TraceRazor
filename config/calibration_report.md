# TAS Weight Calibration Report: `tb`

- Samples: **233**
- Target: efficiency = `1 - recoverable_token_fraction`
- L2 ridge toward prior: `0.1`

## Fit quality (calibrated weights)

| | R² | Pearson r |
|---|---:|---:|
| Train (in-sample) | 0.104 | 0.326 |
| 5-fold cross-validated | 0.083 | 0.288 |
| Default weights (baseline) | -0.244 | 0.077 |

Cross-validated numbers are the honest estimate of generalisation; the
train/CV gap indicates over-fit. Beat the default-weights baseline to
justify recalibration.

## Calibrated weights

| Metric | Calibrated | Default |
|---|---:|---:|
| SRR | 0.0000 | 0.1700 |
| LDI | 0.0000 | 0.1300 |
| TCA | 0.1292 | 0.1300 |
| RDA | 0.1018 | 0.1000 |
| ISR | 0.0000 | 0.1000 |
| TUR | 0.0323 | 0.1000 |
| CCE | 0.0000 | 0.1000 |
| DBO | 0.1340 | 0.0900 |
| VDI | 0.0000 | 0.0800 |
| SHL | 0.0251 | 0.0500 |
| CCR | 0.0979 | 0.0300 |
| GAR | 0.0815 | 0.0700 |
| CSD | 0.0657 | 0.0500 |
| OBS | 0.3325 | 0.0600 |

_Weights sum to 1.0 (convex combination)._
