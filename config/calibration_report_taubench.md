# TAS Weight Calibration Report: `tb_wm`

- Samples: **233**
- Target: efficiency = `1 - recoverable_token_fraction`
- L2 ridge toward prior: `0.1`

## Fit quality (calibrated weights)

| | R² | Pearson r |
|---|---:|---:|
| Train (in-sample) | -0.086 | 0.137 |
| 5-fold cross-validated | -0.110 | 0.110 |
| Default weights (baseline) | -0.373 | 0.038 |

Cross-validated numbers are the honest estimate of generalisation; the
train/CV gap indicates over-fit. Beat the default-weights baseline to
justify recalibration.

## Calibrated weights

| Metric | Calibrated | Default |
|---|---:|---:|
| SRR | 0.0617 | 0.1700 |
| LDI | 0.0937 | 0.1300 |
| TCA | 0.0000 | 0.1300 |
| RDA | 0.2507 | 0.1000 |
| ISR | 0.0000 | 0.1000 |
| TUR | 0.0000 | 0.1000 |
| CCE | 0.0000 | 0.1000 |
| DBO | 0.4200 | 0.0900 |
| VDI | 0.0000 | 0.0800 |
| SHL | 0.0000 | 0.0500 |
| CCR | 0.1739 | 0.0300 |
| GAR | 0.0000 | 0.0700 |
| CSD | 0.0000 | 0.0500 |

_Weights sum to 1.0 (convex combination)._
