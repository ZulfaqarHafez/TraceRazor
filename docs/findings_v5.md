# PRD v5 — Substitutability Classifier Study

Generated: 2026-05-24 16:57

## Objective

Predict whether a cached LLM response (`response_A`) is an adequate substitute
for a fresh response to a new prompt (`prompt_B`), saving one full LLM call per
correct positive prediction.

Pass criteria (pre-committed): **precision >= 80% AND recall >= 30%**
simultaneously at the same operating threshold.

## Dataset

| Split | Records | Positives | Negatives |
|-------|---------|-----------|-----------|
| Train | 126 | 64 | 62 |
| Test  | 60 | 30 | 30 |

Test set: records with `run_id` matching `run_v3`.

## Results

| Model | Tier | AUC-ROC | AUC-PR | Best P | Best R | Passes |
|-------|------|---------|--------|--------|--------|--------|
| logreg | scalar | 0.9856 | 0.9870 | 81.08% | 100.00% | YES |
| gbm | scalar | 0.9978 | 0.9979 | 81.08% | 100.00% | YES |

## Verdict

**PASS** — 2 configuration(s) satisfy both precision and recall criteria.

Recommended deployment config:
- Model: **logreg**, Tier: **scalar**, Threshold: **0.318**
- Expected savings: every substituted call eliminates one full LLM round-trip.

## Best Configuration (by AUC average)

**gbm / scalar**
- AUC-ROC: 0.9978
- AUC-PR: 0.9979

## Feature Tiers

| Tier | Features | Description |
|------|----------|-------------|
| emb | 3 | MiniLM-L6-v2 cosine similarities: cos(pA,pB), cos(rA,pB), cos(rA,pA) |
| scalar | 5 | Jaccard overlaps, length ratios, prefix match fraction |
| both | 8 | Concatenation of emb and scalar |

## Inference

```python
from redundancy.substitutability import build_features, load_labels
import pandas as pd

# Single pair
df = pd.DataFrame([{
    'prompt_A': '...',
    'response_A': '...',
    'prompt_B': '...',
}])
X = build_features(df, tier='both')
prob = model.pipeline.predict_proba(X)[0, 1]
substitutable = prob >= best_threshold
```