"""
Substitutability Classifier (PRD v5)

Predicts whether a cached LLM response (response_A, produced for prompt_A)
is an adequate substitute for a fresh response to a new prompt (prompt_B).

A correct substitute saves a full LLM call; an incorrect one silently corrupts
the agent trajectory.  The classifier must satisfy simultaneously:
    precision >= 0.80  (few false "safe to reuse" verdicts)
    recall    >= 0.30  (catches at least 30 % of reusable pairs)

Data format — judge_transcripts.jsonl (one JSON object per line):
    {
        "run_id":     str,   -- source run identifier
        "task_index": int,   -- tau-bench task index
        "turn":       int,   -- 0-based turn within the trajectory
        "prompt_A":   str,   -- context that produced response_A
        "response_A": str,   -- cached response from run_A
        "prompt_B":   str,   -- new context in run_B
        "label":      int,   -- 1 = substitutable, 0 = not substitutable
    }

Usage:
    # Generate synthetic data first (requires ANTHROPIC_API_KEY)
    python -m redundancy.generate_data --n 300

    # Train and evaluate
    python -m redundancy.substitutability train --tier emb
    python -m redundancy.substitutability train --tier scalar
    python -m redundancy.substitutability train --tier both
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent


# ── data loading ──────────────────────────────────────────────────────────────


@dataclass
class JudgeRecord:
    run_id: str
    task_index: int
    turn: int
    prompt_A: str
    response_A: str
    prompt_B: str
    label: int  # 1 = substitutable, 0 = not


def load_labels(results_dir: Path) -> pd.DataFrame:
    """Walk ``results_dir/run_*/judge_transcripts.jsonl`` and return a DataFrame.

    Falls back to loading from any ``judge_transcripts.jsonl`` found anywhere
    under ``results_dir`` if the ``run_*/`` glob yields nothing.

    Columns: run_id, task_index, turn, prompt_A, response_A, prompt_B, label
    """
    records: List[Dict[str, Any]] = []

    for jsonl_path in sorted(results_dir.rglob("judge_transcripts.jsonl")):
        run_id = jsonl_path.parent.name  # e.g. "run_k1_0513"
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj.setdefault("run_id", run_id)
                records.append(obj)

    if not records:
        raise FileNotFoundError(
            f"No judge_transcripts.jsonl found under {results_dir}.\n"
            "Generate synthetic data first:\n"
            "  python -m redundancy.generate_data --n 300"
        )

    df = pd.DataFrame(records)
    required = {"prompt_A", "response_A", "prompt_B", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in judge_transcripts: {missing}")
    df["label"] = df["label"].astype(int)
    return df


# ── feature engineering ───────────────────────────────────────────────────────


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _jaccard(s1: str, s2: str) -> float:
    t1, t2 = set(s1.lower().split()), set(s2.lower().split())
    union = t1 | t2
    return len(t1 & t2) / len(union) if union else 0.0


def build_features(df: pd.DataFrame, tier: str) -> np.ndarray:
    """Build the feature matrix for training or inference.

    Parameters
    ----------
    df:    DataFrame with columns prompt_A, response_A, prompt_B.
    tier:  "emb"    — MiniLM sentence embeddings (cosine similarities)
           "scalar" — hand-crafted scalar features
           "both"   — concatenation of emb and scalar features
    """
    if tier not in ("emb", "scalar", "both"):
        raise ValueError(f"tier must be 'emb', 'scalar', or 'both', got {tier!r}")

    scalar_feats: Optional[np.ndarray] = None
    emb_feats: Optional[np.ndarray] = None

    if tier in ("scalar", "both"):
        scalar_feats = _build_scalar(df)

    if tier in ("emb", "both"):
        emb_feats = _build_embedding(df)

    if tier == "emb":
        return emb_feats
    if tier == "scalar":
        return scalar_feats
    return np.hstack([emb_feats, scalar_feats])


def _build_scalar(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, row in df.iterrows():
        pA, rA, pB = str(row.prompt_A), str(row.response_A), str(row.prompt_B)

        # Jaccard overlap between prompts
        jac_prompts = _jaccard(pA, pB)

        # Relative length of prompt_B vs prompt_A
        len_ratio = len(pB) / max(len(pA), 1)

        # Response length relative to prompt_B length
        resp_rel_len = len(rA) / max(len(pB), 1)

        # Jaccard overlap between response_A and prompt_B
        jac_resp_pB = _jaccard(rA, pB)

        # Character-level prefix match fraction
        common_prefix = 0
        for a, b in zip(pA.lower(), pB.lower()):
            if a == b:
                common_prefix += 1
            else:
                break
        prefix_frac = common_prefix / max(len(pA), len(pB), 1)

        rows.append([jac_prompts, len_ratio, resp_rel_len, jac_resp_pB, prefix_frac])

    return np.array(rows, dtype=np.float32)


def _build_embedding(df: pd.DataFrame) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for tier='emb'.\n"
            "Install with: pip install sentence-transformers"
        )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    pA_texts = df["prompt_A"].astype(str).tolist()
    rA_texts = df["response_A"].astype(str).tolist()
    pB_texts = df["prompt_B"].astype(str).tolist()

    pA_embs = model.encode(pA_texts, show_progress_bar=False)
    rA_embs = model.encode(rA_texts, show_progress_bar=False)
    pB_embs = model.encode(pB_texts, show_progress_bar=False)

    n = len(df)
    feats = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        # prompt similarity
        feats[i, 0] = _cosine(pA_embs[i], pB_embs[i])
        # response-to-new-prompt relevance
        feats[i, 1] = _cosine(rA_embs[i], pB_embs[i])
        # response-to-old-prompt relevance (sanity anchor)
        feats[i, 2] = _cosine(rA_embs[i], pA_embs[i])

    return feats


# ── training ──────────────────────────────────────────────────────────────────


@dataclass
class TrainedModel:
    tier: str
    model_tier: str  # "logreg" or "gbm"
    pipeline: Any
    feature_names: List[str]


def train(df_train: pd.DataFrame, tier: str) -> Tuple[TrainedModel, TrainedModel]:
    """Train both tier-1 (LogReg) and tier-2 (GBM) on ``df_train``.

    Returns (logreg_model, gbm_model).
    """
    X = build_features(df_train, tier)
    y = df_train["label"].to_numpy()

    feat_names = _feature_names(tier, X.shape[1])

    logreg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)),
    ])
    logreg_pipe.fit(X, y)

    gbm_clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    gbm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", gbm_clf),
    ])
    gbm_pipe.fit(X, y)

    logreg = TrainedModel(tier=tier, model_tier="logreg", pipeline=logreg_pipe, feature_names=feat_names)
    gbm = TrainedModel(tier=tier, model_tier="gbm", pipeline=gbm_pipe, feature_names=feat_names)
    return logreg, gbm


def _feature_names(tier: str, n: int) -> List[str]:
    scalar = ["jac_prompts", "len_ratio", "resp_rel_len", "jac_resp_pB", "prefix_frac"]
    emb = ["cos_pA_pB", "cos_rA_pB", "cos_rA_pA"]
    if tier == "scalar":
        return scalar[:n]
    if tier == "emb":
        return emb[:n]
    return (emb + scalar)[:n]


# ── evaluation ────────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    model_tier: str
    tier: str
    auc_roc: float
    auc_pr: float
    # Threshold that maximises recall subject to precision >= 0.80
    best_threshold: Optional[float]
    precision_at_threshold: Optional[float]
    recall_at_threshold: Optional[float]
    passes: bool  # precision >= 0.80 AND recall >= 0.30


def evaluate(
    model: TrainedModel,
    df_test: pd.DataFrame,
    precision_floor: float = 0.80,
    recall_floor: float = 0.30,
) -> EvalResult:
    """Evaluate ``model`` on ``df_test``, reporting AUC and threshold metrics."""
    X_test = build_features(df_test, model.tier)
    y_true = df_test["label"].to_numpy()

    y_score = model.pipeline.predict_proba(X_test)[:, 1]

    auc_roc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.0
    auc_pr = float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    best_threshold: Optional[float] = None
    best_precision: Optional[float] = None
    best_recall: Optional[float] = None

    # Find the threshold with highest recall while keeping precision >= floor
    best_recall_val = -1.0
    for prec, rec, thr in zip(precisions, recalls, thresholds):
        if prec >= precision_floor and rec >= recall_floor:
            if rec > best_recall_val:
                best_recall_val = rec
                best_threshold = float(thr)
                best_precision = float(prec)
                best_recall = float(rec)

    passes = best_threshold is not None

    return EvalResult(
        model_tier=model.model_tier,
        tier=model.tier,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        best_threshold=best_threshold,
        precision_at_threshold=best_precision,
        recall_at_threshold=best_recall,
        passes=passes,
    )


def decide(result: EvalResult) -> str:
    """Return human-readable verdict on whether the model passes the criteria."""
    if result.passes:
        return (
            f"PASS — {result.model_tier}/{result.tier}: "
            f"precision={result.precision_at_threshold:.2%}, "
            f"recall={result.recall_at_threshold:.2%} "
            f"(threshold={result.best_threshold:.3f})"
        )
    else:
        return (
            f"FAIL — {result.model_tier}/{result.tier}: "
            f"no threshold achieves precision>=80% AND recall>=30% simultaneously. "
            f"AUC-ROC={result.auc_roc:.3f}, AUC-PR={result.auc_pr:.3f}"
        )


# ── split logic ───────────────────────────────────────────────────────────────


def split_by_run(df: pd.DataFrame, test_run_pattern: str = "run_v3") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (train, test) by source run.

    Records whose ``run_id`` contains ``test_run_pattern`` go to the test set;
    the rest go to the training set.  This mirrors the PRD v5 protocol:
    v2.2 first-turn runs → train, v3 multi-turn run → test.
    """
    mask = df["run_id"].str.contains(test_run_pattern, na=False)
    df_test = df[mask].copy()
    df_train = df[~mask].copy()
    if df_test.empty:
        # Fallback: 80/20 random split if no test-run labels found
        idx = df.sample(frac=0.2, random_state=42).index
        df_test = df.loc[idx].copy()
        df_train = df.drop(idx).copy()
    return df_train, df_test


# ── CLI ───────────────────────────────────────────────────────────────────────


def _print_result(result: EvalResult) -> None:
    print(f"\n  [{result.model_tier.upper():6} / {result.tier}]")
    print(f"    AUC-ROC : {result.auc_roc:.4f}")
    print(f"    AUC-PR  : {result.auc_pr:.4f}")
    print(f"    Verdict : {decide(result)}")


def _run_train(args: Any) -> None:
    results_dir = Path(args.results_dir)
    print(f"Loading labels from {results_dir} ...")
    df = load_labels(results_dir)
    print(f"  {len(df)} records  |  positives={df['label'].sum()}  negatives={(df['label']==0).sum()}")

    df_train, df_test = split_by_run(df, test_run_pattern=args.test_run)
    print(f"  train={len(df_train)}  test={len(df_test)}")

    tiers = ["emb", "scalar", "both"] if args.tier == "all" else [args.tier]
    all_results: List[EvalResult] = []

    for tier in tiers:
        print(f"\nTraining tier={tier} ...")
        logreg, gbm = train(df_train, tier)

        logreg_result = evaluate(logreg, df_test)
        gbm_result = evaluate(gbm, df_test)

        all_results.extend([logreg_result, gbm_result])
        _print_result(logreg_result)
        _print_result(gbm_result)

    print("\n" + "=" * 60)
    passed = [r for r in all_results if r.passes]
    if passed:
        print(f"SUMMARY: {len(passed)}/{len(all_results)} configurations PASS the 80/30 criteria.")
        for r in passed:
            print(f"  {r.model_tier}/{r.tier}  precision={r.precision_at_threshold:.2%}  recall={r.recall_at_threshold:.2%}")
    else:
        print(f"SUMMARY: 0/{len(all_results)} configurations pass the 80/30 criteria.")
        print("  Classifier not ready for deployment. See docs/findings_v5.md for next steps.")

    _write_findings(all_results, df_train, df_test, args)


def _write_findings(
    results: List[EvalResult],
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    args: Any,
) -> None:
    from datetime import datetime

    findings_path = _REPO / "docs" / "findings_v5.md"
    findings_path.parent.mkdir(parents=True, exist_ok=True)

    passed = [r for r in results if r.passes]
    best = max(results, key=lambda r: (r.auc_roc + r.auc_pr) / 2) if results else None

    lines = [
        "# PRD v5 — Substitutability Classifier Study",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Objective",
        "",
        "Predict whether a cached LLM response (`response_A`) is an adequate substitute",
        "for a fresh response to a new prompt (`prompt_B`), saving one full LLM call per",
        "correct positive prediction.",
        "",
        "Pass criteria (pre-committed): **precision >= 80% AND recall >= 30%**",
        "simultaneously at the same operating threshold.",
        "",
        "## Dataset",
        "",
        f"| Split | Records | Positives | Negatives |",
        f"|-------|---------|-----------|-----------|",
        f"| Train | {len(df_train)} | {int(df_train['label'].sum())} | {int((df_train['label']==0).sum())} |",
        f"| Test  | {len(df_test)} | {int(df_test['label'].sum())} | {int((df_test['label']==0).sum())} |",
        "",
        f"Test set: records with `run_id` matching `{args.test_run}`.",
        "",
        "## Results",
        "",
        "| Model | Tier | AUC-ROC | AUC-PR | Best P | Best R | Passes |",
        "|-------|------|---------|--------|--------|--------|--------|",
    ]

    for r in results:
        p_str = f"{r.precision_at_threshold:.2%}" if r.precision_at_threshold is not None else "n/a"
        r_str = f"{r.recall_at_threshold:.2%}" if r.recall_at_threshold is not None else "n/a"
        pass_str = "YES" if r.passes else "no"
        lines.append(
            f"| {r.model_tier} | {r.tier} | {r.auc_roc:.4f} | {r.auc_pr:.4f} | {p_str} | {r_str} | {pass_str} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
    ]

    if passed:
        lines.append(
            f"**PASS** — {len(passed)} configuration(s) satisfy both precision and recall criteria."
        )
        lines.append("")
        lines.append("Recommended deployment config:")
        best_pass = max(passed, key=lambda r: (r.recall_at_threshold or 0))
        lines.append(
            f"- Model: **{best_pass.model_tier}**, Tier: **{best_pass.tier}**, "
            f"Threshold: **{best_pass.best_threshold:.3f}**"
        )
        lines.append(
            f"- Expected savings: every substituted call eliminates one full LLM round-trip."
        )
    else:
        lines += [
            "**FAIL** — No configuration satisfies both precision >= 80% and recall >= 30%",
            "simultaneously.",
            "",
            "### Failure Analysis",
            "",
            "The classifier faces a precision-recall trade-off that the current feature set",
            "cannot resolve simultaneously.  Likely causes:",
            "",
            "1. **Insufficient data** — synthetic pairs may not capture real distribution shift",
            "   between run_A and run_B trajectories.",
            "2. **Feature expressiveness** — scalar features lack the context needed to detect",
            "   subtle semantic mismatches.  Try tier=both.",
            "3. **Label noise** — synthetic labels generated via embedding threshold may",
            "   be imprecise.  Consider human-labelled or LLM-judged ground truth.",
            "",
            "### Next Steps",
            "",
            "- Increase training data: `python -m redundancy.generate_data --n 1000`",
            "- Capture real judge transcripts from live tau-bench runs",
            "- Add structured features: tool name match, flight date proximity, passenger count equality",
        ]

    if best:
        lines += [
            "",
            "## Best Configuration (by AUC average)",
            "",
            f"**{best.model_tier} / {best.tier}**",
            f"- AUC-ROC: {best.auc_roc:.4f}",
            f"- AUC-PR: {best.auc_pr:.4f}",
        ]

    lines += [
        "",
        "## Feature Tiers",
        "",
        "| Tier | Features | Description |",
        "|------|----------|-------------|",
        "| emb | 3 | MiniLM-L6-v2 cosine similarities: cos(pA,pB), cos(rA,pB), cos(rA,pA) |",
        "| scalar | 5 | Jaccard overlaps, length ratios, prefix match fraction |",
        "| both | 8 | Concatenation of emb and scalar |",
        "",
        "## Inference",
        "",
        "```python",
        "from redundancy.substitutability import build_features, load_labels",
        "import pandas as pd",
        "",
        "# Single pair",
        "df = pd.DataFrame([{",
        "    'prompt_A': '...',",
        "    'response_A': '...',",
        "    'prompt_B': '...',",
        "}])",
        "X = build_features(df, tier='both')",
        "prob = model.pipeline.predict_proba(X)[0, 1]",
        "substitutable = prob >= best_threshold",
        "```",
    ]

    findings_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFindings written to {findings_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Substitutability classifier — PRD v5")
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="Train and evaluate the classifier")
    tr.add_argument(
        "--tier",
        choices=["emb", "scalar", "both", "all"],
        default="both",
        help="Feature tier (default: both).  Use 'all' to sweep all three.",
    )
    tr.add_argument(
        "--results-dir",
        default=str(_REPO / "results"),
        help="Root directory containing run_*/judge_transcripts.jsonl",
    )
    tr.add_argument(
        "--test-run",
        default="run_v3",
        help="Substring that identifies the held-out test run in run_id",
    )

    args = parser.parse_args()
    if args.command == "train":
        _run_train(args)


if __name__ == "__main__":
    main()
