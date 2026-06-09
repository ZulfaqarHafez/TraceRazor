#!/usr/bin/env python3
"""Calibrate TAS weights to predict *recoverable token waste*.

TraceRazor's composite efficiency is a convex combination of 13 sub-metrics:

    raw_efficiency = sum_k (w_k * m_k) / sum_k w_k          (m_k in [0,1])
    TAS            = 100 * raw_efficiency * (0.7 + 0.3 * task_value)

By default the weights w_k are author-chosen heuristics. This tool replaces them
with weights *fit on data*: given a labelled dataset where each trace has a
ground-truth recoverable-waste fraction, it finds the convex weights that make
`raw_efficiency` best predict the true efficiency `1 - recoverable_fraction`.

The output is a JSON weights file the engine loads via
`tracerazor audit --weights <file>` or the `TRACERAZOR_WEIGHTS` env var, so the
shipped score becomes "fit on dataset X" rather than "chosen by the author".

Dataset manifest (JSON):

    {
      "name": "industry-multiagent-2026q2",
      "entries": [
        {"trace": "runs/agent_01.json", "recoverable_fraction": 0.42},
        {"before": "runs/verbose_02.json", "after": "runs/lean_02.json"}
      ]
    }

For a ``{"before","after"}`` entry the target is computed from the measured
token totals: ``(before_tokens - after_tokens) / before_tokens`` (a measured
before/after re-run at constant task quality, the honest ground truth). The
trace that is *scored* is the "before" run (the real, un-optimised trace).

Usage:
    python -m calibration.calibrate --dataset data/manifest.json \
        --out config/tas_weights.calibrated.json --report config/calibration_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import numpy as np
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - dependency guard
    sys.exit(
        "calibration requires numpy + scipy.\n"
        "Install with: pip install -e '.[calibrate]'"
    )

# Metric codes in the canonical order used by the weights config / engine.
METRICS = ["srr", "ldi", "tca", "rda", "isr", "tur", "cce", "dbo",
           "vdi", "shl", "ccr", "gar", "csd", "obs"]

# Built-in default weights (crates/tracerazor-core/src/scoring.rs::Weights::default),
# used as the baseline to beat in the report.
DEFAULT_WEIGHTS = {
    "srr": 0.17, "ldi": 0.13, "tca": 0.13, "rda": 0.10, "isr": 0.10,
    "tur": 0.10, "cce": 0.10, "dbo": 0.09, "vdi": 0.08, "shl": 0.05,
    "ccr": 0.03, "gar": 0.07, "csd": 0.05, "obs": 0.06,
}


@dataclass
class Sample:
    trace_path: Path
    # Recoverable-waste fraction in [0,1]. Either given directly, or computed at
    # audit time from a before/after token delta (when after_path is set).
    target_fraction: Optional[float] = None
    after_path: Optional[Path] = None
    features: Optional[np.ndarray] = None  # 13 normalised metric values
    extra: Optional[dict] = None           # experimental report.features map
    before_tokens: Optional[int] = None


def recoverable_fraction(before_tokens: int, after_tokens: int) -> float:
    """Measured recoverable fraction from a before/after token delta, clamped."""
    if before_tokens <= 0:
        return 0.0
    return max(0.0, min(1.0, (before_tokens - after_tokens) / before_tokens))


# ── Binary discovery ────────────────────────────────────────────────────────
def find_binary(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    env = os.environ.get("TRACERAZOR_BIN")
    if env and Path(env).is_file():
        return env
    found = shutil.which("tracerazor")
    if found:
        return found
    here = Path(__file__).resolve().parent.parent
    for rel in ("target/release/tracerazor", "target/debug/tracerazor"):
        cand = here / rel
        if cand.is_file():
            return str(cand)
    sys.exit(
        "tracerazor binary not found. Build it (`cargo build --release -p tracerazor`)\n"
        "or pass --binary /path/to/tracerazor (or set TRACERAZOR_BIN)."
    )


def _trace_tokens(path: Path) -> int:
    """Best-effort token total from a raw-schema trace file. Returns 0 (rather
    than raising) when the file is missing or not raw JSON, so the caller can
    fall back to auditing the file."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return 0
    if isinstance(data, dict) and data.get("total_tokens"):
        return int(data["total_tokens"])
    steps = data.get("steps", []) if isinstance(data, dict) else []
    return int(sum(int(s.get("tokens", 0)) for s in steps))


# ── Dataset loading ─────────────────────────────────────────────────────────
def load_dataset(manifest_path: Path) -> Tuple[str, List[Sample]]:
    manifest = json.loads(manifest_path.read_text())
    name = manifest.get("name", manifest_path.stem)
    base = manifest_path.parent
    samples: List[Sample] = []
    for i, e in enumerate(manifest.get("entries", [])):
        if "trace" in e:
            tp = (base / e["trace"]).resolve()
            if "recoverable_fraction" not in e:
                sys.exit(f"entry {i}: 'trace' entries need 'recoverable_fraction'")
            frac = float(e["recoverable_fraction"])
            if not 0.0 <= frac <= 1.0:
                sys.exit(f"entry {i}: recoverable_fraction {frac} out of [0,1]")
            samples.append(Sample(trace_path=tp, target_fraction=frac))
        elif "before" in e and "after" in e:
            # Fraction is computed at audit time from the measured token totals,
            # so this works for any trace format the auditor understands.
            samples.append(Sample(
                trace_path=(base / e["before"]).resolve(),
                after_path=(base / e["after"]).resolve(),
            ))
        else:
            sys.exit(f"entry {i}: need either 'trace'+'recoverable_fraction' or 'before'+'after'")
    if not samples:
        sys.exit("dataset has no entries")
    return name, samples


# ── Feature extraction (run the auditor, read metric_normalised) ────────────
def _try_audit(binary: str, path: Path, env: dict) -> Optional[dict]:
    """Audit a trace and return the report dict, or None if it produced no JSON
    (e.g. fewer than the minimum steps) or could not be run/parsed."""
    try:
        proc = subprocess.run(
            [binary, "audit", str(path), "--format", "json"],
            capture_output=True, text=True, env=env, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except ValueError:
        return None


def _measure_tokens(binary: str, path: Path, env: dict) -> Optional[int]:
    """Total tokens for a trace, measured the same way regardless of size:
    raw schema is read directly (works for any step count), other formats are
    audited. Returns None when the total cannot be determined."""
    t = _trace_tokens(path)
    if t > 0:
        return t
    rep = _try_audit(binary, path, env)
    if rep is None:
        return None
    return int(rep.get("total_tokens", 0)) or None


def extract_features(binary: str, samples: List[Sample]) -> List[Sample]:
    """Populate features and targets, returning only the usable samples. Samples
    that cannot be scored or measured are skipped with a warning rather than
    aborting the whole run (one bad pair should not kill the dataset)."""
    usable: List[Sample] = []
    skipped: List[Tuple[Path, str]] = []
    # Use a throwaway HOME so the audit's persistent store is not touched.
    with tempfile.TemporaryDirectory() as tmp_home:
        env = dict(os.environ, HOME=tmp_home)
        for s in samples:
            report = _try_audit(binary, s.trace_path, env)
            if report is None:
                skipped.append((s.trace_path, "not analysable (need >= 5 steps) or unreadable"))
                continue
            mn = report.get("score", {}).get("metric_normalised")
            if not mn:
                sys.exit(
                    "audit output has no 'metric_normalised' field; rebuild the "
                    "binary from this revision (`cargo build --release -p tracerazor`)."
                )
            s.features = np.array([float(mn[k]) for k in METRICS], dtype=float)
            s.extra = report.get("features", {}) or {}
            # before/after entries: derive the target from measured token totals,
            # measuring both sides the same way so they are comparable.
            if s.after_path is not None:
                before_tok = _measure_tokens(binary, s.trace_path, env)
                after_tok = _measure_tokens(binary, s.after_path, env)
                if not before_tok or after_tok is None:
                    skipped.append((s.trace_path, "could not measure before/after token totals"))
                    continue
                s.before_tokens = before_tok
                s.target_fraction = recoverable_fraction(before_tok, after_tok)
            if s.target_fraction is None:
                skipped.append((s.trace_path, "no recoverable-waste target"))
                continue
            usable.append(s)

    for path, why in skipped:
        print(f"  skipped {path}: {why}", file=sys.stderr)
    if len(usable) < 2:
        sys.exit(f"only {len(usable)} usable sample(s) after skips; need at least 2 to fit.")
    if skipped:
        print(f"Using {len(usable)} samples ({len(skipped)} skipped).", file=sys.stderr)
    return usable


# ── Fitting: convex weights minimising prediction error ─────────────────────
def fit_weights(X: np.ndarray, y: np.ndarray, l2: float = 0.0,
                prior: Optional[np.ndarray] = None) -> np.ndarray:
    """Minimise ||X w - y||^2 + l2*||w - prior||^2 s.t. w >= 0, sum(w) = 1.

    The engine computes raw_efficiency = sum(w*m)/sum(w); fixing sum(w)=1 makes
    that exactly X w, so this fits the engine's actual prediction. `prior` is the
    shrinkage target (defaults to uniform); pass the heuristic default weights to
    treat calibration as a data update of the domain prior, which keeps every
    metric in play instead of collapsing onto a couple of collinear ones.
    """
    n_features = X.shape[1]
    if prior is None:
        prior = np.full(n_features, 1.0 / n_features)

    def obj(w):
        resid = X @ w - y
        return float(resid @ resid + l2 * ((w - prior) @ (w - prior)))

    def grad(w):
        return 2.0 * (X.T @ (X @ w - y) + l2 * (w - prior))

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n_features
    res = minimize(obj, prior, jac=grad, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else prior


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ (w / w.sum())


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def pearson(y: np.ndarray, yhat: np.ndarray) -> float:
    if y.std() == 0 or yhat.std() == 0:
        return 0.0
    return float(np.corrcoef(y, yhat)[0, 1])


def kfold_r2(X: np.ndarray, y: np.ndarray, l2: float, folds: int, seed: int,
             prior: Optional[np.ndarray] = None) -> Tuple[float, float]:
    n = len(y)
    folds = min(folds, n)
    if folds < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    parts = np.array_split(idx, folds)
    preds = np.zeros(n)
    for f in range(folds):
        test = parts[f]
        train = np.concatenate([parts[j] for j in range(folds) if j != f])
        w = fit_weights(X[train], y[train], l2=l2, prior=prior)
        preds[test] = predict(X[test], w)
    return r2(y, preds), pearson(y, preds)


# ── Report ──────────────────────────────────────────────────────────────────
def write_report(path: Path, name: str, n: int, weights: dict,
                 fit: dict, baseline: dict, cv: dict, l2: float) -> None:
    lines = [
        f"# TAS Weight Calibration Report: `{name}`",
        "",
        f"- Samples: **{n}**",
        f"- Target: efficiency = `1 - recoverable_token_fraction`",
        f"- L2 ridge toward prior: `{l2}`",
        "",
        "## Fit quality (calibrated weights)",
        "",
        "| | R² | Pearson r |",
        "|---|---:|---:|",
        f"| Train (in-sample) | {fit['r2']:.3f} | {fit['r']:.3f} |",
        f"| {cv['folds']}-fold cross-validated | {cv['r2']:.3f} | {cv['r']:.3f} |",
        f"| Default weights (baseline) | {baseline['r2']:.3f} | {baseline['r']:.3f} |",
        "",
        "Cross-validated numbers are the honest estimate of generalisation; the",
        "train/CV gap indicates over-fit. Beat the default-weights baseline to",
        "justify recalibration.",
        "",
        "## Calibrated weights",
        "",
        "| Metric | Calibrated | Default |",
        "|---|---:|---:|",
    ]
    for k in METRICS:
        lines.append(f"| {k.upper()} | {weights[k]:.4f} | {DEFAULT_WEIGHTS[k]:.4f} |")
    lines += ["", "_Weights sum to 1.0 (convex combination)._", ""]
    path.write_text("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate TAS weights to recoverable waste.")
    ap.add_argument("--dataset", required=True, type=Path, help="dataset manifest JSON")
    ap.add_argument("--out", type=Path, default=Path("config/tas_weights.calibrated.json"))
    ap.add_argument("--report", type=Path, default=Path("config/calibration_report.md"))
    ap.add_argument("--binary", default=None, help="path to the tracerazor binary")
    ap.add_argument("--cv", type=int, default=5, help="cross-validation folds")
    ap.add_argument("--l2", type=float, default=0.0,
                    help="ridge penalty toward the prior (helps with small/narrow data)")
    ap.add_argument("--prior", choices=["uniform", "default"], default="uniform",
                    help="shrinkage target: 'default' = the heuristic weights, so "
                         "calibration is a data update of the domain prior")
    ap.add_argument("--features", action="store_true",
                    help="also evaluate the experimental report.features (context "
                         "accumulation signals) and report whether they raise CV R^2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    base_vec = np.array([DEFAULT_WEIGHTS[k] for k in METRICS])
    base_vec = base_vec / base_vec.sum()  # normalised heuristic prior (sums to 1)
    prior = base_vec if args.prior == "default" else None

    binary = find_binary(args.binary)
    name, samples = load_dataset(args.dataset)
    print(f"Dataset '{name}': {len(samples)} entries. Auditing with {binary} ...")
    samples = extract_features(binary, samples)

    X = np.vstack([s.features for s in samples])
    y = np.array([1.0 - s.target_fraction for s in samples])

    weights_vec = fit_weights(X, y, l2=args.l2, prior=prior)
    weights = {k: float(round(v, 6)) for k, v in zip(METRICS, weights_vec)}

    yhat = predict(X, weights_vec)
    fit = {"r2": r2(y, yhat), "r": pearson(y, yhat)}
    bpred = predict(X, base_vec)
    baseline = {"r2": r2(y, bpred), "r": pearson(y, bpred)}
    cv_r2, cv_r = kfold_r2(X, y, args.l2, args.cv, args.seed, prior=prior)
    cv = {"r2": cv_r2, "r": cv_r, "folds": min(args.cv, len(y))}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(weights)
    payload["_meta"] = {
        "dataset": name, "samples": len(samples), "target": "recoverable_token_waste",
        "train_r2": round(fit["r2"], 4), "cv_r2": round(cv_r2, 4),
        "baseline_r2": round(baseline["r2"], 4), "l2": args.l2, "prior": args.prior,
    }

    # ── Experimental features: do the context-accumulation signals predict better?
    if args.features:
        # keys present in every sample (so the matrix is not ragged)
        sets = [set((s.extra or {}).keys()) for s in samples]
        feat_names = sorted(set.intersection(*sets)) if sets and all(sets) else []
        if not feat_names:
            print("--features: no feature keys common to all samples; nothing to evaluate.",
                  file=sys.stderr)
        else:
            Xf = np.array([[float(s.extra[k]) for k in feat_names] for s in samples])
            Xmf = np.hstack([X, Xf])
            # metrics-only CV is `cv_r2` above. Features-only and combined:
            f_cv, _ = kfold_r2(Xf, y, args.l2, args.cv, args.seed, prior=None)
            mf_prior = None
            if args.prior == "default":
                eps = 0.1
                mf_prior = np.concatenate([
                    base_vec * (1.0 - eps),
                    np.full(len(feat_names), eps / len(feat_names)),
                ])
                mf_prior = mf_prior / mf_prior.sum()
            mf_cv, _ = kfold_r2(Xmf, y, args.l2, args.cv, args.seed, prior=mf_prior)
            mf_w = fit_weights(Xmf, y, l2=args.l2, prior=mf_prior)
            per_feat_corr = {
                k: round(float(np.corrcoef(Xf[:, i], y)[0, 1]), 3) if Xf[:, i].std() > 0 else 0.0
                for i, k in enumerate(feat_names)
            }
            payload["_features"] = {
                "names": feat_names,
                "weights": {k: round(float(w), 4) for k, w in zip(feat_names, mf_w[len(METRICS):])},
                "per_feature_corr": per_feat_corr,
                "cv_r2_metrics_only": round(cv_r2, 4),
                "cv_r2_features_only": round(f_cv, 4),
                "cv_r2_metrics_plus_features": round(mf_cv, 4),
            }
            print("\n── Experimental features ──")
            print(f"  CV R²  metrics-only        : {cv_r2:+.3f}")
            print(f"  CV R²  features-only       : {f_cv:+.3f}")
            print(f"  CV R²  metrics + features  : {mf_cv:+.3f}")
            print("  per-feature corr with target:")
            for k in feat_names:
                print(f"    {k:<28} {per_feat_corr[k]:+.3f}")

    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    write_report(args.report, name, len(samples), weights, fit, baseline, cv, args.l2)

    print(f"\nTrain R²={fit['r2']:.3f}  CV R²={cv_r2:.3f}  (default baseline R²={baseline['r2']:.3f})")
    print(f"Wrote weights → {args.out}")
    print(f"Wrote report  → {args.report}")
    print("\nUse them:  tracerazor audit <trace> --weights", args.out)
    print("    or set TRACERAZOR_WEIGHTS=" + str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
