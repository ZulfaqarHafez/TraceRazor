"""Auditor detection precision/recall vs. independent step-level labels.

Loads the labelled corpus (from ``make_step_labels.py``) and runs the Teacher
Diagnoser (builtin backend, forced offline) on each trace.  For each WasteKind,
collects predicted-removable step IDs and compares against the ground-truth
removable set, then reports precision / recall / F1 with 95% bootstrap CIs.

Independence: ground-truth labels come from the runner's injection logic, not
from any auditor heuristic.  This breaks the circularity of the current
self-reported "30–60% waste" claim.

Usage:
    python -m benchmark.detection_pr
    python -m benchmark.detection_pr --input benchmark/labels/labelled_traces.jsonl
                                     --output docs/detection_pr.md
                                     --n-boot 1000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teacher.diagnose import Diagnoser
from teacher.schemas import WasteKind

# Ground-truth label → the WasteKind whose detector should flag it
_LABEL_TO_KIND: dict[str, WasteKind] = {
    "preamble":      WasteKind.HEDGING,
    "reformulation": WasteKind.REDUNDANT_STEP,
    "overdepth":     WasteKind.OVER_DEPTH,
    "loop_repeat":   WasteKind.LOOP,
}

_KIND_DISPLAY: dict[WasteKind, str] = {
    WasteKind.HEDGING:        "HEDGING (SHL)",
    WasteKind.REDUNDANT_STEP: "REDUNDANT_STEP (SRR)",
    WasteKind.OVER_DEPTH:     "OVER_DEPTH (RDA)",
    WasteKind.LOOP:           "LOOP (LDI)",
}


class PRSample(NamedTuple):
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _bootstrap_ci(values: list[float], n_boot: int = 1000,
                  ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = int((1 - ci) / 2 * n_boot)
    hi = min(int((1 + ci) / 2 * n_boot), n_boot - 1)
    return (samples[lo], samples[hi])


def _agg(samples: list[PRSample], label: str, n_boot: int, seed_offset: int = 0) -> dict:
    if not samples:
        return {"label": label, "n_traces": 0}
    tp_all = sum(s.tp for s in samples)
    fp_all = sum(s.fp for s in samples)
    fn_all = sum(s.fn for s in samples)
    micro_p = tp_all / (tp_all + fp_all) if (tp_all + fp_all) else 0.0
    micro_r = tp_all / (tp_all + fn_all) if (tp_all + fn_all) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    prec_list = [s.precision for s in samples]
    rec_list = [s.recall for s in samples]
    f1_list = [s.f1 for s in samples]
    p_ci = _bootstrap_ci(prec_list, n_boot, seed=42 + seed_offset)
    r_ci = _bootstrap_ci(rec_list, n_boot, seed=43 + seed_offset)
    f1_ci = _bootstrap_ci(f1_list, n_boot, seed=44 + seed_offset)
    return {
        "label": label,
        "n_traces": len(samples),
        "total_tp": tp_all, "total_fp": fp_all, "total_fn": fn_all,
        "micro_precision": round(micro_p, 3),
        "micro_recall": round(micro_r, 3),
        "micro_f1": round(micro_f1, 3),
        "macro_precision": round(sum(prec_list) / len(prec_list), 3),
        "macro_precision_ci95": [round(p_ci[0], 3), round(p_ci[1], 3)],
        "macro_recall": round(sum(rec_list) / len(rec_list), 3),
        "macro_recall_ci95": [round(r_ci[0], 3), round(r_ci[1], 3)],
        "macro_f1": round(sum(f1_list) / len(f1_list), 3),
        "macro_f1_ci95": [round(f1_ci[0], 3), round(f1_ci[1], 3)],
    }


def evaluate(traces: list[dict], n_boot: int = 1000) -> dict:
    """Run builtin Diagnoser on each trace; compute per-kind and overall P/R."""
    diagnoser = Diagnoser(prefer_auditor=False)  # force builtin for offline reproducibility

    all_kinds = list(set(_LABEL_TO_KIND.values()))
    by_kind: dict[WasteKind, list[PRSample]] = {k: [] for k in all_kinds}
    overall: list[PRSample] = []

    for t in traces:
        # Reconstruct a minimal trace dict from labelled steps
        raw_steps = []
        gt_removable: set[int] = set()
        gt_by_kind: dict[WasteKind, set[int]] = {k: set() for k in all_kinds}

        for s in t.get("steps", []):
            raw_steps.append({
                "id": s["step_id"],
                "type": s["step_type"],
                "content": s["content_preview"],
                "tokens": s["tokens"],
                "tool_name": None,
                "tool_params": None,
            })
            if s["removable"]:
                gt_removable.add(s["step_id"])
            kind = _LABEL_TO_KIND.get(s["label"])
            if kind is not None:
                gt_by_kind[kind].add(s["step_id"])

        # Restore tool_call fields for loop detection
        for s_raw, s_lab in zip(raw_steps, t.get("steps", [])):
            if s_lab["step_type"] == "tool_call":
                tool_name = s_lab.get("tool_name") or _infer_tool_name(s_lab["content_preview"])
                s_raw["tool_name"] = tool_name
                s_raw["tool_params"] = {"arg": "x"}

        trace_dict = {
            "trace_id": t["trace_id"],
            "agent_name": "mock",
            "framework": "langgraph",
            "steps": raw_steps,
        }

        diagnosis = diagnoser.diagnose(trace_dict)

        pred_by_kind: dict[WasteKind, set[int]] = {k: set() for k in all_kinds}
        pred_any: set[int] = set()
        for p in diagnosis.patterns:
            if p.kind in pred_by_kind:
                pred_by_kind[p.kind].update(p.step_ids)
                pred_any.update(p.step_ids)

        for kind in all_kinds:
            pred = pred_by_kind[kind]
            gt = gt_by_kind[kind]
            by_kind[kind].append(PRSample(
                tp=len(pred & gt), fp=len(pred - gt), fn=len(gt - pred),
            ))

        overall.append(PRSample(
            tp=len(pred_any & gt_removable),
            fp=len(pred_any - gt_removable),
            fn=len(gt_removable - pred_any),
        ))

    results: dict[str, dict] = {}
    for i, kind in enumerate(sorted(all_kinds, key=lambda k: k.value)):
        results[kind.value] = _agg(by_kind[kind], _KIND_DISPLAY[kind], n_boot, seed_offset=i * 3)
    results["overall"] = _agg(overall, "OVERALL (any flag vs. any removable step)", n_boot, seed_offset=100)
    return results


def _infer_tool_name(content: str) -> str:
    """Best-effort: extract 'Calling TOOL_NAME' from content preview."""
    if "Calling " in content:
        return content.split("Calling ")[-1].strip()[:40]
    return "unknown_tool"


def render_markdown(results: dict, n_traces: int) -> str:
    lines = [
        "# Auditor Detection Precision / Recall",
        "",
        f"**Corpus:** {n_traces} synthetic traces, MockAgent baseline config (all waste active).  ",
        "**Ground truth:** step labels derived from the runner's injection logic, "
        "independent of the auditor heuristics.  ",
        "**Auditor:** builtin Python heuristics (SHL regex, SRR Jaccard, LDI identical-call, "
        "RDA step-count).  ",
        "**Bootstrap CIs:** 95%, n=1000 resamples.",
        "",
        "## Per-detector results",
        "",
        "| Detector | GT label | Traces | TP | FP | FN | Micro-P | Micro-R | Micro-F1 | "
        "Macro-P [95% CI] | Macro-R [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    label_for_kind = {v: k for k, v in _LABEL_TO_KIND.items()}

    for kind_val in sorted(k for k in results if k != "overall"):
        r = results[kind_val]
        kind = WasteKind(kind_val)
        gt_label = label_for_kind.get(kind, "—")
        p_ci = r.get("macro_precision_ci95", [0, 0])
        r_ci = r.get("macro_recall_ci95", [0, 0])
        lines.append(
            f"| {r['label']} | {gt_label} | {r['n_traces']} "
            f"| {r['total_tp']} | {r['total_fp']} | {r['total_fn']} "
            f"| {r['micro_precision']:.3f} | {r['micro_recall']:.3f} | {r['micro_f1']:.3f} "
            f"| {r['macro_precision']:.3f} [{p_ci[0]:.3f}–{p_ci[1]:.3f}] "
            f"| {r['macro_recall']:.3f} [{r_ci[0]:.3f}–{r_ci[1]:.3f}] |"
        )

    ov = results["overall"]
    p_ci = ov.get("macro_precision_ci95", [0, 0])
    r_ci = ov.get("macro_recall_ci95", [0, 0])
    lines += [
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        f"| **{ov['label']}** | — | {ov['n_traces']} "
        f"| {ov['total_tp']} | {ov['total_fp']} | {ov['total_fn']} "
        f"| {ov['micro_precision']:.3f} | {ov['micro_recall']:.3f} | {ov['micro_f1']:.3f} "
        f"| {ov['macro_precision']:.3f} [{p_ci[0]:.3f}–{p_ci[1]:.3f}] "
        f"| {ov['macro_recall']:.3f} [{r_ci[0]:.3f}–{r_ci[1]:.3f}] |",
        "",
        "## Interpretation",
        "",
        "- **Micro-P/R** are pooled across all traces (dominated by common waste types).",
        "- **Macro-P/R** are per-trace averages (treats each trace equally).",
        "- **TP** = auditor flag fires on a step independently labelled as that waste type.",
        "- **FP** = auditor flag fires on a step that is NOT that waste type.",
        "- **FN** = waste step that the auditor misses.",
        "",
        "This table replaces the previous circular '30–60% waste' claim with independently",
        "verified precision and recall numbers. The ground-truth labels were generated by",
        "the runner's injection code (`teacher/runner.py`), which does not consult any",
        "auditor heuristic.",
        "",
        "_Generated by `python -m benchmark.detection_pr`._",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute auditor detection P/R")
    parser.add_argument("--input", default="benchmark/labels/labelled_traces.jsonl")
    parser.add_argument("--output", default="docs/detection_pr.md")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Labelled corpus not found at {input_path}. Run make_step_labels.py first.")
        sys.exit(1)

    traces = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    print(f"Evaluating {len(traces)} traces with {args.n_boot} bootstrap samples...")
    results = evaluate(traces, n_boot=args.n_boot)

    # Console summary
    for k, r in results.items():
        label = r.get("label", k)
        mp = r.get("micro_precision", 0)
        mr = r.get("micro_recall", 0)
        mf = r.get("micro_f1", 0)
        print(f"  {label:<35} P={mp:.3f}  R={mr:.3f}  F1={mf:.3f}")

    md = render_markdown(results, len(traces))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
