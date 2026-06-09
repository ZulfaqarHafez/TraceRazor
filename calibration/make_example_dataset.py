#!/usr/bin/env python3
"""Generate a *worked-example* calibration dataset with known recoverable waste.

This is a stand-in for the real ground truth (traces from running your products
against industry multi-agent solutions, with measured before/after savings). It
builds synthetic agent traces by taking a clean base run and injecting a *known*
amount of recoverable waste (duplicate reasoning steps and verbose filler) so
the recoverable-waste fraction of each trace is known by construction and is NOT
derived from TraceRazor's own estimates (no circularity).

Output: trace JSON files under ``calibration/example_data/`` plus a
``manifest.json`` consumable by ``calibration/calibrate.py``.

Usage:
    python -m calibration.make_example_dataset --out calibration/example_data --n 36
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Distinct, non-redundant base step contents (so a clean base scores well).
REASONING = [
    "Parse the user request and identify the order id and intent.",
    "The order is within the return window, so it is eligible for a refund.",
    "Determine the correct refund amount from the order total and any discounts.",
    "Choose the payment method to refund based on the original transaction.",
    "Confirm the refund succeeded and prepare a concise customer message.",
    "Check whether a replacement should be offered instead of a refund.",
    "Validate the shipping address before scheduling the replacement.",
    "Summarise the resolution and close the ticket.",
]
TOOLS = [
    ("get_order_details", "Fetch order details for the requested order id."),
    ("check_refund_eligibility", "Check whether the order qualifies for a refund."),
    ("process_refund", "Issue the refund to the original payment method."),
    ("send_confirmation", "Send the customer a confirmation message."),
    ("schedule_replacement", "Schedule a replacement shipment."),
    ("lookup_inventory", "Check inventory for the replacement item."),
]

VERBOSE_FILLER = (
    " I want to be absolutely certain here, so let me carefully and thoroughly "
    "reconsider every detail once more to make sure nothing at all is missed, "
    "because it is very important to be extremely careful and precise."
)


def clean_steps(n_reason: int, n_tool: int, rng: random.Random):
    steps = []
    sid = 1
    reason = rng.sample(REASONING, min(n_reason, len(REASONING)))
    tools = rng.sample(TOOLS, min(n_tool, len(TOOLS)))
    # Interleave reasoning and tool steps.
    for i in range(max(len(reason), len(tools))):
        if i < len(reason):
            steps.append({"id": sid, "step_type": "reasoning",
                          "content": reason[i], "tokens": rng.randint(120, 220)})
            sid += 1
        if i < len(tools):
            name, content = tools[i]
            steps.append({"id": sid, "step_type": "tool_call", "content": content,
                          "tokens": rng.randint(60, 140), "tool_name": name,
                          "tool_success": True})
            sid += 1
    return steps


HEDGING = (
    " I think that maybe, possibly, this could perhaps be roughly correct, but I "
    "am not entirely sure and I might be wrong, so apologies if this is off, it "
    "seems like it may potentially be sort of right in some sense, I believe."
)


def inject_waste(steps, waste_ratio: float, rng: random.Random):
    """Inject a *known* amount of recoverable waste across several categories so
    different sub-metrics respond. Returns (new_steps, injected_tokens).

    Each added step's tokens are recoverable by construction, so the target
    label is exact. Waste types are chosen at random per injection:
      - dup_reason   duplicate an earlier reasoning step      (SRR, ISR)
      - loop_tool    repeat an earlier tool call verbatim      (LDI, SRR, DBO)
      - failed_tool  a failed tool call needing a retry        (TCA)
      - verbose      a filler-heavy reasoning step             (VDI, CCR, TUR)
      - hedge        a sycophantic/hedging reasoning step      (SHL)
      - context_dup  a step that restates earlier context      (CCE, CSD)
    """
    base_tokens = sum(s["tokens"] for s in steps)
    if waste_ratio <= 0:
        return steps, 0
    target_inject = int(base_tokens * waste_ratio / (1.0 - waste_ratio))
    injected = 0
    sid = max(s["id"] for s in steps) + 1
    out = list(steps)
    reasoning = [s for s in steps if s["step_type"] == "reasoning"]
    tools = [s for s in steps if s["step_type"] == "tool_call"]
    # Each sample draws a subset of waste types so the dataset spans the metrics.
    types = rng.sample(
        ["dup_reason", "loop_tool", "failed_tool", "verbose", "hedge", "context_dup"],
        k=rng.randint(2, 4),
    )

    while injected < target_inject:
        wtype = rng.choice(types)
        tok = min(rng.randint(120, 300), target_inject - injected + 1)
        if wtype == "dup_reason" and reasoning:
            src = rng.choice(reasoning)
            out.append({"id": sid, "step_type": "reasoning",
                        "content": src["content"], "tokens": tok})
        elif wtype == "loop_tool" and tools:
            src = rng.choice(tools)
            out.append({"id": sid, "step_type": "tool_call", "content": src["content"],
                        "tokens": tok, "tool_name": src["tool_name"], "tool_success": True})
        elif wtype == "failed_tool" and tools:
            src = rng.choice(tools)
            out.append({"id": sid, "step_type": "tool_call",
                        "content": src["content"], "tokens": tok,
                        "tool_name": src["tool_name"], "tool_success": False,
                        "tool_error": "missing required parameter"})
        elif wtype == "verbose":
            base = rng.choice(reasoning)["content"] if reasoning else "Reconsidering."
            out.append({"id": sid, "step_type": "reasoning",
                        "content": base + VERBOSE_FILLER, "tokens": tok})
        elif wtype == "hedge":
            base = rng.choice(reasoning)["content"] if reasoning else "Let me check."
            out.append({"id": sid, "step_type": "reasoning",
                        "content": base + HEDGING, "tokens": tok})
        elif wtype == "context_dup" and reasoning:
            src = rng.choice(reasoning)
            out.append({"id": sid, "step_type": "reasoning",
                        "content": "As established earlier: " + src["content"], "tokens": tok})
        else:
            out.append({"id": sid, "step_type": "reasoning",
                        "content": (reasoning[0]["content"] if reasoning else "x") + VERBOSE_FILLER,
                        "tokens": tok})
        injected += tok
        sid += 1
    return out, injected


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("calibration/example_data"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(args.n):
        n_reason = rng.randint(3, 5)
        n_tool = rng.randint(2, 4)
        base = clean_steps(n_reason, n_tool, rng)
        ratio = rng.choice([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
        steps, injected = inject_waste(base, ratio, rng)
        total = sum(s["tokens"] for s in steps)
        frac = injected / total if total else 0.0
        trace = {
            "trace_id": f"example-{i:03d}",
            "agent_name": "support-agent",
            "framework": "raw",
            "total_tokens": total,
            "task_value_score": 1.0,
            "steps": steps,
        }
        fname = f"example-{i:03d}.json"
        (args.out / fname).write_text(json.dumps(trace, indent=2))
        entries.append({"trace": fname, "recoverable_fraction": round(frac, 4)})

    manifest = {"name": "synthetic-worked-example", "entries": entries}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(entries)} traces + manifest to {args.out}")
    print(f"Recoverable fractions range "
          f"{min(e['recoverable_fraction'] for e in entries):.2f}"
          f"-{max(e['recoverable_fraction'] for e in entries):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
