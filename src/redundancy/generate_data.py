"""
Synthetic judge transcript generator (PRD v5)

Uses Claude to generate realistic (prompt_A, response_A, prompt_B, label) pairs
for training the substitutability classifier.

Labels:
    1 = response_A is an adequate substitute for a fresh response to prompt_B
    0 = response_A is NOT a suitable substitute

Usage:
    python -m redundancy.generate_data --n 300 --out results/run_synthetic/judge_transcripts.jsonl
    python -m redundancy.generate_data --n 100 --run-id run_v3 --out results/run_v3/judge_transcripts.jsonl
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent

# ── load .env ────────────────────────────────────────────────────────────────

_env_path = _REPO / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


# ── scenario templates ───────────────────────────────────────────────────────

# (positive_seed, negative_seed) pairs — same topic → likely substitutable,
#  different topic → not substitutable
_SCENARIO_PAIRS: List[Tuple[str, str, int]] = [
    # (scenario_A_hint, scenario_B_hint, substitutable)
    # --- POSITIVE pairs (label=1) ---
    ("Book a flight from JFK to LAX on June 15 for one adult", "Book a flight from JFK to LAX on June 16 for one adult", 1),
    ("What is the baggage allowance for economy class?", "How many bags can I bring on economy?", 1),
    ("Cancel reservation CONF-12345", "Cancel my booking CONF-67890", 1),
    ("I need to change my seat to an aisle seat", "Please upgrade my seat to an aisle preference", 1),
    ("What time does flight AA100 depart?", "What is the departure time for AA100?", 1),
    ("Add a meal preference: vegetarian", "I would like to request a vegetarian meal", 1),
    ("I lost my boarding pass, can you resend it?", "Please resend my boarding pass via email", 1),
    ("Is flight UA200 on time?", "Has flight UA200 been delayed?", 1),
    ("Book connecting flight JFK to ORD to LAX on July 4", "Book connecting flight JFK to ORD to LAX on July 5", 1),
    ("What airports does American Airlines serve from New York?", "List American Airlines destinations from NYC", 1),
    # --- NEGATIVE pairs (label=0) ---
    ("Book a flight from JFK to LAX on June 15 for one adult", "Cancel reservation CONF-99999", 0),
    ("What is the baggage allowance?", "Book a business class ticket to London", 0),
    ("I need a vegetarian meal on my flight", "I want to upgrade to first class", 0),
    ("Check in for flight AA100 departing tomorrow", "Cancel my booking and request a refund", 0),
    ("Is there a direct flight from LAX to Miami?", "How do I change a passenger name on a reservation?", 0),
    ("I need to add an infant to my booking", "What is the flight status of DL500?", 0),
    ("My luggage was lost, please file a claim", "Book round trip JFK to Paris in August", 0),
    ("What is the policy for overweight baggage?", "Transfer me to a human agent", 0),
    ("I want to redeem my miles for an upgrade", "What time does the JFK lounge close?", 0),
    ("Confirm my reservation details for booking CONF-11111", "Request a wheelchair for my upcoming flight", 0),
]


# ── Anthropic generator ───────────────────────────────────────────────────────


async def generate_pair(
    client: Any,
    scenario_A: str,
    scenario_B: str,
    label: int,
    task_index: int,
    turn: int,
    run_id: str,
) -> Optional[Dict[str, Any]]:
    """Ask Claude to write prompt_A, response_A, prompt_B for this scenario pair."""

    system = textwrap.dedent("""
        You are a data generator for an airline customer support AI training dataset.
        When given two customer scenarios, you will output:
        1. prompt_A: a realistic multi-turn conversation context ending with the customer request for scenario A
        2. response_A: an ideal agent response to prompt_A (2-4 sentences, professional)
        3. prompt_B: a realistic multi-turn conversation context ending with the customer request for scenario B

        Output ONLY valid JSON with keys: "prompt_A", "response_A", "prompt_B"
        No explanation, no markdown fences, just the JSON object.
    """).strip()

    user_msg = textwrap.dedent(f"""
        Scenario A: {scenario_A}
        Scenario B: {scenario_B}
        Substitutable: {"yes" if label == 1 else "no"}

        Generate the JSON with prompt_A, response_A, and prompt_B.
        If substitutable=yes, prompt_A and prompt_B should be semantically similar
        (same intent, minor variation like date/flight number).
        If substitutable=no, prompt_A and prompt_B should have clearly different intents.
    """).strip()

    loop = asyncio.get_event_loop()
    for attempt in range(5):
        try:
            msg = await loop.run_in_executor(
                None,
                lambda: client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=512,
                    system=system,
                    messages=[{"role": "user", "content": user_msg}],
                ),
            )
            raw = msg.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            obj = json.loads(raw)
            return {
                "run_id": run_id,
                "task_index": task_index,
                "turn": turn,
                "prompt_A": str(obj["prompt_A"]),
                "response_A": str(obj["response_A"]),
                "prompt_B": str(obj["prompt_B"]),
                "label": label,
            }
        except Exception as exc:
            msg_str = str(exc).lower()
            if "rate_limit" in msg_str or "429" in msg_str:
                wait = 20 * (2 ** attempt)
                print(f"  [rate-limit] task={task_index} retrying in {wait}s ...", file=sys.stderr)
                await asyncio.sleep(wait)
            else:
                print(f"  [warn] generate_pair failed (task={task_index}): {exc}", file=sys.stderr)
                return None
    print(f"  [warn] task={task_index} gave up after 5 retries", file=sys.stderr)
    return None


async def generate_batch(
    client: Any,
    n: int,
    run_id: str,
    concurrency: int = 8,
) -> List[Dict[str, Any]]:
    """Generate ``n`` records, cycling through scenario templates."""
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    async def _one(i: int) -> None:
        sA, sB, label = _SCENARIO_PAIRS[i % len(_SCENARIO_PAIRS)]
        # Add mild variation so repeated scenarios differ slightly
        suffix = f" (case {i // len(_SCENARIO_PAIRS) + 1})" if i >= len(_SCENARIO_PAIRS) else ""
        async with sem:
            rec = await generate_pair(
                client=client,
                scenario_A=sA + suffix,
                scenario_B=sB + suffix,
                label=label,
                task_index=i,
                turn=random.randint(0, 5),
                run_id=run_id,
            )
        if rec is not None:
            results.append(rec)
            if len(results) % 10 == 0:
                print(f"  {len(results)}/{n} records generated ...", flush=True)

    await asyncio.gather(*[_one(i) for i in range(n)])
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


async def _main_async(args: Any) -> None:
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n} synthetic judge transcript records (run_id={args.run_id}) ...")
    records = await generate_batch(client, n=args.n, run_id=args.run_id, concurrency=args.concurrency)

    # Shuffle to interleave positives and negatives
    random.shuffle(records)

    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    pos = sum(1 for r in records if r["label"] == 1)
    neg = len(records) - pos
    print(f"\nWrote {len(records)} records to {out_path}")
    print(f"  positives={pos}  negatives={neg}  ratio={pos/max(len(records),1):.1%}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic judge transcripts via Anthropic")
    parser.add_argument("--n", type=int, default=200, help="Number of records to generate (default: 200)")
    parser.add_argument(
        "--out",
        default=str(_REPO / "results" / "run_synthetic" / "judge_transcripts.jsonl"),
        help="Output path for the JSONL file",
    )
    parser.add_argument(
        "--run-id",
        default="run_synthetic",
        help="run_id tag embedded in records (default: run_synthetic)",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Max parallel API calls")
    args = parser.parse_args()

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
