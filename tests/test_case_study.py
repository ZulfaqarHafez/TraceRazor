"""Ship-plan 4.1: the measured case-study harness.

The harness's job is to turn before/after trace pairs into a published table
of *measured* token deltas at constant pass rate, with bootstrap CIs. These
tests pin the CI math and run the full pipeline over synthetic pairs against
the real binary.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmark.case_study import (  # noqa: E402
    bootstrap_ci,
    discover_pairs,
    find_binary,
    main,
    measure,
    render_markdown,
    trace_passes,
)


# ── CI math ────────────────────────────────────────────────────────────────────

def test_bootstrap_ci_contains_mean_and_is_deterministic():
    data = [10.0, 12.0, 8.0, 11.0, 9.0]
    mean, lo, hi = bootstrap_ci(data)
    assert mean == pytest.approx(10.0)
    assert lo <= mean <= hi
    assert lo < hi, "five distinct values must give a non-degenerate CI"
    # Seeded: byte-for-byte reproducible.
    assert bootstrap_ci(data) == (mean, lo, hi)


def test_bootstrap_ci_single_value_degenerates():
    assert bootstrap_ci([7.5]) == (7.5, 7.5, 7.5)


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_bootstrap_ci_narrows_with_more_data():
    tight = bootstrap_ci([10.0] * 20 + [11.0] * 20)
    wide = bootstrap_ci([1.0, 21.0])
    assert (tight[2] - tight[1]) < (wide[2] - wide[1])


# ── Synthetic before/after pairs through the real binary ──────────────────────

def _step(i, content, tokens, tool=None):
    s = {"id": i, "step_type": "tool_call" if tool else "reasoning",
         "content": content, "tokens": tokens}
    if tool:
        s["tool_name"] = tool
        s["tool_success"] = True
    return s


def _trace(trace_id, steps, passed=True):
    return {
        "trace_id": trace_id,
        "agent_name": "case-study-agent",
        "framework": "raw",
        "total_tokens": sum(s["tokens"] for s in steps),
        "task_value_score": 1.0 if passed else 0.0,
        "steps": steps,
    }


def _write_pair(d, task, before_steps, after_steps, after_passed=True):
    (d / f"{task}.before.json").write_text(json.dumps(_trace(f"{task}-before", before_steps)))
    (d / f"{task}.after.json").write_text(
        json.dumps(_trace(f"{task}-after", after_steps, passed=after_passed))
    )


@pytest.fixture
def pairs_dir(tmp_path):
    """Two synthetic tasks where the after-trace drops redundant work."""
    base = [
        _step(1, "Analyse the customer refund request for order ORD-9182", 120),
        _step(2, "Fetch order details from the billing database", 100, tool="get_order"),
        _step(3, "Check the refund eligibility window for this order", 100, tool="check_eligibility"),
        _step(4, "Order is eligible; proceed to process the refund now", 90),
        _step(5, "Process refund transaction for order ORD-9182", 110, tool="process_refund"),
    ]
    waste = [
        _step(6, "Fetch order details from the billing database", 100, tool="get_order"),
        _step(7, "Let me just think about this again basically one more time", 140),
    ]
    d = tmp_path / "pairs"
    d.mkdir()
    _write_pair(d, "task0", base + waste, base)
    _write_pair(d, "task1", base + waste + [
        _step(8, "Fetch order details from the billing database", 100, tool="get_order"),
    ], base)
    return d


needs_binary = pytest.mark.skipif(find_binary() is None, reason="tracerazor binary not built")


@needs_binary
def test_measured_pipeline_end_to_end(pairs_dir, tmp_path):
    pairs = discover_pairs(pairs_dir)
    assert [p[0] for p in pairs] == ["task0", "task1"]

    results = measure(find_binary(), pairs)
    assert len(results) == 2
    for r in results:
        assert r.tokens_saved > 0, "after-trace removed steps: tokens must drop"
        assert r.tokens_before == r.tokens_after + r.tokens_saved
        assert r.pass_held, "synthetic pairs hold the task outcome constant"

    table = render_markdown(results)
    assert "| task0 |" in table and "| task1 |" in table
    assert "95% bootstrap CI" in table
    assert "constant task outcome" in table

    out = tmp_path / "case_study_table.md"
    rc = main(["--pairs-dir", str(pairs_dir), "--out", str(out)])
    assert rc == 0
    assert "mean token reduction" in out.read_text()


@needs_binary
def test_pass_flip_is_called_out_and_fails(pairs_dir, tmp_path):
    # task2's after-trace saves tokens but loses the task: that is a
    # regression, and the harness must say so and exit non-zero.
    base = json.loads((pairs_dir / "task0.before.json").read_text())
    after = json.loads((pairs_dir / "task0.after.json").read_text())
    after["task_value_score"] = 0.0
    (pairs_dir / "task2.before.json").write_text(json.dumps(base))
    (pairs_dir / "task2.after.json").write_text(json.dumps(after))

    rc = main(["--pairs-dir", str(pairs_dir), "--out", str(tmp_path / "t.md")])
    assert rc == 1
    text = (tmp_path / "t.md").read_text()
    assert "FLIPPED" in text
    assert "not a saving" in text


def test_trace_passes_reads_task_value_score(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(_trace("x", [_step(1, "hello world step", 10)], passed=False)))
    assert trace_passes(p) is False
    p.write_text(json.dumps(_trace("x", [_step(1, "hello world step", 10)], passed=True)))
    assert trace_passes(p) is True
