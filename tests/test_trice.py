import json

import pytest

from tracerazor.trice import (
    LearningWeights,
    evaluate_policy,
    render_context,
    render_policy_json,
    segments_from_trace,
    solve_policy,
    update_weights,
)
from tracerazor.trice.render import policy_from_dict


def _trace():
    steps = [
        {
            "id": 1,
            "type": "reasoning",
            "content": "User asks to process refund for order ORD-1.",
            "tokens": 120,
            "input_context": "Refund order ORD-1",
        },
        {
            "id": 2,
            "type": "tool_call",
            "content": "Fetch order details",
            "tokens": 200,
            "tool_name": "get_order",
            "tool_success": True,
            "tool_params": {"order_id": "ORD-1"},
            "output": "Order ORD-1 total $25.00",
        },
        {
            "id": 3,
            "type": "tool_call",
            "content": "Check refund eligibility",
            "tokens": 160,
            "tool_name": "check_refund",
            "tool_success": False,
            "tool_error": "missing required parameter: order_id",
        },
        {
            "id": 4,
            "type": "tool_call",
            "content": "Check refund eligibility with order id",
            "tokens": 180,
            "tool_name": "check_refund",
            "tool_success": True,
            "tool_params": {"order_id": "ORD-1"},
            "output": "Eligible",
        },
        {
            "id": 5,
            "type": "reasoning",
            "content": "Let me re-evaluate whether this is correct. Actually let me double check again.",
            "tokens": 420,
        },
        {
            "id": 6,
            "type": "tool_call",
            "content": "Process refund",
            "tokens": 260,
            "tool_name": "process_refund",
            "tool_success": True,
            "tool_params": {"order_id": "ORD-1"},
            "output": "Refund REF-9 processed",
        },
    ]
    return {
        "trace_id": "trice-test",
        "agent_name": "agent",
        "framework": "raw",
        "task_value_score": 1.0,
        "metadata": {"task": "Process refund for ORD-1"},
        "steps": steps,
    }


def test_segments_label_locked_rehydratable_expired_and_redundant():
    segments = segments_from_trace(_trace())
    by_step = {s.step_id: s for s in segments}
    assert by_step[1].state.value == "essential"
    assert by_step[1].locked is True
    assert by_step[2].state.value == "rehydratable"
    assert by_step[3].state.value == "expired"
    assert by_step[5].state.value in {"redundant", "distractor"}
    assert by_step[6].state.value == "essential"
    assert by_step[6].locked is True
    assert len(by_step[1].receipt) == 64


def test_policy_solver_keeps_locked_anchors_and_reduces_tokens():
    segments = segments_from_trace(_trace())
    policy = solve_policy(segments, budget_ratio=0.55)
    assert policy.algorithm.startswith("trice-v0.1")
    assert policy.policy_tokens < policy.baseline_input_tokens
    for decision in policy.decisions:
        if decision.locked:
            assert decision.action in {"keep", "anchor_prefix"}
            assert decision.policy_tokens == decision.original_tokens
    assert any(d.action in {"lazy_recall", "mask_with_receipt"} for d in policy.decisions)


def test_policy_json_round_trip_context_and_replay_metrics():
    segments = segments_from_trace(_trace())
    policy = solve_policy(segments, budget_ratio=0.60)
    payload = json.loads(render_policy_json(policy))
    loaded = policy_from_dict(payload)
    context = render_context(loaded, segments)
    assert "TRICE_CONTEXT_POLICY" in context
    assert "receipt=" in context
    replay = evaluate_policy(segments, loaded)
    assert replay.evidence_recall == pytest.approx(1.0)
    assert replay.action_divergence == pytest.approx(0.0)
    assert replay.pass_noninferior is True


def test_weight_update_penalizes_quality_and_evidence_failures():
    update = update_weights(
        LearningWeights(),
        features=(0.5, 0.4, 0.3, 0.2, 0.1),
        measured_input_savings=0.25,
        quality_drop=0.08,
        evidence_recall_failure=0.1,
        compression_overhead=0.45,
    )
    assert update.reward < 0.25
    assert update.error != 0.0
    assert update.weights.as_vector() != LearningWeights().as_vector()
