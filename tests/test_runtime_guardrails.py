from __future__ import annotations

import json

import pytest

from tracerazor.runtime import (
    AuditPolicy,
    GuardrailConfig,
    RunContext,
    RuntimeEvent,
    StreamingGuardrailDetector,
    TokenUsage,
    ToolCall,
    TraceRazorProcessor,
)


def _event(
    context: RunContext,
    sequence: int,
    *,
    event_type: str = "reasoning",
    tokens: TokenUsage | None = None,
    input_context: str | None = None,
    tool: ToolCall | None = None,
    output: str | None = None,
    metadata: dict | None = None,
) -> RuntimeEvent:
    return RuntimeEvent.create(
        context,
        event_type=event_type,
        host="test",
        framework="raw",
        tokens=tokens or TokenUsage(),
        input_context=input_context,
        tool=tool,
        output=output,
        metadata=metadata or {},
        sequence=sequence,
    )


def _tool(signature: str, *, status: str = "success", **kwargs) -> ToolCall:
    return ToolCall.from_arguments(signature, {"query": "stable"}, status=status, **kwargs)


def _find(detector: StreamingGuardrailDetector, signal_id: str):
    return [finding for finding in detector.findings if finding.signal_id == signal_id]


def test_repeated_identical_and_equivalent_tool_calls_share_stable_finding_id():
    context = RunContext.create()
    detector = StreamingGuardrailDetector(
        GuardrailConfig(tool_repeat_threshold=3, tool_repeat_window=4)
    )
    signatures = ["MCP__GitHub__Search", "mcp.github.search", "MCP GitHub Search"]
    updates = []
    for sequence, signature in enumerate(signatures, start=1):
        updates.extend(
            detector.observe(
                _event(
                    context,
                    sequence,
                    event_type="tool_call",
                    tool=_tool(signature),
                )
            )
        )
    findings = _find(detector, "guardrail.tool_call_repeat")
    assert len(findings) == 1
    finding = findings[0]
    assert finding in updates
    assert finding.evidence["match"] == "equivalent"
    assert finding.evidence["equivalent_count"] == 3
    assert finding.evidence["normalized_signature"] == "mcp.github.search"
    assert finding.enforcement_eligible is False

    next_update = detector.observe(
        _event(
            context,
            4,
            event_type="tool_call",
            tool=_tool("mcp.github.search"),
        )
    )[0]
    assert next_update.finding_id == finding.finding_id
    assert next_update.last_sequence == 4


def test_child_agent_fanout_tracks_unique_children_and_saturates():
    parent = RunContext.create(agent_id="raw-sensitive-parent-id")
    detector = StreamingGuardrailDetector(
        GuardrailConfig(max_children_per_parent=2, max_total_children=3)
    )
    for sequence in range(1, 5):
        child = RunContext.from_env(parent.spawn_env(child_agent_id=f"child-{sequence}"))
        detector.observe(_event(child, sequence))
    findings = _find(detector, "guardrail.child_agent_fanout")
    assert {finding.evidence["scope"] for finding in findings} == {"parent", "run"}
    run_finding = next(finding for finding in findings if finding.evidence["scope"] == "run")
    assert run_finding.evidence["distinct_parent_child_pairs_at_least"] == 4
    assert run_finding.evidence["tracking_saturated"] is True
    assert parent.agent_id not in json.dumps(run_finding.to_dict())


def test_repeated_context_and_oversized_observation_are_digest_only():
    context = RunContext.create()
    secret_context = "do-not-persist-this-context"
    detector = StreamingGuardrailDetector(
        GuardrailConfig(
            context_repeat_threshold=3,
            context_repeat_window=4,
            min_context_size=1,
            max_observation_size=10,
        )
    )
    for sequence in range(1, 4):
        detector.observe(
            _event(context, sequence, input_context=secret_context)
        )
    detector.observe(
        _event(
            context,
            4,
            event_type="tool_call",
            tool=_tool("read_file", observation_size=11),
        )
    )
    context_finding = _find(detector, "guardrail.context_reinjection")[0]
    observation_finding = _find(detector, "guardrail.oversized_observation")[0]
    assert context_finding.evidence["repeat_count"] == 3
    assert observation_finding.evidence["maximum_observation_size"] == 11
    assert secret_context not in json.dumps(detector.to_dicts())


def test_failed_action_requires_same_state_and_ignores_expected_failures():
    context = RunContext.create()
    detector = StreamingGuardrailDetector(GuardrailConfig(failed_action_threshold=3))
    for sequence in range(1, 4):
        detector.observe(
            _event(
                context,
                sequence,
                event_type="tool_call",
                tool=_tool("edit", status="error", error_type="Conflict"),
                metadata={"state_digest": "unchanged"},
            )
        )
    finding = _find(detector, "guardrail.failed_action_stall")[0]
    assert finding.evidence["failure_count"] == 3
    assert finding.evidence["state_basis"] == "metadata.state_digest"

    changed = StreamingGuardrailDetector(GuardrailConfig(failed_action_threshold=3))
    for sequence in range(1, 4):
        changed.observe(
            _event(
                context,
                sequence,
                event_type="tool_call",
                tool=_tool("edit", status="error", error_type="Conflict"),
                metadata={"state_version": sequence},
            )
        )
    changed.observe(
        _event(
            context,
            4,
            event_type="tool_call",
            tool=_tool(
                "edit",
                status="error",
                error_type="Expected",
                expected_failure=True,
            ),
            metadata={"state_version": 3},
        )
    )
    assert not _find(changed, "guardrail.failed_action_stall")


@pytest.mark.parametrize("provenance", ["estimated", "missing"])
def test_non_provider_usage_never_allows_hard_budget_enforcement(provenance):
    context = RunContext.create()
    detector = StreamingGuardrailDetector(
        GuardrailConfig(token_budget=100, budget_warning_fraction=0.8)
    )
    usage = (
        TokenUsage(input=80, provenance=provenance)
        if provenance == "estimated"
        else TokenUsage(provenance=provenance)
    )
    detector.observe(_event(context, 1, tokens=usage))
    findings = _find(detector, "guardrail.token_budget")
    if provenance == "missing":
        assert findings == []
    else:
        assert findings[0].enforcement_eligible is False
        assert findings[0].estimate_status == "estimated_or_incomplete"
        assert findings[0].evidence["token_provenance"] == "estimated"


def test_provider_reported_token_budget_is_exact_but_cost_remains_opt_in():
    context = RunContext.create()
    detector = StreamingGuardrailDetector(
        GuardrailConfig(
            token_budget=100,
            cost_budget_usd=1.0,
            cost_per_million_tokens_usd=10_000.0,
            budget_warning_fraction=0.8,
        )
    )
    detector.observe(
        _event(
            context,
            1,
            tokens=TokenUsage(input=80, provenance="provider_reported"),
        )
    )
    token_finding = _find(detector, "guardrail.token_budget")[0]
    cost_finding = _find(detector, "guardrail.cost_budget")[0]
    assert token_finding.enforcement_eligible is True
    assert token_finding.estimate_status == "exact"
    assert cost_finding.evidence["estimated_cost_usd"] == 0.8
    assert cost_finding.enforcement_eligible is False
    assert cost_finding.estimate_status == "estimated_cost"


def test_mixed_usage_is_advisory_and_output_is_deterministic():
    context = RunContext.create()
    events = [
        _event(
            context,
            1,
            tokens=TokenUsage(input=60, provenance="provider_reported"),
        ),
        _event(
            context,
            2,
            tokens=TokenUsage(input=20, provenance="estimated"),
        ),
    ]
    config = GuardrailConfig(token_budget=100, budget_warning_fraction=0.8)
    first = StreamingGuardrailDetector(config)
    second = StreamingGuardrailDetector(config)
    assert [finding.to_dict() for finding in first.observe_many(events)] == [
        finding.to_dict() for finding in second.observe_many(events)
    ]
    finding = _find(first, "guardrail.token_budget")[0]
    assert finding.enforcement_eligible is False
    assert finding.evidence["token_provenance"] == "mixed"


def test_processor_persists_streaming_findings_automatically(tmp_path):
    processor = TraceRazorProcessor(
        context=RunContext.create(),
        policy=AuditPolicy(artifact_dir=str(tmp_path)),
        guardrails=GuardrailConfig(tool_repeat_threshold=2, tool_repeat_window=3),
    )
    for _ in range(2):
        processor.record(
            "tool_call",
            tool=_tool("search"),
            tokens=TokenUsage(input=1, provenance="provider_reported"),
        )
    processor.finalize()
    payload = json.loads((processor.run_dir / "findings.json").read_text(encoding="utf-8"))
    findings = payload["findings"]
    assert [finding["signal_id"] for finding in findings] == [
        "guardrail.tool_call_repeat"
    ]
    assert findings[0]["schema_version"] == "tracerazor-guardrail-finding/v1"
    assert findings[0]["enforcement_eligible"] is True


def test_invalid_guardrail_configuration_fails_closed():
    with pytest.raises(ValueError, match="tool_repeat_window"):
        GuardrailConfig(tool_repeat_threshold=3, tool_repeat_window=2)
    with pytest.raises(ValueError, match="configured together"):
        GuardrailConfig(cost_budget_usd=1.0)
    with pytest.raises(ValueError, match="budget_warning_fraction"):
        GuardrailConfig(budget_warning_fraction=0)


def test_processor_rejects_detector_bound_to_another_run(tmp_path):
    first_context = RunContext.create()
    detector = StreamingGuardrailDetector()
    detector.observe(_event(first_context, 1))
    with pytest.raises(ValueError, match="run_id does not match"):
        TraceRazorProcessor(
            context=RunContext.create(),
            policy=AuditPolicy(artifact_dir=str(tmp_path)),
            guardrails=detector,
        )
