from __future__ import annotations

import json
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tracerazor.runtime import (
    AuditPolicy,
    DiskSpoolReceiver,
    PrivacyMode,
    RunContext,
    RuntimeEvent,
    TaskResult,
    TokenUsage,
    ToolCall,
    TraceRazorProcessor,
    recover_partial_run,
)
from tracerazor.errors import BinaryNotFoundError
from tracerazor._launcher import find_binary


def _fake_report(trace, *, secret: str | None = None):
    marker = secret or "derived"
    trace_sha256 = hashlib.sha256(
        json.dumps(
            trace,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "tracerazor-report/v1",
        "trace_id": trace["trace_id"],
        "agent_name": trace["agent_name"],
        "framework": trace["framework"],
        "total_steps": len(trace["steps"]),
        "total_tokens": trace["total_tokens"],
        "score": {
            "score": 81.0,
            "grade": "Good",
            "passes_threshold": True,
            "metric_normalised": {"srr": 0.8},
            marker: 1.0,
        },
        "diff": [
            {
                "action": "trim",
                "step_id": 1,
                "step_type": "reasoning",
                "description": f"excerpt {marker}",
                "justification": f"because {marker}",
                "tokens_actual": 20,
                "tokens_suggested": 10,
            }
        ],
        "savings": {"tokens_saved": 10, "reduction_pct": 5.0, marker: 99},
        "fixes": [
            {
                "fix_type": "dedupe",
                "target": f"target {marker}",
                "patch": f"patch {marker}",
                "estimated_token_savings": 10,
                "risk": "safe",
                marker: "arbitrary secret key",
            }
        ],
        "summary": f"summary {marker}",
        "manifest": {
            "trace_sha256": trace_sha256,
            "tool_version": "1.1.0",
            "created_at": "2026-07-10T00:00:00Z",
            "similarity_backend": "bow",
            "weights": {"srr": 1.0, marker: 1.0},
            "weights_sha256": "b" * 64,
            "threshold": 70.0,
            "min_steps": 2,
            "hermetic": True,
        },
    }


def _successful_auditor(trace, *, hermetic, min_steps):
    assert hermetic is True
    assert len(trace["steps"]) >= min_steps
    return _fake_report(trace)


def _event(context: RunContext, n: int) -> RuntimeEvent:
    return RuntimeEvent.create(
        context,
        event_type="reasoning",
        host="test",
        framework="raw",
        tokens=TokenUsage(input=n, output=1, provenance="provider_reported"),
        content=f"content-{n}",
        sequence=n,
    )


def test_default_persistence_contains_hashes_but_no_raw_content(tmp_path):
    secret = "sk-secret-value-that-must-never-hit-disk"
    captured = {}

    def auditor(trace, *, hermetic, min_steps):
        captured["trace"] = trace
        captured["hermetic"] = hermetic
        captured["min_steps"] = min_steps
        return _fake_report(trace, secret=secret)

    policy = AuditPolicy(artifact_dir=str(tmp_path), min_steps=2)
    processor = TraceRazorProcessor(
        context=RunContext.create(agent_id="agent"),
        policy=policy,
        host="codex",
        framework="openai-agents",
        auditor=auditor,
    )
    processor.record(
        "reasoning",
        content=f"Use credential {secret}",
        input_context=f"system prompt {secret}",
        output=f"result {secret}",
        metadata={"prompt": secret},
        tokens=TokenUsage(input=20, output=5, provenance="provider_reported"),
        task=TaskResult(outcome="passed", verifier="pytest", evidence={"log": secret}),
    )
    processor.record(
        "reasoning",
        content=f"Second raw step {secret}",
        tokens=TokenUsage(input=2, output=1, provenance="provider_reported"),
    )
    manifest = processor.finalize(findings=[{"excerpt": secret}])

    all_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in processor.run_dir.iterdir()
        if path.is_file()
    )
    assert secret not in all_text
    assert "sha256=" in all_text
    assert processor.native_trace["steps"][0]["content"].endswith(secret)
    assert captured["trace"]["steps"][0]["content"].endswith(secret)
    assert captured == {
        "trace": processor.native_trace,
        "hermetic": True,
        "min_steps": 2,
    }
    assert manifest["capture_quality"] == "complete"
    assert manifest["degraded_ingest"] is False
    assert manifest["raw_content_persisted"] is False
    assert set(manifest["files"]) >= {
        "manifest.json",
        "events.jsonl",
        "trace.json",
        "findings.json",
        "validation.json",
        "report.json",
    }
    persisted_trace = json.loads((processor.run_dir / "trace.json").read_text(encoding="utf-8"))
    assert persisted_trace["steps"][0]["content"].startswith("[redacted sha256=")
    assert persisted_trace["metadata"]["tracerazor_redacted"] is True
    assert persisted_trace["metadata"]["persisted_representation"] == "redacted_non_auditable"
    assert persisted_trace["metadata"]["capture_quality"] == "degraded"
    assert len(persisted_trace["metadata"]["source_trace_sha256"]) == 64
    persisted_report = json.loads((processor.run_dir / "report.json").read_text(encoding="utf-8"))
    assert persisted_report["tracerazor_redacted"] is True
    assert persisted_report["fixes"][0]["patch"] == "[redacted fix patch]"
    assert persisted_report["diff"][0]["description"] == "[redacted audit description]"
    persisted_findings = json.loads(
        (processor.run_dir / "findings.json").read_text(encoding="utf-8")
    )["findings"]
    audit_finding = next(
        finding for finding in persisted_findings if finding.get("source") == "native_audit"
    )
    assert audit_finding["fix_type"] == "dedupe"
    assert audit_finding["estimate_status"] == "estimated"
    assert audit_finding["evidence_ref"] == "report.json#/fixes/0"
    assert "patch" not in audit_finding


def test_untrusted_mapping_keys_are_redacted(tmp_path):
    secret_key = "sk-secret-stored-as-a-key"
    processor = TraceRazorProcessor(
        policy=AuditPolicy(artifact_dir=str(tmp_path)),
        context=RunContext.create(),
    )
    processor.record(
        "reasoning",
        metadata={secret_key: "value"},
        tokens=TokenUsage(input=1, output=1, provenance="provider_reported"),
        task=TaskResult(
            outcome="passed",
            verifier="pytest",
            evidence={secret_key: "evidence"},
        ),
    )
    processor.finalize(findings=[{"evidence": {secret_key: "finding"}}])
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in processor.run_dir.iterdir()
        if path.is_file()
    )
    assert secret_key not in persisted
    assert "redacted-key sha256=" in persisted


def test_environment_run_id_cannot_escape_artifact_directory(tmp_path):
    with pytest.raises(ValueError, match="safe path segment"):
        RunContext.from_env(
            {
                "TRACERAZOR_RUN_ID": "../outside",
                "TRACERAZOR_AGENT_ID": "agent",
            }
        )


def test_raw_content_requires_explicit_raw_policy(tmp_path):
    with pytest.raises(ValueError, match="privacy = 'raw'"):
        AuditPolicy(persist_raw_content=True)
    policy = AuditPolicy(
        privacy=PrivacyMode.RAW,
        persist_raw_content=True,
        artifact_dir=str(tmp_path),
    )
    processor = TraceRazorProcessor(policy=policy, context=RunContext.create())
    processor.record(
        "reasoning",
        content="explicitly persisted",
        tokens=TokenUsage(input=1, output=1, provenance="provider_reported"),
    )
    processor.finalize()
    assert "explicitly persisted" in (processor.run_dir / "events.jsonl").read_text(encoding="utf-8")


def test_estimated_usage_is_degraded_and_never_enforcement_eligible(tmp_path):
    policy = AuditPolicy(
        mode="enforce",
        enforcement_enabled=True,
        verifier="pytest",
        artifact_dir=str(tmp_path),
        min_steps=2,
    )
    processor = TraceRazorProcessor(
        policy=policy, context=RunContext.create(), auditor=_successful_auditor
    )
    for _ in range(2):
        processor.record(
            "reasoning",
            content="estimated",
            tokens=TokenUsage(input=10, output=1, provenance="estimated"),
            task=TaskResult(outcome="passed", verifier="pytest"),
        )
    manifest = processor.finalize()
    assert manifest["capture_quality"] == "degraded"
    assert manifest["degraded_ingest"] is True
    assert manifest["enforcement_eligible"] is False
    assert "degraded_or_non_provider_token_usage" in manifest["enforcement_ineligible_reasons"]


def test_self_attested_verifier_never_enables_enforcement(tmp_path):
    policy = AuditPolicy(
        mode="enforce",
        enforcement_enabled=True,
        verifier="pytest",
        artifact_dir=str(tmp_path),
        min_steps=2,
    )
    processor = TraceRazorProcessor(
        policy=policy, context=RunContext.create(), auditor=_successful_auditor
    )
    for _ in range(2):
        processor.record(
            "reasoning",
            content="provider accounted",
            tokens=TokenUsage(input=10, output=1, provenance="provider_reported"),
            task=TaskResult(
                outcome="passed",
                verifier="pytest",
                evidence={
                    "trusted_verifier_receipt": {
                        "verifier": "pytest",
                        "exit_code": 0,
                        "evidence_sha256": "a" * 64,
                    }
                },
            ),
        )
    manifest = processor.finalize()
    assert manifest["capture_quality"] == "complete"
    assert manifest["enforcement_eligible"] is False
    assert "trusted_verifier_receipt_missing" in manifest["enforcement_ineligible_reasons"]


def test_context_manager_marks_exceptional_run_partial_without_leaking_error(tmp_path):
    secret = "exception-secret"
    holder = None
    with pytest.raises(RuntimeError, match=secret):
        with TraceRazorProcessor(
            policy=AuditPolicy(artifact_dir=str(tmp_path)),
            context=RunContext.create(),
        ) as processor:
            holder = processor
            processor.record(
                "reasoning",
                content="started",
                tokens=TokenUsage(input=1, output=0, provenance="provider_reported"),
            )
            raise RuntimeError(secret)
    assert holder is not None
    manifest = json.loads((holder.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert manifest["capture_quality"] == "partial"
    assert manifest["enforcement_eligible"] is False
    assert secret not in (holder.run_dir / "validation.json").read_text(encoding="utf-8")


def test_running_manifest_can_be_recovered_as_partial(tmp_path):
    processor = TraceRazorProcessor(
        policy=AuditPolicy(artifact_dir=str(tmp_path)),
        context=RunContext.create(),
    )
    processor.record(
        "reasoning",
        content="before crash",
        tokens=TokenUsage(input=1, output=0, provenance="provider_reported"),
    )
    recovered = recover_partial_run(processor.run_dir)
    assert recovered["status"] == "partial"
    assert recovered["degraded_ingest"] is True
    assert recovered["enforcement_eligible"] is False


def test_disk_spool_receiver_supports_concurrent_atomic_append(tmp_path):
    context = RunContext.create()
    receiver = DiskSpoolReceiver(tmp_path / context.run_id)
    count = 80
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(receiver.receive, [_event(context, n + 1) for n in range(count)]))
    records = receiver.records()
    assert len(records) == count
    assert len({record["event_id"] for record in records}) == count
    assert not receiver.lock_path.exists()
    assert not list(receiver.run_dir.glob("*.tmp"))


def test_crash_truncated_tail_is_ignored_and_flagged(tmp_path):
    context = RunContext.create()
    receiver = DiskSpoolReceiver(tmp_path / context.run_id)
    receiver.receive(_event(context, 1))
    with receiver.path.open("ab") as handle:
        handle.write(b'{"schema_version":"tracerazor-event/v1"')
    records = receiver.records()
    assert len(records) == 1
    assert receiver.partial_tail is True
    with pytest.raises(ValueError, match="malformed"):
        receiver.records(allow_partial_tail=False)


def test_native_audit_failure_is_explicit_partial_and_has_no_report(tmp_path):
    def missing_binary(trace, *, hermetic, min_steps):
        assert hermetic is True
        raise BinaryNotFoundError("secret path to missing binary")

    processor = TraceRazorProcessor(
        policy=AuditPolicy(artifact_dir=str(tmp_path), min_steps=2),
        context=RunContext.create(),
        auditor=missing_binary,
    )
    for index in range(2):
        processor.record(
            "reasoning",
            content=f"raw-{index}",
            tokens=TokenUsage(input=1, output=1, provenance="provider_reported"),
        )
    manifest = processor.finalize()
    assert manifest["status"] == "partial"
    assert manifest["audit"]["status"] == "failed"
    assert "auditor_binary_missing" in manifest["audit"]["issues"]
    assert manifest["enforcement_eligible"] is False
    assert not (processor.run_dir / "report.json").exists()
    assert "secret path" not in (processor.run_dir / "validation.json").read_text(encoding="utf-8")


def test_short_run_is_partial_and_never_calls_auditor(tmp_path):
    called = False

    def must_not_run(trace, *, hermetic, min_steps):
        nonlocal called
        called = True
        return _fake_report(trace)

    processor = TraceRazorProcessor(
        policy=AuditPolicy(artifact_dir=str(tmp_path), min_steps=2),
        context=RunContext.create(),
        auditor=must_not_run,
    )
    processor.record(
        "reasoning",
        content="one step",
        tokens=TokenUsage(input=1, provenance="provider_reported"),
    )
    manifest = processor.finalize()
    assert called is False
    assert manifest["status"] == "partial"
    assert manifest["audit"]["status"] == "skipped"
    assert "below_min_steps" in manifest["audit"]["issues"]


def test_child_only_spools_and_parent_aggregates_without_overwrite(tmp_path):
    captured = {}

    def auditor(trace, *, hermetic, min_steps):
        captured["trace"] = trace
        return _fake_report(trace)

    policy = AuditPolicy(artifact_dir=str(tmp_path), min_steps=2)
    parent_context = RunContext.create(agent_id="parent")
    parent = TraceRazorProcessor(
        policy=policy,
        context=parent_context,
        auditor=auditor,
    )
    for index in range(2):
        parent.record(
            "reasoning",
            content=f"parent raw {index}",
            tokens=TokenUsage(input=1, provenance="provider_reported"),
        )
    manifest_before_child = (parent.run_dir / "manifest.json").read_bytes()

    child_context = RunContext.from_env(parent_context.spawn_env(child_agent_id="worker"))
    child = TraceRazorProcessor(policy=policy, context=child_context, auditor=auditor)
    for index in range(2):
        child.record(
            "reasoning",
            content=f"child secret raw {index}",
            tokens=TokenUsage(input=1, provenance="provider_reported"),
        )
    receipt = child.finalize()
    assert receipt["schema_version"] == "tracerazor-run-receipt/v1"
    assert receipt["agent_id"] == "worker"
    assert receipt["parent_agent_id"] == "parent"
    assert receipt["audit_status"] == "deferred_to_parent"
    assert (parent.run_dir / "manifest.json").read_bytes() == manifest_before_child
    assert not (parent.run_dir / "trace.json").exists()
    assert not (parent.run_dir / "report.json").exists()
    assert (parent.run_dir / "receipts" / f"{child_context.span_id}.json").is_file()

    manifest = parent.finalize()
    assert manifest["event_count"] == 4
    assert manifest["child_agent_ids"] == ["worker"]
    assert manifest["child_receipt_count"] == 1
    assert manifest["capture_quality"] == "degraded"
    assert "redacted_child_content" in manifest["ingest_quality"]["issues"]
    assert [step["agent_id"] for step in captured["trace"]["steps"]].count("worker") == 2
    child_steps = [step for step in captured["trace"]["steps"] if step["agent_id"] == "worker"]
    assert all(step["content"].startswith("[redacted sha256=") for step in child_steps)
    assert "child secret raw" not in json.dumps(captured["trace"])


@pytest.mark.parametrize("receipt_state", ["missing", "partial", "invalid"])
def test_parent_is_partial_for_incomplete_child_receipt(tmp_path, receipt_state):
    policy = AuditPolicy(artifact_dir=str(tmp_path), min_steps=2)
    parent_context = RunContext.create(agent_id="parent")
    parent = TraceRazorProcessor(
        policy=policy,
        context=parent_context,
        auditor=_successful_auditor,
    )
    parent.record(
        "reasoning",
        content="parent",
        tokens=TokenUsage(input=1, provenance="provider_reported"),
    )
    child = TraceRazorProcessor(
        policy=policy,
        context=RunContext.from_env(parent_context.spawn_env(child_agent_id="worker")),
        auditor=_successful_auditor,
    )
    child.record(
        "reasoning",
        content="child",
        tokens=TokenUsage(input=1, provenance="provider_reported"),
    )
    if receipt_state == "partial":
        child.finalize(status="partial")
    elif receipt_state == "invalid":
        child.finalize()
        receipt_path = parent.run_dir / "receipts" / f"{child.context.span_id}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["event_ids"] = []
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    manifest = parent.finalize()
    assert manifest["status"] == "partial"
    assert manifest["capture_quality"] == "partial"
    assert "incomplete_child_run" in manifest["ingest_quality"]["issues"]
    assert manifest["enforcement_eligible"] is False

    if receipt_state == "missing":
        # A late receipt must not retroactively make already-finalized root
        # artifacts look complete.
        child.finalize()
        assert parent.finalize()["status"] == "partial"


def test_raw_policy_allows_parent_to_audit_raw_child_content(tmp_path):
    captured = {}

    def auditor(trace, *, hermetic, min_steps):
        captured["trace"] = trace
        return _fake_report(trace)

    policy = AuditPolicy(
        privacy=PrivacyMode.RAW,
        persist_raw_content=True,
        artifact_dir=str(tmp_path),
        min_steps=2,
    )
    parent_context = RunContext.create(agent_id="parent")
    parent = TraceRazorProcessor(policy=policy, context=parent_context, auditor=auditor)
    child = TraceRazorProcessor(
        policy=policy,
        context=RunContext.from_env(parent_context.spawn_env(child_agent_id="worker")),
        auditor=auditor,
    )
    parent.record("reasoning", content="parent raw", tokens=TokenUsage(input=1, provenance="provider_reported"))
    child.record("reasoning", content="child raw", tokens=TokenUsage(input=1, provenance="provider_reported"))
    child.finalize()
    manifest = parent.finalize()
    assert manifest["capture_quality"] == "complete"
    assert any(step["content"] == "child raw" for step in captured["trace"]["steps"])
    persisted_trace = json.loads((parent.run_dir / "trace.json").read_text(encoding="utf-8"))
    assert "tracerazor_redacted" not in persisted_trace["metadata"]
    persisted_bytes = (parent.run_dir / "trace.json").read_bytes()
    assert hashlib.sha256(persisted_bytes).hexdigest() == manifest["audit"]["source_trace_sha256"]


def test_raw_native_report_replays_when_binary_is_available(tmp_path):
    binary = find_binary()
    if binary is None:
        pytest.skip("native TraceRazor auditor is unavailable")
    policy = AuditPolicy(
        privacy=PrivacyMode.RAW,
        persist_raw_content=True,
        artifact_dir=str(tmp_path),
        min_steps=2,
    )
    processor = TraceRazorProcessor(policy=policy, context=RunContext.create())
    for index in range(2):
        processor.record(
            "reasoning",
            content=f"native replay {index}",
            tokens=TokenUsage(input=4, output=1, provenance="provider_reported"),
        )
    manifest = processor.finalize()
    assert manifest["audit"]["status"] == "completed"
    result = subprocess.run(
        [
            binary,
            "verify",
            str(processor.run_dir / "report.json"),
            str(processor.run_dir / "trace.json"),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_redaction_covers_bytes_custom_objects_tool_errors_and_keys(tmp_path):
    secret = "TOPSECRET-never-persist"

    class Hostile:
        def __str__(self):
            return secret

    processor = TraceRazorProcessor(
        policy=AuditPolicy(artifact_dir=str(tmp_path)),
        context=RunContext.create(),
    )
    processor.record(
        "tool_call",
        metadata={secret: b"TOPSECRET-bytes", "object": Hostile()},
        tool=ToolCall.from_arguments("tool", {}, status="error", error_type=secret),
        tokens=TokenUsage(input=1, provenance="provider_reported"),
    )
    processor.finalize(findings=[{secret: Hostile()}])
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in processor.run_dir.rglob("*")
        if path.is_file()
    )
    assert secret not in persisted
    assert "TOPSECRET-bytes" not in persisted
    assert "redacted-bytes sha256=" in persisted
    assert "redacted-key sha256=" in persisted


def test_artifact_directory_traversal_and_absolute_policy_path_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="parent-directory traversal"):
        AuditPolicy(artifact_dir="../outside")

    policy_path = tmp_path / "tracerazor.toml"
    policy_path.write_text(
        f'artifact_dir = "{tmp_path.as_posix()}"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="relative to the policy root"):
        AuditPolicy.load(policy_path)


def test_spool_rejects_symlink_parent_and_leaf(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    context = RunContext.create()
    receiver = DiskSpoolReceiver(linked / context.run_id)
    with pytest.raises(ValueError, match="symlink or junction"):
        receiver.receive(_event(context, 1))

    leaf_run = tmp_path / "leaf" / context.run_id
    leaf_run.mkdir(parents=True)
    target = tmp_path / "outside-events.jsonl"
    target.write_text("do not touch", encoding="utf-8")
    (leaf_run / "events.jsonl").symlink_to(target)
    leaf_receiver = DiskSpoolReceiver(leaf_run)
    with pytest.raises(ValueError, match="symlink or junction"):
        leaf_receiver.receive(_event(context, 2))
    assert target.read_text(encoding="utf-8") == "do not touch"


def test_spool_rejects_leaf_link_before_open_even_when_links_are_unavailable(tmp_path, monkeypatch):
    import tracerazor.runtime.persistence as persistence

    context = RunContext.create()
    receiver = DiskSpoolReceiver(tmp_path / context.run_id)
    original = persistence._is_link_or_reparse

    def simulated_link(path):
        return Path(path) == receiver.path or original(Path(path))

    monkeypatch.setattr(persistence, "_is_link_or_reparse", simulated_link)
    with pytest.raises(ValueError, match="symlink or junction"):
        receiver.receive(_event(context, 1))
