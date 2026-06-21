"""Public TRICE library API.

This package is the stable user-facing import path. The implementation still
lives in ``benchmark.trice`` while the research harness matures, but user code
should import from ``tracerazor.trice``.
"""

from benchmark.trice.adapters import JsonPatchAdapter, PatchEdit, RepairAdapter
from benchmark.trice.evidence import (
    EvidenceManifest,
    build_manifest,
    canonical_json,
    verify_manifest,
)
from benchmark.trice.learn import LearningWeights, PolicyUpdate, update_weights
from benchmark.trice.live import LiveRolloutResult, LiveRound, LiveTask, run_live_learning_loop
from benchmark.trice.policy import ContextPolicy, PolicyDecision, solve_policy
from benchmark.trice.replay import ReplayMetrics, evaluate_policy
from benchmark.trice.render import render_context, render_policy_json
from benchmark.trice.score import ActionCandidate, ScoreWeights, SegmentScore, score_segment
from benchmark.trice.segment import Segment, SegmentState, segments_from_trace
from benchmark.trice.schemas import load_schema, schema_path, validate_patch_spec, validate_patch_spec_file, validate_suite_manifest_file
from benchmark.trice.stats import ClaimGate, ConfidenceInterval, bootstrap_mean_ci, claim_gate_from_rounds, clustered_bootstrap_mean_ci, wilson_ci
from benchmark.trice.suite import SuiteRunResult, SuiteTaskRun, SuiteTaskSpec, run_suite_manifest, verify_suite_evidence
from benchmark.trice.user import UserPreferenceProfile

__all__ = [
    "ActionCandidate",
    "ClaimGate",
    "ConfidenceInterval",
    "ContextPolicy",
    "EvidenceManifest",
    "JsonPatchAdapter",
    "LearningWeights",
    "LiveRolloutResult",
    "LiveRound",
    "LiveTask",
    "PatchEdit",
    "PolicyDecision",
    "PolicyUpdate",
    "ReplayMetrics",
    "RepairAdapter",
    "ScoreWeights",
    "Segment",
    "SegmentScore",
    "SegmentState",
    "SuiteRunResult",
    "SuiteTaskRun",
    "SuiteTaskSpec",
    "UserPreferenceProfile",
    "bootstrap_mean_ci",
    "build_manifest",
    "canonical_json",
    "claim_gate_from_rounds",
    "clustered_bootstrap_mean_ci",
    "evaluate_policy",
    "load_schema",
    "render_context",
    "render_policy_json",
    "run_live_learning_loop",
    "run_suite_manifest",
    "score_segment",
    "schema_path",
    "segments_from_trace",
    "solve_policy",
    "update_weights",
    "validate_patch_spec",
    "validate_patch_spec_file",
    "validate_suite_manifest_file",
    "verify_manifest",
    "verify_suite_evidence",
    "wilson_ci",
]
