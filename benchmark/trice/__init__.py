"""TRICE: TraceRazor Information Control Engine.

V1 converts TraceRazor traces into typed context segments, scores each segment
from agent/harness/transformer perspectives, solves a token-budgeted policy,
and replays the policy against recorded traces.

V2 adds live rollout: fresh workspaces, real edits, verifier commands, and a
small user preference profile that adapts budgets from feedback and outcomes.
"""

from .adapters import JsonPatchAdapter, PatchEdit, RepairAdapter
from .evidence import EvidenceManifest, build_manifest, canonical_json, verify_manifest
from .learn import LearningWeights, PolicyUpdate, update_weights
from .policy import ContextPolicy, PolicyDecision, solve_policy
from .replay import ReplayMetrics, evaluate_policy
from .render import render_context, render_policy_json
from .score import ActionCandidate, SegmentScore, ScoreWeights, score_segment
from .segment import Segment, SegmentState, segments_from_trace
from .schemas import load_schema, schema_path, validate_patch_spec, validate_patch_spec_file
from .stats import ClaimGate, ConfidenceInterval, bootstrap_mean_ci, claim_gate_from_rounds, clustered_bootstrap_mean_ci, wilson_ci
from .suite import SuiteRunResult, SuiteTaskRun, SuiteTaskSpec, run_suite_manifest, validate_suite_manifest_file, verify_suite_evidence
from .user import UserPreferenceProfile

__all__ = [
    "ActionCandidate",
    "ContextPolicy",
    "ClaimGate",
    "ConfidenceInterval",
    "EvidenceManifest",
    "JsonPatchAdapter",
    "LearningWeights",
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
