"""Public TRICE library API.

This package is the stable user-facing import path. The implementation still
lives in ``benchmark.trice`` while the research harness matures, but user code
should import from ``tracerazor.trice``.
"""

from benchmark.trice.adapters import CommandRepairAdapter, JsonPatchAdapter, PatchEdit, RepairAdapter
from benchmark.trice.bundle import BundleManifest, export_evidence_bundle, verify_evidence_bundle
from benchmark.trice.claim import build_claim_card, render_claim_card_markdown, render_claim_card_tex, render_claim_ladder_svg, verify_claim_card_file, write_claim_outputs
from benchmark.trice.doctor import doctor_report, render_doctor_text
from benchmark.trice.evidence import (
    EvidenceManifest,
    build_manifest,
    canonical_json,
    verify_manifest,
)
from benchmark.trice.learn import LearningWeights, PolicyUpdate, update_weights
from benchmark.trice.live import LiveRolloutResult, LiveRound, LiveTask, run_live_learning_loop
from benchmark.trice.policy import ContextPolicy, PolicyDecision, solve_policy
from benchmark.trice.provenance import TreeFingerprint, fingerprint_tree, hash_file
from benchmark.trice.receipt import validate_run_receipt, validate_run_receipt_file
from benchmark.trice.readiness import build_suite_readiness, render_readiness_markdown, render_readiness_svg, render_readiness_tex, verify_readiness_file, write_readiness_outputs
from benchmark.trice.replay import ReplayMetrics, evaluate_policy
from benchmark.trice.render import render_context, render_policy_json
from benchmark.trice.score import ActionCandidate, ScoreWeights, SegmentScore, score_segment
from benchmark.trice.segment import Segment, SegmentState, segments_from_trace
from benchmark.trice.schemas import load_schema, schema_path, validate_adapter_profile, validate_adapter_profile_file, validate_patch_spec, validate_patch_spec_file, validate_suite_manifest_file
from benchmark.trice.stats import ClaimGate, ConfidenceInterval, bootstrap_mean_ci, claim_gate_from_rounds, clustered_bootstrap_mean_ci, wilson_ci
from benchmark.trice.suite import SuiteRunResult, SuiteTaskRun, SuiteTaskSpec, run_suite_manifest, scaffold_suite_manifest, verify_suite_evidence
from benchmark.trice.user import UserPreferenceProfile

__all__ = [
    "ActionCandidate",
    "BundleManifest",
    "ClaimGate",
    "CommandRepairAdapter",
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
    "TreeFingerprint",
    "UserPreferenceProfile",
    "bootstrap_mean_ci",
    "build_manifest",
    "build_claim_card",
    "build_suite_readiness",
    "canonical_json",
    "claim_gate_from_rounds",
    "clustered_bootstrap_mean_ci",
    "doctor_report",
    "evaluate_policy",
    "export_evidence_bundle",
    "fingerprint_tree",
    "hash_file",
    "load_schema",
    "render_context",
    "render_claim_card_markdown",
    "render_claim_card_tex",
    "render_claim_ladder_svg",
    "render_readiness_markdown",
    "render_readiness_svg",
    "render_readiness_tex",
    "render_doctor_text",
    "render_policy_json",
    "run_live_learning_loop",
    "run_suite_manifest",
    "scaffold_suite_manifest",
    "score_segment",
    "schema_path",
    "segments_from_trace",
    "solve_policy",
    "update_weights",
    "validate_adapter_profile",
    "validate_adapter_profile_file",
    "validate_patch_spec",
    "validate_patch_spec_file",
    "validate_run_receipt",
    "validate_run_receipt_file",
    "validate_suite_manifest_file",
    "verify_manifest",
    "verify_evidence_bundle",
    "verify_claim_card_file",
    "verify_readiness_file",
    "verify_suite_evidence",
    "write_readiness_outputs",
    "write_claim_outputs",
    "wilson_ci",
]
