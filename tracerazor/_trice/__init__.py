"""TRICE: TraceRazor Information Control Engine.

V1 converts TraceRazor traces into typed context segments, scores each segment
from agent/harness/transformer perspectives, solves a token-budgeted policy,
and replays the policy against recorded traces.

V2 adds live rollout: fresh workspaces, real edits, verifier commands, and a
small user preference profile that adapts budgets from feedback and outcomes.
"""

from .adapters import CommandRepairAdapter, JsonPatchAdapter, PatchEdit, RepairAdapter
from .bundle import BundleManifest, export_evidence_bundle, verify_evidence_bundle
from .contract import build_contract_card, verify_contract_card_file
from .crates import build_crates_card, verify_crates_card_file
from .design import build_design_card, verify_design_card_file
from .evidence import EvidenceManifest, build_manifest, canonical_json, verify_manifest
from .integrity import build_integrity_card, verify_integrity_card_file
from .install import build_install_card, verify_install_card_file
from .learn import LearningWeights, PolicyUpdate, update_weights
from .policy import ContextPolicy, PolicyDecision, solve_policy
from .recall import EvidenceRecallReport, evidence_recall_from_policy
from .protocol import build_protocol_lock, verify_protocol_lock_file
from .provenance import TreeFingerprint, fingerprint_tree, hash_file
from .release import build_release_card, verify_release_card_file
from .release_evidence import build_release_evidence_card, verify_release_evidence_file
from .replay import ReplayMetrics, evaluate_policy
from .render import render_context, render_policy_json
from .reproduction import build_reproduction_card, verify_reproduction_card_file
from .research import build_research_card, verify_research_card_file
from .score import ActionCandidate, SegmentScore, ScoreWeights, score_segment
from .segment import Segment, SegmentState, segments_from_trace
from .receipt import validate_run_receipt, validate_run_receipt_file
from .schemas import load_schema, schema_path, validate_adapter_profile, validate_adapter_profile_file, validate_patch_spec, validate_patch_spec_file
from .stats import ClaimGate, ConfidenceInterval, bootstrap_mean_ci, claim_gate_from_rounds, clustered_bootstrap_mean_ci, wilson_ci
from .suite import SuiteRunResult, SuiteTaskRun, SuiteTaskSpec, run_suite_manifest, validate_suite_manifest_file, verify_suite_evidence
from .user import UserPreferenceProfile

__all__ = [
    "ActionCandidate",
    "BundleManifest",
    "ContextPolicy",
    "ClaimGate",
    "ConfidenceInterval",
    "CommandRepairAdapter",
    "EvidenceManifest",
    "EvidenceRecallReport",
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
    "TreeFingerprint",
    "UserPreferenceProfile",
    "bootstrap_mean_ci",
    "build_manifest",
    "build_crates_card",
    "build_integrity_card",
    "build_install_card",
    "build_contract_card",
    "build_design_card",
    "build_protocol_lock",
    "build_release_card",
    "build_release_evidence_card",
    "build_reproduction_card",
    "build_research_card",
    "canonical_json",
    "claim_gate_from_rounds",
    "clustered_bootstrap_mean_ci",
    "evaluate_policy",
    "evidence_recall_from_policy",
    "export_evidence_bundle",
    "fingerprint_tree",
    "hash_file",
    "load_schema",
    "render_context",
    "render_policy_json",
    "run_suite_manifest",
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
    "verify_crates_card_file",
    "verify_integrity_card_file",
    "verify_install_card_file",
    "verify_contract_card_file",
    "verify_evidence_bundle",
    "verify_design_card_file",
    "verify_protocol_lock_file",
    "verify_release_card_file",
    "verify_release_evidence_file",
    "verify_reproduction_card_file",
    "verify_research_card_file",
    "verify_suite_evidence",
    "wilson_ci",
]
