"""Public TRICE library API.

This package is the stable user-facing import path. The implementation still
lives in ``benchmark.trice`` while the research harness matures, but user code
should import from ``tracerazor.trice``.
"""

from benchmark.trice.adapters import CommandRepairAdapter, JsonPatchAdapter, PatchEdit, RepairAdapter
from benchmark.trice.artifact import build_artifact_card, render_artifact_markdown, render_artifact_svg, render_artifact_tex, verify_artifact_card_file, write_artifact_outputs
from benchmark.trice.bundle import BundleManifest, export_evidence_bundle, verify_evidence_bundle
from benchmark.trice.claim import build_claim_card, render_claim_card_markdown, render_claim_card_tex, render_claim_ladder_svg, verify_claim_card_file, write_claim_outputs
from benchmark.trice.contract import build_contract_card, render_contract_markdown, render_contract_svg, render_contract_tex, verify_contract_card_file, write_contract_outputs
from benchmark.trice.crates import build_crates_card, render_crates_markdown, render_crates_svg, render_crates_tex, verify_crates_card_file, write_crates_outputs
from benchmark.trice.design import build_design_card, render_design_markdown, render_design_svg, render_design_tex, verify_design_card_file, write_design_outputs
from benchmark.trice.doctor import doctor_report, render_doctor_text
from benchmark.trice.evidence import (
    EvidenceManifest,
    build_manifest,
    canonical_json,
    verify_manifest,
)
from benchmark.trice.integrity import build_integrity_card, render_integrity_markdown, render_integrity_svg, render_integrity_tex, verify_integrity_card_file, write_integrity_outputs
from benchmark.trice.install import build_install_card, render_install_markdown, render_install_svg, render_install_tex, verify_install_card_file, write_install_outputs
from benchmark.trice.learn import LearningWeights, PolicyUpdate, update_weights
from benchmark.trice.live import LiveRolloutResult, LiveRound, LiveTask, run_live_learning_loop
from benchmark.trice.policy import ContextPolicy, PolicyDecision, solve_policy
from benchmark.trice.recall import EvidenceRecallReport, evidence_recall_from_policy
from benchmark.trice.protocol import build_protocol_lock, render_protocol_markdown, render_protocol_svg, render_protocol_tex, verify_protocol_lock_file, write_protocol_outputs
from benchmark.trice.provenance import TreeFingerprint, fingerprint_tree, hash_file
from benchmark.trice.receipt import validate_run_receipt, validate_run_receipt_file
from benchmark.trice.readiness import build_suite_readiness, render_readiness_markdown, render_readiness_svg, render_readiness_tex, verify_readiness_file, write_readiness_outputs
from benchmark.trice.release import build_release_card, render_release_markdown, render_release_svg, render_release_tex, verify_release_card_file, write_release_outputs
from benchmark.trice.release_evidence import build_release_evidence_card, render_release_evidence_markdown, render_release_evidence_svg, render_release_evidence_tex, verify_release_evidence_file, write_release_evidence_outputs
from benchmark.trice.reproduction import build_reproduction_card, render_reproduction_markdown, render_reproduction_svg, render_reproduction_tex, verify_reproduction_card_file, write_reproduction_outputs
from benchmark.trice.research import build_research_card, render_research_markdown, render_research_svg, render_research_tex, verify_research_card_file, write_research_outputs
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
    "EvidenceRecallReport",
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
    "build_crates_card",
    "build_integrity_card",
    "build_install_card",
    "build_claim_card",
    "build_artifact_card",
    "build_contract_card",
    "build_design_card",
    "build_protocol_lock",
    "build_release_card",
    "build_release_evidence_card",
    "build_reproduction_card",
    "build_research_card",
    "build_suite_readiness",
    "canonical_json",
    "claim_gate_from_rounds",
    "clustered_bootstrap_mean_ci",
    "doctor_report",
    "evaluate_policy",
    "evidence_recall_from_policy",
    "export_evidence_bundle",
    "fingerprint_tree",
    "hash_file",
    "load_schema",
    "render_context",
    "render_claim_card_markdown",
    "render_claim_card_tex",
    "render_claim_ladder_svg",
    "render_contract_markdown",
    "render_crates_markdown",
    "render_crates_svg",
    "render_crates_tex",
    "render_contract_svg",
    "render_contract_tex",
    "render_artifact_markdown",
    "render_artifact_svg",
    "render_artifact_tex",
    "render_design_markdown",
    "render_design_svg",
    "render_design_tex",
    "render_protocol_markdown",
    "render_protocol_svg",
    "render_protocol_tex",
    "render_release_markdown",
    "render_release_svg",
    "render_release_tex",
    "render_release_evidence_markdown",
    "render_release_evidence_svg",
    "render_release_evidence_tex",
    "render_reproduction_markdown",
    "render_reproduction_svg",
    "render_reproduction_tex",
    "render_research_markdown",
    "render_research_svg",
    "render_research_tex",
    "render_readiness_markdown",
    "render_readiness_svg",
    "render_readiness_tex",
    "render_doctor_text",
    "render_integrity_markdown",
    "render_integrity_svg",
    "render_integrity_tex",
    "render_install_markdown",
    "render_install_svg",
    "render_install_tex",
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
    "verify_crates_card_file",
    "verify_integrity_card_file",
    "verify_install_card_file",
    "verify_evidence_bundle",
    "verify_artifact_card_file",
    "verify_claim_card_file",
    "verify_contract_card_file",
    "verify_design_card_file",
    "verify_protocol_lock_file",
    "verify_readiness_file",
    "verify_release_card_file",
    "verify_release_evidence_file",
    "verify_reproduction_card_file",
    "verify_research_card_file",
    "verify_suite_evidence",
    "write_readiness_outputs",
    "write_artifact_outputs",
    "write_claim_outputs",
    "write_contract_outputs",
    "write_crates_outputs",
    "write_design_outputs",
    "write_protocol_outputs",
    "write_release_outputs",
    "write_release_evidence_outputs",
    "write_integrity_outputs",
    "write_install_outputs",
    "write_reproduction_outputs",
    "write_research_outputs",
    "wilson_ci",
]
