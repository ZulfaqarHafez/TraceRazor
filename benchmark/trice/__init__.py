"""TRICE: TraceRazor Information Control Engine.

V1 converts TraceRazor traces into typed context segments, scores each segment
from agent/harness/transformer perspectives, solves a token-budgeted policy,
and replays the policy against recorded traces.

V2 adds live rollout: fresh workspaces, real edits, verifier commands, and a
small user preference profile that adapts budgets from feedback and outcomes.
"""

from .learn import LearningWeights, PolicyUpdate, update_weights
from .policy import ContextPolicy, PolicyDecision, solve_policy
from .replay import ReplayMetrics, evaluate_policy
from .render import render_context, render_policy_json
from .score import ActionCandidate, SegmentScore, ScoreWeights, score_segment
from .segment import Segment, SegmentState, segments_from_trace
from .user import UserPreferenceProfile

__all__ = [
    "ActionCandidate",
    "ContextPolicy",
    "LearningWeights",
    "PolicyDecision",
    "PolicyUpdate",
    "ReplayMetrics",
    "ScoreWeights",
    "Segment",
    "SegmentScore",
    "SegmentState",
    "UserPreferenceProfile",
    "evaluate_policy",
    "render_context",
    "render_policy_json",
    "score_segment",
    "segments_from_trace",
    "solve_policy",
    "update_weights",
]
