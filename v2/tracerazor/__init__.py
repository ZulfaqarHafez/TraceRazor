"""
TraceRazor 1.0.0

TraceRazor is a token efficiency auditor and adaptive sampling library for
production AI agents. It does two things:

1. Audit: record your agent's reasoning steps and tool calls, then submit
   the trace to the TraceRazor analyzer to get a Token Adequacy Score (TAS),
   per-metric breakdowns, fix recommendations, and savings estimates.

2. Sample: replace your agent's single LLM call per step with a parallel
   ensemble. TraceRazor's AdaptiveKNode samples K candidates per step and
   picks the consensus winner, increasing task success rates while keeping
   token overhead low.

Both features work independently. Use just the auditor, just the sampler,
or both together.


AUDIT QUICKSTART
----------------
Record steps manually with Tracer:

    from tracerazor import Tracer

    with Tracer(agent_name="support-agent", framework="openai") as t:
        response = llm.invoke(prompt)
        t.reasoning(response.text, tokens=response.usage.total_tokens)

        result = lookup_order(order_id="ORD-123")
        t.tool("lookup_order", params={"order_id": "ORD-123"},
               output=str(result), success=True, tokens=80)

    report = t.analyse()
    print(report.summary())
    report.assert_passes()   # raises AssertionError in CI if TAS < 70


SAMPLING QUICKSTART
-------------------
Drop AdaptiveKNode into a LangGraph graph as a replacement for your ReAct node:

    from tracerazor import AdaptiveKNode, openai_llm
    from openai import AsyncOpenAI

    llm = openai_llm(AsyncOpenAI(), model="gpt-4.1")
    node = AdaptiveKNode(llm=llm, tools=my_tools, k_max=5)

    graph = StateGraph(AgentState)
    graph.add_node("agent", node)

After the graph runs, the consensus report is available at:
    result["consensus_report"].summary()


AUDIT API
---------
Tracer             Record steps and submit for analysis.
TraceRazorClient   Lower-level client for submitting trace dicts directly.
TraceRazorReport   Parsed audit result with TAS score, metrics, and fixes.
TraceStep          Data class for a single recorded step.

SAMPLING API
------------
AdaptiveKNode        LangGraph node. Samples K candidates per step and picks
                     the consensus winner. K adapts dynamically.
ExactMatchConsensus  Aggregates K branch proposals by exact-match comparison.
MutationMetadata     Classifies tools as mutating vs read-only. Used by
                     AdaptiveKNode to reset K after state-changing calls.
NaiveKEnsemble       Baseline: runs K independent full-task agents and picks
                     the majority result.
SelfConsistencyBaseline
                     Baseline: deterministic tool calls, then re-samples the
                     final response K times and picks the best candidate.

LLM ADAPTERS
------------
openai_llm       Create an async LLM adapter from an AsyncOpenAI client.
anthropic_llm    Create an async LLM adapter from an AsyncAnthropic client.
mock_llm         Deterministic mock for tests and offline demos.

DATA TYPES
----------
BranchProposal   One candidate response from a single LLM sample.
ConsensusResult  Output of ExactMatchConsensus.aggregate().
ConsensusReport  Full trajectory record with consensus rates and token counts.
Outcome          Enum: FULL_CONSENSUS, MAJORITY, DIVERGENT.
StepRecord       Per-step detail inside a ConsensusReport.
TokenCounts      Input, output, cached, and fresh token breakdown.
NaiveRunResult   Result of one agent run inside NaiveKEnsemble.
SCResult         Result of a SelfConsistencyBaseline run.
"""

# Audit
from ._audit_tracer import Tracer
from ._audit_client import TraceRazorClient, TraceRazorReport
from ._audit_trace import TraceStep

# Sampling
from ._adaptive_k import AdaptiveKNode
from ._consensus import BranchProposal, ConsensusResult, ExactMatchConsensus, Outcome
from ._mutation import MutationMetadata
from ._naive_ensemble import NaiveKEnsemble, NaiveRunResult
from ._report import ConsensusReport, StepRecord, TokenCounts
from ._self_consistency import SCResult, SelfConsistencyBaseline

# LLM adapters
from ._adapters import anthropic_llm, mock_llm, openai_llm

__version__ = "1.0.0"
__author__ = "Zulfaqar Hafez"

__all__ = [
    # Audit
    "Tracer",
    "TraceRazorClient",
    "TraceRazorReport",
    "TraceStep",
    # Sampling
    "AdaptiveKNode",
    "ExactMatchConsensus",
    "ConsensusReport",
    "MutationMetadata",
    "NaiveKEnsemble",
    "NaiveRunResult",
    "SelfConsistencyBaseline",
    "SCResult",
    "BranchProposal",
    "ConsensusResult",
    "Outcome",
    "StepRecord",
    "TokenCounts",
    # LLM adapters
    "openai_llm",
    "anthropic_llm",
    "mock_llm",
]
