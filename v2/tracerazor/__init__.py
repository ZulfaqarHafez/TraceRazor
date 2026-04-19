"""TraceRazor v2: Adaptive-K speculative ensembles for agent reliability.

Public API
----------
AdaptiveKNode       — LangGraph node (drop-in ReAct replacement)
ExactMatchConsensus — aggregates K branch proposals via exact-match comparison
ConsensusReport     — structured output: trajectory, consensus_rate, divergences, tokens
MutationMetadata    — classifies tools as mutating vs read-only
openai_llm          — adapter factory for AsyncOpenAI
anthropic_llm       — adapter factory for AsyncAnthropic
mock_llm            — deterministic mock for tests / offline demos

Quick start
-----------
>>> from tracerazor import AdaptiveKNode, openai_llm
>>> from openai import AsyncOpenAI
>>> from langgraph.graph import StateGraph
>>>
>>> llm = openai_llm(AsyncOpenAI(), model="gpt-4.1")
>>> node = AdaptiveKNode(llm=llm, tools=my_tools, k_max=5)
>>>
>>> graph = StateGraph(AgentState)
>>> graph.add_node("agent", node)
>>> # ... add edges, compile ...
>>> result = await graph.ainvoke({"messages": [{"role": "user", "content": "..."}]})
>>> print(result["consensus_report"].summary())
"""

from ._adaptive_k import AdaptiveKNode
from ._adapters import anthropic_llm, mock_llm, openai_llm
from ._consensus import BranchProposal, ConsensusResult, ExactMatchConsensus, Outcome
from ._mutation import MutationMetadata
from ._naive_ensemble import NaiveKEnsemble, NaiveRunResult
from ._report import ConsensusReport, StepRecord, TokenCounts
from ._self_consistency import SCResult, SelfConsistencyBaseline

__version__ = "2.0.0"

__all__ = [
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
    "openai_llm",
    "anthropic_llm",
    "mock_llm",
]
