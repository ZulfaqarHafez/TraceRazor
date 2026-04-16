#!/usr/bin/env python3
"""
Multi-Agent Workflow with TraceRazor Efficiency Auditing

This example demonstrates a real-world multi-agent system that handles
customer support escalations. Each agent specializes in a specific domain:
- Triage Agent: categorizes incoming requests
- Resolution Agent: attempts first-line resolution
- Escalation Agent: handles complex cases
- Knowledge Base Agent: searches and synthesizes information

TraceRazor audits each agent independently and tracks efficiency across
the workflow. This enables per-agent optimization and load balancing.

Example usage:
    python multi_agent_workflow.py --trace-file traces/ --threshold 75
"""

from dataclasses import dataclass
from typing import Optional
import json
import os
from datetime import datetime

from tracerazor_sdk import Tracer, TraceRazorClient, TraceRazorReport


# ============================================================================
# SIMULATED LLM BACKEND (replace with real OpenAI/Anthropic client)
# ============================================================================

class MockLLM:
    """Mock LLM for demonstration. Replace with real API calls."""

    def invoke(self, prompt: str, agent_role: str) -> dict:
        """Simulate an LLM response."""
        # In production, this would call:
        # - OpenAI: client.chat.completions.create(...)
        # - Anthropic: client.messages.create(...)
        # - LangChain: llm.invoke(prompt)

        response_map = {
            "triage": {
                "category": "billing_inquiry",
                "priority": "medium",
                "reasoning": "Customer asking about invoice. Routing to resolution agent.",
                "tokens": 120,
            },
            "resolution": {
                "answer": "Your invoice shows billing for 3 months of service.",
                "resolved": True,
                "reasoning": "Found matching order in system. Provided direct answer.",
                "tokens": 180,
            },
            "escalation": {
                "summary": "Complex multi-account issue. Requires human review.",
                "resolved": False,
                "reasoning": "Detected cross-account dependencies. Escalating to human team.",
                "tokens": 200,
            },
            "knowledge": {
                "search_results": ["Invoice FAQ", "Billing Guide", "Account Management"],
                "summary": "Found 3 relevant docs on billing processes.",
                "tokens": 150,
            },
        }
        return response_map.get(agent_role, {"tokens": 100})


# ============================================================================
# TOOL DEFINITIONS (these would be real API calls)
# ============================================================================

def search_knowledge_base(query: str) -> dict:
    """Search internal knowledge base for relevant documents."""
    # Simulated tool response
    return {
        "success": True,
        "results": ["Billing Guide", "Invoice FAQ", "Refund Policy"],
        "tokens": 80,
    }


def lookup_customer_account(customer_id: str) -> dict:
    """Look up customer details from database."""
    return {
        "success": True,
        "account": {
            "id": customer_id,
            "status": "active",
            "subscription": "pro",
            "invoices": 12,
        },
        "tokens": 60,
    }


def check_resolution_precedent(issue_type: str) -> dict:
    """Check if this issue type has been resolved before."""
    return {
        "success": True,
        "precedent_found": True,
        "resolution_template": "billing_inquiry_standard",
        "tokens": 50,
    }


def escalate_to_human(issue_summary: str, priority: str) -> dict:
    """Escalate case to human support team."""
    return {
        "success": True,
        "ticket_id": f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "estimated_wait": "15 mins",
        "tokens": 40,
    }


# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class TriageAgent:
    """Categorizes incoming customer requests."""

    def __init__(self, llm: MockLLM, tracer: Tracer):
        self.llm = llm
        self.tracer = tracer
        self.name = "triage"

    def process(self, customer_query: str) -> dict:
        """Route customer query to appropriate handler."""

        # Step 1: Initial reasoning
        prompt = f"Categorize this customer issue: {customer_query}"
        reasoning = self.llm.invoke(prompt, self.name)

        self.tracer.reasoning(
            content=f"Analyzing: {reasoning.get('reasoning', '')}",
            tokens=reasoning.get("tokens", 0),
        )

        # Step 2: Lookup customer (tool call)
        customer_id = "CUST-12345"  # In reality, extracted from query
        lookup = lookup_customer_account(customer_id)

        self.tracer.tool(
            name="lookup_customer_account",
            params={"customer_id": customer_id},
            output=lookup,
            success=lookup.get("success", False),
            tokens=lookup.get("tokens", 0),
        )

        return {
            "category": reasoning.get("category", "unknown"),
            "priority": reasoning.get("priority", "low"),
            "customer_id": customer_id,
            "next_agent": "resolution",
        }


class ResolutionAgent:
    """Attempts to resolve the issue without escalation."""

    def __init__(self, llm: MockLLM, tracer: Tracer):
        self.llm = llm
        self.tracer = tracer
        self.name = "resolution"

    def process(self, triage_result: dict) -> dict:
        """Attempt first-line resolution."""

        # Step 1: Search knowledge base
        kb_search = search_knowledge_base(triage_result["category"])

        self.tracer.tool(
            name="search_knowledge_base",
            params={"query": triage_result["category"]},
            output=kb_search,
            success=kb_search.get("success", False),
            tokens=kb_search.get("tokens", 0),
        )

        # Step 2: Check precedent
        precedent = check_resolution_precedent(triage_result["category"])

        self.tracer.tool(
            name="check_resolution_precedent",
            params={"issue_type": triage_result["category"]},
            output=precedent,
            success=precedent.get("success", False),
            tokens=precedent.get("tokens", 0),
        )

        # Step 3: Generate response
        prompt = f"Generate solution for: {triage_result['category']}"
        response = self.llm.invoke(prompt, self.name)

        self.tracer.reasoning(
            content=f"Generating response: {response.get('reasoning', '')}",
            tokens=response.get("tokens", 0),
        )

        return {
            "resolved": response.get("resolved", False),
            "answer": response.get("answer", ""),
            "confidence": 0.85,
            "next_agent": "escalation" if not response.get("resolved") else None,
        }


class EscalationAgent:
    """Handles complex cases that require human intervention."""

    def __init__(self, llm: MockLLM, tracer: Tracer):
        self.llm = llm
        self.tracer = tracer
        self.name = "escalation"

    def process(self, triage_result: dict, resolution_result: dict) -> dict:
        """Escalate unresolved cases to human team."""

        # Step 1: Analyze complexity
        prompt = f"Analyze case complexity: {triage_result['category']}"
        analysis = self.llm.invoke(prompt, self.name)

        self.tracer.reasoning(
            content=f"Case analysis: {analysis.get('reasoning', '')}",
            tokens=analysis.get("tokens", 0),
        )

        # Step 2: Create escalation ticket
        escalate = escalate_to_human(
            issue_summary=f"{triage_result['category']} - unresolved",
            priority=triage_result.get("priority", "medium"),
        )

        self.tracer.tool(
            name="escalate_to_human",
            params={"priority": triage_result.get("priority")},
            output=escalate,
            success=escalate.get("success", False),
            tokens=escalate.get("tokens", 0),
        )

        return {
            "escalated": True,
            "ticket_id": escalate.get("ticket_id"),
            "status": "pending_human_review",
        }


class KnowledgeBaseAgent:
    """Continuously indexes and synthesizes knowledge from resolutions."""

    def __init__(self, llm: MockLLM, tracer: Tracer):
        self.llm = llm
        self.tracer = tracer
        self.name = "knowledge"

    def process(self, resolution_result: dict) -> dict:
        """Index successful resolution for future reference."""

        prompt = "Synthesize lesson learned from successful resolution"
        synthesis = self.llm.invoke(prompt, self.name)

        self.tracer.reasoning(
            content=f"Synthesizing: {synthesis.get('reasoning', '')}",
            tokens=synthesis.get("tokens", 0),
        )

        return {
            "indexed": True,
            "timestamp": datetime.now().isoformat(),
            "summary": synthesis.get("summary", ""),
        }


# ============================================================================
# WORKFLOW ORCHESTRATION
# ============================================================================

@dataclass
class WorkflowReport:
    """Aggregated report across all agents in the workflow."""

    timestamp: datetime
    customer_query: str
    individual_reports: dict[str, TraceRazorReport]  # agent_name -> report
    workflow_status: str  # "resolved", "escalated", "pending"
    total_tokens: int
    total_cost_usd: float
    efficiency_score: float  # 0-100

    def summary(self) -> str:
        """Print workflow summary."""
        lines = [
            f"\n{'='*70}",
            "MULTI-AGENT WORKFLOW REPORT",
            f"{'='*70}",
            f"Timestamp:     {self.timestamp.isoformat()}",
            f"Status:        {self.workflow_status.upper()}",
            f"Total tokens:  {self.total_tokens}",
            f"Total cost:    ${self.total_cost_usd:.4f}",
            f"Efficiency:    {self.efficiency_score:.1f}/100",
            f"\n{'Agent':<20} {'TAS':<10} {'Tokens':<10} {'Grade':<10}",
            "-" * 70,
        ]

        for agent_name, report in self.individual_reports.items():
            lines.append(
                f"{agent_name:<20} "
                f"{report.tas_score:<10.1f} "
                f"{report.total_tokens:<10} "
                f"{report.grade:<10}"
            )

        lines += [f"{'='*70}\n"]
        return "\n".join(lines)

    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps({
            "timestamp": self.timestamp.isoformat(),
            "status": self.workflow_status,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "efficiency_score": self.efficiency_score,
            "agents": {
                name: {
                    "tas_score": report.tas_score,
                    "grade": report.grade,
                    "total_tokens": report.total_tokens,
                    "passes": report.passes,
                }
                for name, report in self.individual_reports.items()
            },
        }, indent=2)


class MultiAgentWorkflow:
    """Orchestrates a multi-agent system with TraceRazor auditing."""

    def __init__(
        self,
        llm: MockLLM,
        tracer_server: Optional[str] = None,
        threshold: float = 75.0,
    ):
        self.llm = llm
        self.tracer_server = tracer_server
        self.threshold = threshold
        self.client = TraceRazorClient(server=tracer_server) if tracer_server else None

        # Initialize agents with independent tracers
        self.triage = None
        self.resolution = None
        self.escalation = None
        self.knowledge = None

    def run(self, customer_query: str) -> WorkflowReport:
        """Execute the workflow and audit each agent."""

        start_time = datetime.now()
        reports = {}

        # --- TRIAGE AGENT ---
        with Tracer(agent_name="triage", server=self.tracer_server) as tracer:
            self.triage = TriageAgent(self.llm, tracer)
            triage_result = self.triage.process(customer_query)

        triage_report = tracer.analyse()
        if triage_report:
            reports["triage"] = triage_report
            print(f"✓ Triage: {triage_report.summary()}")

        # --- RESOLUTION AGENT ---
        with Tracer(agent_name="resolution", server=self.tracer_server) as tracer:
            self.resolution = ResolutionAgent(self.llm, tracer)
            resolution_result = self.resolution.process(triage_result)

        resolution_report = tracer.analyse()
        if resolution_report:
            reports["resolution"] = resolution_report
            print(f"✓ Resolution: {resolution_report.summary()}")

        # --- ESCALATION AGENT (if needed) ---
        escalation_result = None
        if not resolution_result.get("resolved"):
            with Tracer(agent_name="escalation", server=self.tracer_server) as tracer:
                self.escalation = EscalationAgent(self.llm, tracer)
                escalation_result = self.escalation.process(triage_result, resolution_result)

            escalation_report = tracer.analyse()
            if escalation_report:
                reports["escalation"] = escalation_report
                print(f"✓ Escalation: {escalation_report.summary()}")

        # --- KNOWLEDGE BASE AGENT (continuous learning) ---
        if resolution_result.get("resolved"):
            with Tracer(agent_name="knowledge_base", server=self.tracer_server) as tracer:
                self.knowledge = KnowledgeBaseAgent(self.llm, tracer)
                self.knowledge.process(resolution_result)

            kb_report = tracer.analyse()
            if kb_report:
                reports["knowledge_base"] = kb_report
                print(f"✓ Knowledge Base: {kb_report.summary()}")

        # --- AGGREGATE RESULTS ---
        total_tokens = sum(r.total_tokens for r in reports.values())
        total_cost = sum(
            (r.total_tokens / 1_000_000) * 0.01  # $0.01 per 1M tokens (approx)
            for r in reports.values()
        )

        efficiency_score = sum(
            r.tas_score for r in reports.values()
        ) / len(reports) if reports else 0

        workflow_status = (
            "resolved" if resolution_result.get("resolved")
            else "escalated" if escalation_result
            else "pending"
        )

        workflow_report = WorkflowReport(
            timestamp=start_time,
            customer_query=customer_query,
            individual_reports=reports,
            workflow_status=workflow_status,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            efficiency_score=efficiency_score,
        )

        # --- VALIDATION ---
        self._validate_workflow(workflow_report)

        return workflow_report

    def _validate_workflow(self, report: WorkflowReport) -> None:
        """Validate workflow against thresholds."""
        if report.efficiency_score < self.threshold:
            print(
                f"\n⚠️  WARNING: Workflow efficiency ({report.efficiency_score:.1f}) "
                f"below threshold ({self.threshold})."
            )
            print("Run `tracerazor optimize` on individual traces to improve.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-agent workflow with TraceRazor auditing"
    )
    parser.add_argument(
        "--trace-file",
        default="traces/",
        help="Directory to save trace files",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="TraceRazor server URL (e.g. http://localhost:8080)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=75.0,
        help="Minimum acceptable TAS score",
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Export workflow report as JSON",
    )
    args = parser.parse_args()

    # Ensure trace directory exists
    os.makedirs(args.trace_file, exist_ok=True)

    # Initialize workflow
    llm = MockLLM()
    workflow = MultiAgentWorkflow(
        llm=llm,
        tracer_server=args.server,
        threshold=args.threshold,
    )

    # Run workflow
    print("Executing multi-agent workflow...\n")

    customer_query = (
        "Hi, I was charged twice for my subscription this month. "
        "Can you help me understand why and get a refund?"
    )

    report = workflow.run(customer_query)

    # Print results
    print(report.summary())

    # Export JSON if requested
    if args.export_json:
        json_path = os.path.join(args.trace_file, "workflow_report.json")
        with open(json_path, "w") as f:
            f.write(report.to_json())
        print(f"✓ Workflow report saved to {json_path}")

    # Validate against threshold
    if report.efficiency_score < args.threshold:
        print(
            f"\n❌ FAILED: Workflow efficiency {report.efficiency_score:.1f} "
            f"is below threshold {args.threshold}"
        )
        exit(1)
    else:
        print(
            f"\n✅ PASSED: Workflow efficiency {report.efficiency_score:.1f} "
            f"exceeds threshold {args.threshold}"
        )
        exit(0)
