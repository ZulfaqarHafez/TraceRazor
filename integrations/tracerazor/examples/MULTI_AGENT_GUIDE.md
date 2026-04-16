# Multi-Agent Workflow with TraceRazor

This guide shows how to audit and optimize a complex multi-agent system using TraceRazor.

## The Scenario

A customer support system handles escalating levels of complexity:

1. **Triage Agent** — Categorizes incoming requests (billing, technical, account)
2. **Resolution Agent** — Attempts first-line resolution using knowledge base + precedent lookup
3. **Escalation Agent** — Routes unresolved cases to human support
4. **Knowledge Base Agent** — Indexes successful resolutions for future reference

Each agent runs independently but feeds data to the next. TraceRazor audits each one separately, enabling targeted optimization.

---

## Key Features Demonstrated

### Per-Agent Auditing
Each agent gets its own TraceRazor report showing:
- **TAS (Token Audit Score)** — Overall efficiency (0–100)
- **Grade** — Excellent/Good/Fair/Poor
- **Token count** — What was consumed
- **Metric breakdown** — Which waste patterns dominate (redundancy, loops, verbosity, drift, etc.)
- **Fixes** — Automated remediation suggestions

### Workflow-Level Reporting
Aggregate metrics across all agents:
- Total tokens consumed by the workflow
- Per-agent efficiency trends
- Which agent(s) are bottlenecks
- Estimated cost per run (e.g., at 50K runs/month)

### Optimization Validation
After applying fixes:
```bash
# Before optimization
python multi_agent_workflow.py

# Apply fixes to agents' system prompts...

# Re-run and validate improvement
python multi_agent_workflow.py --threshold 80
```

TraceRazor compares before/after scores and shows **Adherence Score** — what % of fix types actually improved.

---

## Running the Example

### 1. CLI Mode (No Server Required)

Requires `tracerazor` binary in PATH:

```bash
# Install Rust binary
cargo install --path crates/tracerazor-cli

# Run workflow (audits each agent automatically)
python examples/multi_agent_workflow.py

# Export as JSON
python examples/multi_agent_workflow.py --export-json

# Set efficiency threshold
python examples/multi_agent_workflow.py --threshold 85
```

### 2. Server Mode (Optional)

For persistent storage, dashboards, and REST API:

```bash
# Terminal 1: Start TraceRazor server
./target/release/tracerazor-server

# Terminal 2: Run workflow with server backend
python examples/multi_agent_workflow.py --server http://localhost:8080

# View dashboard
# http://localhost:8080
```

### 3. With Real LLM

Replace the `MockLLM` with actual API calls:

```python
from openai import OpenAI

class RealLLM:
    def __init__(self):
        self.client = OpenAI()  # requires OPENAI_API_KEY

    def invoke(self, prompt: str, agent_role: str) -> dict:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": f"You are a {agent_role} agent. Be concise and efficient.",
            }, {
                "role": "user",
                "content": prompt,
            }],
        )
        return {
            "text": response.choices[0].message.content,
            "tokens": response.usage.total_tokens,
        }

# Then:
llm = RealLLM()
workflow = MultiAgentWorkflow(llm=llm)
report = workflow.run(customer_query)
```

---

## Example Output

```
Executing multi-agent workflow...

✓ Triage: TAS 78.5/100 [GOOD] | 4 steps, 620 tokens | Saved 180 tokens (29%)
✓ Resolution: TAS 72.3/100 [FAIR] | 6 steps, 890 tokens | Saved 310 tokens (35%)
✓ Knowledge Base: TAS 85.0/100 [EXCELLENT] | 2 steps, 280 tokens | Saved 40 tokens (14%)

======================================================================
MULTI-AGENT WORKFLOW REPORT
======================================================================
Timestamp:     2026-04-16T14:23:45.123456
Status:        RESOLVED
Total tokens:  1,790
Total cost:    $0.0018
Efficiency:    78.6/100

Agent              TAS        Tokens     Grade
triage             78.5       620        GOOD
resolution         72.3       890        FAIR
knowledge_base     85.0       280        EXCELLENT
======================================================================

✅ PASSED: Workflow efficiency 78.6 exceeds threshold 75.0
```

---

## Detailed Metrics for Multi-Agent Systems

When running a workflow, TraceRazor tracks:

### Structural Efficiency
- **Redundancy (SRR)** — Are agents re-reasoning steps already explored?
- **Loops (LDI)** — Do agents retry the same tool call?
- **Tool Accuracy (TCA)** — Are tools misconfigured?
- **Reasoning Depth (RDA)** — Are agents thinking too hard for simple tasks?
- **Information Sufficiency (ISR)** — Does each step add value?
- **Token Utilization (TUR)** — What % of tokens are task-relevant?
- **Context Efficiency (CCE)** — Is context carried forward without duplication?
- **Decision Optimality (DBO)** — Are tool sequences optimal?

### Semantic Continuity (NEW)
- **Semantic Drift (CSD)** — Does the agent's reasoning wander off-topic mid-workflow?
  - Identifies agents that lose focus across steps
  - Example: Triage agent starts discussing billing, then shifts to technical issues

### Verbosity Intelligence
- **Density (VDI)** — Are outputs verbose with filler?
- **Sycophancy/Hedging (SHL)** — Is the agent over-apologetic? ("I'd be happy to...")
- **Compression (CCR)** — How much could be removed without losing meaning?

---

## Optimization Workflow

### Step 1: Audit Each Agent

```python
report = workflow.run(customer_query)
print(report.summary())  # Shows metrics + fixes
```

### Step 2: Apply Fixes

For each agent's fixes, update the system prompt:

```python
# Example fix suggestion:
# [context_compression] → system_prompt
# "Before each tool call, summarize the conversation to the last three
#  relevant facts. Do not re-include information already established."

# Old system prompt:
system_prompt = "You are a resolution agent. Use tools to help customers."

# New system prompt:
system_prompt = """You are a resolution agent. Use tools to help customers.

EFFICIENCY RULES:
• Before each tool call, summarize the conversation to the last three
  relevant facts. Do not re-include information already established.
• Call each tool at most once per unique input.
• Reply immediately once the answer is known — no closing preamble.
"""
```

### Step 3: Validate Improvement

Re-run the workflow and check:

```bash
# Before: TAS 72.3
# After:  TAS 81.5
# Delta:  +9.2 ✓

# Adherence Score: 4/5 fix types improved = 80%
# This confirms the optimizer's recommendations were effective.
```

### Step 4: Monitor Anomalies

TraceRazor tracks per-agent baselines (after 5+ runs):

```
ANOMALY ALERTS
[REGRESSION] triage/redundancy: 0.22 (z=2.4)
  → Triage agent's step redundancy increased 2.4σ from baseline
  → Action: Review recent triage prompt changes

[IMPROVEMENT] resolution/loop_detection: 0.08 (z=-1.8)
  → Resolution agent's loops decreased 1.8σ from baseline
  → Likely due to context_compression fix applied last run
```

---

## Cost Analysis

Each agent's report includes a **Savings Estimate**:

| Agent | Tokens | Est. Savings | Monthly (50K runs) |
|---|---|---|---|
| Triage | 620 | 180 (29%) | $900 |
| Resolution | 890 | 310 (35%) | $1,550 |
| Escalation | 420 | 70 (17%) | $350 |
| Knowledge Base | 280 | 40 (14%) | $200 |
| **Total** | **2,210** | **600 (27%)** | **$3,000** |

At typical pricing (gpt-4o-mini ~$0.01 per 1M input tokens), a 30% efficiency gain on 50K monthly runs = **$2,700–3,600 saved per month**.

---

## Integration with Your Stack

### LangGraph
```python
from langgraph.graph import StateGraph
from tracerazor_sdk import Tracer

def triage_node(state):
    with Tracer(agent_name="triage") as t:
        # ... triage logic
        t.reasoning(...)
        t.tool(...)
    return state

# Workflow automatically traces each node
```

### CrewAI
```python
from crewai import Agent, Crew, Task
from tracerazor_crewai import TraceRazorCallback

triage_agent = Agent(role="Triage", goal="...", tools=[...])
resolution_agent = Agent(role="Resolution", goal="...", tools=[...])

crew = Crew(
    agents=[triage_agent, resolution_agent],
    tasks=[...],
    callbacks=[TraceRazorCallback(threshold=75)],
)

crew.kickoff()
```

### Anthropic SDK
```python
from anthropic import Anthropic
from tracerazor_sdk import Tracer

client = Anthropic()

with Tracer(agent_name="my-agent") as t:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[...],
    )
    t.reasoning(response.content[0].text, tokens=response.usage.input_tokens)
```

---

## Troubleshooting

### Q: My agents are audited but report shows no fixes?
**A:** Agents are performing well. This is normal! Fixes only appear when metrics fail their targets.

### Q: How do I optimize a specific agent?
**A:** Run that agent with `--export-json`, then use:
```bash
tracerazor optimize traces/agent_trace.json --output optimized_prompt.txt
```

### Q: Can I compare two runs of the same agent?
**A:** Yes:
```bash
tracerazor bench \
  --before traces/run1.json \
  --after traces/run2.json \
  --regression-threshold 5.0
```

### Q: Server mode shows 0 agents on dashboard?
**A:** Traces are auto-captured only if TAS ≥ 85. Use `--threshold 0` to capture all.

---

## What's New in v0.2.0

✅ **Semantic Continuity Metric (CSD)** — Detects reasoning drift across steps  
✅ **Adherence Scoring (IAR)** — Validates that fixes improve metrics on re-audit  
✅ **Multi-agent reporting** — Aggregate efficiency across 2+ agents  
✅ **Enhanced Python SDK** — Workflow-level reporting and JSON export  
✅ **Real-time monitoring** — Anomaly detection per-agent per-run  

---

## Next Steps

1. **Run the example**: `python multi_agent_workflow.py`
2. **Read the README changes**: Semantic Continuity and Adherence sections
3. **Optimize your first agent**: Apply the generated fixes
4. **Monitor**: Track improvements across runs

Questions? Open an issue: https://github.com/ZulfaqarHafez/tracerazor/issues
