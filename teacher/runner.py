"""Offline mock agent + evaluation harness.

The ``MockAgent`` is a deterministic stand-in for a real tool-using LLM agent.
Its behaviour is *controlled by its ``AgentConfig``* so that interventions the
Teacher applies genuinely change the traces it emits -- which is what lets the
closed loop demonstrate real token reduction with preserved task success, all
offline with no API keys.

A task needs N tool calls to succeed. By default the agent wastes tokens in
exactly the structural ways TraceRazor detects:

* hedging / preamble in every reasoning step      (SHL / VDI)
* a reformulated "let me re-read the request" step (SRR / CCE)
* one redundant repeat of the first tool call      (LDI)
* an extra over-deep reasoning step                (RDA)

Each waste source is switched off by the corresponding policy/section that the
remediation engine installs -- so a remediated config emits a leaner trace.
The agent only marks the task ``success`` when it actually performs all N
required tool calls and emits a final answer; an over-aggressive intervention
(e.g. a step cap below N) therefore *breaks* success and must be rejected by
the quality gate.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schemas import AgentConfig, EvalResult


HEDGE = (
    "Certainly! I'd be happy to help with this. Let me take a careful look. "
    "I think that, generally speaking, it might possibly be the case that "
)
FILLER = "Basically, to be honest, at the end of the day, essentially "


@dataclass
class Task:
    task_id: str
    goal: str
    required_tools: list[str]   # ordered tool calls needed for success


def _reasoning_tokens(config: AgentConfig, base: int) -> int:
    """Tokens for one reasoning step, inflated by un-suppressed verbosity."""
    tokens = base
    if "NO_HEDGING" not in config.system_prompt_sections:
        tokens += len(HEDGE.split()) * 4          # hedging preamble
    if "EFFICIENCY_RULES" not in config.system_prompt_sections:
        tokens += len(FILLER.split()) * 4         # filler density
    return tokens


def run_task(config: AgentConfig, task: Task) -> dict:
    """Produce a trace dict (auditor schema) + bookkeeping for one task."""
    steps: list[dict] = []
    sid = 0

    def add(step_type: str, content: str, tokens: int, **extra) -> None:
        nonlocal sid
        sid += 1
        steps.append(
            {"id": sid, "type": step_type, "content": content, "tokens": tokens, **extra}
        )

    # 1. Initial reasoning (always present).
    add("reasoning", HEDGE + "I need to handle: " + task.goal,
        _reasoning_tokens(config, 120), input_context=task.goal)

    # 2. Reformulation step -- removed once NO_REFORMULATION is installed.
    if "NO_REFORMULATION" not in config.system_prompt_sections:
        add("reasoning", FILLER + "Let me re-read the request again: " + task.goal,
            _reasoning_tokens(config, 110), input_context=task.goal)

    # 3. Over-deep speculative reasoning -- removed once a step budget exists.
    step_cap = config.policies.get("step_cap")
    if "STEP_BUDGET" not in config.system_prompt_sections and step_cap is None:
        add("reasoning", FILLER + "Let me also consider several unlikely edge cases.",
            _reasoning_tokens(config, 130))

    # 4. The required tool calls (the actual work).
    calls_made = 0
    loop_broken = bool(config.policies.get("loop_breaker"))
    for i, tool in enumerate(task.required_tools):
        # Respect a hard step cap: a too-low cap starves real tool calls.
        if step_cap is not None and len(steps) >= step_cap:
            break
        add("tool_call", f"Calling {tool}", 90, tool_name=tool,
            tool_params={"arg": "x"}, tool_success=True, output="ok")
        calls_made += 1
        # First tool call is redundantly repeated unless a loop breaker exists.
        if i == 0 and not loop_broken:
            if step_cap is None or len(steps) < step_cap:
                add("tool_call", f"Calling {tool}", 90, tool_name=tool,
                    tool_params={"arg": "x"}, tool_success=True, output="ok")

    # 5. Final answer.
    add("reasoning", "Final answer: done. " + task.goal, _reasoning_tokens(config, 60))

    success = calls_made == len(task.required_tools)
    total_tokens = sum(s["tokens"] for s in steps)
    trace = {
        "trace_id": f"{config.version}-{task.task_id}",
        "agent_name": "mock-support-agent",
        "framework": "langgraph",
        "task_value_score": 1.0 if success else 0.4,
        "steps": steps,
    }
    return {"trace": trace, "tokens": total_tokens, "success": success}


def evaluate(config: AgentConfig, holdout: list[Task], diagnoser=None) -> EvalResult:
    """Run the agent over the holdout set; collect tokens + success (+ real TAS)."""
    tokens, success, last_trace = [], [], None
    for task in holdout:
        out = run_task(config, task)
        tokens.append(out["tokens"])
        success.append(out["success"])
        last_trace = out["trace"]

    tas = 0.0
    if diagnoser is not None and last_trace is not None:
        try:
            tas = diagnoser.diagnose(last_trace).tas_score
        except Exception:
            tas = 0.0
    return EvalResult(tokens=tokens, success=success, tas=tas)
