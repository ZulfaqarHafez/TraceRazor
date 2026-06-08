"""Real online verification.

Runs an actual tool-calling agent against a live OpenAI-compatible LLM endpoint,
measures real token usage (from the API `usage` block) and real task success,
applies a candidate config (system-prompt sections + runtime policies), re-runs,
and gates on the statistical non-inferiority test in ``teacher.stats``.

The same code path targets:
  * a real provider  -- set TRACERAZOR_LLM_BASE_URL + TRACERAZOR_LLM_API_KEY
                        (or OPENAI_BASE_URL + OPENAI_API_KEY), and a model;
  * any OpenAI-compatible gateway (Azure, Groq, vLLM, Ollama, LiteLLM);
  * the bundled stdlib mock server (``teacher._mockserver``) for offline CI.

Runtime interventions are enforced *live* here, not just suggested:
  * ``loop_breaker`` -- an identical repeated tool call is not re-executed; the
    cached result is returned and a guard marker is added to the request so a
    cooperative model stops looping.
  * ``step_cap``     -- the agent loop is hard-stopped after N turns.
System-prompt interventions (NO_HEDGING, EFFICIENCY_RULES, STEP_BUDGET, ...) are
sent in the real system prompt, so a real model genuinely emits fewer tokens.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

from .interventions import apply, propose
from .schemas import AgentConfig, EvalResult, Intervention, Tier
from .stats import GateEvidence, StatGate


# --------------------------------------------------------------------------- #
# LLM client (OpenAI chat-completions wire format, over requests).
# --------------------------------------------------------------------------- #
@dataclass
class LLMClient:
    base_url: str
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    timeout: float = 60.0

    @classmethod
    def from_env(cls, base_url: Optional[str] = None) -> "LLMClient":
        base = (base_url or os.environ.get("TRACERAZOR_LLM_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
        key = (os.environ.get("TRACERAZOR_LLM_API_KEY")
               or os.environ.get("OPENAI_API_KEY") or "")
        model = (os.environ.get("TRACERAZOR_LLM_MODEL") or "gpt-4o-mini")
        return cls(base_url=base.rstrip("/"), api_key=key, model=model)

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None,
             seed: Optional[int] = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict = {"model": self.model, "messages": messages,
                         "temperature": self.temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if seed is not None:
            payload["seed"] = seed
        r = requests.post(f"{self.base_url}/chat/completions",
                          headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


# --------------------------------------------------------------------------- #
# Tools + task.
# --------------------------------------------------------------------------- #
@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict                 # JSON schema
    fn: Callable[..., str]

    def openai(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


@dataclass
class OnlineTask:
    task_id: str
    goal: str
    required_tools: list[str]        # all must be called for success

    def user_message(self) -> str:
        return (f"{self.goal}  | TOOLS: {','.join(self.required_tools)} "
                f"| ID: {self.task_id}")


@dataclass
class RunOutcome:
    tokens: int
    success: bool
    n_calls: int
    executed_tools: list[str]
    final_answer: str
    trace: dict


# --------------------------------------------------------------------------- #
# The online agent loop.
# --------------------------------------------------------------------------- #
class OnlineAgent:
    def __init__(self, client: LLMClient, tools: dict[str, ToolSpec],
                 max_turns: int = 12):
        self.client = client
        self.tools = tools
        self.max_turns = max_turns

    def _system_prompt(self, cfg: AgentConfig) -> str:
        parts = [cfg.render_prompt()]
        # Surface active runtime policies into the request so a cooperative
        # model also respects them (belt-and-suspenders with hard enforcement).
        if cfg.policies.get("loop_breaker"):
            parts.append("[runtime] loop_guard on: never repeat an identical tool call.")
        if cfg.policies.get("step_cap"):
            parts.append(f"[runtime] step_cap={cfg.policies['step_cap']}.")
        return "\n".join(p for p in parts if p).strip() or "You are a helpful agent."

    def run(self, cfg: AgentConfig, task: OnlineTask,
            seed: Optional[int] = None) -> RunOutcome:
        system = self._system_prompt(cfg)
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task.user_message()},
        ]
        tool_defs = [t.openai() for t in self.tools.values()]
        step_cap = cfg.policies.get("step_cap")
        loop_breaker = bool(cfg.policies.get("loop_breaker"))

        total_tokens, n_calls = 0, 0
        executed: list[str] = []
        seen: dict[tuple, str] = {}
        steps: list[dict] = []
        final_answer = ""

        for turn in range(self.max_turns):
            if step_cap is not None and turn >= step_cap:
                break
            resp = self.client.chat(messages, tool_defs, seed=seed)
            n_calls += 1
            usage = resp.get("usage", {})
            tok = int(usage.get("total_tokens", 0))
            total_tokens += tok
            msg = resp["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                messages.append({"role": "assistant", "content": msg.get("content"),
                                 "tool_calls": tool_calls})
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    raw_args = tc["function"].get("arguments", "{}")
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = {}
                    sig = (name, json.dumps(args, sort_keys=True))
                    if loop_breaker and sig in seen:
                        result = seen[sig]            # do NOT re-execute the loop
                        suppressed = True
                    else:
                        spec = self.tools.get(name)
                        result = spec.fn(**args) if spec else f"unknown tool {name}"
                        seen[sig] = result
                        executed.append(name)
                        suppressed = False
                    steps.append({"id": len(steps) + 1, "type": "tool_call",
                                  "content": f"Calling {name}", "tokens": tok,
                                  "tool_name": name, "tool_params": args,
                                  "tool_success": True, "output": result})
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": (("[loop_guard: duplicate suppressed] "
                                                  if suppressed else "") + str(result))})
            else:
                final_answer = msg.get("content") or ""
                steps.append({"id": len(steps) + 1, "type": "reasoning",
                              "content": final_answer, "tokens": tok})
                break

        success = set(task.required_tools).issubset(set(executed)) and bool(final_answer)
        trace = {"trace_id": f"{cfg.version}-{task.task_id}",
                 "agent_name": "online-agent", "framework": "openai",
                 "task_value_score": 1.0 if success else 0.3, "steps": steps}
        return RunOutcome(total_tokens, success, n_calls, executed, final_answer, trace)


# --------------------------------------------------------------------------- #
# Evaluation + closed-loop verification.
# --------------------------------------------------------------------------- #
def evaluate_online(agent: OnlineAgent, cfg: AgentConfig, holdout: list[OnlineTask],
                    repeats: int = 1) -> EvalResult:
    tokens, success = [], []
    for task in holdout:
        for r in range(repeats):
            out = agent.run(cfg, task, seed=1000 + r)
            tokens.append(out.tokens)
            success.append(out.success)
    return EvalResult(tokens=tokens, success=success)


class OnlineRunner:
    """Real HTTP agent runner -- a drop-in ``Runner`` for ``Teacher.improve``.

    Pair it with ``gate=StatGate()`` on the Teacher so the curriculum is gated
    by the statistical non-inferiority test on live re-runs.
    """
    name = "online"

    def __init__(self, agent: "OnlineAgent", holdout: list[OnlineTask],
                 diagnoser, repeats: int = 3, seed: int = 0):
        if not holdout:
            raise ValueError("OnlineRunner needs at least one OnlineTask")
        self.agent = agent
        self.holdout = holdout
        self.diagnoser = diagnoser
        self.repeats = repeats
        self.seed = seed

    def evaluate(self, cfg: AgentConfig) -> EvalResult:
        res = evaluate_online(self.agent, cfg, self.holdout, self.repeats)
        try:
            sample = self.agent.run(cfg, self.holdout[0], seed=self.seed).trace
            res.tas = self.diagnoser.diagnose(sample).tas_score
        except Exception:
            pass
        return res

    def sample_trace(self, cfg: AgentConfig) -> dict:
        return self.agent.run(cfg, self.holdout[0], seed=self.seed).trace


@dataclass
class OnlineVerification:
    base_config: AgentConfig
    final_config: AgentConfig
    accepted: list[Intervention] = field(default_factory=list)
    rounds: list[tuple] = field(default_factory=list)   # (intervention, GateEvidence)
    baseline: Optional[EvalResult] = None

    def render(self) -> str:
        out = ["=" * 74, "REAL ONLINE VERIFICATION", "=" * 74]
        if self.baseline:
            out.append(f"  baseline: {self.baseline.mean_tokens:.0f} tok/run, "
                       f"success {self.baseline.success_rate*100:.0f}%  "
                       f"(n={len(self.baseline.tokens)})")
        out.append("\n  intervention            verdict          evidence")
        for iv, ev in self.rounds:
            out.append(f"  {iv.key:<22} {ev.decision.value:<15} {ev.summary()}")
        out.append("")
        first = self.baseline.mean_tokens if self.baseline else 0
        # net saving recomputed against the last accepted trial is in evidence;
        # report accepted set.
        out.append(f"  ACCEPTED: {[iv.key for iv in self.accepted] or 'none'}")
        out.append("=" * 74)
        return "\n".join(out)


def verify_online(base_config: AgentConfig, holdout: list[OnlineTask],
                  agent: OnlineAgent, diagnoser, gate: Optional[StatGate] = None,
                  repeats: int = 3, max_interventions: int = 6) -> OnlineVerification:
    """Diagnose -> greedily apply gate-passing interventions, verified by REAL
    online re-runs with statistical non-inferiority.  Tier order: cheapest first.
    """
    gate = gate or StatGate()
    cfg = base_config.clone()
    baseline = evaluate_online(agent, cfg, holdout, repeats)

    # Diagnose from a real baseline trace.
    sample = agent.run(cfg, holdout[0], seed=0).trace
    diagnosis = diagnoser.diagnose(sample)
    candidates = sorted(propose(diagnosis), key=lambda iv: int(iv.tier))

    result = OnlineVerification(base_config=base_config, final_config=cfg,
                                baseline=baseline)
    tried = 0
    for iv in candidates:
        if tried >= max_interventions:
            break
        if iv.tier is Tier.STRUCT:
            # still verify it -- the gate should reject the unsafe ones.
            pass
        trial_cfg = apply(iv, cfg)
        trial = evaluate_online(agent, trial_cfg, holdout, repeats)
        ev = gate.evaluate(baseline, trial)
        result.rounds.append((iv, ev))
        tried += 1
        if ev.decision.value == "ACCEPT":
            cfg = trial_cfg
            baseline = trial            # ratchet the baseline
            result.accepted.append(iv)
    result.final_config = cfg
    return result
