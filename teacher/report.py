"""Human-readable rendering of a teaching run (COACH mode output + diffs)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import AgentConfig, Intervention
from .teacher import TeacherResult

_TIER_NAME = {0: "INLINE", 1: "PROMPT", 2: "TOOL", 3: "STRUCT"}


@dataclass
class Recommendation:
    intervention: Intervention
    total_predicted_savings: int
    n_traces: int                 # how many traces exhibited this
    prior_winrate: float          # from the playbook (0.5 == no history)


@dataclass
class CoachReport:
    """Output of COACH mode over real, non-rerunnable captured traces.

    Promotes nothing -- it ranks the auditor's own + taxonomy interventions by
    predicted savings (and playbook prior) and emits a proposed config diff for
    a human to approve.
    """
    n_traces: int
    mean_tas: float
    total_tokens: int
    recommendations: list[Recommendation] = field(default_factory=list)
    base_config: AgentConfig = field(default_factory=AgentConfig)
    proposed_config: AgentConfig = field(default_factory=AgentConfig)
    backend: str = "builtin"

    @property
    def total_predicted_savings(self) -> int:
        return sum(r.total_predicted_savings for r in self.recommendations)

    def render(self) -> str:
        out = ["=" * 70,
               f"TRACERAZOR TEACHER -- COACH report   (backend: {self.backend})",
               "=" * 70,
               f"  traces analysed : {self.n_traces}",
               f"  mean TAS        : {self.mean_tas:.1f}",
               f"  total tokens    : {self.total_tokens}",
               f"  predicted save  : {self.total_predicted_savings} tokens "
               f"({100*self.total_predicted_savings/max(self.total_tokens,1):.1f}%)",
               "",
               "RECOMMENDED INTERVENTIONS (ranked; promote nothing automatically)",
               f"  {'tier':<7}{'intervention':<22}{'save~':>7}{'traces':>7}{'prior':>7}"]
        for r in self.recommendations:
            tier = _TIER_NAME.get(int(r.intervention.tier), "?")
            out.append(
                f"  {tier:<7}{r.intervention.key:<22}{r.total_predicted_savings:>7}"
                f"{r.n_traces:>7}{r.prior_winrate*100:>6.0f}%")
            if r.intervention.rationale:
                out.append(f"          - {r.intervention.rationale}")
        out.append("")
        out.append("PROPOSED CONFIG DIFF (for human approval)")
        out.append("  " + config_diff(self.base_config, self.proposed_config)
                   .replace("\n", "\n  "))
        out.append("=" * 70)
        return "\n".join(out)


def config_diff(before: AgentConfig, after: AgentConfig) -> str:
    lines: list[str] = []
    for k, v in after.system_prompt_sections.items():
        if k not in before.system_prompt_sections:
            lines.append(f"+ [system_prompt:{k}] {v}")
    for k, v in after.policies.items():
        if k not in before.policies:
            lines.append(f"+ [runtime_policy:{k}] = {v}")
    for k, v in after.tool_required_params.items():
        if before.tool_required_params.get(k) != v:
            lines.append(f"~ [tool_def:{k}] required = {v}")
    return "\n".join(lines) if lines else "(no changes)"


def render(result: TeacherResult) -> str:
    out: list[str] = []
    out.append("=" * 66)
    out.append(f"TRACERAZOR TEACHER  --  mode: {result.mode.value}")
    out.append("=" * 66)

    out.append("\nPER-ROUND DECISIONS")
    out.append(f"  {'tier':<7}{'intervention':<20}{'tok%':>7} {'succ':>6}  decision")
    for vr in result.history:
        tier = _TIER_NAME.get(int(vr.intervention.tier), "?")
        out.append(
            f"  {tier:<7}{vr.intervention.key:<20}"
            f"{vr.token_delta_pct:>6.1f}% {vr.success_after*100:>5.0f}%  {vr.decision.value}")

    out.append("\nTRAJECTORY")
    tas = " -> ".join(f"{t:.1f}" for t in result.tas_trajectory)
    tok = " -> ".join(f"{t:.0f}" for t in result.tokens_trajectory)
    out.append(f"  TAS    : {tas}")
    out.append(f"  tokens : {tok}")
    out.append(f"  net token saving: {result.total_token_saving_pct:.1f}% "
               f"(task success preserved)")

    out.append("\nPROMOTED CONFIG DIFF" if result.promoted
               else "\nPROPOSED CONFIG DIFF (coach -- not applied)")
    out.append("  " + config_diff(result.base_config, result.final_config).replace("\n", "\n  "))
    out.append("=" * 66)
    return "\n".join(out)
