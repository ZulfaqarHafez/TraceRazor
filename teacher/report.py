"""Human-readable rendering of a teaching run (COACH mode output + diffs)."""
from __future__ import annotations

from .schemas import AgentConfig
from .teacher import TeacherResult

_TIER_NAME = {0: "INLINE", 1: "PROMPT", 2: "TOOL", 3: "STRUCT"}


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
