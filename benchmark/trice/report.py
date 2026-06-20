"""Markdown reporting helpers for TRICE experiments."""

from __future__ import annotations

from .policy import ContextPolicy
from .replay import ReplayMetrics


def render_markdown_report(policy: ContextPolicy, replay: ReplayMetrics) -> str:
    lines = [
        "# TRICE Run Report",
        "",
        "## Summary",
        "",
        f"- Baseline input tokens: {policy.baseline_input_tokens}",
        f"- Policy input tokens: {policy.policy_tokens}",
        f"- Projected input savings: {policy.projected_input_savings_pct:.2f}%",
        f"- Evidence recall: {replay.evidence_recall:.3f}",
        f"- Pass noninferior proxy: {str(replay.pass_noninferior).lower()}",
        "",
        "## Token Flow",
        "",
        "```mermaid",
        "flowchart LR",
        f'  A["Baseline {policy.baseline_input_tokens} tokens"] --> B["TRICE policy {policy.policy_tokens} tokens"]',
        f'  B --> C["Projected savings {policy.projected_input_savings_pct:.1f}%"]',
        "```",
        "",
        "## Context Portfolio",
        "",
        "| State | Decisions | Policy tokens |",
        "|---|---:|---:|",
    ]
    by_state: dict[str, tuple[int, int]] = {}
    for d in policy.decisions:
        n, tokens = by_state.get(d.state, (0, 0))
        by_state[d.state] = (n + 1, tokens + d.policy_tokens)
    for state, (n, tokens) in sorted(by_state.items()):
        lines.append(f"| {state} | {n} | {tokens} |")
    lines += [
        "",
        "## Replay Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| evidence_recall | {replay.evidence_recall:.3f} |",
        f"| action_divergence | {replay.action_divergence:.3f} |",
        f"| expired_info_retention | {replay.expired_info_retention:.3f} |",
        f"| rehydration_success | {replay.rehydration_success:.3f} |",
        f"| compression_overhead | {replay.compression_overhead:.3f} |",
    ]
    return "\n".join(lines) + "\n"
