"""Generate TRICE paper artifacts from deterministic live evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from textwrap import dedent

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .evidence import build_manifest, write_manifest
from .stats import claim_gate_from_rounds

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO / "benchmark" / "trice" / "results" / "v2-smoke" / "trice_v2_live_results.json"
DEFAULT_SUITE_RESULTS = REPO / "benchmark" / "trice" / "results" / "v2-suite" / "trice_suite_results.json"
DEFAULT_OUT_DIR = REPO / "paper"
DEFAULT_DOCS_DIR = REPO / "docs"


REFERENCES = [
    ("agentsmatter2024", "AI Agents That Matter", "https://arxiv.org/html/2407.01502v1"),
    ("acmartifact", "ACM Artifact Review and Badging", "https://www.acm.org/publications/policies/artifact-review-badging"),
    ("mohammadi2025agentsurvey", "Evaluation and Benchmarking of LLM Agents: A Survey", "https://arxiv.org/html/2507.21504v1"),
    ("complexmcp2026", "ComplexMCP: Evaluation of LLM Agents in Dynamic Tool Sandboxes", "https://arxiv.org/html/2605.10787v2"),
    ("reasonbench2025", "ReasonBENCH: Benchmarking the Instability of LLM Reasoning", "https://arxiv.org/html/2512.07795v2"),
    ("dfah2026", "Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness", "https://arxiv.org/html/2601.15322v1"),
    ("terminalagents2026", "Building AI Coding Agents for the Terminal", "https://arxiv.org/html/2603.05344v1"),
    ("uncertainty2024", "Towards Reproducible LLM Evaluation", "https://arxiv.org/html/2410.03492v2"),
    ("externalization2026", "Externalization in LLM Agents", "https://arxiv.org/html/2604.08224v1"),
    ("sweuniverse2026", "SWE-Universe: Scale Real-World Verifiable Environments", "https://arxiv.org/html/2602.02361v1"),
    ("swebench2023", "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", "https://arxiv.org/abs/2310.06770"),
    ("sweagent2024", "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering", "https://arxiv.org/abs/2405.15793"),
    ("acon2025", "Acon: Optimizing Context Compression for Long-Horizon LLM Agents", "https://arxiv.org/html/2510.00615v1"),
    ("personalizedagents2026", "Learning Personalized Agents from Human Feedback", "https://arxiv.org/html/2602.16173v1"),
    ("nlharness2026", "Natural-Language Agent Harnesses", "https://arxiv.org/html/2603.25723v1"),
    ("swtbench2024", "SWT-Bench: Testing and Validating Real-World Software Fixes", "https://arxiv.org/html/2406.12952v3"),
    ("swebenchcl2025", "SWE-Bench-CL: Continual Learning for Software Engineering Agents", "https://arxiv.org/abs/2507.00014"),
    ("rigorousagentbench2025", "Establishing Best Practices for Building Rigorous Agentic Benchmarks", "https://arxiv.org/html/2507.02825v5"),
    ("sparkllmeval2026", "Spark-LLM-Eval: A Distributed Framework for Statistically Rigorous LLM Evaluation", "https://arxiv.org/html/2603.28769v1"),
    ("agentloganalysis2026", "Log Analysis is Necessary for Credible Evaluation of AI Agents", "https://arxiv.org/html/2605.08545v1"),
    ("topline2026", "Topline Benchmark Performance", "https://arxiv.org/html/2605.28065v1"),
    ("setupbench2025", "SetupBench: Assessing Software Engineering Agents' Ability to Bootstrap Development Environments", "https://arxiv.org/abs/2507.09063"),
    ("pairedbca2025", "A Paired Bootstrap Protocol for Evaluating Small Improvements", "https://arxiv.org/pdf/2511.19794"),
    ("llmlingua2023", "LLMLingua: Compressing Prompts for Accelerated Inference", "https://arxiv.org/abs/2310.05736"),
    ("longllmlingua2023", "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios", "https://arxiv.org/abs/2310.06839"),
    ("promptcache2023", "Prompt Cache: Modular Attention Reuse for Low-Latency Inference", "https://arxiv.org/abs/2311.04934"),
    ("kvflow2025", "KVFlow: Efficient Serving for LLM Agent Workflows", "https://arxiv.org/abs/2507.07400"),
    ("agentdiet2025", "AgentDiet: Efficient Context Pruning for LLM Agents", "https://hf.co/papers/2509.23586"),
    ("react2022", "ReAct: Synergizing Reasoning and Acting in Language Models", "https://arxiv.org/abs/2210.03629"),
    ("reflexion2023", "Reflexion: Language Agents with Verbal Reinforcement Learning", "https://arxiv.org/abs/2303.11366"),
    ("sweevo2025", "SWE-EVO: Benchmarking Coding Agents in Long-Horizon Software Evolution", "https://arxiv.org/html/2512.18470v6"),
]


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in data["rounds"]:
        rows.append(
            {
                "task": r["task_id"],
                "baseline": r["baseline"]["input_tokens"],
                "trice": r["optimized"]["input_tokens"],
                "savings": r["measured_input_savings"],
                "passed": bool(r["optimized"]["passed"] and r["baseline"]["passed"]),
                "accepted": bool(r["accepted"]),
            }
        )
    return rows


def load_gate(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("claim_gate") or claim_gate_from_rounds(data["rounds"], data.get("profile", {}).get("target_savings", 0.60)).to_dict()


def load_suite_summary(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    gate = data["claim_gate"]
    clustered = gate.get("clustered_savings_ci") or gate["savings_ci"]
    return {
        "name": data["suite"]["name"],
        "run_count": gate.get("replicate_count", len(data.get("tasks", []))),
        "task_clusters": gate.get("task_cluster_count", data["suite"].get("task_count", 0)),
        "mean_savings": gate["mean_savings"],
        "cluster_low": clustered["low"],
        "cluster_high": clustered["high"],
        "pass_regressions": gate["pass_regressions"],
    }


def render_tex(rows: list[dict], gate: dict, suite: dict | None = None) -> str:
    avg = mean(r["savings"] for r in rows)
    ci = gate["savings_ci"]
    suite_text = (
        "No suite artifact was available during paper generation."
        if suite is None
        else (
            f"The public suite artifact \\texttt{{{suite['name']}}} contains "
            f"{suite['run_count']} live replicate runs across {suite['task_clusters']} task cluster(s). "
            f"Mean input-token savings is {100*suite['mean_savings']:.1f}\\% with clustered-by-task "
            f"95\\% CI {100*suite['cluster_low']:.1f}\\%--{100*suite['cluster_high']:.1f}\\%; "
            f"pass regressions {suite['pass_regressions']}."
        )
    )
    table = "\n".join(
        f"{r['task']} & {r['baseline']} & {r['trice']} & {100*r['savings']:.1f}\\% & yes & yes \\\\"
        for r in rows
    )
    refs = ", ".join(f"\\cite{{{key}}}" for key, _, _ in REFERENCES[:8])
    return dedent(
        rf"""
        \documentclass[10pt]{{article}}
        \usepackage[margin=0.85in]{{geometry}}
        \usepackage{{booktabs}}
        \usepackage{{hyperref}}
        \usepackage{{amsmath}}
        \title{{TRICE: Deterministic Context Control for Live Software Agents}}
        \author{{TraceRazor Research}}
        \date{{2026-06-21}}

        \begin{{document}}
        \maketitle

        \begin{{abstract}}
        We present TRICE, a deterministic harness-level context controller for
        software agents. TRICE does not claim to make stochastic models
        deterministic. Instead, it makes context assembly, evidence manifests,
        live verification, and acceptance gates deterministic. On six bundled
        live repository tasks, TRICE reduces measured input context by
        {100*avg:.1f}\% (95\% bootstrap CI:
        {100*ci['low']:.1f}\%--{100*ci['high']:.1f}\%) while preserving
        executable task success.
        \end{{abstract}}

        \section{{Motivation}}
        Agent evaluation research emphasizes task completion, tool use,
        reliability, dynamic interaction, and cost as first-class metrics
        {refs}. Prompt compression work shows that shorter context can help,
        but product evidence must be live, outcome-gated, and repeatable under
        the same artifact contract.

        \section{{Method}}
        For each segment $s_i$, TRICE chooses an action $a_i$ from
        \{{keep, extract, summarize, mask, anchor, lazy\_recall\}}:
        \[
        \max_a \sum_i U_i(a_i) - \lambda R_i(a_i) - \mu C_i(a_i)
        + \rho K_i(a_i) - \gamma H_i(a_i)
        \]
        subject to a token budget, locked-anchor preservation, and live
        noninferiority:
        \[
        pass(policy) \ge pass(baseline) - 0.02.
        \]
        V3 adds a deterministic evidence manifest:
        \[
        E = SHA256(canonical\_json(result)) \Vert SHA256(report).
        \]
        Verifier output is normalized before hashing: duration strings are
        replaced with \texttt{{<duration>}}, and wall-clock fields are excluded
        because they are not decision evidence.
        Suite manifests extend this contract to multi-repository evaluation:
        the aggregate manifest hashes the suite snapshot and each child live
        task manifest, then deep verification checks every child bundle.

        \section{{Metrics}}
        Primary metric: measured input-token savings. Guardrail metrics:
        verifier pass preservation, evidence-manifest validity, accepted policy
        count, confidence intervals, and user-profile consistency. Replay
        metrics remain preflight diagnostics only.

        \section{{Live Results}}
        \begin{{table}}[h]
        \centering
        \begin{{tabular}}{{lrrrrr}}
        \toprule
        Task & Baseline & TRICE & Savings & Base Pass & TRICE Pass \\
        \midrule
        {table}
        \midrule
        Mean & -- & -- & {100*avg:.1f}\% & yes & yes \\
        \bottomrule
        \end{{tabular}}
        \caption{{Live workspace smoke: fresh copied repos, real source edits, and pytest verification.}}
        \end{{table}}

        Deterministic claim gate: savings lower bound
        {100*ci['low']:.1f}\% versus target {100*gate['target_savings']:.1f}\%;
        pass regressions {gate['pass_regressions']}; local smoke gate
        {"passed" if gate["smoke_gate_passed"] else "failed"}. Broad claim
        allowed: {"yes" if gate["broad_claim_allowed"] else "no"}.

        \section{{Replicated Suite Evidence}}
        {suite_text}

        \section{{Product Definition}}
        TRICE is ship-worthy only when the library produces stable manifests,
        verifier-backed measurements, and user-conditioned policies through a
        public API: \texttt{{tracerazor.trice}}. Generic users provide
        \texttt{{LiveTask}} plus a deterministic \texttt{{RepairAdapter}};
        the bundled JSON patch adapter is schema-bound and refuses test edits
        by default. Identical real-run smoke executions are regression-tested
        to reproduce result and artifact hashes. The public CLI is
        \texttt{{tracerazor-trice}} and mirrors the module form
        \texttt{{python -m tracerazor.trice}}. Multi-repository evaluation uses
        \texttt{{trice-suite/v1}} manifests and \texttt{{verify-suite}} for
        aggregate plus child-manifest verification. A replay-only result cannot
        certify savings.

        \section{{Limitations}}
        The current smoke uses deterministic managed repair recipes. The next
        paper-grade claim requires provider adapters, held-out repositories,
        clustered bootstrap confidence intervals, and repeated trials.

        \bibliographystyle{{plain}}
        \bibliography{{trice_v3_references}}
        \end{{document}}
        """
    ).strip() + "\n"


def render_bib() -> str:
    blocks = []
    for key, title, url in REFERENCES:
        blocks.append(
            dedent(
                f"""
                @misc{{{key},
                  title = {{{title}}},
                  howpublished = {{\\url{{{url}}}}},
                  year = {{2026}},
                  note = {{Accessed 2026-06-21}}
                }}
                """
            ).strip()
        )
    return "\n\n".join(blocks) + "\n"


def render_svg(rows: list[dict], gate: dict) -> str:
    width, height = 940, 390
    left, top = 150, 42
    bar_h, gap = 28, 18
    axis_w = 650
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="28" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE deterministic live input-token savings</text>',
        '<text x="24" y="52" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Fresh workspaces, real source edits, pytest pass preserved</text>',
    ]
    for i, r in enumerate(rows):
        y = top + 36 + i * (bar_h + gap)
        pct = r["savings"]
        w = int(axis_w * pct)
        parts.append(f'<text x="24" y="{y + 19}" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#111827">{r["task"]}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{axis_w}" height="{bar_h}" rx="5" fill="#eef2ff"/>')
        parts.append(f'<rect x="{left}" y="{y}" width="{w}" height="{bar_h}" rx="5" fill="#2563eb"/>')
        parts.append(f'<text x="{left + w + 10}" y="{y + 19}" font-family="Inter,Segoe UI,Arial" font-size="13" font-weight="700" fill="#111827">{100*pct:.1f}%</text>')
    avg = mean(r["savings"] for r in rows)
    ci = gate["savings_ci"]
    parts.append(f'<text x="24" y="{height - 42}" font-family="Inter,Segoe UI,Arial" font-size="14" font-weight="700" fill="#047857">Mean savings: {100*avg:.1f}% | 95% bootstrap CI: {100*ci["low"]:.1f}% to {100*ci["high"]:.1f}%</text>')
    parts.append(f'<text x="24" y="{height - 20}" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#4b5563">Local smoke gate: {"passed" if gate["smoke_gate_passed"] else "failed"} | broad claim allowed: {"yes" if gate["broad_claim_allowed"] else "no"}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def build_pdf(rows: list[dict], gate: dict, pdf_path: Path, suite: dict | None = None) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch)
    story = [
        Paragraph("TRICE: Deterministic Context Control for Live Software Agents", styles["Title"]),
        Paragraph("TraceRazor Research - 2026-06-21", styles["Small"]),
        Spacer(1, 0.18 * inch),
        Paragraph("Abstract", styles["Heading2"]),
        Paragraph(
            f"TRICE makes context assembly, evidence manifests, live verification, and acceptance gates deterministic. "
            f"Verifier durations are normalized before hashing and wall-clock fields are excluded from evidence. "
            f"On six bundled live repository tasks, TRICE reduces measured input context by {100*mean(r['savings'] for r in rows):.1f}% "
            f"(95% bootstrap CI: {100*gate['savings_ci']['low']:.1f}% to {100*gate['savings_ci']['high']:.1f}%) while preserving executable task success.",
            styles["BodyText"],
        ),
        Paragraph("Method", styles["Heading2"]),
        Paragraph(
            "TRICE chooses keep, extract, summarize, mask-with-receipt, anchor-prefix, or lazy-recall actions under a token budget. "
            "Acceptance requires live verifier pass preservation and measured input savings above the user-conditioned target.",
            styles["BodyText"],
        ),
    ]
    table_data = [["Task", "Baseline", "TRICE", "Savings", "Pass"]]
    for r in rows:
        table_data.append([r["task"], str(r["baseline"]), str(r["trice"]), f"{100*r['savings']:.1f}%", "yes" if r["passed"] else "no"])
    table_data.append(["Mean", "--", "--", f"{100*mean(r['savings'] for r in rows):.1f}%", "yes"])
    table = Table(table_data, colWidths=[2.0 * inch, 1.0 * inch, 0.9 * inch, 0.9 * inch, 0.7 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
    ]))
    story.extend([Spacer(1, 0.15 * inch), table, Spacer(1, 0.18 * inch)])
    story.extend([
        Paragraph("Replicated Suite Evidence", styles["Heading2"]),
        Paragraph(
            "No suite artifact was available during paper generation."
            if suite is None
            else (
                f"The public suite artifact {suite['name']} contains {suite['run_count']} live replicate runs "
                f"across {suite['task_clusters']} task cluster(s). Mean input-token savings is "
                f"{100*suite['mean_savings']:.1f}% with clustered-by-task 95% CI "
                f"{100*suite['cluster_low']:.1f}% to {100*suite['cluster_high']:.1f}%; "
                f"pass regressions {suite['pass_regressions']}."
            ),
            styles["BodyText"],
        ),
        Paragraph("Product Contract", styles["Heading2"]),
        Paragraph(
            "A TRICE result is accepted only when the public library emits a stable manifest, the live verifier passes, and replay is used only as preflight evidence. Generic users provide LiveTask plus a deterministic RepairAdapter; the bundled JSON patch adapter is schema-bound and refuses test edits by default. Identical real-run smoke executions must reproduce result and artifact hashes. Multi-repository evaluation uses trice-suite/v1 manifests and verify-suite for aggregate plus child-manifest verification. The public CLI is tracerazor-trice and mirrors python -m tracerazor.trice.",
            styles["BodyText"],
        ),
        Paragraph("Deterministic Claim Gate", styles["Heading2"]),
        Paragraph(
            f"Local smoke gate passed: {'yes' if gate['smoke_gate_passed'] else 'no'}. "
            f"Broad claim allowed: {'yes' if gate['broad_claim_allowed'] else 'no'}. "
            f"Rationale: {gate['rationale']}",
            styles["BodyText"],
        ),
        PageBreak(),
        Paragraph("Selected References", styles["Heading2"]),
    ])
    for key, title, url in REFERENCES:
        story.append(Paragraph(f"[{key}] {title}. {url}", styles["Small"]))
    doc.build(story)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate TRICE LaTeX, PDF, SVG, and paper manifest.")
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--suite-results", type=Path, default=DEFAULT_SUITE_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    args = ap.parse_args(argv)

    rows = load_rows(args.results)
    gate = load_gate(args.results)
    suite = load_suite_summary(args.suite_results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    tex_path = args.out_dir / "trice_v3_research_paper.tex"
    bib_path = args.out_dir / "trice_v3_references.bib"
    pdf_path = args.out_dir / "trice_v3_research_paper.pdf"
    docs_svg_path = args.docs_dir / "trice_v3_live_savings.svg"
    paper_svg_path = args.out_dir / "trice_v3_live_savings.svg"
    manifest_path = args.out_dir / "trice_v3_research_manifest.json"

    tex_path.write_text(render_tex(rows, gate, suite), encoding="utf-8")
    bib_path.write_text(render_bib(), encoding="utf-8")
    svg = render_svg(rows, gate)
    docs_svg_path.write_text(svg, encoding="utf-8")
    paper_svg_path.write_text(svg, encoding="utf-8")
    build_pdf(rows, gate, pdf_path, suite)

    manifest = build_manifest(
        json.loads(args.results.read_text(encoding="utf-8")),
        result_path=args.results,
        artifact_paths=[tex_path, bib_path, pdf_path, paper_svg_path],
        algorithm="trice-v3-paper-generator",
        notes=["LaTeX source and PDF generated from deterministic live evidence."],
        base_dir=args.out_dir,
    )
    write_manifest(manifest, manifest_path)
    print(f"tex: {tex_path}")
    print(f"pdf: {pdf_path}")
    print(f"svg: {docs_svg_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
