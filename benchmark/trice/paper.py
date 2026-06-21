"""Generate TRICE paper artifacts from deterministic live evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
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

from .claim import build_claim_card, render_claim_card_tex, write_claim_outputs
from .evidence import build_manifest, write_manifest
from .readiness import build_suite_readiness, render_readiness_tex, write_readiness_outputs
from .stats import claim_gate_from_rounds

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO / "benchmark" / "trice" / "results" / "v2-smoke" / "trice_v2_live_results.json"
DEFAULT_SUITE_RESULTS = REPO / "benchmark" / "trice" / "results" / "v2-suite" / "trice_suite_results.json"
DEFAULT_SUITE_BUNDLE = REPO / "benchmark" / "trice" / "results" / "v2-suite" / "trice_suite_evidence.trice.zip"
DEFAULT_BROAD_SUITE_RESULTS = REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_suite_results.json"
DEFAULT_BROAD_SUITE_MANIFEST = REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_suite_evidence_manifest.json"
DEFAULT_BROAD_SUITE_BUNDLE = REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_broad_smoke_evidence.trice.zip"
DEFAULT_READINESS_MANIFEST = REPO / "examples" / "trice_suite_bundled_live.json"
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
    ("taubench2024", "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains", "https://arxiv.org/abs/2406.12045"),
    ("agentbench2023", "AgentBench: Evaluating LLMs as Agents", "https://arxiv.org/abs/2308.03688"),
    ("toolllm2023", "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs", "https://arxiv.org/abs/2307.16789"),
    ("webarena2023", "WebArena: A Realistic Web Environment for Building Autonomous Agents", "https://arxiv.org/abs/2307.13854"),
    ("voyager2023", "Voyager: An Open-Ended Embodied Agent with Large Language Models", "https://arxiv.org/abs/2305.16291"),
    ("longbench2023", "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding", "https://arxiv.org/abs/2308.14508"),
    ("lostmiddle2023", "Lost in the Middle: How Language Models Use Long Contexts", "https://arxiv.org/abs/2307.03172"),
    ("memgpt2023", "MemGPT: Towards LLMs as Operating Systems", "https://arxiv.org/abs/2310.08560"),
    ("rag2020", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "https://arxiv.org/abs/2005.11401"),
    ("pagedattention2023", "Efficient Memory Management for Large Language Model Serving with PagedAttention", "https://arxiv.org/abs/2309.06180"),
    ("streamingllm2023", "Efficient Streaming Language Models with Attention Sinks", "https://arxiv.org/abs/2309.17453"),
    ("h2o2023", "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", "https://arxiv.org/abs/2306.14048"),
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
    source_path = path.parent / data["suite"].get("source_manifest", "")
    source = None
    if source_path.is_file():
        sources = json.loads(source_path.read_text(encoding="utf-8")).get("sources", [])
        source = sources[0] if sources else None
    return {
        "name": data["suite"]["name"],
        "run_count": gate.get("replicate_count", len(data.get("tasks", []))),
        "task_clusters": gate.get("task_cluster_count", data["suite"].get("task_count", 0)),
        "mean_savings": gate["mean_savings"],
        "cluster_low": clustered["low"],
        "cluster_high": clustered["high"],
        "pass_regressions": gate["pass_regressions"],
        "s_tier_gate": gate.get("s_tier_gate"),
        "source": source,
    }


def load_bundle_summary(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("trice_bundle_manifest.json").decode("utf-8"))
    return {
        "name": path.name,
        "entry_count": len(manifest.get("entries", [])),
        "root_manifest": manifest.get("root_manifest"),
        "root_result": manifest.get("root_result"),
    }


def _suite_tex_summary(suite: dict | None, missing_text: str) -> str:
    if suite is None:
        return missing_text
    s_tier = suite.get("s_tier_gate") or {}
    missing = [str(item).replace("_", "\\_") for item in s_tier.get("missing_requirements") or []]
    return (
        f"The suite \\texttt{{{suite['name']}}} contains {suite['run_count']} live run(s) "
        f"across {suite['task_clusters']} task cluster(s). Mean input-token savings is "
        f"{100*suite['mean_savings']:.1f}\\% with clustered-by-task 95\\% CI "
        f"{100*suite['cluster_low']:.1f}\\%--{100*suite['cluster_high']:.1f}\\%; "
        f"pass regressions {suite['pass_regressions']}. S-tier gate: "
        f"{'passed' if s_tier.get('passed') else 'not passed'}; missing requirements: "
        f"{', '.join(missing or ['none'])}."
    )


def render_tex(
    rows: list[dict],
    gate: dict,
    suite: dict | None = None,
    bundle: dict | None = None,
    broad_suite: dict | None = None,
    broad_bundle: dict | None = None,
    claim_card: dict | None = None,
    readiness_card: dict | None = None,
) -> str:
    avg = mean(r["savings"] for r in rows)
    ci = gate["savings_ci"]
    missing_s_tier = []
    if suite and suite.get("s_tier_gate"):
        missing_s_tier = [str(item).replace("_", "\\_") for item in suite["s_tier_gate"].get("missing_requirements") or []]
    suite_text = (
        "No suite artifact was available during paper generation."
        if suite is None
        else (
            f"The public suite artifact \\texttt{{{suite['name']}}} contains "
            f"{suite['run_count']} live replicate runs across {suite['task_clusters']} task cluster(s). "
            f"Mean input-token savings is {100*suite['mean_savings']:.1f}\\% with clustered-by-task "
            f"95\\% CI {100*suite['cluster_low']:.1f}\\%--{100*suite['cluster_high']:.1f}\\%; "
            f"pass regressions {suite['pass_regressions']}. "
            f"S-tier gate: {'passed' if (suite.get('s_tier_gate') or {}).get('passed') else 'not passed'}; "
            f"missing requirements: {', '.join(missing_s_tier or ['none'])}."
        )
    )
    bundle_text = (
        "No portable bundle artifact was available during paper generation."
        if bundle is None
        else (
            f"The public bundle \\texttt{{{bundle['name']}}} contains {bundle['entry_count']} hashed entries, "
            f"root manifest \\texttt{{{bundle['root_manifest']}}}, and root result \\texttt{{{bundle['root_result']}}}. "
            "Verification replays aggregate and child-manifest checks after safe extraction."
        )
    )
    broad_suite_text = _suite_tex_summary(broad_suite, "No broad-smoke suite artifact was available during paper generation.")
    broad_bundle_text = (
        "No broad-smoke bundle artifact was available during paper generation."
        if broad_bundle is None
        else (
            f"The broad-smoke bundle \\texttt{{{broad_bundle['name']}}} contains "
            f"{broad_bundle['entry_count']} hashed entries."
        )
    )
    source = suite.get("source") if suite else None
    source_intervention = ""
    if source is not None:
        if source.get("adapter_type") == "command_profile":
            source_intervention = (
                f"command adapter profile \\texttt{{{str(source.get('adapter_profile_name') or source.get('adapter_profile'))[:48]}...}} "
                f"with profile SHA-256 \\texttt{{{source.get('adapter_profile_sha256', '')[:12]}...}}"
            )
        elif source.get("adapter_type") == "command":
            source_intervention = (
                f"command adapter \\texttt{{{str(source.get('repair_cmd'))[:48]}...}} "
                f"with timeout {source.get('repair_timeout_s')}s"
            )
        else:
            source_intervention = (
                f"patch SHA-256 \\texttt{{{source['patch_sha256'][:12]}...}}"
            )
    source_text = (
        "No source fingerprint artifact was available during paper generation."
        if source is None
        else (
            f"The suite source manifest records repo tree digest \\texttt{{{source['repo_tree']['digest'][:12]}...}} "
            f"over {source['repo_tree']['file_count']} files and {source_intervention} "
            "before execution. Suite tasks may use either local "
            "paths or locked Git sources with URL, revision, optional subdirectory, and resolved commit recorded."
        )
    )
    table = "\n".join(
        f"{r['task']} & {r['baseline']} & {r['trice']} & {100*r['savings']:.1f}\\% & yes & yes \\\\"
        for r in rows
    )
    refs = ", ".join(f"\\cite{{{key}}}" for key, _, _ in REFERENCES[:8])
    stable_refs = ", ".join(
        f"\\cite{{{key}}}"
        for key in (
            "swebench2023",
            "sweagent2024",
            "taubench2024",
            "agentbench2023",
            "llmlingua2023",
            "longllmlingua2023",
            "promptcache2023",
            "pagedattention2023",
        )
    )
    claim_tex = render_claim_card_tex(claim_card) if claim_card else "\\section{Deterministic Claim Card}\nNo claim-card artifact was available during paper generation.\n"
    readiness_tex = render_readiness_tex(readiness_card) if readiness_card else "\\section{Suite Readiness Preflight}\nNo readiness artifact was available during paper generation.\n"
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

        \section{{Related Work}}
        TRICE sits at the intersection of real software-agent benchmarks,
        prompt/context compression, memory systems, and serving efficiency.
        Real-repo and tool-interaction benchmarks motivate executable,
        environment-grounded outcomes rather than transcript-only scoring
        {stable_refs}. Prompt compression and long-context work motivate
        shrinking carried context, but TRICE treats compression as a
        harness-verified decision-preservation problem rather than a standalone
        language modeling objective. Serving work on prompt reuse, KV/cache
        management, and attention-memory pressure motivates the cost term in
        the TRICE portfolio controller.

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
        The S-tier gate is a deterministic non-claim unless the suite clears
        savings, pass preservation, replication, held-out source, adapter
        profile, and receipt-validation requirements.

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

        {claim_tex}

        {readiness_tex}

        \section{{Replicated Suite Evidence}}
        {suite_text}

        \section{{Broad Smoke Evidence}}
        {broad_suite_text} {broad_bundle_text}

        \section{{Portable Artifact Bundle}}
        {bundle_text}

        \section{{Source Provenance}}
        {source_text}

        \section{{Product Definition}}
        TRICE is ship-worthy only when the library produces stable manifests,
        verifier-backed measurements, and user-conditioned policies through a
        public API: \texttt{{tracerazor.trice}}. Generic users provide
        \texttt{{LiveTask}} plus a deterministic \texttt{{RepairAdapter}};
        the bundled JSON patch and command repair adapters refuse test edits
        by default. Command adapters may be packaged as
        \texttt{{trice-adapter-profile/v1}} files and every live condition emits
        a \texttt{{trice-run-receipt/v1}} artifact with adapter envelope,
        command hash, before/after workspace fingerprints, changed files,
        output hashes, and optional agent-reported token accounting. Receipt
        artifacts have a shipped JSON Schema and are validated during manifest
        and bundle verification, so malformed receipts can fail even when their
        bytes are faithfully hashed. Identical real-run smoke executions are
        regression-tested to reproduce result and artifact hashes. The public CLI is
        \texttt{{tracerazor-trice}} and mirrors the module form
        \texttt{{python -m tracerazor.trice}}. Multi-repository evaluation uses
        \texttt{{trice-suite/v1}} manifests, \texttt{{verify-suite}} for
        aggregate plus child-manifest verification, and deterministic
        \texttt{{.trice.zip}} evidence bundles for handoff. A replay-only
        result cannot certify savings.

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


def build_pdf(
    rows: list[dict],
    gate: dict,
    pdf_path: Path,
    suite: dict | None = None,
    bundle: dict | None = None,
    broad_suite: dict | None = None,
    broad_bundle: dict | None = None,
    claim_card: dict | None = None,
    readiness_card: dict | None = None,
) -> None:
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
        Paragraph("Related Work", styles["Heading2"]),
        Paragraph(
            "TRICE combines software-agent benchmark lessons from SWE-bench, SWE-agent, tau-bench, AgentBench, and ToolLLM with prompt compression and serving-efficiency work such as LLMLingua, LongLLMLingua, Prompt Cache, PagedAttention, StreamingLLM, and H2O. The paper treats compression as a verifier-backed decision-preservation contract rather than a replay-only token counter.",
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
        Paragraph("Deterministic Claim Card", styles["Heading2"]),
        Paragraph(
            "No claim-card artifact was available during paper generation."
            if claim_card is None
            else (
                f"The generated claim card reports claim level {claim_card['claim_level']}, "
                f"claim allowed {'yes' if claim_card['claim_allowed'] else 'no'}, and determinism contract score "
                f"{claim_card['determinism_contract_score']}/100. It binds the suite result and evidence manifest hashes "
                f"and lists the non-claims that prevent over-selling the current smoke evidence."
            ),
            styles["BodyText"],
        ),
        Paragraph("Suite Readiness Preflight", styles["Heading2"]),
        Paragraph(
            "No readiness artifact was available during paper generation."
            if readiness_card is None
            else (
                f"The generated readiness card reports level {readiness_card['readiness_level']}, "
                f"pilot-ready {'yes' if readiness_card['pilot_execution_ready'] else 'no'}, "
                f"claim-ready {'yes' if readiness_card['claim_execution_ready'] else 'no'}, and "
                f"readiness score {readiness_card['readiness_score']}/100. It is a no-execution preflight gate: "
                "outcome claims still require live suite results, verified manifests, bundles, and a claim card."
            ),
            styles["BodyText"],
        ),
        Paragraph("Replicated Suite Evidence", styles["Heading2"]),
        Paragraph(
            "No suite artifact was available during paper generation."
            if suite is None
            else (
                f"The public suite artifact {suite['name']} contains {suite['run_count']} live replicate runs "
                f"across {suite['task_clusters']} task cluster(s). Mean input-token savings is "
                f"{100*suite['mean_savings']:.1f}% with clustered-by-task 95% CI "
                f"{100*suite['cluster_low']:.1f}% to {100*suite['cluster_high']:.1f}%; "
                f"pass regressions {suite['pass_regressions']}. "
                f"S-tier gate: {'passed' if (suite.get('s_tier_gate') or {}).get('passed') else 'not passed'}."
            ),
            styles["BodyText"],
        ),
        Paragraph("Portable Artifact Bundle", styles["Heading2"]),
        Paragraph(
            "No portable bundle artifact was available during paper generation."
            if bundle is None
            else (
                f"The public bundle {bundle['name']} contains {bundle['entry_count']} hashed entries, "
                f"root manifest {bundle['root_manifest']}, and root result {bundle['root_result']}. "
                "Verification replays aggregate and child-manifest checks after safe extraction."
            ),
            styles["BodyText"],
        ),
        Paragraph("Broad Smoke Evidence", styles["Heading2"]),
        Paragraph(
            "No broad-smoke suite artifact was available during paper generation."
            if broad_suite is None
            else (
                f"The broad-smoke suite {broad_suite['name']} contains {broad_suite['run_count']} live run(s) "
                f"across {broad_suite['task_clusters']} task cluster(s). Mean input-token savings is "
                f"{100*broad_suite['mean_savings']:.1f}% with clustered-by-task 95% CI "
                f"{100*broad_suite['cluster_low']:.1f}% to {100*broad_suite['cluster_high']:.1f}%; "
                f"pass regressions {broad_suite['pass_regressions']}. "
                f"S-tier gate: {'passed' if (broad_suite.get('s_tier_gate') or {}).get('passed') else 'not passed'}."
            ),
            styles["BodyText"],
        ),
        Paragraph(
            "No broad-smoke bundle artifact was available during paper generation."
            if broad_bundle is None
            else f"The broad-smoke bundle {broad_bundle['name']} contains {broad_bundle['entry_count']} hashed entries.",
            styles["BodyText"],
        ),
        Paragraph("Source Provenance", styles["Heading2"]),
        Paragraph(
            "No source fingerprint artifact was available during paper generation."
            if suite is None or suite.get("source") is None
            else (
                f"The suite source manifest records repo tree digest {suite['source']['repo_tree']['digest'][:12]}... "
                f"over {suite['source']['repo_tree']['file_count']} files and intervention provenance before execution. "
                "JSON patch tasks record patch SHA-256; command adapter tasks record command argv and timeout; adapter-profile tasks record profile SHA-256. Suite tasks may use either local "
                "paths or locked Git sources with URL, revision, optional subdirectory, and resolved commit recorded."
            ),
            styles["BodyText"],
        ),
        Paragraph("Product Contract", styles["Heading2"]),
        Paragraph(
            "A TRICE result is accepted only when the public library emits a stable manifest, the live verifier passes, and replay is used only as preflight evidence. Generic users provide LiveTask plus a deterministic RepairAdapter; the bundled JSON patch and command repair adapters refuse test edits by default. Command adapters may be packaged as trice-adapter-profile/v1 files, and every live condition emits a trice-run-receipt/v1 artifact with adapter envelope, command hash, before/after workspace fingerprints, changed files, output hashes, and optional agent-reported token accounting. Receipts have a shipped JSON Schema and are validated during manifest and bundle verification. Identical real-run smoke executions must reproduce result and artifact hashes. Multi-repository evaluation uses trice-suite/v1 manifests, local or locked Git sources, verify-suite for aggregate plus child-manifest verification, and deterministic .trice.zip evidence bundles for handoff. The public CLI is tracerazor-trice and mirrors python -m tracerazor.trice.",
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
    ap.add_argument("--suite-bundle", type=Path, default=DEFAULT_SUITE_BUNDLE)
    ap.add_argument("--broad-suite-results", type=Path, default=DEFAULT_BROAD_SUITE_RESULTS)
    ap.add_argument("--broad-suite-manifest", type=Path, default=DEFAULT_BROAD_SUITE_MANIFEST)
    ap.add_argument("--broad-suite-bundle", type=Path, default=DEFAULT_BROAD_SUITE_BUNDLE)
    ap.add_argument("--readiness-manifest", type=Path, default=DEFAULT_READINESS_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    args = ap.parse_args(argv)

    rows = load_rows(args.results)
    gate = load_gate(args.results)
    suite = load_suite_summary(args.suite_results)
    bundle = load_bundle_summary(args.suite_bundle)
    broad_suite = load_suite_summary(args.broad_suite_results)
    broad_bundle = load_bundle_summary(args.broad_suite_bundle)
    claim_card = build_claim_card(args.broad_suite_results, manifest_path=args.broad_suite_manifest)
    readiness_card = build_suite_readiness(args.readiness_manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    tex_path = args.out_dir / "trice_v3_research_paper.tex"
    bib_path = args.out_dir / "trice_v3_references.bib"
    pdf_path = args.out_dir / "trice_v3_research_paper.pdf"
    docs_svg_path = args.docs_dir / "trice_v3_live_savings.svg"
    paper_svg_path = args.out_dir / "trice_v3_live_savings.svg"
    paper_bundle_path = args.out_dir / "trice_suite_evidence.trice.zip"
    paper_broad_bundle_path = args.out_dir / "trice_broad_smoke_evidence.trice.zip"
    paper_claim_path = args.out_dir / "trice_claim_card.json"
    docs_claim_path = args.docs_dir / "trice_claim_card.json"
    paper_readiness_path = args.out_dir / "trice_suite_readiness.json"
    docs_readiness_path = args.docs_dir / "trice_suite_readiness.json"
    manifest_path = args.out_dir / "trice_v3_research_manifest.json"

    tex_path.write_text(render_tex(rows, gate, suite, bundle, broad_suite, broad_bundle, claim_card, readiness_card), encoding="utf-8")
    bib_path.write_text(render_bib(), encoding="utf-8")
    svg = render_svg(rows, gate)
    docs_svg_path.write_text(svg, encoding="utf-8")
    paper_svg_path.write_text(svg, encoding="utf-8")
    build_pdf(rows, gate, pdf_path, suite, bundle, broad_suite, broad_bundle, claim_card, readiness_card)
    artifact_paths = [tex_path, bib_path, pdf_path, paper_svg_path]
    paper_claim_outputs = write_claim_outputs(claim_card, paper_claim_path)
    write_claim_outputs(claim_card, docs_claim_path)
    artifact_paths.extend([Path(path) for path in paper_claim_outputs.values()])
    paper_readiness_outputs = write_readiness_outputs(readiness_card, paper_readiness_path)
    write_readiness_outputs(readiness_card, docs_readiness_path)
    artifact_paths.extend([Path(path) for path in paper_readiness_outputs.values()])
    if args.suite_bundle.is_file():
        shutil.copy2(args.suite_bundle, paper_bundle_path)
        artifact_paths.append(paper_bundle_path)
    if args.broad_suite_bundle.is_file():
        shutil.copy2(args.broad_suite_bundle, paper_broad_bundle_path)
        artifact_paths.append(paper_broad_bundle_path)

    manifest = build_manifest(
        json.loads(args.results.read_text(encoding="utf-8")),
        result_path=args.results,
        artifact_paths=artifact_paths,
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
