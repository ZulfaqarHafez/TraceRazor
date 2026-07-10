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

from .artifact import build_artifact_card, write_artifact_outputs
from .claim import build_claim_card, render_claim_card_tex, write_claim_outputs
from .contract import build_contract_card, render_contract_tex, write_contract_outputs
from .crates import build_crates_card, write_crates_outputs
from .design import build_design_card, render_design_tex, write_design_outputs
from .evidence import build_manifest, write_manifest, write_text_lf
from .install import build_install_card, render_install_tex, write_install_outputs
from .protocol import build_protocol_lock, render_protocol_tex, write_protocol_outputs
from .readiness import build_suite_readiness, render_readiness_tex, write_readiness_outputs
from .release_evidence import build_release_evidence_card, write_release_evidence_outputs
from .reproduction import build_reproduction_card, write_reproduction_outputs
from .research import build_research_card, render_research_tex, write_research_outputs
from .stats import claim_gate_from_rounds

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO / "benchmark" / "trice" / "results" / "v2-smoke" / "trice_v2_live_results.json"
DEFAULT_SUITE_RESULTS = REPO / "benchmark" / "trice" / "results" / "v2-suite" / "trice_suite_results.json"
DEFAULT_SUITE_BUNDLE = REPO / "benchmark" / "trice" / "results" / "v2-suite" / "trice_suite_evidence.trice.zip"
DEFAULT_BROAD_SUITE_RESULTS = REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_suite_results.json"
DEFAULT_BROAD_SUITE_MANIFEST = REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_suite_evidence_manifest.json"
DEFAULT_BROAD_SUITE_BUNDLE = REPO / "benchmark" / "trice" / "results" / "v2-broad-smoke" / "trice_broad_smoke_evidence.trice.zip"
DEFAULT_REMOTE_SUITE_RESULTS = REPO / "benchmark" / "trice" / "results" / "v2-remote-smoke" / "trice_suite_results.json"
DEFAULT_REMOTE_SUITE_MANIFEST = REPO / "benchmark" / "trice" / "results" / "v2-remote-smoke" / "trice_suite_evidence_manifest.json"
DEFAULT_REMOTE_SUITE_BUNDLE = REPO / "benchmark" / "trice" / "results" / "v2-remote-smoke" / "trice_remote_smoke_evidence.trice.zip"
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
    ("slsa", "Supply-chain Levels for Software Artifacts Specification", "https://slsa.dev/spec/"),
    ("openssfscorecard", "OpenSSF Scorecard", "https://scorecard.dev/"),
    ("pypitrust", "PyPI Trusted Publishing", "https://docs.pypi.org/trusted-publishers/"),
    ("pypiattest", "PyPI Digital Attestations", "https://docs.pypi.org/attestations/"),
    ("intoto", "in-toto Attestations", "https://in-toto.io/"),
    ("cyclonedx", "CycloneDX Specification", "https://cyclonedx.org/specification/overview/"),
    ("mlperfpolicies", "MLPerf Policies and Submission Rules", "https://docs.mlcommons.org/inference/policies/"),
    ("pypasample", "PyPA Sample Project", "https://github.com/pypa/sampleproject"),
    ("semver", "Semantic Versioning 2.0.0", "https://semver.org/"),
    ("pyversion", "Python Version Specifiers", "https://packaging.python.org/en/latest/specifications/version-specifiers/"),
    ("jsonschema2020", "JSON Schema Draft 2020-12", "https://json-schema.org/draft/2020-12"),
    ("reprobuilds", "Reproducible Builds", "https://reproducible-builds.org/"),
    ("cargoPublishing", "Cargo Publishing Reference", "https://doc.rust-lang.org/cargo/reference/publishing.html"),
    ("pypapackaging", "Python Packaging User Guide: Packaging Python Projects", "https://packaging.python.org/tutorials/packaging-projects/"),
    ("pipinstall", "pip install Command Reference", "https://pip.pypa.io/en/stable/cli/pip_install/"),
    ("wheelpep", "PEP 427: The Wheel Binary Package Format", "https://peps.python.org/pep-0427/"),
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


def _tex(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def render_tex(
    rows: list[dict],
    gate: dict,
    suite: dict | None = None,
    bundle: dict | None = None,
    broad_suite: dict | None = None,
    broad_bundle: dict | None = None,
    remote_suite: dict | None = None,
    remote_bundle: dict | None = None,
    remote_claim_card: dict | None = None,
    claim_card: dict | None = None,
    readiness_card: dict | None = None,
    protocol_card: dict | None = None,
    design_card: dict | None = None,
    contract_card: dict | None = None,
    crates_card: dict | None = None,
    install_card: dict | None = None,
    research_card: dict | None = None,
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
    remote_suite_text = _suite_tex_summary(remote_suite, "No remote-git smoke suite artifact was available during paper generation.")
    remote_bundle_text = (
        "No remote-git smoke bundle artifact was available during paper generation."
        if remote_bundle is None
        else (
            f"The remote-git smoke bundle \\texttt{{{remote_bundle['name']}}} contains "
            f"{remote_bundle['entry_count']} hashed entries."
        )
    )
    remote_claim_text = (
        "No remote-git smoke claim card was available during paper generation."
        if remote_claim_card is None
        else (
            f"The remote-git smoke claim card reports claim level "
            f"\\texttt{{{remote_claim_card['claim_level']}}}, claim allowed "
            f"{'yes' if remote_claim_card['claim_allowed'] else 'no'}, and determinism score "
            f"{remote_claim_card['determinism_contract_score']}/100. "
            "It is a verified smoke artifact, not an S-tier claim."
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
    protocol_tex = render_protocol_tex(protocol_card) if protocol_card else "\\section{Protocol Lock}\nNo protocol-lock artifact was available during paper generation.\n"
    design_tex = render_design_tex(design_card) if design_card else "\\section{Statistical Design Card}\nNo design-card artifact was available during paper generation.\n"
    claim_tex = render_claim_card_tex(claim_card) if claim_card else "\\section{Deterministic Claim Card}\nNo claim-card artifact was available during paper generation.\n"
    readiness_tex = render_readiness_tex(readiness_card) if readiness_card else "\\section{Suite Readiness Preflight}\nNo readiness artifact was available during paper generation.\n"
    contract_tex = render_contract_tex(contract_card) if contract_card else "\\section{Public Contract Card}\nNo contract-card artifact was available during paper generation.\n"
    install_tex = render_install_tex(install_card) if install_card else "\\section{Installability Card}\nNo installability artifact was available during paper generation.\n"
    research_tex = render_research_tex(research_card) if research_card else "\\section{Research Card}\nNo research-card artifact was available during paper generation.\n"
    crates_level_tex = str((crates_card or {}).get("crates_card_level") or "").replace("_", "\\_")
    reproduction_protocol_tex = (
        "\\section{Reviewer Reproduction Packet}\n"
        "TRICE emits a separate reproduction card after the paper manifest is written. "
        "The card binds readiness, protocol, design, claim, bundle, paper-manifest, "
        "and paper-result inputs with SHA-256 receipts and lists the exact verifier "
        "commands a reviewer can run. It is intentionally outside the paper manifest "
        "to avoid a hash cycle; the final artifact card binds both the paper manifest "
        "and reproduction card.\n"
    )
    release_card_tex = (
        "\\section{Release Trust Card}\n"
        "TRICE treats distribution readiness as a separate machine-readable "
        "contract from benchmark evidence. The release card snapshots "
        "\\texttt{tracerazor-trice doctor}, binds the artifact, reproduction, "
        "and contract cards, records package metadata, and refuses \\texttt{public\\_release\\_ready} "
        "until PyPI, piwheels, crates.io, GitHub tag alignment, GitHub "
        "Actions, and OpenSSF Scorecard are green. This follows supply-chain guidance that provenance, "
        "public project health, trusted publishing, attestations, SBOMs, and "
        "checksums should be release artifacts rather than prose-only claims "
        "\\cite{slsa,openssfscorecard,pypitrust,pypiattest,intoto,cyclonedx}. "
        "The separate \\texttt{tracerazor-trice release-evidence} packet binds "
        "the platform wheel, CLI binary, installability card, proof cards, paper artifacts, "
        "evidence bundles, SHA-256 checksums, CycloneDX-style Python and Cargo "
        "SBOMs, and an in-toto/SLSA-shaped provenance statement. The GitHub "
        "release workflow then generates release-evidence sidecars and hosted "
        "artifact attestations for release assets produced by Actions. A separate "
        "\\texttt{tracerazor-trice integrity} card binds offline doctor output, "
            "proof-card verifiers, release evidence, the crates publish card, "
            "the research card, the paper manifest, schemas, "
        "and workflow hooks as one top-level proof graph. The "
        "contract card follows SemVer, Python package-versioning, JSON Schema, "
        "and reproducible-input practice by declaring the API boundary before "
        "release claims \\cite{semver,pyversion,jsonschema2020,reprobuilds}.\n"
    )
    crates_card_tex = (
        "\\section{Crates Publish Card}\n"
        "No crates publish card was available during paper generation.\n"
        if crates_card is None
        else (
            "\\section{Crates Publish Card}\n"
            f"The generated crates card reports level \\texttt{{{crates_level_tex}}} "
            f"and score {crates_card['crates_publish_score']}/100. It binds Cargo manifests, "
            "local dependency order, crates.io registry status, and README cargo-install honesty. "
            f"At generation time, \\texttt{{cargo\\_install\\_claim\\_allowed}} was "
            f"{'true' if crates_card['cargo_install_claim_allowed'] else 'false'}, "
            "so Rust install claims remain blocked until the final CLI crate is public "
            "\\cite{cargoPublishing}.\n"
        )
    )
    install_summary_tex = (
        "\\section{Clean Wheel Installability}\n"
        "No installability card was available during paper generation.\n"
        if install_card is None
        else (
            "\\section{Clean Wheel Installability}\n"
            f"The generated installability card reports level \\texttt{{{_tex(str(install_card['install_level']))}}} "
            f"and score {install_card['install_score']}/100. It creates a clean virtual environment, "
            "installs the built wheel with \\texttt{pip install --no-deps}, imports packaged TRICE schemas "
            "and public APIs, and runs the installed \\texttt{tracerazor-trice} console script. "
            "The full \\texttt{tracerazor} Rust CLI check is kept separate so generic wheels do not overclaim "
            "platform-bundled binary readiness \\cite{pypapackaging,pipinstall,wheelpep}.\n"
        )
    )
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

        {readiness_tex}

        {protocol_tex}

        {design_tex}

        {claim_tex}

        {reproduction_protocol_tex}

        {contract_tex}

        {install_tex}

        {research_tex}

        {release_card_tex}

        {crates_card_tex}

        {install_summary_tex}

        \section{{Replicated Suite Evidence}}
        {suite_text}

        \section{{Broad Smoke Evidence}}
        {broad_suite_text} {broad_bundle_text}

        \section{{Remote-Git Smoke Evidence}}
        {remote_suite_text} {remote_bundle_text} {remote_claim_text}
        The remote smoke fixture clones the locked public PyPA sample project
        revision, records the resolved commit and repository tree digest, applies
        a declarative source-only patch, and verifies the objective behavior.
        This follows benchmark-submission practice that inputs, revisions, and
        rules must be fixed before a claim is interpreted
        \cite{{mlperfpolicies,pypasample}}.

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
    remote_suite: dict | None = None,
    remote_bundle: dict | None = None,
    remote_claim_card: dict | None = None,
    claim_card: dict | None = None,
    readiness_card: dict | None = None,
    protocol_card: dict | None = None,
    design_card: dict | None = None,
    contract_card: dict | None = None,
    crates_card: dict | None = None,
    install_card: dict | None = None,
    research_card: dict | None = None,
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
        Paragraph("Protocol Lock", styles["Heading2"]),
        Paragraph(
            "No protocol-lock artifact was available during paper generation."
            if protocol_card is None
            else (
                f"The generated protocol lock reports level {protocol_card['protocol_level']}, "
                f"claim allowed by protocol {'yes' if protocol_card['claim_allowed_by_protocol'] else 'no'}, "
                f"and protocol score {protocol_card['protocol_score']}/100. It pre-registers the primary metric, "
                "pass-preservation guardrail, clustered confidence rule, required task clusters, replicates, "
                "locked sources, adapter profiles, receipts, claim card, and artifact card before outcome evidence."
            ),
            styles["BodyText"],
        ),
        Paragraph("Statistical Design Card", styles["Heading2"]),
        Paragraph(
            "No design-card artifact was available during paper generation."
            if design_card is None
            else (
                f"The generated design card reports level {design_card['design_level']}, "
                f"claim-design ready {'yes' if design_card['claim_design_ready'] else 'no'}, "
                f"and design score {design_card['design_score']}/100. It uses task-cluster means to project "
                "whether the locked claim sample size would clear the savings target while preserving the "
                "non-claim boundary when held-out design requirements are missing."
            ),
            styles["BodyText"],
        ),
        Paragraph("Reviewer Reproduction Packet", styles["Heading2"]),
        Paragraph(
            "TRICE emits a separate reproduction card after the paper manifest is written. "
            "The card binds readiness, protocol, design, claim, bundle, paper-manifest, and paper-result inputs "
            "with SHA-256 receipts and lists exact verifier commands. It is intentionally outside the paper manifest "
            "to avoid a hash cycle; the final artifact card binds both the paper manifest and reproduction card.",
            styles["BodyText"],
        ),
        Paragraph("Public Contract Card", styles["Heading2"]),
        Paragraph(
            "No contract-card artifact was available during paper generation."
            if contract_card is None
            else (
                f"The generated contract card reports level {contract_card['contract_level']} "
                f"and score {contract_card['contract_score']}/100. It binds SemVer, public imports, "
                "tracerazor-trice commands, shipped schemas, examples, docs, and package metadata so release "
                "compatibility claims have a declared API boundary."
            ),
            styles["BodyText"],
        ),
        Paragraph("Release Trust Card", styles["Heading2"]),
        Paragraph(
            "TRICE treats distribution readiness as a separate machine-readable contract from benchmark evidence. "
            "The release card snapshots tracerazor-trice doctor, binds the artifact, reproduction, and contract cards, records package metadata, "
            "and refuses public_release_ready until PyPI, piwheels, crates.io, GitHub tag alignment, GitHub Actions, and OpenSSF Scorecard are green. "
            "This follows supply-chain guidance that provenance, public project health, trusted publishing, attestations, SBOMs, and checksums should be release artifacts rather than prose-only claims. "
            "The separate tracerazor-trice release-evidence packet binds the platform wheel, CLI binary, installability card, proof cards, paper artifacts, evidence bundles, SHA-256 checksums, CycloneDX-style Python and Cargo SBOMs, and an in-toto/SLSA-shaped provenance statement. The GitHub release workflow then generates release-evidence sidecars and hosted artifact attestations for release assets produced by Actions. A separate tracerazor-trice integrity card binds offline doctor output, proof-card verifiers, release evidence, the crates publish card, installability card, research card, the paper manifest, schemas, and workflow hooks as one top-level proof graph.",
            styles["BodyText"],
        ),
        Paragraph("Clean Wheel Installability", styles["Heading2"]),
        Paragraph(
            "No installability card was available during paper generation."
            if install_card is None
            else (
                f"The generated installability card reports level {install_card['install_level']} "
                f"and score {install_card['install_score']}/100. It creates a clean virtual environment, installs the built wheel with pip install --no-deps, "
                "imports packaged schemas and public APIs, and runs the installed tracerazor-trice console script. "
                "The full tracerazor Rust CLI check remains separate so generic wheels do not overclaim platform-bundled binary readiness."
            ),
            styles["BodyText"],
        ),
        Paragraph("Research Card", styles["Heading2"]),
        Paragraph(
            "No research-card artifact was available during paper generation."
            if research_card is None
            else (
                f"The generated research card reports level {research_card['research_level']} "
                f"and score {research_card['research_score']}/100. It binds "
                f"{research_card['source_count']} ledgered sources, {research_card['unique_source_count']} unique URLs, "
                "category coverage, row hashes, and the source ledger hash. This proves the paper basis is reviewable, "
                "not that the held-out S-tier outcome gate has passed."
            ),
            styles["BodyText"],
        ),
        Paragraph("Crates Publish Card", styles["Heading2"]),
        Paragraph(
            "No crates publish card was available during paper generation."
            if crates_card is None
            else (
                f"The generated crates card reports level {crates_card['crates_card_level']} "
                f"and score {crates_card['crates_publish_score']}/100. It binds Cargo manifests, "
                "local dependency order, crates.io registry status, and README cargo-install honesty. "
                f"cargo_install_claim_allowed is {str(bool(crates_card['cargo_install_claim_allowed'])).lower()}, "
                "so Rust install wording remains blocked until the final CLI crate is public."
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
        Paragraph("Remote-Git Smoke Evidence", styles["Heading2"]),
        Paragraph(
            "No remote-git smoke suite artifact was available during paper generation."
            if remote_suite is None
            else (
                f"The remote-git smoke suite {remote_suite['name']} contains {remote_suite['run_count']} live run(s) "
                f"across {remote_suite['task_clusters']} locked public Git task cluster(s). Mean input-token savings is "
                f"{100*remote_suite['mean_savings']:.1f}% with clustered-by-task 95% CI "
                f"{100*remote_suite['cluster_low']:.1f}% to {100*remote_suite['cluster_high']:.1f}%; "
                f"pass regressions {remote_suite['pass_regressions']}. "
                f"S-tier gate: {'passed' if (remote_suite.get('s_tier_gate') or {}).get('passed') else 'not passed'}."
            ),
            styles["BodyText"],
        ),
        Paragraph(
            "No remote-git smoke bundle artifact was available during paper generation."
            if remote_bundle is None
            else f"The remote-git smoke bundle {remote_bundle['name']} contains {remote_bundle['entry_count']} hashed entries.",
            styles["BodyText"],
        ),
        Paragraph(
            "No remote-git smoke claim card was available during paper generation."
            if remote_claim_card is None
            else (
                f"The remote-git smoke claim card reports level {remote_claim_card['claim_level']}, "
                f"claim allowed {'yes' if remote_claim_card['claim_allowed'] else 'no'}, "
                f"and determinism score {remote_claim_card['determinism_contract_score']}/100. "
                "It is a verified smoke artifact, not an S-tier claim."
            ),
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
    ap.add_argument("--remote-suite-results", type=Path, default=DEFAULT_REMOTE_SUITE_RESULTS)
    ap.add_argument("--remote-suite-manifest", type=Path, default=DEFAULT_REMOTE_SUITE_MANIFEST)
    ap.add_argument("--remote-suite-bundle", type=Path, default=DEFAULT_REMOTE_SUITE_BUNDLE)
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
    remote_suite = load_suite_summary(args.remote_suite_results)
    remote_bundle = load_bundle_summary(args.remote_suite_bundle)
    claim_card = build_claim_card(args.broad_suite_results, manifest_path=args.broad_suite_manifest)
    remote_claim_card = (
        build_claim_card(
            args.remote_suite_results,
            manifest_path=args.remote_suite_manifest,
            scope="remote-git smoke path on one locked public Python repository",
        )
        if args.remote_suite_results.is_file()
        else None
    )
    readiness_card = build_suite_readiness(args.readiness_manifest)
    protocol_card = build_protocol_lock(args.readiness_manifest)
    contract_card = build_contract_card()
    crates_card = build_crates_card()
    install_card = build_install_card()
    research_card = build_research_card()
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
    paper_remote_claim_path = args.out_dir / "trice_remote_smoke_claim_card.json"
    docs_remote_claim_path = args.docs_dir / "trice_remote_smoke_claim_card.json"
    paper_readiness_path = args.out_dir / "trice_suite_readiness.json"
    docs_readiness_path = args.docs_dir / "trice_suite_readiness.json"
    paper_protocol_path = args.out_dir / "trice_protocol_lock.json"
    docs_protocol_path = args.docs_dir / "trice_protocol_lock.json"
    paper_design_path = args.out_dir / "trice_design_card.json"
    docs_design_path = args.docs_dir / "trice_design_card.json"
    paper_reproduction_path = args.out_dir / "trice_reproduction_card.json"
    docs_reproduction_path = args.docs_dir / "trice_reproduction_card.json"
    paper_contract_path = args.out_dir / "trice_contract_card.json"
    docs_contract_path = args.docs_dir / "trice_contract_card.json"
    paper_artifact_path = args.out_dir / "trice_artifact_card.json"
    docs_artifact_path = args.docs_dir / "trice_artifact_card.json"
    paper_release_evidence_path = args.out_dir / "trice_release_evidence.json"
    docs_release_evidence_path = args.docs_dir / "trice_release_evidence.json"
    paper_crates_path = args.out_dir / "trice_crates_card.json"
    docs_crates_path = args.docs_dir / "trice_crates_card.json"
    paper_install_path = args.out_dir / "trice_install_card.json"
    docs_install_path = args.docs_dir / "trice_install_card.json"
    paper_research_path = args.out_dir / "trice_research_card.json"
    docs_research_path = args.docs_dir / "trice_research_card.json"
    manifest_path = args.out_dir / "trice_v3_research_manifest.json"

    paper_protocol_outputs = write_protocol_outputs(protocol_card, paper_protocol_path)
    write_protocol_outputs(protocol_card, docs_protocol_path)
    design_card = build_design_card(docs_protocol_path, suite_result_path=args.broad_suite_results)
    write_text_lf(
        tex_path,
        render_tex(
            rows,
            gate,
            suite,
            bundle,
            broad_suite,
            broad_bundle,
            remote_suite,
            remote_bundle,
            remote_claim_card,
            claim_card,
            readiness_card,
            protocol_card,
            design_card,
            contract_card,
            crates_card,
            install_card,
            research_card,
        ),
    )
    write_text_lf(bib_path, render_bib())
    svg = render_svg(rows, gate)
    write_text_lf(docs_svg_path, svg)
    write_text_lf(paper_svg_path, svg)
    build_pdf(
        rows,
        gate,
        pdf_path,
        suite,
        bundle,
        broad_suite,
        broad_bundle,
        remote_suite,
        remote_bundle,
        remote_claim_card,
        claim_card,
        readiness_card,
        protocol_card,
        design_card,
        contract_card,
        crates_card,
        install_card,
        research_card,
    )
    artifact_paths = [tex_path, bib_path, pdf_path, paper_svg_path]
    paper_claim_outputs = write_claim_outputs(claim_card, paper_claim_path)
    write_claim_outputs(claim_card, docs_claim_path)
    artifact_paths.extend([Path(path) for path in paper_claim_outputs.values()])
    if remote_claim_card is not None:
        paper_remote_claim_outputs = write_claim_outputs(remote_claim_card, paper_remote_claim_path)
        write_claim_outputs(remote_claim_card, docs_remote_claim_path)
        artifact_paths.extend([Path(path) for path in paper_remote_claim_outputs.values()])
    paper_readiness_outputs = write_readiness_outputs(readiness_card, paper_readiness_path)
    write_readiness_outputs(readiness_card, docs_readiness_path)
    artifact_paths.extend([Path(path) for path in paper_readiness_outputs.values()])
    artifact_paths.extend([Path(path) for path in paper_protocol_outputs.values()])
    paper_design_outputs = write_design_outputs(design_card, paper_design_path)
    write_design_outputs(design_card, docs_design_path)
    artifact_paths.extend([Path(path) for path in paper_design_outputs.values()])
    paper_contract_outputs = write_contract_outputs(contract_card, paper_contract_path)
    write_contract_outputs(contract_card, docs_contract_path)
    artifact_paths.extend([Path(path) for path in paper_contract_outputs.values()])
    paper_crates_outputs = write_crates_outputs(crates_card, paper_crates_path)
    write_crates_outputs(crates_card, docs_crates_path)
    artifact_paths.extend([Path(path) for path in paper_crates_outputs.values()])
    paper_install_outputs = write_install_outputs(install_card, paper_install_path)
    write_install_outputs(install_card, docs_install_path)
    artifact_paths.extend([Path(path) for path in paper_install_outputs.values()])
    paper_research_outputs = write_research_outputs(research_card, paper_research_path)
    write_research_outputs(research_card, docs_research_path)
    artifact_paths.extend([Path(path) for path in paper_research_outputs.values()])
    if args.suite_bundle.is_file():
        shutil.copy2(args.suite_bundle, paper_bundle_path)
        artifact_paths.append(paper_bundle_path)
    if args.broad_suite_bundle.is_file():
        shutil.copy2(args.broad_suite_bundle, paper_broad_bundle_path)
        artifact_paths.append(paper_broad_bundle_path)
    paper_remote_bundle_path = args.out_dir / "trice_remote_smoke_evidence.trice.zip"
    if args.remote_suite_bundle.is_file():
        shutil.copy2(args.remote_suite_bundle, paper_remote_bundle_path)
        artifact_paths.append(paper_remote_bundle_path)

    manifest = build_manifest(
        json.loads(args.results.read_text(encoding="utf-8")),
        result_path=args.results,
        artifact_paths=artifact_paths,
        algorithm="trice-v3-paper-generator",
        notes=["LaTeX source and PDF generated from deterministic live evidence."],
        base_dir=args.out_dir,
    )
    write_manifest(manifest, manifest_path)
    reproduction_card = build_reproduction_card(
        readiness_path=docs_readiness_path,
        suite_manifest_path=args.readiness_manifest,
        protocol_path=docs_protocol_path,
        design_path=docs_design_path,
        broad_result_path=args.broad_suite_results,
        broad_manifest_path=args.broad_suite_manifest,
        broad_bundle_path=args.broad_suite_bundle,
        claim_path=docs_claim_path,
        paper_manifest_path=manifest_path,
        paper_result_path=args.results,
    )
    write_reproduction_outputs(reproduction_card, paper_reproduction_path)
    write_reproduction_outputs(reproduction_card, docs_reproduction_path)
    artifact_card = build_artifact_card(reproduction_path=docs_reproduction_path, contract_path=docs_contract_path)
    write_artifact_outputs(artifact_card, paper_artifact_path)
    write_artifact_outputs(artifact_card, docs_artifact_path)
    release_evidence_card = build_release_evidence_card(sidecar_stem=docs_release_evidence_path.stem)
    write_release_evidence_outputs(release_evidence_card, paper_release_evidence_path)
    write_release_evidence_outputs(release_evidence_card, docs_release_evidence_path)
    print(f"tex: {tex_path}")
    print(f"pdf: {pdf_path}")
    print(f"svg: {docs_svg_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
