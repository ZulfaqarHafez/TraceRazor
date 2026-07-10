"""Deterministic research-basis cards for TRICE."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

from .evidence import canonical_json, sha256_file, write_text_lf

RESEARCH_CARD_SCHEMA_VERSION = "trice-research-card/v1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = REPO / "docs" / "trice_research_ledger.md"
DEFAULT_OUT = REPO / "docs" / "trice_research_card.json"
DEFAULT_MIN_SOURCES = 150

ROW_RE = re.compile(
    r"^\|\s*(?P<index>\d+)\s*\|\s*(?P<area>[^|]+?)\s*\|\s*\[(?P<title>[^\]]+)\]\((?P<url>[^)]+)\)\s*\|\s*(?P<takeaway>.*?)\s*\|\s*$"
)

CATEGORY_RULES = {
    "agent_evaluation": (
        "agent",
        "benchmark",
        "bench",
        "harness",
        "repo",
        "software",
        "testing",
        "environment",
        "reliability",
        "live",
        "github",
    ),
    "context_efficiency": (
        "prompt",
        "context",
        "compression",
        "memory",
        "retrieval",
        "cache",
        "kv",
        "token",
        "attention",
        "serving",
        "routing",
    ),
    "statistical_quality": (
        "calibration",
        "confidence",
        "bootstrap",
        "psychometric",
        "evaluation",
        "metric",
        "contamination",
        "reproducible",
    ),
    "public_trust": (
        "supply",
        "provenance",
        "package",
        "pypi",
        "pip",
        "cargo",
        "scorecard",
        "attestation",
        "sbom",
        "artifact",
        "semver",
        "schema",
        "release",
        "wheel",
        "security",
    ),
    "trice_internal": ("trice", "release evidence", "remote smoke", "proof graph"),
}

MIN_CATEGORY_COUNTS = {
    "agent_evaluation": 30,
    "context_efficiency": 30,
    "public_trust": 20,
    "statistical_quality": 5,
}


def build_research_card(
    *,
    ledger_path: str | Path = DEFAULT_LEDGER,
    min_sources: int = DEFAULT_MIN_SOURCES,
    scope: str = "TRICE deterministic research basis",
) -> dict[str, Any]:
    """Build a machine-verifiable card from the research ledger."""

    ledger = Path(ledger_path)
    rows = parse_research_ledger(ledger)
    category_counts = _category_counts(rows)
    domain_counts = _domain_counts(rows)
    unique_urls = sorted({row["url"] for row in rows})
    checks = [
        _check("ledger_present", ledger.is_file(), _display_path(ledger), "ledger file exists"),
        _check("minimum_sources", len(rows) >= min_sources, len(rows), f">= {min_sources} ledger rows"),
        _check("unique_source_urls", len(unique_urls) >= int(min_sources * 0.85), len(unique_urls), f">= {int(min_sources * 0.85)} unique source URLs"),
        _check("category_coverage", _categories_pass(category_counts), category_counts, MIN_CATEGORY_COUNTS),
        _check("valid_source_links", _valid_link_fraction(rows) >= 0.95, round(_valid_link_fraction(rows), 4), ">= 0.95 valid http/local links"),
        _check("takeaways_present", all(row["takeaway"] for row in rows), _takeaway_summary(rows), "every row has a takeaway"),
        _check("index_sequence", _index_sequence_ok(rows), _index_summary(rows), "rows are uniquely indexed in ascending order"),
        _check("product_decision_section", _ledger_contains_product_decision(ledger), "Product Decision From The Ledger", "ledger includes product decision synthesis"),
    ]
    card = {
        "schema_version": RESEARCH_CARD_SCHEMA_VERSION,
        "scope": scope,
        "ledger": _input_row(ledger),
        "research_level": _research_level(checks),
        "research_score": _research_score(checks),
        "source_count": len(rows),
        "unique_source_count": len(unique_urls),
        "category_counts": category_counts,
        "domain_counts": domain_counts,
        "thresholds": {
            "min_sources": min_sources,
            "min_category_counts": MIN_CATEGORY_COUNTS,
            "valid_link_fraction_min": 0.95,
        },
        "checks": checks,
        "sources": rows,
        "research_basis": [
            "Agent evaluation research motivates fixed task environments, repeated trials, and cost-aware metrics.",
            "Prompt-compression and long-context research motivate decision-preserving context control rather than shortest-prompt contests.",
            "RAG and evidence-recall research motivate explicit recall floors for hidden or compressed evidence.",
            "Reproducible-build and supply-chain practice motivates hash-bound research, package, release, and provenance artifacts.",
            "TRICE treats its literature base as a versioned product input: if the ledger drifts, the paper, README, and integrity proof must be regenerated.",
        ],
        "non_claims": [
            "A green research card does not prove the 60% S-tier outcome gate.",
            "A green research card does not prove that every cited paper was replicated.",
            "A green research card proves that the current ledger is broad, hashed, categorized, and bound to generated artifacts.",
        ],
        "next_actions": _next_actions(checks, len(rows)),
    }
    card["research_card_sha256"] = hashlib.sha256(canonical_json(_without_research_hash(card)).encode("utf-8")).hexdigest()
    return card


def parse_research_ledger(path: str | Path) -> list[dict[str, Any]]:
    ledger = Path(path)
    if not ledger.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        match = ROW_RE.match(line)
        if not match:
            continue
        area = _clean_cell(match.group("area"))
        title = _clean_cell(match.group("title"))
        url = _clean_cell(match.group("url"))
        takeaway = _clean_cell(match.group("takeaway"))
        rows.append(
            {
                "index": int(match.group("index")),
                "area": area,
                "category": _classify(area, title, takeaway, url),
                "title": title,
                "url": url,
                "domain": _domain(url),
                "takeaway": takeaway,
                "row_sha256": hashlib.sha256(canonical_json([area, title, url, takeaway]).encode("utf-8")).hexdigest(),
            }
        )
    return rows


def verify_research_card_file(path: str | Path) -> dict[str, Any]:
    """Verify a research card self hash, bound ledger hash, and deterministic rebuild."""

    card_path = Path(path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if card.get("schema_version") != RESEARCH_CARD_SCHEMA_VERSION:
        errors.append(f"schema_version must be {RESEARCH_CARD_SCHEMA_VERSION}")
    expected_hash = str(card.get("research_card_sha256") or "")
    actual_hash = hashlib.sha256(canonical_json(_without_research_hash(card)).encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        errors.append("research_card_sha256 mismatch")

    ledger_row = card.get("ledger") if isinstance(card.get("ledger"), dict) else {}
    ledger_path = _resolve_path(str(ledger_row.get("path") or ""))
    checked_inputs: list[str] = []
    if not ledger_path.is_file():
        errors.append(f"ledger missing: {ledger_row.get('path')}")
    else:
        checked_inputs.append("ledger")
        if ledger_row.get("sha256") != sha256_file(ledger_path):
            errors.append("ledger sha256 mismatch")
        if ledger_row.get("bytes") != ledger_path.stat().st_size:
            errors.append("ledger byte count mismatch")

    thresholds = card.get("thresholds") if isinstance(card.get("thresholds"), dict) else {}
    try:
        rebuilt = build_research_card(
            ledger_path=ledger_path,
            min_sources=int(thresholds.get("min_sources", DEFAULT_MIN_SOURCES)),
            scope=str(card.get("scope") or "TRICE deterministic research basis"),
        )
        if canonical_json(_without_research_hash(rebuilt)) != canonical_json(_without_research_hash(card)):
            errors.append("research card does not match deterministic rebuild from bound ledger")
    except Exception as exc:
        errors.append(f"research card rebuild failed: {exc}")

    return {
        "ok": not errors,
        "schema_version": card.get("schema_version"),
        "research_level": card.get("research_level"),
        "research_score": card.get("research_score"),
        "source_count": card.get("source_count"),
        "unique_source_count": card.get("unique_source_count"),
        "research_card_sha256": expected_hash,
        "computed_research_card_sha256": actual_hash,
        "checked_inputs": checked_inputs,
        "errors": errors,
    }


def render_research_markdown(card: dict[str, Any]) -> str:
    lines = [
        "# TRICE Research Card",
        "",
        f"- Scope: `{card['scope']}`",
        f"- Research level: `{card['research_level']}`",
        f"- Research score: **{card['research_score']}/100**",
        f"- Sources: **{card['source_count']}** total, **{card['unique_source_count']}** unique URLs",
        f"- Ledger: `{card['ledger']['path']}`",
        "",
        "## Checks",
        "",
        "| Check | Passed | Observed | Required |",
        "|---|---:|---|---|",
    ]
    for row in card["checks"]:
        lines.append(f"| {row['name']} | {'yes' if row['passed'] else 'no'} | {_md(row['observed'])} | {_md(row['required'])} |")
    lines.extend(["", "## Category Coverage", "", "| Category | Count |", "|---|---:|"])
    for name, count in sorted(card["category_counts"].items()):
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "## Source Domains", "", "| Domain | Count |", "|---|---:|"])
    for name, count in sorted(card["domain_counts"].items(), key=lambda item: (-item[1], item[0]))[:12]:
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "## Research Basis", ""])
    for item in card["research_basis"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Non-Claims", ""])
    for item in card["non_claims"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Actions", ""])
    for item in card["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Hash", "", f"- research card: `{card['research_card_sha256']}`", ""])
    return "\n".join(lines)


def render_research_tex(card: dict[str, Any]) -> str:
    check_rows = "\n".join(
        f"{_tex(row['name'])} & {'yes' if row['passed'] else 'no'} & {_tex(str(row['required']))} \\\\"
        for row in card["checks"]
    )
    category_rows = "\n".join(
        f"{_tex(name)} & {count} \\\\"
        for name, count in sorted(card["category_counts"].items())
    )
    basis = "\n".join(f"\\item {_tex(item)}" for item in card["research_basis"])
    return (
        "\\section{Research Card}\n"
        f"The TRICE research card binds {card['source_count']} ledgered sources "
        f"({card['unique_source_count']} unique URLs) with research level "
        f"\\texttt{{{_tex(card['research_level'])}}} and score {card['research_score']}/100.\n\n"
        "\\begin{table}[h]\n\\centering\n"
        "\\begin{tabular}{lrl}\n\\toprule\nCheck & Passed & Required \\\\\n\\midrule\n"
        f"{check_rows}\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{Deterministic checks over the TRICE research ledger.}\n"
        "\\end{table}\n\n"
        "\\begin{table}[h]\n\\centering\n"
        "\\begin{tabular}{lr}\n\\toprule\nCategory & Sources \\\\\n\\midrule\n"
        f"{category_rows}\n"
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{Research coverage categories bound by the research card.}\n"
        "\\end{table}\n\n"
        "\\noindent Research basis:\n\\begin{itemize}\n"
        f"{basis}\n"
        "\\end{itemize}\n"
    )


def render_research_svg(card: dict[str, Any]) -> str:
    counts = dict(card.get("category_counts") or {})
    categories = sorted(counts.items(), key=lambda item: item[0])
    max_count = max([count for _, count in categories] or [1])
    width, height = 980, 340
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE research basis card</text>',
        f'<text x="28" y="58" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Score {card["research_score"]}/100 | level {html.escape(str(card["research_level"]))} | {card["source_count"]} sources</text>',
    ]
    y = 92
    for name, count in categories:
        bar = int(560 * count / max_count)
        color = "#2563eb" if count >= MIN_CATEGORY_COUNTS.get(name, 0) else "#f59e0b"
        parts.append(f'<text x="28" y="{y + 17}" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#111827">{html.escape(name)}</text>')
        parts.append(f'<rect x="190" y="{y}" width="580" height="22" rx="4" fill="#e5e7eb"/>')
        parts.append(f'<rect x="190" y="{y}" width="{bar}" height="22" rx="4" fill="{color}"/>')
        parts.append(f'<text x="786" y="{y + 16}" font-family="Inter,Segoe UI,Arial" font-size="13" fill="#111827">{count}</text>')
        y += 36
    parts.append(f'<text x="28" y="302" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">Research-card green means the literature basis is hashed and reviewable, not that the held-out S-tier outcome gate passed.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_research_outputs(card: dict[str, Any], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(out, json.dumps(card, indent=2, sort_keys=True) + "\n")
    md = out.with_suffix(".md")
    tex = out.with_suffix(".tex")
    svg = out.with_suffix(".svg")
    write_text_lf(md, render_research_markdown(card))
    write_text_lf(tex, render_research_tex(card))
    write_text_lf(svg, render_research_svg(card))
    return {"json": str(out), "markdown": str(md), "tex": str(tex), "svg": str(svg)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a deterministic TRICE research card.")
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-sources", type=int, default=DEFAULT_MIN_SOURCES)
    ap.add_argument("--format", choices=["json", "markdown", "tex"], default="json")
    args = ap.parse_args(argv)
    card = build_research_card(ledger_path=args.ledger, min_sources=args.min_sources)
    outputs = write_research_outputs(card, args.out)
    if args.format == "markdown":
        print(render_research_markdown(card))
    elif args.format == "tex":
        print(render_research_tex(card))
    else:
        print(json.dumps({"research_card": card, "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if all(row["passed"] for row in card["checks"]) else 1


def verify_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a deterministic TRICE research card.")
    ap.add_argument("research_card", type=Path)
    args = ap.parse_args(argv)
    verdict = verify_research_card_file(args.research_card)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def _clean_cell(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value).replace("`", "").strip()


def _classify(area: str, title: str, takeaway: str, url: str) -> str:
    text = f"{area} {title} {takeaway} {url}".lower()
    scores = {
        name: sum(1 for token in tokens if token in text)
        for name, tokens in CATEGORY_RULES.items()
    }
    best, score = max(scores.items(), key=lambda item: (item[1], item[0]))
    return best if score else "other"


def _domain(url: str) -> str:
    if url.startswith("../") or url.startswith("./") or not re.match(r"^https?://", url):
        return "local"
    return re.sub(r"^https?://", "", url).split("/", 1)[0].lower()


def _category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in sorted({*CATEGORY_RULES.keys(), "other"})}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return counts


def _domain_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["domain"]] = counts.get(row["domain"], 0) + 1
    return counts


def _categories_pass(counts: dict[str, int]) -> bool:
    return all(counts.get(name, 0) >= minimum for name, minimum in MIN_CATEGORY_COUNTS.items())


def _valid_link_fraction(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    valid = 0
    for row in rows:
        url = str(row["url"])
        if url.startswith(("http://", "https://", "../", "./")) or "/" in url:
            valid += 1
    return valid / len(rows)


def _takeaway_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    missing = sum(1 for row in rows if not row["takeaway"])
    return {"present": len(rows) - missing, "missing": missing}


def _index_sequence_ok(rows: list[dict[str, Any]]) -> bool:
    indexes = [row["index"] for row in rows]
    return len(indexes) == len(set(indexes)) and indexes == sorted(indexes)


def _index_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexes = [row["index"] for row in rows]
    return {
        "first": min(indexes) if indexes else None,
        "last": max(indexes) if indexes else None,
        "count": len(indexes),
        "unique": len(set(indexes)),
    }


def _ledger_contains_product_decision(path: Path) -> bool:
    return path.is_file() and "## Product Decision From The Ledger" in path.read_text(encoding="utf-8")


def _research_level(checks: list[dict[str, Any]]) -> str:
    passed = {row["name"] for row in checks if row["passed"]}
    required = {"ledger_present", "minimum_sources", "category_coverage", "valid_source_links", "takeaways_present", "index_sequence", "product_decision_section"}
    if required.issubset(passed):
        return "research_basis_locked"
    if {"ledger_present", "minimum_sources"}.issubset(passed):
        return "working_research_ledger"
    return "research_basis_incomplete"


def _research_score(checks: list[dict[str, Any]]) -> int:
    weights = {
        "ledger_present": 10,
        "minimum_sources": 20,
        "unique_source_urls": 10,
        "category_coverage": 20,
        "valid_source_links": 10,
        "takeaways_present": 10,
        "index_sequence": 10,
        "product_decision_section": 10,
    }
    return min(100, sum(weights.get(row["name"], 0) for row in checks if row["passed"]))


def _next_actions(checks: list[dict[str, Any]], source_count: int) -> list[str]:
    missing = [row["name"] for row in checks if not row["passed"]]
    if missing:
        actions = []
        if "minimum_sources" in missing:
            actions.append("Add enough primary paper/source rows to meet the configured minimum.")
        if "category_coverage" in missing:
            actions.append("Add sources in under-covered categories before relying on the ledger for product direction.")
        if "valid_source_links" in missing:
            actions.append("Replace stale or malformed source links with primary URLs or local artifact paths.")
        if "takeaways_present" in missing:
            actions.append("Fill every ledger row with a TRICE-specific takeaway.")
        if "index_sequence" in missing:
            actions.append("Fix duplicate or out-of-order ledger indexes.")
        return actions or ["Regenerate the research card after fixing ledger drift."]
    actions = [
        "Regenerate this card whenever the ledger, paper, README, or proof graph changes.",
        "Keep S-tier outcome claims gated by live held-out suite results, not by research-card status.",
    ]
    if source_count < 300:
        actions.append("Scale the ledger toward 300+ primary sources before submitting the paper outside the repo.")
    return actions


def _input_row(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "path": _display_path(path),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def _without_research_hash(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out.pop("research_card_sha256", None)
    return out


def _md(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, sort_keys=True).replace("|", "\\|") + "`"
    return "`" + str(value).replace("|", "\\|") + "`"


def _tex(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


if __name__ == "__main__":
    raise SystemExit(main())
