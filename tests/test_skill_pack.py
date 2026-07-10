"""Guard rails for the canonical cross-agent TraceRazor Agent Skill.

Verifies the two SKILL.md copies stay byte-identical, that the frontmatter
parses and stays within the host trigger-surface budget, that the body stays
small enough to remain self-contained, and that every `tracerazor <subcommand>`
the skill instructs an agent to run actually exists in the CLI's clap
definitions. Parsing is stdlib-only (no PyYAML dependency).
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL_A = os.path.join(ROOT, "skills", "tracerazor", "SKILL.md")
SKILL_B = os.path.join(ROOT, ".claude", "skills", "tracerazor", "SKILL.md")
SKILL_C = os.path.join(ROOT, ".agents", "skills", "tracerazor", "SKILL.md")
SKILL_D = os.path.join(ROOT, "plugins", "tracerazor", "skills", "tracerazor", "SKILL.md")
SKILL_E = os.path.join(
    ROOT,
    "crates",
    "tracerazor-cli",
    "assets",
    "tracerazor-skill",
    "SKILL.md",
)
SKILL_F = os.path.join(
    ROOT, "extensions", "claude-code", "tracerazor", "skills", "tracerazor", "SKILL.md"
)
SKILL_G = os.path.join(
    ROOT, "extensions", "gemini-cli", "tracerazor", "skills", "tracerazor", "SKILL.md"
)
CLI_MAIN = os.path.join(ROOT, "crates", "tracerazor-cli", "src", "main.rs")

# The Agent Skills specification exposes only ``description`` as trigger
# metadata and limits it to 1024 characters. Extra top-level fields are not
# portable; vendor-specific values belong below ``metadata``.
DESCRIPTION_BUDGET = 1024
ALLOWED_FRONTMATTER = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}
# Hosts/models are lazy about following reference links, so the body must stay
# self-contained and short.
MAX_BODY_LINES = 160


def _read_text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read().replace("\r\n", "\n").replace("\r", "\n")


def _split_skill(text):
    """Return (frontmatter_dict, body_str) from a SKILL.md string.

    Minimal top-level `key: value` YAML parsing — every value in this skill is a
    single-line scalar, so a full YAML engine is unnecessary.
    """
    lines = text.split("\n")
    assert lines[0].strip() == "---", "SKILL.md must open with a --- frontmatter fence"
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    assert end is not None, "SKILL.md frontmatter is not closed with ---"

    fm = {}
    for ln in lines[1:end]:
        if not ln.strip() or ln[0] in " \t" or ":" not in ln:
            continue
        key, _, val = ln.partition(":")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        fm[key.strip()] = val

    body = "\n".join(lines[end + 1:]).strip("\n")
    return fm, body


def test_all_discovery_and_distribution_copies_are_byte_identical():
    paths = [SKILL_A, SKILL_B, SKILL_C, SKILL_D, SKILL_E, SKILL_F, SKILL_G]
    payloads = []
    for path in paths:
        with open(path, "rb") as handle:
            payloads.append(handle.read())
    assert all(payload == payloads[0] for payload in payloads[1:]), (
        "canonical, Claude, Agent Skills, Codex plugin, and embedded skill "
        "copies must be byte-identical"
    )


def test_codex_skill_declares_openai_metadata():
    metadata = os.path.join(
        ROOT,
        "plugins",
        "tracerazor",
        "skills",
        "tracerazor",
        "agents",
        "openai.yaml",
    )
    text = _read_text(metadata)
    assert 'display_name: "TraceRazor"' in text
    assert "allow_implicit_invocation: true" in text


def test_frontmatter_parses_and_names_the_skill():
    fm, _ = _split_skill(_read_text(SKILL_A))
    assert fm.get("name") == "tracerazor"
    assert fm.get("license") == "MIT"
    assert fm.get("description"), "description (the trigger surface) is required"
    assert set(fm) <= ALLOWED_FRONTMATTER


def test_trigger_surface_within_host_budget():
    fm, _ = _split_skill(_read_text(SKILL_A))
    length = len(fm["description"])
    assert length <= DESCRIPTION_BUDGET, (
        f"description is {length} chars; must be <= {DESCRIPTION_BUDGET}"
    )


def test_body_stays_self_contained():
    _, body = _split_skill(_read_text(SKILL_A))
    line_count = body.count("\n") + 1
    assert line_count <= MAX_BODY_LINES, (
        f"body is {line_count} lines; must be <= {MAX_BODY_LINES}"
    )


def test_referenced_subcommands_exist_in_cli():
    _, body = _split_skill(_read_text(SKILL_A))
    with open(CLI_MAIN, encoding="utf-8") as fh:
        cli_src = fh.read().lower()

    # First token after a same-line `tracerazor ` invocation (skip flags like
    # `--version`, and hyphenated siblings such as `tracerazor-trice`).
    tokens = sorted(set(re.findall(r"tracerazor[ \t]+([a-z][a-z-]*)", body)))
    assert tokens, "expected at least one `tracerazor <subcommand>` in the body"
    for tok in tokens:
        assert re.search(r"\b" + re.escape(tok) + r"\b", cli_src), (
            f"skill references `tracerazor {tok}` but the CLI defines no such subcommand"
        )
