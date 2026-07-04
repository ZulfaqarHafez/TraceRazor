# TraceRazor MCP server

`tracerazor-mcp` is a stdio [Model Context Protocol](https://modelcontextprotocol.io)
server that exposes TraceRazor's deterministic audit surface to MCP-capable
agent hosts. It is a thin wrapper over the same `tracerazor` CLI the
[Agent Skill](../skills/tracerazor/SKILL.md) drives — every tool shells the CLI
and returns its JSON.

> The **Agent Skill** (`skills/tracerazor`) is the recommended primary
> integration: it carries the honesty rules, the workflow, and the output
> contract inline. The MCP server is **complementary** — reach for it when a
> host speaks MCP but cannot load skills, or when you want the auditor available
> as callable tools alongside other MCP servers.

## Tools

All four tools return JSON-serializable data. Audits are **hermetic by default**,
so a score is a pure function of `(trace, config, version)`.

| Tool | Purpose |
| --- | --- |
| `audit_trace(path, hermetic=True, min_steps=None, threshold=None)` | Run `tracerazor audit --format json` and return the parsed report. Exit 0/1 are both success; exit 1 only means an explicit `threshold` gate failed and is surfaced as `passed: false`. Exit 2 returns a structured error. |
| `convert_transcript(path, format="auto")` | Normalize an external export (LangSmith, Langfuse, Arize Phoenix, OTel GenAI, raw, or a Claude Code `.jsonl`) into a TraceRazor trace JSON. |
| `list_claude_sessions(cwd=".")` | Return the parsed `.tracerazor/claude-code/index.json` written by the Claude Code SessionEnd hook, or an empty list. Needs no auditor binary. |
| `verify_report(report_path, trace_path=None)` | Re-verify a report against its trace, or an evidence bundle (`.zip`) on its own. Returns the verify verdict. |

Honesty still applies to whatever a host does with the output: TAS is an
**ordinal** heuristic, and every token/dollar figure is an **estimate** until
measured with `tracerazor bench`. See the skill for the full rules.

## Install

```bash
pip install "tracerazor[mcp]"
```

### Prerequisite: the auditor binary

`audit_trace`, `convert_transcript`, and `verify_report` shell the Rust
`tracerazor` binary. A platform wheel bundles it; a pure-Python (sdist) install
does not. If a tool raises a "no auditor binary found" error, follow the exact
recovery steps in the message — install a platform wheel, point `TRACERAZOR_BIN`
at a build, or `cargo build --release -p tracerazor`. `list_claude_sessions`
works without the binary.

Sanity-check the tool catalog without starting the server:

```bash
python -m tracerazor.mcp_server --selftest
```

## Registering the server

All hosts launch the same stdio command: `tracerazor-mcp`.

**Claude Code**

```bash
claude mcp add tracerazor -- tracerazor-mcp
```

**Cursor** — `.cursor/mcp.json`

```json
{
  "mcpServers": {
    "tracerazor": { "command": "tracerazor-mcp" }
  }
}
```

**VS Code** — `.vscode/mcp.json`

```json
{
  "servers": {
    "tracerazor": { "type": "stdio", "command": "tracerazor-mcp" }
  }
}
```

**Windsurf** — `~/.codeium/windsurf/mcp_config.json`

```json
{
  "mcpServers": {
    "tracerazor": { "command": "tracerazor-mcp" }
  }
}
```

If `tracerazor-mcp` is not on the host's PATH, use the interpreter form instead:
set `"command": "python"` with `"args": ["-m", "tracerazor.mcp_server"]`.
