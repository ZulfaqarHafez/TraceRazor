# TraceRazor MCP server

`tracerazor-mcp` is a local stdio Model Context Protocol server for
TraceRazor. It exposes auditing, run discovery, comparison, policy checks,
dry-run fix previews, validation receipts, and evidence verification without a
hosted TraceRazor service.

The Agent Skill remains the workflow and honesty contract. MCP is the callable
control surface used by Codex, Claude Code, Gemini CLI, and other MCP hosts.

## Install and inspect

```sh
pip install "tracerazor[mcp]>=1.1,<2"
python -m tracerazor.mcp_server --selftest
```

Platform wheels include the native auditor. The MCP module and catalog can be
inspected without the optional SDK, but starting the stdio server requires the
`mcp` extra.

## Result contract

New tools return a `tracerazor-mcp/v1` envelope:

```json
{
  "schema_version": "tracerazor-mcp/v1",
  "ok": true,
  "run_id": "run-id-or-null",
  "ingest_quality": null,
  "estimate_status": "provider_reported",
  "warnings": [],
  "data": {},
  "evidence_ref": ".tracerazor/runs/run-id/report.json",
  "error": null
}
```

The original four 1.x tools keep their legacy return shapes and receive
additive `_tracerazor` metadata. This avoids breaking existing MCP consumers.

TAS is ordinal. Savings fields are estimates until a before/after rerun is
measured with `tracerazor bench`. Estimated, missing, partial, or degraded
usage is never eligible for hard enforcement.

## Tools

| Tool | Behavior |
| --- | --- |
| `doctor(cwd=".")` | Report native binary, project policy, run store, and package readiness without requiring the MCP SDK. |
| `audit_trace(path, hermetic=True, min_steps=None, threshold=None, cwd=".")` | Audit a workspace-contained trace. Exit 1 is an explicit gate result, not a transport error. |
| `audit_current_run(run_id=None, cwd=".", ...)` | Audit a raw selected/newest trace. For default redacted persistence, return the report created from raw content in memory instead of re-scoring placeholders. |
| `convert_transcript(path, format="auto", cwd=".")` | Normalize a supported exporter or transcript into native trace JSON. |
| `latest_findings(run_id=None, cwd=".")` | Read findings from the selected/newest run, falling back to report fixes. |
| `compare_runs(baseline, target, cwd=".", regression_threshold=10.0)` | Compare run IDs or trace paths through the native JSON comparison command. |
| `explain_signal(signal, run_id=None, cwd=".")` | Explain a named metric and attach run-specific score details and matching fixes. |
| `preview_fix(run_id, target_path, cwd=".", include_needs_review=False)` | Invoke `apply --dry-run` only. It never writes the target and never includes dangerous fixes. |
| `record_validation(run_id, validation, cwd=".")` | Atomically write privacy-filtered advisory validation metadata. It is marked untrusted and never authorizes enforcement. |
| `check_policy(run_id=None, cwd=".")` | Parse `tracerazor.toml` and refuse enforcement without exact usage, complete ingest, and a trusted executed-verifier receipt. |
| `verify_evidence(report_path, trace_path=None, cwd=".")` | Return verified, tampered, or `non_replayable_redacted`; redaction is never mislabeled as tampering. |
| `verify_report(...)` | Legacy 1.x verification shape. |
| `list_claude_sessions(cwd=".")` | Legacy Claude-session index shape. |

All path-taking tools resolve paths inside the selected workspace, reject
traversal and symlink escapes, and detach native child processes from the MCP
stdio protocol pipe. At server startup the workspace boundary is fixed to
`TRACERAZOR_MCP_ROOT` or the host-selected process directory; model arguments
cannot retarget the server at another checkout.

## Host registration

Every host starts the same local command:

```text
tracerazor-mcp
```

The repository provides ready-to-package configurations:

- Codex: `plugins/tracerazor/.mcp.json`
- Claude Code: `extensions/claude-code/tracerazor/.mcp.json`
- Gemini CLI: `extensions/gemini-cli/tracerazor/gemini-extension.json`

A generic host can use:

```json
{
  "mcpServers": {
    "tracerazor": {
      "command": "tracerazor-mcp",
      "args": []
    }
  }
}
```

If console scripts are unavailable to the host, use `python -m
tracerazor.mcp_server`. TraceRazor makes no network connection unless the
embedding-enhanced auditor or another explicitly configured exporter is used.
