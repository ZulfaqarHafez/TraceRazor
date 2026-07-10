# TraceRazor for Claude Code

This plugin bundles the TraceRazor Agent Skill, local stdio MCP server, and
advisory lifecycle hooks. Install TraceRazor 1.1.0 or newer first, then test
the plugin without installing it:

```sh
claude --plugin-dir ./extensions/claude-code/tracerazor
```

The plugin never applies fixes or enables enforcement. Hooks capture lifecycle
events and expose prior-session coaching only after the user has installed and
trusted the plugin. SessionEnd audits the main `transcript_path`; SubagentStop
audits the host-provided `agent_transcript_path`. Raw transcript content is not
persisted unless project policy explicitly opts into raw persistence.
