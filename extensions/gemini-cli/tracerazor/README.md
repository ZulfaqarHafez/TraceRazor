# TraceRazor for Gemini CLI

Install TraceRazor 1.1.0 or newer, then link this extension for development:

```sh
gemini extensions link ./extensions/gemini-cli/tracerazor
```

The extension bundles the TraceRazor Agent Skill, local stdio MCP server, and
JSON-only advisory hooks. Its awaited AfterAgent hook audits the current
session JSONL after each completed turn; SessionEnd is a best-effort fallback.
It does not enable enforcement, retain raw content by default, or make
automatic edits.
