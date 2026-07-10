# TRICE Installability Card

- Scope: `TraceRazor wheel installability`
- Install level: `full_cli_install_ready`
- Install score: **100/100**
- Expected version: `1.1.0`

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| wheel_present | yes | dist/tracerazor-1.1.0-py3-none-win_amd64.whl | built wheel exists |
| venv_created | yes | 0 | clean virtual environment can be created |
| wheel_installs | yes | 0 | wheel and MCP runtime dependencies install cleanly |
| version_matches | yes | 1.1.0 | 1.1.0 |
| schemas_importable | yes | {"crates_schema_title": "TRICE crates publish card", "install_schema_title": "TRICE installability card", "research_schema_title": "TRICE research card"} | install, crates, and research schemas import from wheel |
| trice_api_importable | yes | {"build_crates_card": true, "build_install_card": true, "build_research_card": true, "verify_install_card_file": true, "verify_research_card_file": true} | public tracerazor.trice install/crates/research APIs import |
| runtime_api_importable | yes | {"event_schema": "TraceRazor runtime event v1", "runtime_api": true} | runtime API and tracerazor-event/v1 schema import from wheel |
| agent_assets_shipped | yes | {"agent_policy": true, "claude_plugin": true, "codex_plugin": true, "gemini_extension": true, "sample_trace": true} | policy, sample trace, and Codex/Claude/Gemini assets ship in wheel |
| mcp_catalog_importable | yes | ["audit_current_run", "audit_trace", "check_policy", "compare_runs", "convert_transcript", "doctor", "explain_signal", "latest_findings", "list_claude_sessions", "preview_fix", "record_validation", "verify_evidence", "verify_report"] | versioned MCP tool catalog imports without the optional SDK |
| trice_console_works | yes | 0 | tracerazor-trice console script works after wheel install |
| rust_cli_bundled | yes | exit=0; tracerazor 1.1.0
 | tracerazor console script can find a bundled Rust auditor binary |
| rust_cli_from_distribution | yes | True | resolved auditor binary lives inside the installed distribution |
| agent_console_works | yes | exit=0; {
  "auto_host": "generic",
  "command": "doctor",
  "executable": "C:\\Users\\zulfa\\AppData\\Local\\Temp\\trice-install-card-nfxt6txp\\venv\\Lib\\site-packages\\tracerazor\\bin\\ | tracerazor agent doctor works after wheel install |
| mcp_selftest_works | yes | exit=0; [
  {
    "name": "audit_trace",
    "description": "Audit a trace hermetically and return the 1.x report shape with additive TraceRazor metadata."
  },
  {
    "name": "convert_tr | installed MCP server constructs and exposes its tool catalog |
| sample_audit_works | yes | exit=0; {
  "schema_version": "tracerazor-report/v1",
  "trace_id": "support-agent-run-2847",
  "agent_name": "customer-support-v3",
  "framework": "langgraph",
  "total_steps": 11,
  "tot | installed bundled sample audits outside the source checkout |

## Commands

| Command | Exit | Status |
|---|---:|---|
| create_venv | 0 | ok |
| install_wheel | 0 | ok |
| import_probe | 0 | ok |
| trice_console | 0 | ok |
| rust_console | 0 | ok |
| agent_console | 0 | ok |
| mcp_selftest | 0 | ok |
| sample_audit | 0 | ok |

## Next Actions

- Publish the install card with the release evidence bundle.

## Hash

- install card: `912d52ec9613334504acbf420e8876e956bc5eba691e19eaa935bb4ff50632a9`
