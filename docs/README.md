# TraceRazor docs map

Use this index to distinguish stable reference material from generated
point-in-time evidence.

## Reference

- [Agent guide](AGENT_GUIDE.md) - end-to-end machine recipe for agent runs.
- [Trace format](trace-format.md) - native trace fields, validation rules, and
  supported import formats.
- [MCP server](MCP.md) - callable MCP tools and their JSON contracts.
- [Python API](python_api.md) - Python tracer, client, and package usage.
- [Metric effectiveness](metric_effectiveness.md) - current metric semantics.

## Generated Snapshots

Files named `trice_*` are generated proof-card artifacts. Regenerate them with
`tracerazor-trice`; do not hand-edit their contents.

The external audit reports below are historical snapshots generated under older
scoring models and should not be quoted as current product measurements without
rerunning the commands they document:

- [External agent audits](external_agent_audits.md)
- [Hugging Face AgentInstruct audit](huggingface_agentinstruct_audit.md)

## Research And Planning

- [TRICE paper outline and proof-card material](trice_reproduction_card.md)
- [Ship plan](ship_plan.md)
- [Refactor backlog](REFACTOR_BACKLOG.md)
