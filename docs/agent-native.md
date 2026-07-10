# Agent-native TraceRazor

TraceRazor becomes automatic only after an explicit install or image
provisioning step. Installing the Python package never edits an agent host's
settings.

## Trusted bootstrap

Preview the exact paths first:

```sh
tracerazor agent doctor --format json
tracerazor agent install --host auto --scope project --mode coach --dry-run
```

Then install after review:

```sh
tracerazor agent install --host auto --scope project --mode coach
tracerazor agent status --format json
```

Bootstrap adapters exist for Codex, Claude Code, Gemini CLI, and a generic wrapper.
`project` writes only inside the current repository. `user` uses the host's
user-level skill/config location. `image` requires
`TRACERAZOR_IMAGE_ROOT`, making container provisioning the trust event.

Install is idempotent and ownership-recorded. Existing differing files are
backed up rather than silently replaced. Uninstall removes only content owned
by TraceRazor:

```sh
tracerazor agent uninstall --host auto --scope project
```

The legacy Claude command remains supported during 1.x.

## Host bundles

- Codex plugin: `plugins/tracerazor`
- Repo-discoverable Agent Skill: `.agents/skills/tracerazor`
- Claude Code plugin: `extensions/claude-code/tracerazor`
- Gemini CLI extension: `extensions/gemini-cli/tracerazor`

Plugin and extension hooks are advisory. Codex and Gemini require users to
review changed hooks before they execute. None of the bundles installs itself,
applies fixes, or enables enforcement.

Capture capability is reported, not inferred. Command hooks consume the JSON
object supplied on stdin and treat its transcript path as trusted host input:

- Claude Code audits `transcript_path` at SessionEnd and
  `agent_transcript_path` at SubagentStop.
- Codex audits `transcript_path` at Stop and `agent_transcript_path` at
  SubagentStop. Stop hooks always return valid JSON (`{}`) on success, as the
  Codex hook contract requires. The normalizer supports the current rollout
  JSONL records (`session_meta`, `turn_context`, `event_msg`, and
  `response_item`); Codex documents the transcript representation as unstable,
  so an unknown future shape remains explicitly partial.
- Gemini CLI audits its session JSONL after each awaited AfterAgent event.
  SessionEnd is retained as a best-effort final fallback because Gemini does
  not wait for that hook during shutdown.

Transcript reads reject relative paths, non-files, symlink leaves, invalid
UTF-8, and files over 64 MiB. Missing paths, hook-event mismatches, unreadable
files, unrecognized formats, and below-minimum-step traces write a degraded or
partial manifest instead of a false pass. Raw transcript bytes stay in memory
under `local-redacted`; persistence still requires the explicit `privacy =
"raw"` plus `persist_raw_content = true` policy opt-in.

Provider-reported usage is preserved when available. When a host supplies only
a session or message total, per-step allocation is marked
`token_distribution_estimated` and the audit is degraded; missing usage is
marked `missing_provider_usage` and is never replaced with a character-count
estimate. The generic `agent run -- <command>` and Python runtime paths
propagate context for hosts without lifecycle transcript support.

## Project policy

Commit `tracerazor.toml`; keep `.tracerazor/` disposable:

```toml
schema_version = 1
mode = "coach"
capture = "auto"
hermetic = true
privacy = "local-redacted"
persist_raw_content = false
artifact_dir = ".tracerazor/runs"
min_steps = 5

[quality]
verifier = ""

[enforcement]
enabled = false
```

Modes:

- `off`: no capture.
- `passive`: capture and report only.
- `coach`: capture, report, and next-session advisory; default.
- `enforce`: may make a hard decision only when enforcement is explicitly
  enabled, token provenance is acceptable, ingest is complete, and the task
  verifier is present.

## Runtime API

```python
from tracerazor.runtime import (
    TaskOutcome,
    TaskResult,
    TokenUsage,
    ToolCall,
    ToolStatus,
    configure,
)

runtime = configure(
    policy_path="tracerazor.toml",
    host="openai-agents",
    framework="openai-agents",
    agent_id="planner",
)

runtime.record(
    "reasoning",
    content="Select the smallest applicable tool.",
    tokens=TokenUsage.reported(input_tokens=120, output_tokens=32),
)

runtime.record(
    "tool_call",
    tool=ToolCall.from_arguments(
        "search",
        {"query": "example"},
        status=ToolStatus.SUCCESS,
        observation_size=420,
    ),
    output="result omitted from persisted local-redacted artifacts",
    tokens=TokenUsage.reported(input_tokens=80, output_tokens=10),
)

child_environment = runtime.spawn_env(child_agent_id="researcher")

runtime.finalize(
    task=TaskResult(
        outcome=TaskOutcome.PASSED,
        verifier="pytest -q",
        evidence={"exit_code": 0},
    )
)
```

`tracerazor.runtime.auto_instrument("openai_agents")` registers the native
OpenAI Agents tracing processor when that optional SDK is available.

LangGraph uses its supported per-invocation LangChain callback configuration;
TraceRazor does not patch a graph or a process-global callback manager:

```python
from tracerazor.runtime import auto_instrument, configure

runtime = configure(policy_path="tracerazor.toml", framework="langgraph")
instrumentation = auto_instrument("langgraph", processor=runtime)
langgraph = instrumentation.handles["langgraph"]

# Recommended: adds the callback and guarantees finalization on return/error.
result = langgraph.invoke(graph, {"messages": messages})

# Equivalent manual attachment for invoke/ainvoke integrations:
config = langgraph.attach({"metadata": {"session": "support"}})
result = graph.invoke({"messages": messages}, config=config)
langgraph.finish(output=result)  # idempotent if the root callback already ended
```

`langgraph.ainvoke(...)`, `langgraph.stream(...)`, and `langgraph.astream(...)`
follow the same contract. Streams finalize on exhaustion, record an error on
failure, and mark the run partial if the consumer closes the stream early. Create one
runtime/handle per concurrent root invocation. Provider usage is captured from
`LLMResult.llm_output`, `usage_metadata`, or generation-message metadata when a
provider supplies it; TraceRazor does not infer missing counts from text.

CrewAI exposes an official process-global event bus. Discovery remains
side-effect free: `auto_instrument("crewai")` returns an enabled but detached
handle. Registration happens only after an explicit `attach()` call and is
reversible:

```python
runtime = configure(policy_path="tracerazor.toml", framework="crewai")
instrumentation = auto_instrument("crewai", processor=runtime)
crewai = instrumentation.handles["crewai"]

crewai.attach(crew)  # scope to this crew's known crew/agent/task identifiers
try:
    output = crew.kickoff(inputs={"topic": "agent efficiency"})
    crewai.finish(output=output)  # safe if CrewKickoffCompleted already finalized
except Exception as exc:
    crewai.fail(exc)
    raise
finally:
    crewai.detach()
```

The CrewAI listener maps kickoff completion/failure, LLM completion/failure,
and supported tool completion/error events. It uses provider usage only when the
event includes it. Tool events generally do not carry model-token usage and are
therefore marked `missing`, which degrades ingest rather than fabricating a
count. If a provider reports only a total, that total is preserved but marked
`estimated`/degraded because the input/output split is unavailable. Crew scoping
is best-effort because some provider events omit
crew, agent, or task identifiers; avoid overlapping unscoped handles in one
process.

These runtime handles are the supported provenance-aware path. The older
`tracerazor.integrations.langgraph` and `tracerazor.integrations.crewai`
callbacks remain compatibility trace builders: they may estimate counts and use
their historical absolute-threshold workflow, so their output is not eligible
for agent-native enforcement.

## Local OTLP receiver

`tracerazor serve` accepts both standard OTLP/HTTP JSON and OTLP/HTTP binary
Protobuf at `POST /v1/traces`. The server binds to loopback by default. Bearer
tokens do not encrypt HTTP: a non-loopback bind is refused unless a trusted
reverse proxy terminates TLS and the operator explicitly sets both
`TRACERAZOR_API_TOKEN` and `TRACERAZOR_TLS_TERMINATED=true`. The latter is an
operator assertion, not native TLS support; never expose the plaintext backend
port outside that protected network boundary.

Local loopback example:

```sh
export TRACERAZOR_API_TOKEN="replace-with-a-secret"
export TRACERAZOR_OTLP_SPOOL_DIR=".tracerazor/otlp-spool"
tracerazor serve --bind 127.0.0.1 --port 8080

curl -H "Authorization: Bearer $TRACERAZOR_API_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @otlp-export.json \
  http://127.0.0.1:8080/v1/traces

# Binary ExportTraceServiceRequest produced by an OTLP http/protobuf exporter:
curl -H "Authorization: Bearer $TRACERAZOR_API_TOKEN" \
  -H "Content-Type: application/x-protobuf" \
  --data-binary @otlp-export.pb \
  http://127.0.0.1:8080/v1/traces
```

For a proxy-to-container/private-interface deployment, provision the bearer
secret, terminate HTTPS at the proxy, block direct access to the backend port,
then set `TRACERAZOR_TLS_TERMINATED=true` before using a non-loopback bind.
TraceRazor fails closed if either the token or this assertion is absent.
The repository's default [Compose topology](container.md) implements this with
a Caddy `tls internal` gateway published only on host `127.0.0.1`; the backend
is merely exposed on the private Compose network. The standalone dashboard
image keeps its loopback default, so `docker run -p` is intentionally
unreachable rather than silently insecure.

Both encodings support identity or gzip content encoding, and responses use the
same content type as valid requests. Compressed and decompressed bodies are
limited to 16 MiB. A complete multi-trace batch is normalized, local-redacted,
fsynced, and atomically renamed in the spool before SQLite ingestion. Prompt,
output, error, arbitrary metadata, and agent/service identifiers persist only
as digests and sizes. Invalid auth, malformed bodies, unsafe spool paths, and
storage failures return explicit `google.rpc.Status`-shaped errors instead of a
false success. The receiver does not provide OTLP/gRPC; configure exporters with
`http/protobuf` (the usual OTLP HTTP default) or `http/json`.

Normalized traces retain an additive `metadata.otlp` ledger with every span ID,
parent span ID, per-category token counts (input, output, cache read, cache
creation/write, reasoning, and total), and per-field provenance. Missing usage
is stored as `null` with `estimate_status: "missing"`; the compatibility
`TraceStep.tokens` value may remain zero, but the ledger marks the trace
`degraded_ingest: true` and `enforcement_eligible: false`, so it cannot look like
an exact provider-reported zero. Structured GenAI messages, tool arguments, and
tool results are normalized in memory and redacted before persistence. If
content or usage remains unusable, the receiver returns an OTLP partial-success
warning with zero rejected spans rather than a false full-success response.

## Event and run contracts

`tracerazor-event/v1` records:

- run, trace, span, session, agent, and parent identifiers;
- host/framework versions;
- provider-reported, estimated, or missing token provenance;
- input/output/cache/reasoning token splits;
- privacy-preserving tool metadata;
- task outcome and verifier evidence;
- capture quality and redaction state.

W3C `TRACEPARENT` (with lowercase compatibility), `TRACERAZOR_RUN_ID`,
`TRACERAZOR_PARENT_AGENT_ID`, and `TRACERAZOR_POLICY` propagate through
`agent run` and `spawn_env()`.

Each run uses:

```text
.tracerazor/runs/<run-id>/
  manifest.json
  events.jsonl
  trace.json
  findings.json
  validation.json
  report.json
```

Files are replaced atomically. An interrupted process leaves a recoverable
partial run instead of a false successful report.

## Offline run receipts

Completed lifecycle audits write `run-receipt.json` using the public
`tracerazor-run-receipt/v1` contract. The receipt binds the run identity,
trace/session/agent/parent-agent linkage, privacy and replayability modes,
normalized audit-trace hash, persisted-trace hash, and report hash. Its
canonical JSON field order is defined by the Rust
`RunReceiptV1` type; the optional `signature` envelope is excluded from those
canonical bytes, while `signed` remains covered.

Set the same 32-byte Ed25519 seed used by report signing before the host starts:

```sh
export TRACERAZOR_SIGNING_KEY="<64 lowercase hex characters>"
```

The lifecycle hook then emits `signed: true` and an envelope containing
`algorithm: "Ed25519"`, the public key, and signature. An absent or invalid
environment key never becomes a silent authenticity claim: the receipt remains
explicitly unsigned and the hook writes a warning to stderr.

Verify a returned receipt on the parent machine with:

```sh
export TRACERAZOR_VERIFY_KEY="<public key provisioned by the parent>"
tracerazor agent verify-receipt child-run/run-receipt.json --format json
# Equivalent explicit pin: --verify-key "$TRACERAZOR_VERIFY_KEY"
```

Verification validates the receipt schema, run identity, RFC 3339 timestamp,
privacy/replayability invariants, and lowercase SHA-256 fields before checking
the signature. If sibling `trace.json`, `report.json`, or `manifest.json` files
are present, it also verifies their hashes and manifest identity bindings.
Exit codes follow the machine contract: `0` means a valid signed receipt or an
explicitly unsigned well-formed receipt, `1` means a signature or available
artifact mismatch, and `2` means malformed/unreadable input. Read the JSON
`status` and `authenticated` fields: a cross-machine offline receipt is trusted
only when `status` is `valid`, `authenticated` is `true`, and `signer_pinned`
is `true`. A signature that verifies only against its self-declared embedded
key remains valid but untrusted until `--verify-key` or
`TRACERAZOR_VERIFY_KEY` pins the parent-provisioned public key. `status:
"unsigned"` is intentionally accepted for local compatibility but is never a
trusted remote receipt.

## Evaluation status

`benchmark/agent_native/` contains the locked 50-task, three-condition,
three-repetition evaluation protocol and its fail-closed evaluator. It is
machinery, not a positive efficacy result. Estimated or missing token counts
are excluded, task success is primary, and incomplete evidence exits 2.

TRICE and verified sandbox optimization remain Labs surfaces until real
held-out results satisfy the preregistered efficacy and non-inferiority gates.

## Privacy and enforcement

The default `local-redacted` mode processes content in memory and persists
digests, usage, tool metadata, similarity-safe data, and redacted placeholders.
Set both `privacy = "raw"` and `persist_raw_content = true` to retain raw
content.

Estimated or missing token usage sets degraded ingest and is never
enforcement-eligible. A verified task outcome remains primary; lower token use
alone is not an accepted improvement.
