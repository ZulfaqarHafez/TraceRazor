# Teacher / orchestrator (Labs prototype)

> **Labs, not product.** `teacher/` is an unshipped research prototype: it is not
> part of the `tracerazor` wheel, and CI does not run its tests. Nothing here is
> a product claim. Token savings from it are real only after a measured
> before/after `tracerazor bench` at constant task success.

The teacher observes a target agent's traces, diagnoses token waste with the
TraceRazor auditor, and proposes or applies agent changes behind a
quality-preservation gate, learning across runs via a shared Playbook.

## Try it

```bash
python examples/demo_teacher_offline.py     # closed-loop remediation + quality gate (mock agent)
python examples/demo_langgraph_coach.py     # LangGraph ingest -> COACH recommendations
python examples/demo_online_verification.py # online verification (HTTP agent + stat gate)
python -m pytest teacher/tests              # or: python teacher/tests/run.py (no pytest needed)
```

Everything runs offline with no API keys.

### Online verification (Layer 3)

`teacher.online` runs a tool-calling agent over HTTP against an
OpenAI-compatible endpoint, measures token usage (from the API `usage` block)
and task success, applies a candidate config, re-runs online, and gates on a
statistical non-inferiority test (`teacher.stats.StatGate`: paired bootstrap CI
on token savings plus a one-sided non-inferiority bound on the success
proportion). It promotes only changes that cut tokens without regressing
success on the evaluated tasks.

- **Runtime interventions:** `loop_breaker` suppresses a repeated identical tool
  call; `step_cap` hard-stops the loop. Prompt-section interventions ride in the
  system prompt.
- **Any OpenAI-compatible provider:** set `TRACERAZOR_LLM_BASE_URL` and
  `TRACERAZOR_LLM_API_KEY` (plus `TRACERAZOR_LLM_MODEL`) and run with `--live`.
- **Offline default:** without a key the demo runs against the bundled stdlib
  mock server (`teacher/_mockserver.py`), whose token output responds to the
  installed interventions. Mock-server results are not measurements of a real
  model.
- **Pluggable runner:** `Teacher.improve` takes a `runner`. Pass
  `teacher.online.OnlineRunner(agent, holdout, ...)` (with `gate=StatGate()`) to
  drive the loop live, or omit it to use the deterministic `OfflineRunner`.

```python
from teacher import Teacher, Mode, StatGate, AgentConfig
from teacher.online import OnlineRunner, OnlineAgent, LLMClient
agent  = OnlineAgent(LLMClient.from_env(), tools)
runner = OnlineRunner(agent, holdout, diagnoser, repeats=3)
result = Teacher(AgentConfig(), mode=Mode.CURRICULUM, gate=StatGate()).improve(runner=runner)
```

### Diagnostic backends (Layer 1)

`teacher.Diagnoser` selects a backend automatically and parses the auditor's
full JSON report (metrics, step-level `diff`, `savings`, and `fixes[]`) into
`WastePattern`s with severity, step-id and token attribution.

| Backend | When | How |
|---|---|---|
| **native** | `import tracerazor_native` succeeds | PyO3 binding (`crates/tracerazor-py`), in-process |
| **subprocess** | `tracerazor` binary present | shells `tracerazor audit --format json` |
| **builtin** | neither | transparent pure-Python heuristic (CI/offline) |

The PyO3 crate (`crates/tracerazor-py`) is excluded from the Cargo workspace and
is not built by CI. Build it with
`maturin develop -m crates/tracerazor-py/Cargo.toml`; the subprocess backend is
the default.

### Framework adapters (Layer 2)

`teacher.LangGraphAdapter` ingests LangGraph/LangChain runs into auditor-schema
traces, via a dependency-free `RunRecorder` or `from_tracerazor_callback(cb)`.
Because captured live traces can't be re-run offline, `Teacher.coach(traces)`
ranks interventions by the auditor's predicted savings and playbook priors and
emits a proposed config diff for human approval; it promotes nothing.
