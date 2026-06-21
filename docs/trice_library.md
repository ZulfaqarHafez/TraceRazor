# TRICE Library Contract

TRICE is a deterministic context-control library for live software-agent
evidence. The public import path is:

```python
import tracerazor.trice as trice
```

The product contract is deliberately narrow:

1. Build context from a real workspace/task.
2. Apply a deterministic adapter.
3. Run an objective verifier command.
4. Measure input-token savings against full context.
5. Emit a hash-verifiable evidence manifest.
6. Accept only if the claim gate passes.

Replay is useful as preflight evidence; it is not final proof.

## Determinism Rules

TRICE records objective verifier output, but strips clock noise such as
`passed in 0.23s` before writing traces or manifests. Wall-clock duration is
not evidence for a context-control claim, so it is excluded from live traces.
For the same task, adapter, profile, verifier command, and output path, reruns
must produce the same result hash and artifact hashes.

## Generic Repo Run

Create a deterministic patch spec. A runnable example lives at
[`examples/trice_patch_fix_offbyone.json`](../examples/trice_patch_fix_offbyone.json):

```json
{
  "name": "fix-offby-one-demo",
  "edits": [
    {
      "op": "replace",
      "path": "chunker.py",
      "old": "size - 1",
      "new": "size"
    }
  ]
}
```

Run TRICE on any local repo/seed directory:

```powershell
tracerazor-trice run -- `
  --repo benchmark\live\tasks\fix-offby-one\seed `
  --task-id fix-offby-one-generic `
  --prompt "Fix chunker.py without editing tests." `
  --verify-cmd "python -m pytest -q --tb=short" `
  --patch-spec examples\trice_patch_fix_offbyone.json `
  --out-dir benchmark\trice\results\generic-example `
  --rounds 1 `
  --user-feedback "real runs, not replay; target 60% savings"
```

Verify the evidence:

```powershell
tracerazor-trice verify benchmark\trice\results\generic-example\trice_v2_evidence_manifest.json
```

The module form is equivalent:

```powershell
python -m tracerazor.trice verify benchmark\trice\results\generic-example\trice_v2_evidence_manifest.json
```

## Command Repair Adapter

For v2 real-run evaluation, TRICE can also run a deterministic repair command
inside the fresh workspace. This is the bridge for a user CLI, scripted repair,
or wrapped coding agent:

```powershell
tracerazor-trice run -- `
  --repo benchmark\live\tasks\fix-offby-one\seed `
  --task-id fix-offby-one-command `
  --prompt "Fix chunker.py without editing tests." `
  --verify-cmd "python -m pytest -q --tb=short" `
  --repair-cmd "python path\to\repair_agent.py" `
  --repair-timeout-s 600 `
  --out-dir benchmark\trice\results\command-example `
  --rounds 1 `
  --user-feedback "real runs, not replay; target 60% savings"
```

The command runs with `cwd` set to the copied workspace. TRICE passes
`TRICE_TASK_ID`, `TRICE_PROMPT`, `TRICE_WORKSPACE`, `TRICE_INPUT_TOKENS`,
`TRICE_BASELINE_INPUT_TOKENS`, `TRICE_CONTEXT_MODE`, policy hashes, and budget
fields in the environment, fingerprints files before and after the command,
and records the changed file list. By default it refuses edits under `tests/`
and `test/`; use
`--allow-test-edits` only for suites where benchmark mutation is intentional.
TRICE also passes `TRICE_AGENT_RECEIPT`; a wrapped agent can write JSON there
to report model/token metadata:

```json
{
  "schema_version": "trice-agent-receipt/v1",
  "model": "my-agent-or-model",
  "token_accounting": {
    "input_tokens": 577,
    "baseline_input_tokens": 2470,
    "output_tokens": 1208
  },
  "tool_envelope": {
    "tools": ["read_file", "edit_file", "pytest"]
  }
}
```

Every live condition writes `run_receipt.json`, hashes it into the evidence
manifest, and records its SHA-256 in the trace metadata. The receipt includes
the adapter type, command argv hash, workspace fingerprint before and after the
intervention, changed files, command exit code, output hashes, and any
agent-reported token accounting.
The receipt also stores a `trice_context` envelope with condition, context
mode, measured input tokens, baseline tokens, policy hashes, budget ratio, and
policy action counts. Manifest and bundle verification validate the envelope,
so malformed measurement metadata fails the same evidence path as bad hashes.

## Adapter Profiles

For reusable command adapters, keep the command envelope in a profile file:

```json
{
  "schema_version": "trice-adapter-profile/v1",
  "name": "my-agent",
  "type": "command",
  "command": ["python", "scripts/repair_agent.py"],
  "timeout_s": 600,
  "allow_test_edits": false,
  "agent_receipt_path": ".trice/agent_receipt.json"
}
```

Use it from the CLI:

```powershell
tracerazor-trice run -- `
  --repo path\to\repo `
  --task-id my-task `
  --prompt "Fix the issue without editing tests." `
  --verify-cmd "python -m pytest -q" `
  --adapter-profile path\to\adapter-profile.json `
  --out-dir benchmark\trice\results\profile-example
```

## Suite Runs

For repeatable real-repo evaluation across users, define a suite manifest. A
runnable example lives at
[`examples/trice_suite_fix_offbyone.json`](../examples/trice_suite_fix_offbyone.json).

```powershell
tracerazor-trice suite examples\trice_suite_fix_offbyone.json `
  --out-dir benchmark\trice\results\v2-suite `
  --rounds 1 `
  --replicates 3
```

Deep-verify the aggregate manifest and every child live task manifest:

```powershell
tracerazor-trice verify-suite benchmark\trice\results\v2-suite\trice_suite_evidence_manifest.json
```

Export the complete suite evidence as a portable deterministic bundle:

```powershell
tracerazor-trice bundle benchmark\trice\results\v2-suite\trice_suite_evidence_manifest.json `
  --out benchmark\trice\results\v2-suite\trice_suite_evidence.trice.zip
tracerazor-trice verify-bundle benchmark\trice\results\v2-suite\trice_suite_evidence.trice.zip
```

The bundled broad-smoke suite exercises all six local live tasks through a
reusable command adapter profile:

```powershell
tracerazor-trice suite examples\trice_suite_bundled_live.json `
  --out-dir benchmark\trice\results\v2-broad-smoke `
  --rounds 1 `
  --replicates 1
tracerazor-trice verify-suite benchmark\trice\results\v2-broad-smoke\trice_suite_evidence_manifest.json
tracerazor-trice bundle benchmark\trice\results\v2-broad-smoke\trice_suite_evidence_manifest.json `
  --out benchmark\trice\results\v2-broad-smoke\trice_broad_smoke_evidence.trice.zip
tracerazor-trice verify-bundle benchmark\trice\results\v2-broad-smoke\trice_broad_smoke_evidence.trice.zip
```

This broad smoke is stronger than the one-task example because it spans six
task clusters and uses `adapter_profile`, but it still fails the default
S-tier gate because it is local, single-replicate, and not 50 held-out locked
Git task clusters.

The suite manifest is hashed into `trice_suite_manifest.snapshot.json`. Each
child task keeps its own live report, result JSON, trace artifacts, and evidence
manifest under `tasks/<task_id>/`.

Replicates are independent fresh child live runs. Suite reports include the
ordinary savings CI plus a clustered-by-task savings CI, so repeated runs of the
same repo do not masquerade as independent held-out repositories.

## S-Tier Claim Gate

Every suite result includes an `s_tier_gate` verdict under `claim_gate`. The
default gate is intentionally strict and will reject local smoke evidence:

- mean savings must meet the target;
- clustered-by-task CI lower bound must meet the target;
- pass regressions must be zero;
- every run must be accepted;
- at least 50 task clusters must be present;
- each task cluster must have at least 3 replicates;
- all tasks must use locked Git sources;
- all locked Git sources must be remote URLs by default;
- all tasks must use adapter profiles;
- run receipts must validate.

Suites can lower thresholds for local testing, including
`require_remote_git_sources`, but that does not make a broad claim honest:

```json
{
  "s_tier_gate": {
    "min_task_clusters": 50,
    "min_replicates_per_task": 3,
    "min_mean_savings": 0.6,
    "min_clustered_savings_ci_low": 0.6,
    "max_pass_regressions": 0,
    "require_locked_git_sources": true,
    "require_remote_git_sources": true,
    "require_adapter_profiles": true,
    "require_receipt_validation": true
  }
}
```

Tasks may reference either a local `repo` path or a locked Git source. Git
sources are cloned with `GIT_TERMINAL_PROMPT=0`, checked out detached at `rev`,
and stripped of `.git` before execution:

```json
{
  "task_id": "real-repo-task",
  "git": {
    "url": "https://github.com/example/project.git",
    "rev": "0123456789abcdef0123456789abcdef01234567",
    "subdir": ""
  },
  "patch_spec": "patches/fix.json",
  "verify_cmd": ["python", "-m", "pytest", "-q"]
}
```

Instead of `patch_spec`, a task can provide `repair_cmd`:

```json
{
  "task_id": "real-repo-command-task",
  "repo": "../path/to/repo",
  "repair_cmd": ["python", "scripts/repair_agent.py"],
  "repair_timeout_s": 600,
  "prompt": "Fix the issue without editing tests.",
  "verify_cmd": ["python", "-m", "pytest", "-q"]
}
```

Or a reusable `adapter_profile`:

```json
{
  "task_id": "real-repo-profile-task",
  "repo": "../path/to/repo",
  "adapter_profile": "adapters/my-agent.json",
  "prompt": "Fix the issue without editing tests.",
  "verify_cmd": ["python", "-m", "pytest", "-q"]
}
```

Bundles are ZIP files with deterministic file order and timestamps. Each bundle
contains `trice_bundle_manifest.json`, `ro-crate-metadata.json`, the aggregate
result, child manifests, and every trace/context artifact needed for deep
verification.

The source manifest `trice_suite_sources.json` records each task's repo tree
fingerprint (`trice-tree-sha256/v1`), intervention provenance, verifier command,
file count, and bytes before any live run starts. JSON patch tasks record a
patch-spec SHA-256. Command tasks record command argv, timeout, and whether test
edits were allowed. Adapter-profile tasks additionally record the profile
SHA-256 and profile name. Suite reports break out runs by adapter type and
failure mode.

## Schemas

- Patch spec: [`schemas/trice_patch_spec.schema.json`](../schemas/trice_patch_spec.schema.json)
- Evidence manifest: [`schemas/trice_evidence_manifest.schema.json`](../schemas/trice_evidence_manifest.schema.json)
- Suite manifest: [`schemas/trice_suite_manifest.schema.json`](../schemas/trice_suite_manifest.schema.json)
- Bundle manifest: [`schemas/trice_bundle_manifest.schema.json`](../schemas/trice_bundle_manifest.schema.json)
- Adapter profile: [`schemas/trice_adapter_profile.schema.json`](../schemas/trice_adapter_profile.schema.json)
- Run receipt: [`schemas/trice_run_receipt.schema.json`](../schemas/trice_run_receipt.schema.json)

## Python API

```python
from tracerazor.trice import CommandRepairAdapter, JsonPatchAdapter, LiveTask, run_live_learning_loop, verify_manifest

task = LiveTask.from_repo(
    "benchmark/live/tasks/fix-offby-one/seed",
    task_id="fix-offby-one-generic",
    prompt="Fix chunker.py without editing tests.",
    verify_cmd=["python", "-m", "pytest", "-q", "--tb=short"],
)
adapter = JsonPatchAdapter.from_dict({
    "name": "fix-offby-one-demo",
    "edits": [{"op": "replace", "path": "chunker.py", "old": "size - 1", "new": "size"}],
})
result = run_live_learning_loop([task], adapter=adapter, rounds=1)
assert verify_manifest(result.manifest_path)["ok"]
```

By default `JsonPatchAdapter` refuses test-file edits and path traversal. This
keeps the measured run tied to source edits and verifier outcomes.
`CommandRepairAdapter` applies the same test-edit refusal after comparing
workspace fingerprints before and after the command.

## CLI Helpers

```powershell
tracerazor-trice schema patch
tracerazor-trice schema manifest
tracerazor-trice schema suite
tracerazor-trice schema bundle
tracerazor-trice schema adapter-profile
tracerazor-trice schema receipt
tracerazor-trice validate-patch examples\trice_patch_fix_offbyone.json
tracerazor-trice validate-adapter examples\trice_adapter_profile_echo.json
tracerazor-trice validate-receipt benchmark\trice\results\v2-suite\tasks\fix-offby-one-suite\replicate-1\fix-offby-one-suite\round-1\trice-v2\run_receipt.json
tracerazor-trice validate-suite examples\trice_suite_fix_offbyone.json
```
