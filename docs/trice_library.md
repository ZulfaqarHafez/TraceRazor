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

The suite manifest is hashed into `trice_suite_manifest.snapshot.json`. Each
child task keeps its own live report, result JSON, trace artifacts, and evidence
manifest under `tasks/<task_id>/`.

Replicates are independent fresh child live runs. Suite reports include the
ordinary savings CI plus a clustered-by-task savings CI, so repeated runs of the
same repo do not masquerade as independent held-out repositories.

## Schemas

- Patch spec: [`schemas/trice_patch_spec.schema.json`](../schemas/trice_patch_spec.schema.json)
- Evidence manifest: [`schemas/trice_evidence_manifest.schema.json`](../schemas/trice_evidence_manifest.schema.json)
- Suite manifest: [`schemas/trice_suite_manifest.schema.json`](../schemas/trice_suite_manifest.schema.json)

## Python API

```python
from tracerazor.trice import JsonPatchAdapter, LiveTask, run_live_learning_loop, verify_manifest

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

## CLI Helpers

```powershell
tracerazor-trice schema patch
tracerazor-trice schema manifest
tracerazor-trice schema suite
tracerazor-trice validate-patch examples\trice_patch_fix_offbyone.json
tracerazor-trice validate-suite examples\trice_suite_fix_offbyone.json
```
