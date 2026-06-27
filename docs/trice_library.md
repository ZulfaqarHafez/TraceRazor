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

The checked-in remote smoke fixture uses the same public flow against a locked
PyPA `sampleproject` commit. It is deliberately tiny: one public Git source,
one source-only patch spec, one objective verifier, and one replicate.

```powershell
tracerazor-trice suite scaffold `
  --source examples\trice_remote_smoke_source.json `
  --out examples\trice_remote_smoke_suite.json
tracerazor-trice suite examples\trice_remote_smoke_suite.json `
  --out-dir benchmark\trice\results\v2-remote-smoke `
  --rounds 1 `
  --replicates 1
tracerazor-trice verify-suite benchmark\trice\results\v2-remote-smoke\trice_suite_evidence_manifest.json
tracerazor-trice claim `
  --suite-result benchmark\trice\results\v2-remote-smoke\trice_suite_results.json `
  --manifest benchmark\trice\results\v2-remote-smoke\trice_suite_evidence_manifest.json `
  --scope "remote-git smoke path on one locked public Python repository" `
  --out docs\trice_remote_smoke_claim_card.json
tracerazor-trice verify-claim docs\trice_remote_smoke_claim_card.json
tracerazor-trice bundle benchmark\trice\results\v2-remote-smoke\trice_suite_evidence_manifest.json `
  --out benchmark\trice\results\v2-remote-smoke\trice_remote_smoke_evidence.trice.zip
tracerazor-trice verify-bundle benchmark\trice\results\v2-remote-smoke\trice_remote_smoke_evidence.trice.zip
```

Current remote smoke evidence reports 83.2% measured input-token savings, zero
pass regressions, 100% evidence recall with zero recall failures, a verified
17-entry bundle, and `s_tier_gate.passed = false` because one public repo and
one replicate are not a broad held-out claim.

The bundled broad-smoke suite exercises all six local live tasks through a
reusable command adapter profile:

```powershell
tracerazor-trice suite examples\trice_suite_bundled_live.json `
  --out-dir benchmark\trice\results\v2-broad-smoke `
  --rounds 1 `
  --replicates 1
tracerazor-trice suite readiness examples\trice_suite_bundled_live.json `
  --out docs\trice_suite_readiness.json
tracerazor-trice suite verify-readiness docs\trice_suite_readiness.json `
  --manifest examples\trice_suite_bundled_live.json
tracerazor-trice protocol --manifest examples\trice_suite_bundled_live.json `
  --out docs\trice_protocol_lock.json
tracerazor-trice verify-protocol docs\trice_protocol_lock.json `
  --manifest examples\trice_suite_bundled_live.json
tracerazor-trice design `
  --protocol docs\trice_protocol_lock.json `
  --suite-result benchmark\trice\results\v2-broad-smoke\trice_suite_results.json `
  --out docs\trice_design_card.json
tracerazor-trice verify-design docs\trice_design_card.json
tracerazor-trice verify-suite benchmark\trice\results\v2-broad-smoke\trice_suite_evidence_manifest.json
tracerazor-trice bundle benchmark\trice\results\v2-broad-smoke\trice_suite_evidence_manifest.json `
  --out benchmark\trice\results\v2-broad-smoke\trice_broad_smoke_evidence.trice.zip
tracerazor-trice verify-bundle benchmark\trice\results\v2-broad-smoke\trice_broad_smoke_evidence.trice.zip
tracerazor-trice claim `
  --suite-result benchmark\trice\results\v2-broad-smoke\trice_suite_results.json `
  --manifest benchmark\trice\results\v2-broad-smoke\trice_suite_evidence_manifest.json `
  --out docs\trice_claim_card.json
tracerazor-trice verify-claim docs\trice_claim_card.json
tracerazor-trice artifact --out docs\trice_artifact_card.json
tracerazor-trice verify-artifact docs\trice_artifact_card.json
```

This broad smoke is stronger than the one-task example because it spans six
task clusters, uses `adapter_profile`, and currently reports 100% evidence
recall with zero recall failures. It still fails the default S-tier gate because
it is local, single-replicate, and not 50 held-out locked Git task clusters.

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
- evidence recall must be at least 95% on every accepted optimized run;
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
    "min_evidence_recall": 0.95,
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

## Public Proof Cards

TRICE separates preflight, outcome, and artifact-review claims:

- `suite readiness` emits `trice-suite-readiness/v1`, a no-execution preflight
  that says whether a manifest is smoke-ready, pilot-ready, or claim-ready to
  run. It never claims savings.
- `protocol` emits `trice-protocol-lock/v1`, a pre-outcome contract that binds
  the suite manifest to the primary metric, quality guardrail, clustered CI
  rule, source-locking rule, adapter-profile rule, receipt rule, claim-card
  rule, and artifact-card rule.
- `verify-protocol` recomputes the protocol hash and rebuilds the lock from the
  suite manifest so post-hoc edits are caught before a live claim run.
- `design` emits `trice-design-card/v1`, a statistical design review that uses
  task-cluster means from suite results to project whether the locked claim-run
  sample size would clear the target, without overriding held-out source or
  replicate requirements.
- `verify-design` recomputes the design hash and rebuilds the card from the
  protocol lock plus suite result.
- `claim` emits `trice-claim-card/v1`, an outcome boundary that binds a suite
  result and evidence manifest. It may say `claim_allowed = true` only when the
  suite gate passes.
- `reproduction` emits `trice-reproduction-card/v1`, a reviewer runbook that
  binds exact verifier commands plus readiness, protocol, design, claim,
  bundle, paper-manifest, and paper-result hashes.
- `verify-reproduction` checks the reproduction card hash, all bound inputs,
  and a deterministic rebuild from the bound files.
- `contract` emits `trice-contract-card/v1`, a public library contract that
  binds SemVer, import exports, CLI commands, shipped schemas, examples, docs,
  and package metadata before release claims are made.
- `verify-contract` checks the contract card hash, bound files, and a
  deterministic rebuild from the current public surface.
- `artifact` emits `trice-artifact-card/v1`, a reviewer packet that binds
  README, paper source/PDF, paper manifest, readiness card, protocol lock,
  design card, reproduction card, contract card, installability card, claim
  card, evidence bundle, library doc, and schemas together.
- `verify-artifact` recomputes the card hash and every bound artifact/schema
  hash, making the public review packet independently checkable.
- `release` emits `trice-release-card/v1`, a distribution trust snapshot that
  binds `doctor` output, package metadata, proof cards, the public contract
  card, and release docs. It can be a `local_release_candidate` while refusing
  `public_release_ready`.
- `verify-release` checks the release card hash, bound inputs, doctor snapshot
  hash, and deterministic rebuild. Public release readiness requires PyPI,
  piwheels, crates.io, GitHub tag, and GitHub Actions to be green.
- `release-evidence` emits `trice-release-evidence/v1`, a release asset packet
  that binds wheels, sdist, Rust CLI binary, proof cards including the crates
  publish card and installability card, paper artifacts, evidence bundles, SHA-256 checksums,
  CycloneDX-style Python and Cargo SBOMs, and an in-toto/SLSA-shaped provenance
  statement.
- `verify-release-evidence` checks the release evidence hash, bound artifact
  hashes, sidecar byte counts, sidecar hashes, and deterministic rebuild from
  the bound `dist` directory and CLI binary.
- `crates` emits `trice-crates-card/v1`, a staged crates.io publish card that
  binds workspace manifests, local dependency order, registry state, and README
  cargo-install honesty. It can lock the publish plan before crates.io is green.
- `verify-crates` checks the crates card hash, bound manifest hashes, registry
  snapshot hash, and deterministic rebuild from the bound Cargo inputs.
- `install` emits `trice-install-card/v1`, a clean-wheel installability card
  that creates a virtual environment, installs the built wheel, imports shipped
  schemas and public APIs, runs `tracerazor-trice`, and separately checks the
  bundled Rust CLI path.
- `verify-install` checks the install card hash and bound wheel/package input
  hashes without rerunning the virtual environment.
- `research` emits `trice-research-card/v1`, a paper-basis card that parses the
  research ledger, checks source and category coverage, binds row hashes, and
  renders JSON, Markdown, SVG, and LaTeX.
- `verify-research` checks the research card hash, bound ledger hash, and
  deterministic rebuild from the current ledger.
- `integrity` emits `trice-integrity-card/v1`, the top-level proof graph card
  that binds offline doctor output, contract, artifact, reproduction, release,
  release-evidence, crates, installability, research, paper-manifest, schema, and
  workflow-hook checks.
- `verify-integrity` checks the integrity card hash, bound input hashes, schema
  hashes, workflow hashes, doctor snapshot hash, and deterministic rebuild from
  the bound proof graph.

## Schemas

- Patch spec: [`schemas/trice_patch_spec.schema.json`](../schemas/trice_patch_spec.schema.json)
- Evidence manifest: [`schemas/trice_evidence_manifest.schema.json`](../schemas/trice_evidence_manifest.schema.json)
- Suite manifest: [`schemas/trice_suite_manifest.schema.json`](../schemas/trice_suite_manifest.schema.json)
- Bundle manifest: [`schemas/trice_bundle_manifest.schema.json`](../schemas/trice_bundle_manifest.schema.json)
- Adapter profile: [`schemas/trice_adapter_profile.schema.json`](../schemas/trice_adapter_profile.schema.json)
- Run receipt: [`schemas/trice_run_receipt.schema.json`](../schemas/trice_run_receipt.schema.json)
- Claim card: [`schemas/trice_claim_card.schema.json`](../schemas/trice_claim_card.schema.json)
- Suite readiness: [`schemas/trice_suite_readiness.schema.json`](../schemas/trice_suite_readiness.schema.json)
- Artifact card: [`schemas/trice_artifact_card.schema.json`](../schemas/trice_artifact_card.schema.json)
- Protocol lock: [`schemas/trice_protocol_lock.schema.json`](../schemas/trice_protocol_lock.schema.json)
- Design card: [`schemas/trice_design_card.schema.json`](../schemas/trice_design_card.schema.json)
- Reproduction card: [`schemas/trice_reproduction_card.schema.json`](../schemas/trice_reproduction_card.schema.json)
- Release card: [`schemas/trice_release_card.schema.json`](../schemas/trice_release_card.schema.json)
- Contract card: [`schemas/trice_contract_card.schema.json`](../schemas/trice_contract_card.schema.json)
- Release evidence: [`schemas/trice_release_evidence.schema.json`](../schemas/trice_release_evidence.schema.json)
- Integrity card: [`schemas/trice_integrity_card.schema.json`](../schemas/trice_integrity_card.schema.json)
- Crates publish card: [`schemas/trice_crates_card.schema.json`](../schemas/trice_crates_card.schema.json)
- Installability card: [`schemas/trice_install_card.schema.json`](../schemas/trice_install_card.schema.json)
- Research card: [`schemas/trice_research_card.schema.json`](../schemas/trice_research_card.schema.json)

## Python API

```python
from tracerazor.trice import (
    CommandRepairAdapter,
    JsonPatchAdapter,
    LiveTask,
    build_artifact_card,
    build_claim_card,
    build_contract_card,
    build_crates_card,
    build_design_card,
    build_integrity_card,
    build_install_card,
    build_research_card,
    build_protocol_lock,
    build_release_card,
    build_release_evidence_card,
    build_reproduction_card,
    build_suite_readiness,
    run_live_learning_loop,
    verify_artifact_card_file,
    verify_contract_card_file,
    verify_crates_card_file,
    verify_design_card_file,
    verify_integrity_card_file,
    verify_install_card_file,
    verify_research_card_file,
    verify_release_card_file,
    verify_release_evidence_file,
    verify_reproduction_card_file,
    verify_manifest,
    verify_protocol_lock_file,
)

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
assert build_suite_readiness("examples/trice_suite_bundled_live.json")["readiness_level"] == "smoke_ready"
assert build_protocol_lock("examples/trice_suite_bundled_live.json")["protocol_level"] == "smoke_protocol_locked"
assert build_design_card()["design_level"] == "smoke_design_observed"
assert build_integrity_card()["integrity_level"] == "proof_graph_integrity_locked"
assert build_claim_card("benchmark/trice/results/v2-broad-smoke/trice_suite_results.json")["claim_allowed"] is False
assert build_contract_card()["contract_level"] == "library_contract_locked"
assert build_artifact_card()["artifact_level"] == "review_ready_smoke"
assert build_crates_card()["local_publish_plan_locked"] is True
assert build_research_card()["research_level"] == "research_basis_locked"
assert verify_install_card_file("docs/trice_install_card.json")["ok"]
assert verify_research_card_file("docs/trice_research_card.json")["ok"]
assert verify_protocol_lock_file("docs/trice_protocol_lock.json")["ok"]
assert verify_design_card_file("docs/trice_design_card.json")["ok"]
assert verify_integrity_card_file("docs/trice_integrity_card.json")["ok"]
assert verify_contract_card_file("docs/trice_contract_card.json")["ok"]
assert verify_artifact_card_file("docs/trice_artifact_card.json")["ok"]
assert verify_crates_card_file("docs/trice_crates_card.json")["ok"]
assert verify_release_evidence_file("docs/trice_release_evidence.json")["ok"]
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
tracerazor-trice schema claim-card
tracerazor-trice schema suite-readiness
tracerazor-trice schema artifact-card
tracerazor-trice schema protocol-lock
tracerazor-trice schema design-card
tracerazor-trice schema contract-card
tracerazor-trice schema reproduction-card
tracerazor-trice schema release-card
tracerazor-trice schema release-evidence-card
tracerazor-trice schema integrity-card
tracerazor-trice schema crates-card
tracerazor-trice schema install-card
tracerazor-trice schema research-card
tracerazor-trice install --out docs\trice_install_card.json --dist-dir dist
tracerazor-trice verify-install docs\trice_install_card.json
tracerazor-trice research --out docs\trice_research_card.json
tracerazor-trice verify-research docs\trice_research_card.json
tracerazor-trice validate-patch examples\trice_patch_fix_offbyone.json
tracerazor-trice validate-adapter examples\trice_adapter_profile_echo.json
tracerazor-trice validate-receipt benchmark\trice\results\v2-suite\tasks\fix-offby-one-suite\replicate-1\fix-offby-one-suite\round-1\trice-v2\run_receipt.json
tracerazor-trice validate-suite examples\trice_suite_fix_offbyone.json
```
