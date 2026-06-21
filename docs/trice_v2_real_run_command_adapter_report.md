# TRICE V2 Real-Run Command Adapter Report

Date: 2026-06-21

## Research Delta

This v2 iteration targets real-run evaluation rather than replay. The design is
grounded in four practical requirements from current agent-evaluation research:

- Cost and quality must be optimized together, not reported as separate
  afterthoughts. See [AI Agents That Matter](https://arxiv.org/abs/2407.01502).
- Software-agent evaluation should use real repositories, edits, and objective
  execution checks. See [SWE-bench](https://arxiv.org/abs/2310.06770).
- Agent harness design matters because the interface controls what an agent can
  inspect, edit, and test. See [SWE-agent](https://arxiv.org/abs/2405.15793).
- Claims need portable, independently checkable artifacts. See
  [ACM Artifact Review and Badging](https://www.acm.org/publications/policies/artifact-review-badging).

## Product Change

TRICE now supports a `CommandRepairAdapter` and reusable adapter profiles in
addition to the deterministic JSON patch adapter. The command adapter runs a
user-supplied repair command with `cwd` set to a fresh copied workspace, passes
`TRICE_TASK_ID`, `TRICE_PROMPT`, `TRICE_WORKSPACE`, `TRICE_AGENT_RECEIPT`,
`TRICE_INPUT_TOKENS`, `TRICE_BASELINE_INPUT_TOKENS`, `TRICE_CONTEXT_MODE`, and
policy/budget metadata in the environment, fingerprints the workspace before
and after execution, records changed files, captures a structured
`run_receipt.json`, and refuses edits under `tests/` and `test/` by default.

Public surfaces:

- `tracerazor-trice run -- --repair-cmd "..."`
- `tracerazor-trice run -- --adapter-profile profile.json`
- Suite task field `repair_cmd`, mutually exclusive with `patch_spec`
- Suite task field `adapter_profile`, mutually exclusive with `repair_cmd` and
  `patch_spec`
- Suite provenance field `adapter_type`, with command argv, timeout, and
  test-edit policy for command tasks
- Run receipt artifact with adapter envelope, command hash, before/after
  workspace digest, changed files, output hashes, TRICE context envelope, and
  optional agent-reported token accounting
- Python API export `tracerazor.trice.CommandRepairAdapter`

## Real-Run Evidence

The regenerated public suite remains a live run, not replay:

- Suite: `benchmark/trice/results/v2-suite/trice_suite_results.json`
- Replicates: 3 fresh workspace runs
- Mean input-token savings: 76.6%
- Pass regressions: 0
- Smoke gate: passed
- Bundle: `benchmark/trice/results/v2-suite/trice_suite_evidence.trice.zip`
- Bundle entries: regenerated after each evidence run
- Deep verification: aggregate and all child manifests passed

The broader bundled smoke now exercises all six local live task repositories
through `trice-adapter-profile/v1`:

- Suite: `benchmark/trice/results/v2-broad-smoke/trice_suite_results.json`
- Task clusters: 6
- Replicates: 1 per task
- Mean input-token savings: 81.5%
- Pass regressions: 0
- Adapter breakdown: `command_profile`, 6 accepted runs
- Bundle: `benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip`
- Bundle entries: 65
- Sample optimized receipt: `TRICE_INPUT_TOKENS=577` versus
  `TRICE_BASELINE_INPUT_TOKENS=2470` on `fix-offby-one`
- S-tier gate: failed honestly because the suite is local, single-replicate, and
  not 50 held-out locked remote Git task clusters

This still is not an S-tier broad claim. It proves the local deterministic
contract and the command-adapter path, but held-out external repositories and
provider-backed agent adapters remain the next gate. The default S-tier gate
now distinguishes remote Git URLs from local file Git sources, so local mirrors
cannot satisfy the final proof requirement.

## Next V2 Iteration

The next product step is to add first-class provider wrappers for real coding
agents that emit deterministic command boundaries:

- Provider-backed adapter profiles should be added for common coding-agent CLIs.
- Agent commands should write model, prompt hash, tool envelope, and
  environment-derived input-token accounting to `TRICE_AGENT_RECEIPT`.
- Suite reports should split savings by adapter type, repo cluster, and failure
  mode.
- The held-out gate should run 20 to 50 locked Git tasks with at least 3
  replicates per condition and clustered confidence intervals.
