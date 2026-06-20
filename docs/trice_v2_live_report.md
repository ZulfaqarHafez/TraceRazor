# TRICE V2: Live, User-Conditioned Context Control

Date: 2026-06-21

## Research Update

V1 proved that TRICE could preserve evidence on recorded traces. V2 changes the
acceptance standard: a policy is not accepted unless it survives a live run in a
fresh workspace with an objective verifier.

Fresh research anchors:

- [Acon: Optimizing Context Compression for Long-horizon LLM Agents](https://arxiv.org/html/2510.00615v1): context compression should be optimized from failure cases and judged on task success, not token count alone.
- [Learning Personalized Agents from Human Feedback](https://arxiv.org/html/2602.16173v1): useful agents need explicit user memory plus pre/post-action feedback, especially when preferences drift.
- [Natural-Language Agent Harnesses](https://arxiv.org/html/2603.25723v1): context engineering belongs inside a harness that owns tools, validation, state, and feedback channels.
- [SWE-agent](https://arxiv.org/abs/2405.15793): real software-agent quality depends on the agent-computer interface, repository navigation, file edits, and tests.
- [SWT-Bench](https://arxiv.org/html/2406.12952v3): repository tasks should use real issues, real code, and executable tests as the oracle.
- [SWE-Bench-CL](https://arxiv.org/abs/2507.00014): coding agents need continual learning metrics, including transfer, forgetting, and tool-use efficiency.

The V2 product rule from this research is:

```text
accept(policy, user, task) iff
  live_workspace_run(policy).verifier_passes
  and pass_noninferior(policy, baseline)
  and measured_input_savings >= user.target_savings
  and user.require_live_rollout is satisfied
```

Replay remains useful as a cheap preflight, but it cannot certify the product.

## Implemented V2

Added `benchmark.trice.user`:

- persistent `UserPreferenceProfile`
- learns target savings from user feedback such as "60% savings"
- records "real runs, not replay" as a hard proof requirement
- adapts budget ratio after outcomes: relax on pass failure, tighten when
  savings miss target, remember accepted live wins

Added `benchmark.trice.live`:

- `LiveTask` loads real task repos from `benchmark/live/tasks`
- `ManagedPythonRepairAdapter` applies real source edits in fresh copied
  workspaces
- each condition runs `python -m pytest -q --tb=short`
- baseline uses full assembled context
- TRICE V2 uses the V1 policy solver only to assemble compressed context
- acceptance uses measured input-token reduction plus live pass preservation
- emits Markdown and JSON evidence under `benchmark/trice/results`

CLI:

```powershell
python -m benchmark.trice.live `
  --out-dir benchmark\trice\results\v2-smoke `
  --rounds 1 `
  --user-feedback "real runs, not replay runs; learn from the user; target 60% token savings; do not modify tests"
```

## Live Smoke Evidence

Evidence artifact:

- `benchmark/trice/results/v2-smoke/trice_v2_live_report.md`
- `benchmark/trice/results/v2-smoke/trice_v2_live_results.json`

Clean six-task run:

| Task | Baseline tokens | TRICE V2 tokens | Savings | Baseline pass | TRICE pass | Accepted |
|---|---:|---:|---:|---|---|---|
| csv-filter | 1895 | 430 | 77.3% | yes | yes | yes |
| dedupe-helpers | 1826 | 361 | 80.2% | yes | yes | yes |
| fix-imports | 1806 | 341 | 81.1% | yes | yes | yes |
| fix-offby-one | 2527 | 634 | 74.9% | yes | yes | yes |
| implement-median | 1889 | 424 | 77.6% | yes | yes | yes |
| rename-api | 1814 | 349 | 80.8% | yes | yes | yes |

Mean measured input-token savings: 78.7%.

This is not a broad S-tier claim yet. It is a V2 smoke gate on six bundled live
repo tasks with a deterministic managed adapter. The next proof step is the
same harness with provider adapters and held-out real repositories.

## Next Iteration

1. Add provider adapters for Claude Code, Codex CLI, SWE-agent, and mini-SWE-agent
   that report assembled input tokens and objective verifier results.
2. Persist live workspace traces selectively instead of committing full copied
   workspaces.
3. Replace deterministic repair recipes with model-driven repair under the same
   gate.
4. Add clustered bootstrap CIs across repo/task families.
5. Promote a policy only when user profile, live verifier, and noninferiority
   gate agree.
