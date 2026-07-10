# Agent-native efficacy evaluation

This directory is a reproducible evaluation scaffold, **not an efficacy
result**. Synthetic fixtures test the evaluator; they are never evidence that
TraceRazor reduces tokens or preserves task quality.

## Locked protocol and held-out tasks

[`protocol.json`](protocol.json) is the machine-readable preregistration for
`tracerazor-agent-native-efficacy-v2`. Its canonical SHA-256 digest is compiled
into the evaluator. Lowering a threshold, dropping an invariant, or otherwise
editing it is rejected. A legitimate amendment needs a new `study_id` and an
intentional code-lock update.

Before collection, create a private task manifest conforming to
[`task_manifest.schema.json`](task_manifest.schema.json). It must contain:

- at least 50 unique task IDs and unique content digests;
- at least five tasks in every host-by-workload cell;
- Codex, Claude Code, and Gemini CLI across coding, tool-heavy research, and
  support;
- the task-quality verifier selected before the run; and
- a precommitted, position-balanced condition order for every one of the three
  repetitions.

Prepare that manifest from an identifier-and-digest-only private catalog
conforming to [`task_catalog.schema.json`](task_catalog.schema.json):

```bash
python -m benchmark.agent_native.prepare \
  --catalog path/to/private-task-catalog.json \
  --output path/to/locked-held-out-manifest.json \
  --generated-at 2026-07-10T00:00:00Z
```

The planner never accepts task prompts. It sorts task IDs, derives a seeded
base order per task, rotates it across the three repetitions so every condition
appears exactly once in each order position, and validates the result against
the locked protocol before writing it. Commit or otherwise timestamp the
resulting manifest digest before collecting the first run.

The external manifest's canonical digest is bound into the study JSONL. The
evaluator rejects duplicate content hidden behind different task IDs, unbalanced
cells, fixed condition order, task substitution, and a manifest changed after
collection.

Print the locked protocol digest:

```bash
python -m benchmark.agent_native.evaluate --print-protocol-sha256
```

## Results and pair integrity

Results are UTF-8 JSONL with one record per line, described by
[`result.schema.json`](result.schema.json). Run records include:

- task, host, model, agent configuration, verifier, and randomization identity;
- the preregistered order position and actual start timestamp;
- explicit provider-reported, estimated, or missing token provenance;
- task outcome and task-level optimizer acceptance;
- unique run and verifier receipt digests; and
- unique recommendation, independent adjudication, and adjudication-receipt
  identity.

For each task and repetition, baseline and intervention runs must match on every
locked pair invariant. Every condition starts in a fresh sandbox restored from
the same clean initial-state digest (repository, fixture data, sessions, and
cleared caches), and separately records the same base image/host, toolchain,
dependency lock, and allowed-environment digest. Every run receives a unique
disposable-workspace ID; reuse invalidates the study. These fields must be bound
into `trace.metadata.evaluation_binding`; the signed audit report must bind the
exact trace hash. The evaluator compares that authenticated binding with the
JSONL run record before accepting release evidence. Actual timestamp
order must match the precommitted condition order. One receipt cannot be reused for two observations. Optimizer
acceptance is all-or-none at task level: all three optimizer repetitions must
pass their task verifier and be accepted.

Estimated or missing token counts are retained in provenance totals but never
enter reductions or confidence intervals. Efficacy uses only pairs where both
baseline and intervention succeeded and both token counts are provider-reported.
A failed run with an attractive token count cannot improve the result. Missing
usage on a successful pair makes the study incomplete.

## Analysis

The evaluator computes each intervention separately:

- matched-success pair coverage and median measured token reduction;
- a deterministic task-cluster bootstrap 95% interval for token reduction;
- a paired task-cluster bootstrap interval for task-success difference; and
- the preregistered non-inferiority test on that interval's lower bound.

This prevents a strong optimizer result from hiding a harmful coach result and
prevents three repetitions of one task from being treated as independent tasks.
Baseline solvability is gated before efficacy is interpreted.

Accepted optimizer performance is task-level and requires at least ten tasks,
at least 20% of held-out tasks, and at least 10% median measured reduction.
Recommendation precision requires at least 30 unique, independently adjudicated
recommendations with receipt digests.

## Signed release evidence

A statistical pass is deliberately not a release pass. Real studies must also
provide an evidence index conforming to
[`evidence_index.schema.json`](evidence_index.schema.json). Each indexed report:

1. must remain inside the evidence directory and may not be a symlink;
2. must hash to the receipt digest committed in the JSONL;
3. must have exact coverage—no missing or unexpected run, verifier, or
   adjudication receipts; and
4. must pass `tracerazor verify --format json` with an Ed25519 signature and
   matching trace hash.

Without this verification, a real statistical pass is `release_incomplete`.
Synthetic mode is always `release_incomplete`, even if a caller supplies an
apparently verified evidence object.

Run a study:

```bash
python -m benchmark.agent_native.evaluate \
  --input path/to/results.jsonl \
  --task-manifest path/to/held-out-tasks.json \
  --evidence-index path/to/evidence/index.json \
  --output path/to/evaluation-report.json
```

Exit codes:

- `0`: real, complete statistical pass with every signed receipt authenticated;
- `1`: a complete statistical or safety gate failed;
- `2`: malformed/incomplete input, missing release authentication, or any
  synthetic study.

The report exposes `statistical_status`, `release_status`, and the combined
`status` so a statistical result cannot be mistaken for publishable evidence.

## Tests

```bash
python -m pytest -q benchmark/agent_native/tests
```

The focused suite covers synthetic pass/fail/incomplete cases plus attacks on
the protocol lock, task identity and balance, randomization, pair invariants,
receipt reuse, task-level acceptance, recommendation adjudication, selection on
successful runs, estimated-token laundering, per-condition pooling, quality
non-inferiority, evidence coverage, unsigned receipts, and path traversal.
