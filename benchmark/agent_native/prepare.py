"""Prepare a balanced, protocol-bound manifest before efficacy collection.

The input catalog contains identifiers and content digests only. Task prompts,
credentials, and private verifier configuration stay in the design partner's
own execution system. This command fixes condition order without reading any
outcome, which prevents post-hoc assignment from entering the study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluate import (
    CONDITIONS,
    EvaluationError,
    TASK_MANIFEST_SCHEMA,
    _is_sha256,
    _validate_task_manifest,
    load_protocol,
    protocol_sha256,
    task_manifest_sha256,
)


CATALOG_SCHEMA = "tracerazor-agent-task-catalog/v1"
DEFAULT_PROTOCOL = Path(__file__).with_name("protocol.json")
CATALOG_TASK_FIELDS = {
    "task_id",
    "content_sha256",
    "host_category",
    "workload_stratum",
    "verifier_id",
}


def load_catalog(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        catalog = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationError(f"cannot read task catalog {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{source}: invalid task catalog JSON: {exc.msg}") from exc
    if not isinstance(catalog, dict):
        raise EvaluationError("task catalog must be a JSON object")
    if set(catalog) != {"schema_version", "tasks"}:
        raise EvaluationError("task catalog fields must be exactly ['schema_version', 'tasks']")
    if catalog.get("schema_version") != CATALOG_SCHEMA:
        raise EvaluationError("unsupported task catalog schema")
    tasks = catalog.get("tasks")
    if not isinstance(tasks, list):
        raise EvaluationError("task catalog tasks must be an array")
    for number, task in enumerate(tasks, 1):
        label = f"task catalog task {number}"
        if not isinstance(task, dict) or set(task) != CATALOG_TASK_FIELDS:
            raise EvaluationError(
                f"{label} fields must be exactly {sorted(CATALOG_TASK_FIELDS)!r}"
            )
        if not isinstance(task.get("task_id"), str) or not task["task_id"]:
            raise EvaluationError(f"{label} task_id must be non-empty")
        if not _is_sha256(task.get("content_sha256")):
            raise EvaluationError(f"{label} content_sha256 is invalid")
        if not isinstance(task.get("verifier_id"), str) or not task["verifier_id"]:
            raise EvaluationError(f"{label} verifier_id must be non-empty")
    return catalog


def build_task_manifest(
    catalog: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    seed: int = 1729,
    generated_at: str | None = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Build and validate one precommitted, position-balanced task manifest."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise EvaluationError("seed must be an integer")
    timestamp = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvaluationError("generated_at must be an ISO-8601 timestamp") from exc
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
        raise EvaluationError("generated_at must include a timezone offset")
    tasks: list[dict[str, Any]] = []
    raw_tasks = catalog.get("tasks")
    if not isinstance(raw_tasks, list):
        raise EvaluationError("task catalog tasks must be an array")

    for raw_task in sorted(raw_tasks, key=lambda item: str(item.get("task_id", ""))):
        task_id = str(raw_task.get("task_id", ""))
        base_order = list(CONDITIONS)
        task_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).digest()[:8],
            "big",
        )
        random.Random(task_seed).shuffle(base_order)
        randomization = []
        for repetition in range(1, 4):
            shift = repetition - 1
            order = base_order[shift:] + base_order[:shift]
            block_digest = hashlib.sha256(
                (
                    f"{protocol.get('study_id')}:{seed}:{task_id}:{repetition}:"
                    + ",".join(order)
                ).encode("utf-8")
            ).hexdigest()
            randomization.append(
                {
                    "repetition": repetition,
                    "block_id": f"block-{block_digest[:24]}",
                    "condition_order": order,
                }
            )
        tasks.append(
            {
                "task_id": task_id,
                "content_sha256": raw_task.get("content_sha256"),
                "host_category": raw_task.get("host_category"),
                "workload_stratum": raw_task.get("workload_stratum"),
                "verifier_id": raw_task.get("verifier_id"),
                "randomization": randomization,
            }
        )

    manifest = {
        "schema_version": TASK_MANIFEST_SCHEMA,
        "study_id": protocol.get("study_id"),
        "protocol_sha256": protocol_sha256(protocol),
        "synthetic": synthetic,
        "generated_at": timestamp,
        "tasks": tasks,
    }
    errors, _, _ = _validate_task_manifest(manifest, protocol)
    if errors:
        raise EvaluationError("invalid prepared task manifest: " + "; ".join(errors))
    return manifest


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a balanced held-out task manifest before collection."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--generated-at", help="fixed ISO-8601 timestamp for reproducible planning")
    parser.add_argument("--synthetic", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
        protocol = load_protocol(args.protocol)
        manifest = build_task_manifest(
            catalog,
            protocol,
            seed=args.seed,
            generated_at=args.generated_at,
            synthetic=args.synthetic,
        )
        write_manifest(args.output, manifest)
    except EvaluationError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "task_count": len(manifest["tasks"]),
                "protocol_sha256": manifest["protocol_sha256"],
                "task_manifest_sha256": task_manifest_sha256(manifest),
                "synthetic": manifest["synthetic"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
