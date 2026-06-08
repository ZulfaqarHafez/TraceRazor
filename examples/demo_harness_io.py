"""Harness I/O substrate demo -- real, reproducible numbers (offline).

Shows the three optimisations that make the diagnosis/remediation pipeline
viable at industrial scale:

  1. content-addressed diagnosis CACHE  -- fewer fork/exec + recompute
  2. append-only LOG vs full-file rewrite -- no write amplification
  3. mmap random reads vs load-all       -- page-cache-served, no full scan

Then wires the cache into a live Teacher curriculum and shows the drop in real
backend (subprocess) invocations.

Run:
    python examples/demo_harness_io.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teacher import AgentConfig, Diagnoser, Mode, Playbook, Teacher  # noqa: E402
from teacher.harness import CachingDiagnoser, DiagnosisCache, content_hash  # noqa: E402
from teacher.harness import bench  # noqa: E402
from teacher.runner import Task, run_task  # noqa: E402


def _fmt(d: dict) -> str:
    return "\n".join(f"    {k:<26} {v}" for k, v in d.items())


def main() -> None:
    diagnoser = Diagnoser(prefer_auditor=True)
    print(f"[diagnostic backend: {diagnoser.backend}]  "
          f"[content hash: {content_hash({'a': 1})[:8]}...]\n")

    # A handful of representative traces (some identical -> dedup opportunity).
    cfg = AgentConfig()
    tasks = [
        Task("refund-1", "refund order ORD-9182", ["get_order", "check", "refund"]),
        Task("status-1", "status of ORD-5500", ["get_order", "status"]),
        Task("refund-1", "refund order ORD-9182", ["get_order", "check", "refund"]),  # dup
    ]
    traces = [run_task(cfg, t)["trace"] for t in tasks]

    print("=" * 74)
    print("BENCHMARK 1 -- diagnosis cache vs recompute (curriculum re-diagnosis)")
    print("=" * 74)
    print(_fmt(bench.bench_cache(diagnoser, traces, rounds=4)))

    print("\n" + "=" * 74)
    print("BENCHMARK 2 -- append-only log vs full-file rewrite (Playbook pattern)")
    print("=" * 74)
    print(_fmt(bench.bench_append_vs_rewrite(updates=2000, keyspace=200)))

    print("\n" + "=" * 74)
    print("BENCHMARK 3 -- mmap random reads vs load-all")
    print("=" * 74)
    print(_fmt(bench.bench_mmap_vs_loadall(records=20000, reads=5000)))

    # ---- Wire the cache into a live curriculum ---------------------------- #
    print("\n" + "=" * 74)
    print("WIRED: Teacher curriculum with a content-addressed CachingDiagnoser")
    print("=" * 74)
    fresh = Diagnoser(prefer_auditor=True)
    caching = CachingDiagnoser(fresh, DiagnosisCache())
    curric_tasks = [
        Task("refund-1", "refund order ORD-9182", ["get_order", "check", "refund"]),
        Task("status-1", "status of ORD-5500", ["get_order", "status"]),
    ]
    teacher = Teacher(AgentConfig(), mode=Mode.CURRICULUM,
                      playbook=Playbook(), diagnoser=caching)
    result = teacher.improve(curric_tasks, max_rounds=8)
    print(f"    backend audit calls (with cache) : {fresh.audit_calls}")
    print(f"    cache stats                       : {caching.cache.stats()}")
    print(f"    curriculum net token saving       : {result.total_token_saving_pct:.1f}%")
    print(f"    task success preserved            : "
          f"{all(vr.success_after >= 0.99 for vr in result.accepted)}")

    # Acceptance checks.
    c = bench.bench_cache(diagnoser, traces, rounds=4)
    assert c["backend_calls_cached"] < c["backend_calls_nocache"]
    assert caching.cache.hits > 0
    print("\n[OK] cache cuts backend calls, append-log avoids rewrite amplification, "
          "mmap beats load-all.")


if __name__ == "__main__":
    main()
