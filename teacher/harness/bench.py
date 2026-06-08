"""Benchmarks for the harness I/O substrate -- real, reproducible numbers.

Measures the three optimisations against their status-quo baselines:

  1. CACHE      -- caching vs recompute on repeated diagnoses (the curriculum
                   re-diagnoses the baseline every round). Reports the drop in
                   real backend (subprocess/native) invocations + wall time.
  2. APPEND-LOG -- append-only writes vs full-file rewrite for an evolving
                   key/value set (the Playbook update pattern). Reports bytes
                   written and wall time.
  3. MMAP READ  -- mmap random reads vs load-the-whole-file-then-index.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

from .cache import CachingDiagnoser, DiagnosisCache
from .store import KVStore


def _timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, (time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
# 1. Diagnosis cache vs recompute
# --------------------------------------------------------------------------- #
def bench_cache(diagnoser, traces: list[dict], rounds: int = 4) -> dict:
    """Re-diagnose ``traces`` ``rounds`` times (curriculum-like) with/without cache."""
    # Baseline: no cache, every diagnose hits the backend.
    base_calls0 = getattr(diagnoser, "audit_calls", 0)
    _, t_nocache = _timed(lambda: [diagnoser.diagnose(t)
                                   for _ in range(rounds) for t in traces])
    nocache_calls = getattr(diagnoser, "audit_calls", 0) - base_calls0

    cached = CachingDiagnoser(diagnoser, DiagnosisCache())
    base_calls1 = getattr(diagnoser, "audit_calls", 0)
    _, t_cache = _timed(lambda: [cached.diagnose(t)
                                 for _ in range(rounds) for t in traces])
    cache_calls = getattr(diagnoser, "audit_calls", 0) - base_calls1

    return {
        "rounds": rounds, "unique_traces": len(traces),
        "ops": rounds * len(traces),
        "backend_calls_nocache": nocache_calls,
        "backend_calls_cached": cache_calls,
        "call_reduction_pct": round(100 * (1 - cache_calls / max(nocache_calls, 1)), 1),
        "wall_nocache_s": round(t_nocache, 4),
        "wall_cached_s": round(t_cache, 4),
        "speedup_x": round(t_nocache / t_cache, 2) if t_cache else float("inf"),
        "cache_stats": cached.cache.stats(),
    }


# --------------------------------------------------------------------------- #
# 2. Append-only log vs full-file rewrite (the Playbook pattern)
# --------------------------------------------------------------------------- #
def bench_append_vs_rewrite(updates: int = 2000, keyspace: int = 200) -> dict:
    """Simulate ``updates`` single-record updates over ``keyspace`` keys."""
    tmp = tempfile.mkdtemp(prefix="trz_bench_")

    # Status quo: rewrite the whole JSON dict every update (write amplification).
    rewrite_path = os.path.join(tmp, "playbook.json")
    state: dict[str, dict] = {}
    bytes_rewrite = 0

    def do_rewrite():
        nonlocal bytes_rewrite
        for i in range(updates):
            state[f"k{i % keyspace}"] = {"trials": i, "wins": i // 2}
            blob = json.dumps(state).encode()
            bytes_rewrite += len(blob)
            with open(rewrite_path, "wb") as fh:
                fh.write(blob)
    _, t_rewrite = _timed(do_rewrite)

    # Optimised: append-only KV log (sequential, no rewrite amplification).
    kv = KVStore(os.path.join(tmp, "playbook.log"))
    bytes_append = 0

    def do_append():
        nonlocal bytes_append
        for i in range(updates):
            v = json.dumps({"trials": i, "wins": i // 2}).encode()
            bytes_append += len(v)
            kv.put(f"k{i % keyspace}", v)
    _, t_append = _timed(do_append)
    kv.close()

    return {
        "updates": updates, "keyspace": keyspace,
        "bytes_rewrite": bytes_rewrite, "bytes_append": bytes_append,
        "write_amplification_x": round(bytes_rewrite / max(bytes_append, 1), 1),
        "wall_rewrite_s": round(t_rewrite, 4),
        "wall_append_s": round(t_append, 4),
        "speedup_x": round(t_rewrite / t_append, 2) if t_append else float("inf"),
    }


# --------------------------------------------------------------------------- #
# 3. mmap random read vs load-all
# --------------------------------------------------------------------------- #
def bench_mmap_vs_loadall(records: int = 20000, reads: int = 5000) -> dict:
    import random
    tmp = tempfile.mkdtemp(prefix="trz_bench_")
    # Build a JSONL file (load-all baseline) and a KV log (mmap) with same data.
    jsonl = os.path.join(tmp, "data.jsonl")
    kv = KVStore(os.path.join(tmp, "data.log"))
    items = [(f"k{i}", json.dumps({"i": i, "pad": "x" * 64}).encode())
             for i in range(records)]
    with open(jsonl, "w") as fh:
        for k, v in items:
            fh.write(json.dumps({"k": k, "v": v.decode()}) + "\n")
    kv.put_many(items)

    rng = random.Random(7)
    targets = [f"k{rng.randrange(records)}" for _ in range(reads)]

    def load_all():
        index = {}
        with open(jsonl) as fh:
            for line in fh:                       # full scan into memory
                rec = json.loads(line)
                index[rec["k"]] = rec["v"]
        return [index[t] for t in targets]
    _, t_loadall = _timed(load_all)

    _, t_mmap = _timed(lambda: [kv.get(t) for t in targets])
    kv.close()

    return {
        "records": records, "reads": reads,
        "wall_loadall_s": round(t_loadall, 4),
        "wall_mmap_s": round(t_mmap, 4),
        "speedup_x": round(t_loadall / t_mmap, 2) if t_mmap else float("inf"),
    }


def run_all(diagnoser, traces) -> dict:
    return {
        "cache": bench_cache(diagnoser, traces),
        "append_vs_rewrite": bench_append_vs_rewrite(),
        "mmap_vs_loadall": bench_mmap_vs_loadall(),
    }
