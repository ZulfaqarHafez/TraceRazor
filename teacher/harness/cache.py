"""Content-addressed diagnosis cache (harness engineering, cache vs recompute).

The dominant per-trace cost in the curriculum / online loop is the *audit* --
a subprocess fork+exec (~1-4 ms) plus a temp-file write, or an in-process native
call -- not the cheap parse. The same trace is frequently re-audited (the
baseline is re-diagnosed every curriculum round; identical traces recur across a
holdout). Hashing the trace and memoising the raw report turns those repeats
into an O(1) cache hit -- no fork, no temp file, no recompute.

Design:
  * **content address** -- BLAKE2b over the canonical (sorted-key) trace JSON.
    BLAKE3 (multi-GB/s) is a drop-in if available; BLAKE2b is stdlib and already
    fast enough that hashing is cheaper than a single fork.
  * **in-memory LRU** -- hot set, O(1) get/put.
  * **append-only journal (optional)** -- a sequential, SSD-friendly on-disk log
    so the cache survives across process restarts; load replays the log.

Immutable, content-addressed blobs make this safe: the same trace always hashes
to the same key and always yields the same report.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from typing import Optional


def _hasher():
    """Prefer BLAKE3 if installed (multi-GB/s); else stdlib BLAKE2b."""
    try:
        import blake3  # type: ignore
        return ("blake3", lambda b: blake3.blake3(b).hexdigest(16))
    except Exception:
        return ("blake2b", lambda b: hashlib.blake2b(b, digest_size=16).hexdigest())


_HASH_NAME, _HASH = _hasher()


def content_hash(trace: dict) -> str:
    """Stable content address of a trace (canonical JSON -> 128-bit hex)."""
    blob = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    return _HASH(blob)


class DiagnosisCache:
    """LRU cache of raw auditor reports, keyed by trace content hash.

    Optionally backed by an append-only journal for cross-run persistence.
    """

    def __init__(self, capacity: int = 8192, journal_path: Optional[str] = None):
        self.capacity = capacity
        self.journal_path = journal_path
        self._lru: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if journal_path and os.path.exists(journal_path):
            self._replay_journal()

    # -- core ops ----------------------------------------------------------- #
    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                self.hits += 1
                return self._lru[key]
            self.misses += 1
            return None

    def put(self, key: str, report: dict) -> None:
        with self._lock:
            self._lru[key] = report
            self._lru.move_to_end(key)
            while len(self._lru) > self.capacity:
                self._lru.popitem(last=False)        # evict LRU
        if self.journal_path is not None:
            self._append_journal(key, report)

    # -- append-only journal (sequential writes) ---------------------------- #
    def _append_journal(self, key: str, report: dict) -> None:
        line = json.dumps({"k": key, "v": report}, separators=(",", ":")) + "\n"
        # Append mode == sequential write; cheap on SSD and HDD alike.
        with open(self.journal_path, "a") as fh:
            fh.write(line)

    def _replay_journal(self) -> None:
        with open(self.journal_path) as fh:
            for line in fh:                          # sequential scan
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._lru[rec["k"]] = rec["v"]   # last write wins
                except Exception:
                    continue
        while len(self._lru) > self.capacity:
            self._lru.popitem(last=False)

    def compact(self) -> None:
        """Rewrite the journal with only the live (deduped) entries."""
        if self.journal_path is None:
            return
        tmp = self.journal_path + ".compact"
        with open(tmp, "w") as fh:
            for k, v in self._lru.items():
                fh.write(json.dumps({"k": k, "v": v}, separators=(",", ":")) + "\n")
        os.replace(tmp, self.journal_path)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hit_rate, 3), "size": len(self._lru),
                "hash": _HASH_NAME}


class CachingDiagnoser:
    """Wraps a ``Diagnoser`` with a content-addressed report cache.

    On a cache hit it skips the subprocess/native audit entirely and only runs
    the cheap parse, so identical traces cost a hash lookup instead of a fork.
    Drop-in: exposes ``diagnose`` and ``audit_calls`` like ``Diagnoser``.
    """

    def __init__(self, inner, cache: Optional[DiagnosisCache] = None):
        self.inner = inner
        self.cache = cache or DiagnosisCache()
        self.backend = getattr(inner, "backend", "builtin")

    @property
    def audit_calls(self) -> int:
        return getattr(self.inner, "audit_calls", 0)

    def diagnose(self, trace: dict):
        # Builtin backend has no expensive audit to memoise -> passthrough.
        if getattr(self.inner, "_audit", None) is None:
            return self.inner.diagnose(trace)
        key = content_hash(trace)
        data = self.cache.get(key)
        if data is None:
            try:
                data = self.inner.audit(trace)
                self.cache.put(key, data)
            except Exception:
                return self.inner._diagnose_builtin(trace)
        return self.inner._parse_auditor(data, trace)
