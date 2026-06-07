"""Harness I/O substrate -- cache-aware, SSD-optimized I/O for the agent harness.

* ``content_hash`` / ``DiagnosisCache`` / ``CachingDiagnoser`` -- content-
  addressed memoisation of the expensive audit (cache vs recompute).
* ``AppendLog`` / ``KVStore`` -- append-only sequential writes + mmap random
  reads (SSD-friendly, no rewrite amplification).
* ``bench`` -- reproducible benchmarks with real numbers.
"""
from .cache import CachingDiagnoser, DiagnosisCache, content_hash
from .store import AppendLog, KVStore

__all__ = [
    "content_hash", "DiagnosisCache", "CachingDiagnoser",
    "AppendLog", "KVStore",
]
