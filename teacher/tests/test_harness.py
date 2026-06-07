"""Tests for the harness I/O substrate (cache + append-only/mmap store)."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from teacher.harness import (  # noqa: E402
    AppendLog, CachingDiagnoser, DiagnosisCache, KVStore, content_hash,
)
from teacher.schemas import Diagnosis  # noqa: E402

_TRACE = {"trace_id": "t1", "steps": [{"id": 1, "type": "reasoning",
                                       "content": "hi", "tokens": 10}]}


class _FakeAuditor:
    """A Diagnoser-shaped stub with an expensive `audit` we can count."""
    backend = "fake"
    _audit = True

    def __init__(self):
        self.audit_calls = 0

    def audit(self, trace):
        self.audit_calls += 1
        return {"score": {"score": 88.0}, "total_tokens":
                sum(s["tokens"] for s in trace["steps"])}

    def _parse_auditor(self, data, trace):
        return Diagnosis(trace.get("trace_id", "t"), "a", "f",
                         data["score"]["score"], data["total_tokens"], [],
                         source="auditor", backend="fake")

    def _diagnose_builtin(self, trace):
        return Diagnosis("t", "a", "f", 100.0, 0, [], source="builtin")


# --------------------------------------------------------------------------- #
# content hash + cache
# --------------------------------------------------------------------------- #
def test_content_hash_stable_and_order_insensitive():
    a = content_hash({"x": 1, "y": 2})
    b = content_hash({"y": 2, "x": 1})        # key order must not matter
    assert a == b
    assert content_hash({"x": 1}) != a


def test_cache_get_put_and_lru_eviction():
    c = DiagnosisCache(capacity=2)
    c.put("a", {"v": 1}); c.put("b", {"v": 2})
    assert c.get("a") == {"v": 1}             # touch a -> b is now LRU
    c.put("c", {"v": 3})                       # evicts b
    assert c.get("b") is None
    assert c.get("c") == {"v": 3}
    assert 0.0 <= c.hit_rate <= 1.0


def test_cache_journal_persists_across_instances():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "cache.jsonl")
    c1 = DiagnosisCache(journal_path=path)
    c1.put("k1", {"report": 1})
    c2 = DiagnosisCache(journal_path=path)    # replay the append-only journal
    assert c2.get("k1") == {"report": 1}


def test_caching_diagnoser_skips_repeat_audits():
    inner = _FakeAuditor()
    cd = CachingDiagnoser(inner, DiagnosisCache())
    d1 = cd.diagnose(_TRACE)
    d2 = cd.diagnose(_TRACE)                    # identical -> cache hit
    assert inner.audit_calls == 1              # backend hit only once
    assert cd.cache.hits >= 1
    assert d1.tas_score == d2.tas_score == 88.0
    # a different trace forces a new audit
    cd.diagnose({"trace_id": "t2", "steps": [{"id": 1, "type": "reasoning",
                                              "content": "x", "tokens": 5}]})
    assert inner.audit_calls == 2


# --------------------------------------------------------------------------- #
# append-only log + mmap reads
# --------------------------------------------------------------------------- #
def test_appendlog_roundtrip_and_scan():
    tmp = tempfile.mkdtemp()
    log = AppendLog(os.path.join(tmp, "a.log"))
    o1 = log.append(b"hello")
    o2 = log.append(b"world!!")
    assert log.read_at(o1) == b"hello"        # mmap random read by offset
    assert log.read_at(o2) == b"world!!"
    recs = [r for _, r in log.scan()]
    assert recs == [b"hello", b"world!!"]
    log.close()


def test_kvstore_last_write_wins_and_persist():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "kv.log")
    kv = KVStore(path)
    kv.put("k", b"v1")
    kv.put("k", b"v2")                          # supersedes
    assert kv.get("k") == b"v2"
    kv.put_many([("a", b"1"), ("b", b"2")])
    kv.close()
    kv2 = KVStore(path)                         # rebuild index from log
    assert kv2.get("k") == b"v2"
    assert kv2.get("a") == b"1" and kv2.get("b") == b"2"
    kv2.close()


def test_kvstore_compact_reclaims_and_preserves():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "kv.log")
    kv = KVStore(path)
    for i in range(50):
        kv.put("k", str(i).encode())           # 50 superseding writes
    size_before = os.path.getsize(path)
    kv.compact()
    size_after = os.path.getsize(path)
    assert kv.get("k") == b"49"                 # value preserved
    assert size_after < size_before            # space reclaimed
    kv.close()
