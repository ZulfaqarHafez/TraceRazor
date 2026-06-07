"""SSD-optimized append-only store with memory-mapped random reads.

Two filesystem anti-patterns dominate the harness today:
  * ``Playbook.save()`` rewrites the *entire* JSON file on every update --
    O(n) write amplification per single-record change.
  * the SQLite store loads *all* rows into memory per request.

This module provides the primitives to fix both:

  * **AppendLog** -- length-prefixed, append-only records. Writes are purely
    sequential (the access pattern SSDs *and* HDDs handle fastest, with no read-
    modify-write and no write amplification). Each append returns a byte offset.
  * **mmap random reads** -- the log is read back through a single ``mmap`` so
    any record is reachable by offset with no ``seek`` syscall, served straight
    from the OS page cache (no user/kernel copy).
  * **KVStore** -- a content-addressed key/value layer over AppendLog with an
    in-memory ``key -> offset`` index, so a get is one mmap slice. ``compact``
    reclaims space from superseded records.

This is the storage substrate a production agent harness needs to ingest
millions of traces / intervention outcomes without rewrite amplification.
"""
from __future__ import annotations

import mmap
import os
import struct
from typing import Iterator, Optional

_LEN = struct.Struct("<I")        # 4-byte little-endian length prefix


class AppendLog:
    """Append-only, length-prefixed record log with mmap random reads."""

    _MAX_RECORD = (1 << 32) - 1            # 4-byte length prefix limit

    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            open(path, "ab").close()
        self._wf = open(path, "ab")            # sequential append handle
        # Track the write offset explicitly from the file size rather than
        # relying on tell() in append mode (its value before the first write is
        # platform-dependent), so reopen+append computes correct offsets.
        self._end = os.path.getsize(path)
        self._mm: Optional[mmap.mmap] = None
        self._mm_len = 0

    # -- writes (sequential) ------------------------------------------------ #
    def append(self, record: bytes) -> int:
        """Append one record; return its byte offset. Sequential write."""
        if len(record) > self._MAX_RECORD:
            raise ValueError(f"record {len(record)} bytes exceeds 4GB length prefix")
        offset = self._end
        self._wf.write(_LEN.pack(len(record)))
        self._wf.write(record)
        self._wf.flush()
        self._end += 4 + len(record)
        return offset

    def append_many(self, records: list[bytes]) -> list[int]:
        """Batch append -- one flush amortised over many records."""
        offsets = []
        for rec in records:
            if len(rec) > self._MAX_RECORD:
                raise ValueError(f"record {len(rec)} bytes exceeds 4GB length prefix")
            offsets.append(self._end)
            self._wf.write(_LEN.pack(len(rec)))
            self._wf.write(rec)
            self._end += 4 + len(rec)
        self._wf.flush()
        return offsets

    # -- reads (mmap, random by offset) ------------------------------------- #
    def _ensure_map(self) -> Optional[mmap.mmap]:
        size = os.path.getsize(self.path)
        if size == 0:
            return None
        if self._mm is None or self._mm_len != size:
            if self._mm is not None:
                self._mm.close()
            with open(self.path, "rb") as rf:
                self._mm = mmap.mmap(rf.fileno(), size, access=mmap.ACCESS_READ)
            self._mm_len = size
        return self._mm

    def read_at(self, offset: int) -> bytes:
        """Random read of the record at ``offset`` via mmap (no seek syscall)."""
        mm = self._ensure_map()
        if mm is None:
            raise IndexError("empty log")
        if offset < 0 or offset + 4 > self._mm_len:
            raise IndexError(f"offset {offset} past end of log ({self._mm_len})")
        (length,) = _LEN.unpack(mm[offset:offset + 4])
        start = offset + 4
        if start + length > self._mm_len:
            # Guards against a torn/partial tail (e.g. a concurrent in-flight
            # append) being read back as a silently truncated record.
            raise ValueError("record extends past end of log (torn write?)")
        return mm[start:start + length]

    def scan(self) -> Iterator[tuple[int, bytes]]:
        """Sequential scan -> (offset, record). Streams via mmap, O(1) memory."""
        mm = self._ensure_map()
        if mm is None:
            return
        pos = 0
        while pos + 4 <= self._mm_len:
            (length,) = _LEN.unpack(mm[pos:pos + 4])
            end = pos + 4 + length
            if end > self._mm_len:
                break          # torn/incomplete tail record -> stop (crash recovery)
            yield pos, mm[pos + 4:end]
            pos = end

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        self._wf.close()


class KVStore:
    """Content-addressed append-only KV over AppendLog (mmap reads)."""

    def __init__(self, path: str):
        self.log = AppendLog(path)
        self._index: dict[str, int] = {}       # key -> latest record offset
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        for offset, rec in self.log.scan():
            klen = _LEN.unpack(rec[:4])[0]
            key = rec[4:4 + klen].decode()
            self._index[key] = offset          # last write wins

    @staticmethod
    def _encode(key: str, value: bytes) -> bytes:
        kb = key.encode()
        return _LEN.pack(len(kb)) + kb + value

    def put(self, key: str, value: bytes) -> None:
        offset = self.log.append(self._encode(key, value))
        self._index[key] = offset

    def put_many(self, items: list[tuple[str, bytes]]) -> None:
        recs = [self._encode(k, v) for k, v in items]
        offsets = self.log.append_many(recs)
        for (k, _), off in zip(items, offsets):
            self._index[k] = off

    def get(self, key: str) -> Optional[bytes]:
        off = self._index.get(key)
        if off is None:
            return None
        rec = self.log.read_at(off)
        klen = _LEN.unpack(rec[:4])[0]
        return rec[4 + klen:]

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def keys(self):
        return self._index.keys()

    def compact(self) -> None:
        """Rewrite the log with only live records; rebuild the index."""
        tmp = self.log.path + ".compact"
        if os.path.exists(tmp):
            os.remove(tmp)        # discard any leftover from a crashed compaction
        new = AppendLog(tmp)
        live = {k: self.get(k) for k in self._index}
        for k, v in live.items():
            new.append(self._encode(k, v))
        new.close()
        self.log.close()
        os.replace(tmp, self.log.path)
        self.log = AppendLog(self.log.path)
        self._index = {}
        self._rebuild_index()

    def close(self) -> None:
        self.log.close()
