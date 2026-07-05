"""Compatibility shim for the old ``benchmark.trice`` import path.

The packaged TRICE implementation now lives under ``tracerazor._trice`` so the
wheel no longer installs a generic top-level ``benchmark`` package. Source-tree
benchmark scripts can keep importing ``benchmark.trice`` while they migrate.
"""

from __future__ import annotations

import sys

from tracerazor import _trice as _impl
from tracerazor._trice import *  # noqa: F401,F403

__all__ = list(_impl.__all__)
__path__ = list(_impl.__path__)

sys.modules.setdefault("benchmark.trice", sys.modules[__name__])
