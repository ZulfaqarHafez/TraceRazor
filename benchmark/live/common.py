"""Shared helpers for the live case-study kit (run_live / reconvert)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

#: Trailing release-date suffix on model ids, e.g. "claude-haiku-4-5-20251001".
_MODEL_DATE_RE = re.compile(r"-\d{8}$")


def short_model(model: str) -> str:
    """Model id without its release-date suffix, for trace agent names."""
    return _MODEL_DATE_RE.sub("", model)


def pytest_passes(sandbox: Path) -> bool:
    """The objective task outcome: does the sandbox's test suite pass?"""
    check = subprocess.run(
        ["python3", "-m", "pytest", "-q", "--tb=no"],
        cwd=sandbox,
        capture_output=True,
        timeout=120,
    )
    return check.returncode == 0
