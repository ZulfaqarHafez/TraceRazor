"""Normalize trusted image-install timestamps to ``SOURCE_DATE_EPOCH``.

The native installer intentionally records wall-clock installation time for
normal hosts. OCI image construction is different: provisioning is part of a
reproducible release build, so its generated receipts and ownership ledger use
the tagged source commit's fixed epoch instead.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


MIN_ZIP_EPOCH = 315532800  # 1980-01-01; Python wheel ZIP timestamps need this.


def _timestamp() -> str:
    raw = os.environ.get("SOURCE_DATE_EPOCH", "")
    try:
        epoch = int(raw)
    except ValueError as exc:
        raise ValueError("SOURCE_DATE_EPOCH must be an integer") from exc
    if epoch < MIN_ZIP_EPOCH:
        raise ValueError("SOURCE_DATE_EPOCH must be at least 315532800")
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _normalize(value: Any, timestamp: str) -> int:
    changed = 0
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "installed_at":
                value[key] = timestamp
                changed += 1
            else:
                changed += _normalize(child, timestamp)
    elif isinstance(value, list):
        for child in value:
            changed += _normalize(child, timestamp)
    return changed


def normalize_file(path: Path, timestamp: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if _normalize(payload, timestamp) == 0:
        raise ValueError(f"{path} contains no installed_at field")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    paths = [Path(item) for item in (sys.argv[1:] if argv is None else argv)]
    if not paths:
        raise ValueError("at least one receipt or ledger path is required")
    timestamp = _timestamp()
    for path in paths:
        normalize_file(path, timestamp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
