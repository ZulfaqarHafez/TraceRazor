"""Loader for the Hugging Face ``zai-org/AgentInstruct`` agent-trajectory dataset.

Mirrors :class:`benchmark.tau2_loader.Tau2Loader`: three sources, same offline
default so tests stay hermetic.

* **bundled** (default) — the vendored real sample in
  ``benchmark/data/_agentinstruct_hf_sample.py`` (9 real trajectories pulled
  from the Hub). No network, no extra deps.
* **disk** — a JSONL snapshot of dataset rows (one ``{"id","conversations"}``
  object per line), e.g. produced by ``datasets`` or the live fetch below.
* **live** — fetch rows from the Hugging Face dataset-viewer REST API
  (``https://datasets-server.huggingface.co/rows``). Reproducible wherever the
  Hub is reachable; raises a clear error when it is not.

Each loaded item is the raw dataset row; convert it to a TraceRazor trace with
:func:`tools.convert_agentinstruct.convert_row`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.data._agentinstruct_hf_sample import (
    HF_CONFIG,
    HF_DATASET,
    ROWS as _BUNDLED_ROWS,
)


class HFAgentInstructLoader:
    """Load AgentInstruct rows from the bundled sample, a JSONL snapshot, or live.

    Parameters
    ----------
    source:
        ``"bundled"`` (default), ``"disk"``, or ``"live"``.
    split:
        Dataset split to filter/fetch (``os``, ``db``, ``kg``, ``alfworld``,
        ``mind2web``, ``webshop``). ``None`` → all splits in the source.
    jsonl_path:
        Required when ``source="disk"``.
    max_rows:
        Truncate to the first N rows.
    """

    def __init__(
        self,
        source: str = "bundled",
        split: Optional[str] = None,
        jsonl_path: Optional[Path] = None,
        max_rows: Optional[int] = None,
    ) -> None:
        if source not in ("bundled", "disk", "live"):
            raise ValueError(f"unknown source '{source}'")
        self.source = source
        self.split = split
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.max_rows = max_rows

    def load(self) -> List[Dict[str, Any]]:
        if self.source == "bundled":
            rows = self._load_bundled()
        elif self.source == "disk":
            rows = self._load_disk()
        else:
            rows = self._load_live()
        if self.split is not None:
            rows = [r for r in rows if r.get("domain", r.get("split")) == self.split]
        if self.max_rows is not None:
            rows = rows[: self.max_rows]
        return rows

    def _load_bundled(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in _BUNDLED_ROWS]

    def _load_disk(self) -> List[Dict[str, Any]]:
        if self.jsonl_path is None or not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"source='disk' requires an existing JSONL snapshot, got {self.jsonl_path}"
            )
        rows = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def _load_live(self) -> List[Dict[str, Any]]:
        """Fetch rows from the Hugging Face dataset-viewer REST API.

        Uses only the stdlib so it carries no hard dependency on ``datasets``.
        The split must be specified for a live fetch.
        """
        import urllib.parse
        import urllib.request

        if self.split is None:
            raise ValueError("source='live' requires an explicit split")
        length = min(self.max_rows or 100, 100)
        params = urllib.parse.urlencode(
            {
                "dataset": HF_DATASET,
                "config": HF_CONFIG,
                "split": self.split,
                "offset": 0,
                "length": length,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{params}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                payload = json.load(resp)
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                f"live fetch of {HF_DATASET}:{self.split} failed ({exc}). "
                "The Hugging Face Hub is not reachable from this environment; "
                "use source='bundled' (default) or source='disk'."
            ) from exc
        rows = []
        for item in payload.get("rows", []):
            row = item.get("row", {})
            row.setdefault("domain", self.split)
            rows.append(row)
        return rows
