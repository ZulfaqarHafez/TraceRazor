"""Adapter contracts for deterministic TRICE live rollouts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class RepairAdapter(Protocol):
    """Deterministic edit adapter used by ``run_live_learning_loop``."""

    name: str

    def apply_fix(self, task: Any, workspace: Path) -> list[str]:
        """Apply a deterministic intervention and return modified relative paths."""


@dataclass(frozen=True)
class PatchEdit:
    op: str
    path: str
    old: str | None = None
    new: str | None = None
    content: str | None = None


@dataclass
class JsonPatchAdapter:
    """Apply a declarative JSON patch spec in a fresh workspace.

    Supported edit operations:
    - ``replace``: replace ``old`` with ``new`` in ``path``.
    - ``write``: write ``content`` to ``path``.

    The adapter is intentionally small. It is for deterministic evaluation and
    evidence generation, not a general patch language.
    """

    edits: list[PatchEdit]
    name: str = "json-patch-adapter"
    allow_test_edits: bool = False
    forbidden_prefixes: tuple[str, ...] = ("tests/", "test/")
    applied_empty_ok: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "JsonPatchAdapter":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JsonPatchAdapter":
        edits = [
            PatchEdit(
                op=str(e.get("op") or e.get("type") or ""),
                path=str(e["path"]),
                old=e.get("old"),
                new=e.get("new"),
                content=e.get("content"),
            )
            for e in data.get("edits", [])
        ]
        return cls(
            edits=edits,
            name=str(data.get("name") or "json-patch-adapter"),
            allow_test_edits=bool(data.get("allow_test_edits", False)),
            forbidden_prefixes=tuple(data.get("forbidden_prefixes") or ("tests/", "test/")),
            applied_empty_ok=bool(data.get("applied_empty_ok", False)),
            metadata=dict(data.get("metadata") or {}),
        )

    def apply_fix(self, task: Any, workspace: Path) -> list[str]:
        if not self.edits:
            raise ValueError("patch spec has no edits")
        changed: list[str] = []
        for edit in self.edits:
            rel = _clean_rel_path(edit.path)
            if not self.allow_test_edits and _is_forbidden(rel, self.forbidden_prefixes):
                raise ValueError(f"refusing to edit forbidden path: {rel}")
            target = _resolve_in_workspace(workspace, rel)
            if edit.op == "replace":
                if edit.old is None or edit.new is None:
                    raise ValueError(f"replace edit for {rel} needs old and new")
                text = target.read_text(encoding="utf-8")
                if edit.old not in text:
                    if self.applied_empty_ok:
                        continue
                    raise ValueError(f"old text not found in {rel}")
                target.write_text(text.replace(edit.old, edit.new), encoding="utf-8")
                changed.append(rel)
            elif edit.op == "write":
                if edit.content is None:
                    raise ValueError(f"write edit for {rel} needs content")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(edit.content, encoding="utf-8")
                changed.append(rel)
            else:
                raise ValueError(f"unsupported patch op: {edit.op!r}")
        return sorted(set(changed))


def _clean_rel_path(path: str) -> str:
    rel = Path(path.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"patch path must stay inside workspace: {path}")
    return rel.as_posix()


def _resolve_in_workspace(workspace: Path, rel: str) -> Path:
    root = workspace.resolve()
    target = (root / rel).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"patch path escapes workspace: {rel}")
    return target


def _is_forbidden(rel: str, prefixes: tuple[str, ...]) -> bool:
    rel_l = rel.lower().replace("\\", "/")
    return any(rel_l == p.rstrip("/") or rel_l.startswith(p.lower()) for p in prefixes)
