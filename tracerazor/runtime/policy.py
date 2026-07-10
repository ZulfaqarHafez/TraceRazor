"""Runtime policy loading and enforcement eligibility."""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - package requires 3.10
    tomllib = None  # type: ignore[assignment]

from .models import PrivacyMode, RuntimeEvent, TaskResult


_MODES = {"off", "passive", "coach", "enforce"}
_CAPTURE_MODES = {"auto", "manual", "off"}
_TOML_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _bool_field(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"policy.{name} must be a boolean")
    return value


def _minimal_toml_load(payload: bytes) -> dict[str, Any]:
    """Parse the small policy subset on Python 3.10 without a dependency.

    The policy format needs tables plus string, boolean, and integer scalars;
    unsupported TOML features fail closed instead of being guessed.
    """

    root: dict[str, Any] = {}
    current = root
    for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            if not section:
                raise ValueError(f"empty TOML table at line {line_number}")
            current = root
            for component in section.split("."):
                if not _TOML_KEY_RE.fullmatch(component):
                    raise ValueError(f"unsupported TOML table at line {line_number}")
                child = current.setdefault(component, {})
                if not isinstance(child, dict):
                    raise ValueError(f"TOML table conflicts with a value at line {line_number}")
                current = child
            continue
        if "=" not in line:
            raise ValueError(f"invalid TOML policy line {line_number}")
        key, raw_value = (part.strip() for part in line.split("=", 1))
        if not _TOML_KEY_RE.fullmatch(key):
            raise ValueError(f"unsupported TOML key at line {line_number}")
        # Policy strings do not need inline '#'; reject ambiguous unquoted use.
        if raw_value.startswith('"'):
            try:
                value: Any = json.loads(raw_value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid TOML string at line {line_number}") from exc
        elif raw_value.startswith("'") and raw_value.endswith("'"):
            value = raw_value[1:-1]
        elif raw_value == "true":
            value = True
        elif raw_value == "false":
            value = False
        else:
            try:
                value = int(raw_value)
            except ValueError as exc:
                raise ValueError(f"unsupported TOML value at line {line_number}") from exc
        current[key] = value
    return root


@dataclass(frozen=True)
class AuditPolicy:
    """Project policy for capture, persistence, and optional enforcement.

    The default is intentionally safe: local-only redacted artifacts, coaching,
    and no enforcement.  Merely selecting ``mode="enforce"`` is insufficient;
    the policy must also explicitly enable enforcement and name a verifier.
    """

    schema_version: int = 1
    mode: str = "coach"
    capture: str = "auto"
    hermetic: bool = True
    privacy: PrivacyMode = PrivacyMode.LOCAL_REDACTED
    persist_raw_content: bool = False
    artifact_dir: str = ".tracerazor/runs"
    min_steps: int = 5
    verifier: str | None = None
    enforcement_enabled: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only tracerazor policy schema_version = 1 is supported")
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}")
        if self.capture not in _CAPTURE_MODES:
            raise ValueError(f"capture must be one of {sorted(_CAPTURE_MODES)}")
        object.__setattr__(self, "privacy", PrivacyMode(self.privacy))
        if self.persist_raw_content and self.privacy is not PrivacyMode.RAW:
            raise ValueError("persist_raw_content requires privacy = 'raw'")
        if not isinstance(self.artifact_dir, str) or not self.artifact_dir.strip():
            raise ValueError("artifact_dir must be a non-empty path")
        if ".." in Path(self.artifact_dir).parts:
            raise ValueError("artifact_dir must not contain parent-directory traversal")
        if isinstance(self.min_steps, bool) or not isinstance(self.min_steps, int) or self.min_steps < 2:
            raise ValueError("min_steps must be an integer greater than or equal to 2")
        if self.verifier is not None and not str(self.verifier).strip():
            object.__setattr__(self, "verifier", None)

    @property
    def captures(self) -> bool:
        return self.mode != "off" and self.capture != "off"

    @property
    def configured_for_enforcement(self) -> bool:
        return self.mode == "enforce" and self.enforcement_enabled and bool(self.verifier)

    def enforcement_eligibility(
        self,
        events: list[RuntimeEvent],
        task: TaskResult | None,
        *,
        partial: bool = False,
    ) -> tuple[bool, list[str]]:
        """Return whether a run may make a hard decision and why not."""

        reasons: list[str] = []
        if not self.configured_for_enforcement:
            reasons.append("enforcement_not_explicitly_configured")
        if partial:
            reasons.append("partial_run")
        if any(not event.enforcement_eligible for event in events):
            reasons.append("degraded_or_non_provider_token_usage")
        if task is None or not task.verified:
            reasons.append("task_outcome_not_verified")
        elif self.verifier and task.verifier != self.verifier:
            reasons.append("verifier_mismatch")
        # ``TaskResult`` is supplied by the instrumented process and is only a
        # claim.  Until a trusted verifier runner issues an unforgeable receipt,
        # it cannot authorize a hard gate or mutation.  Evidence embedded by a
        # caller is deliberately not accepted as proof of execution.
        reasons.append("trusted_verifier_receipt_missing")
        return not reasons, reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "capture": self.capture,
            "hermetic": self.hermetic,
            "privacy": self.privacy.value,
            "persist_raw_content": self.persist_raw_content,
            "artifact_dir": self.artifact_dir,
            "min_steps": self.min_steps,
            "quality": {"verifier": self.verifier or ""},
            "enforcement": {"enabled": self.enforcement_enabled},
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AuditPolicy":
        quality = value.get("quality") or {}
        enforcement = value.get("enforcement") or {}
        if not isinstance(quality, Mapping):
            raise ValueError("policy.quality must be a table")
        if not isinstance(enforcement, Mapping):
            raise ValueError("policy.enforcement must be a table")
        verifier = quality.get("verifier")
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            mode=str(value.get("mode", "coach")),
            capture=str(value.get("capture", "auto")),
            hermetic=_bool_field(value.get("hermetic", True), "hermetic"),
            privacy=PrivacyMode(value.get("privacy", "local-redacted")),
            persist_raw_content=_bool_field(
                value.get("persist_raw_content", False), "persist_raw_content"
            ),
            artifact_dir=os.fspath(value.get("artifact_dir", ".tracerazor/runs")),
            min_steps=int(value.get("min_steps", 5)),
            verifier=str(verifier).strip() if verifier else None,
            enforcement_enabled=_bool_field(enforcement.get("enabled", False), "enforcement.enabled"),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str] = "tracerazor.toml") -> "AuditPolicy":
        policy_path = Path(path)
        try:
            payload = policy_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"could not read TraceRazor policy {policy_path}: {exc}") from exc
        if tomllib is None:  # pragma: no cover - exercised on Python 3.10
            value = _minimal_toml_load(payload)
        else:
            value = tomllib.loads(payload.decode("utf-8"))
        if "tracerazor" in value:
            section = value["tracerazor"]
            if not isinstance(section, Mapping):
                raise ValueError("[tracerazor] policy section must be a table")
            value = section
        policy = cls.from_mapping(value)
        if Path(policy.artifact_dir).is_absolute():
            raise ValueError("artifact_dir in tracerazor.toml must be relative to the policy root")
        return policy


__all__ = ["AuditPolicy"]
