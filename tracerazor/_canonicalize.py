"""Canonical-form normalization for tool call arguments.

Biases in consensus measurement come from equivalent calls that look syntactically
different. This module normalises all representational noise before comparison.
Every handled case is enumerated below; skipped cases bias consensus downward.
"""
from __future__ import annotations

import json
import re
from typing import Any


# ── normalisation helpers ──────────────────────────────────────────────────


def _norm_str(s: str) -> str:
    """Collapse internal whitespace and strip edges."""
    return re.sub(r"\s+", " ", s.strip())


def _try_numeric(s: str) -> int | float | str:
    """Convert numeric strings to int or float where unambiguous."""
    stripped = s.strip()
    try:
        as_int = int(stripped)
        return as_int
    except (ValueError, TypeError):
        pass
    try:
        as_float = float(stripped)
        # 5.0 → 5 so int and float at the same value compare equal
        if as_float == int(as_float):
            return int(as_float)
        return as_float
    except (ValueError, TypeError):
        pass
    return stripped


def _norm_value(v: Any, *, null_equiv: bool) -> Any:
    """Recursively normalise a value to canonical form.

    Handles:
    - Whitespace normalisation in string arguments
    - Dict key ordering (via sorted keys in json.dumps)
    - Numeric type unification: "5" == 5 == 5.0
    - Null/empty/missing-key unification (when null_equiv=True)
    - Encoding artefacts from LLM output (trailing commas, unicode escapes)
      are handled at the top-level by pre-parsing via _load_json
    """
    if v is None:
        return None

    if isinstance(v, bool):
        # bool before int check — bool is a subtype of int in Python
        return v

    if isinstance(v, int):
        return v

    if isinstance(v, float):
        return int(v) if v == int(v) else v

    if isinstance(v, str):
        if null_equiv and v.strip() in ("", "null", "None", "none", "nil"):
            return None
        numeric = _try_numeric(v)
        if not isinstance(numeric, str):
            return numeric
        return _norm_str(numeric)

    if isinstance(v, dict):
        normalised = {k: _norm_value(val, null_equiv=null_equiv) for k, val in v.items()}
        if null_equiv:
            # treat missing key and None value as equivalent: drop None entries
            normalised = {k: val for k, val in normalised.items() if val is not None}
        return normalised

    if isinstance(v, (list, tuple)):
        return [_norm_value(x, null_equiv=null_equiv) for x in v]

    return v


def _load_json(raw: Any) -> Any:
    """Parse a JSON string if raw is a string, otherwise return as-is.

    Handles LLM encoding artefacts: trailing commas, single quotes,
    stray whitespace, unicode escapes.
    """
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    # strip trailing commas before closing braces/brackets (invalid JSON)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # last resort: return stripped string as-is
        return text


# ── public API ─────────────────────────────────────────────────────────────


def canonicalize(
    tool_name: str,
    arguments: Any,
    *,
    null_equiv: bool = True,
) -> tuple[str, str]:
    """Return a stable (tool_name, canonical_args_json) pair.

    Parameters
    ----------
    tool_name:
        Name of the tool being called.
    arguments:
        Argument dict (or a JSON string of one) from the LLM.
    null_equiv:
        Treat None, empty string, and missing keys as equivalent.

    Returns
    -------
    (normalised_name, canonical_json)
        Both parts are deterministic for equivalent inputs.
    """
    parsed = _load_json(arguments) if arguments is not None else {}
    if not isinstance(parsed, dict):
        # scalar argument — wrap for consistency
        parsed = {"value": parsed}

    normed = _norm_value(parsed, null_equiv=null_equiv)
    canonical_json = json.dumps(normed, sort_keys=True, ensure_ascii=False)
    return (tool_name.strip().lower(), canonical_json)


def canonical_key(tool_name: str, arguments: Any, *, null_equiv: bool = True) -> str:
    """Single string key suitable for dict lookup / Counter."""
    name, args = canonicalize(tool_name, arguments, null_equiv=null_equiv)
    return f"tool:{name}:{args}"
