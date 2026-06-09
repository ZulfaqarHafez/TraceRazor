#!/usr/bin/env python3
"""Convert messages-format agent trajectories into TraceRazor traces + a manifest.

Public agent-trajectory datasets (e.g. SWE-agent trajectories, CC-Bench,
AgentTrove) store each run as a conversation in OpenAI "messages" format
(`{"role","content","tool_calls"}`) or ShareGPT format (`{"from","value"}`),
plus metadata such as an instance id, the model/config, and a resolved/correct
flag. This connector turns each such record into a TraceRazor raw trace and,
for instances solved by more than one configuration, emits before/after pairs
(verbose run vs lean run, both successful) so `calibrate.py` can fit weights to
measured recoverable waste on real runs.

Input: a JSONL file, one trajectory record per line. Field names are
configurable so this fits whatever a given dataset uses.

Usage:
    python -m calibration.sources.from_messages --jsonl traj.jsonl \
        --out-dir converted --manifest manifest.json \
        --id-field instance_id --model-field model --resolved-field resolved \
        --messages-field messages
    python -m calibration.calibrate --dataset manifest.json --out config/tas_weights.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def _ntokens(text: str) -> int:
        return len(_ENC.encode(text or ""))
except Exception:  # pragma: no cover - fallback when tiktoken is unavailable
    def _ntokens(text: str) -> int:
        # ~1.3 tokens/word is a reasonable rough estimate.
        return int(len((text or "").split()) * 1.3) + 1


def _norm_messages(rec: Dict[str, Any], field: str) -> List[Dict[str, Any]]:
    """Return a list of {role, content, tool_calls} from an OpenAI- or
    ShareGPT-style messages field."""
    raw = rec.get(field)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        if "role" in m:  # OpenAI format
            out.append({"role": m.get("role", ""), "content": m.get("content") or "",
                        "tool_calls": m.get("tool_calls")})
        elif "from" in m:  # ShareGPT format
            role = {"gpt": "assistant", "human": "user", "system": "system",
                    "tool": "tool", "observation": "tool"}.get(m.get("from", ""), m.get("from", ""))
            out.append({"role": role, "content": m.get("value") or "", "tool_calls": None})
    return out


def _tool_name(content: str) -> Optional[str]:
    """Heuristically pull a tool/command name from a tool-call content string."""
    m = re.search(r"[A-Za-z_][A-Za-z0-9_\-]{1,40}", content or "")
    return m.group(0) if m else None


def messages_to_trace(rec: Dict[str, Any], args) -> Optional[Dict[str, Any]]:
    msgs = _norm_messages(rec, args.messages_field)
    steps: List[Dict[str, Any]] = []
    sid = 1
    for m in msgs:
        role, content = m["role"], (m["content"] or "").strip()
        if role == "assistant":
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                    name = fn.get("name") or "tool"
                    cargs = fn.get("arguments") or ""
                    text = f"{name} {cargs}".strip()
                    steps.append({"id": sid, "step_type": "tool_call", "content": text,
                                  "tokens": _ntokens(text), "tool_name": name,
                                  "tool_success": True})
                    sid += 1
                if content:
                    steps.append({"id": sid, "step_type": "reasoning", "content": content,
                                  "tokens": _ntokens(content)})
                    sid += 1
            elif content:
                steps.append({"id": sid, "step_type": "reasoning", "content": content,
                              "tokens": _ntokens(content)})
                sid += 1
        elif role == "tool":
            # A tool result; attach as a tool_call step marking success/failure.
            ok = not re.search(r"\b(error|traceback|exception|failed)\b", content, re.I)
            steps.append({"id": sid, "step_type": "tool_call",
                          "content": content[:500], "tokens": _ntokens(content),
                          "tool_name": _tool_name(content) or "tool", "tool_success": bool(ok)})
            sid += 1
        # system/user messages are context, not agent work; skipped.
    if len(steps) < 5:
        return None
    instance = str(rec.get(args.id_field, "unknown"))
    model = str(rec.get(args.model_field, "model"))
    return {
        "trace_id": f"{instance}__{model}",
        "agent_name": model,
        "framework": "raw",
        "task_value_score": 1.0,
        "steps": steps,
        "_instance": instance,
        "_model": model,
    }


def _is_resolved(rec: Dict[str, Any], field: str) -> bool:
    v = rec.get(field)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v > 0
    if isinstance(v, str):
        return v.strip().lower() in ("true", "resolved", "1", "yes", "pass", "passed")
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert messages-format trajectories to TraceRazor traces + manifest.")
    ap.add_argument("--jsonl", required=True, type=Path, help="trajectory records, one JSON per line")
    ap.add_argument("--out-dir", type=Path, default=Path("calibration/converted"))
    ap.add_argument("--manifest", type=Path, default=Path("manifest.json"))
    ap.add_argument("--id-field", default="instance_id")
    ap.add_argument("--model-field", default="model")
    ap.add_argument("--resolved-field", default="resolved")
    ap.add_argument("--messages-field", default="messages")
    ap.add_argument("--require-resolved", action="store_true", default=True,
                    help="only pair runs marked resolved/correct (default on)")
    args = ap.parse_args(argv)

    if not args.jsonl.is_file():
        sys.exit(f"not found: {args.jsonl}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_instance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    n_records = n_written = 0
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        n_records += 1
        rec = json.loads(line)
        if args.require_resolved and not _is_resolved(rec, args.resolved_field):
            continue
        trace = messages_to_trace(rec, args)
        if trace is None:
            continue
        instance, model = trace.pop("_instance"), trace.pop("_model")
        total = sum(s["tokens"] for s in trace["steps"])
        path = args.out_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', instance+'__'+model)}.json"
        path.write_text(json.dumps(trace, indent=2))
        n_written += 1
        by_instance[instance].append({"path": str(path.resolve()), "tokens": total, "model": model})

    # Pair: for each instance solved by >=2 configs, before=most tokens, after=fewest.
    entries = []
    for instance, runs in by_instance.items():
        if len(runs) < 2:
            continue
        runs.sort(key=lambda r: r["tokens"])
        after, before = runs[0], runs[-1]
        if before["tokens"] > after["tokens"]:
            entries.append({"before": before["path"], "after": after["path"]})

    if not entries:
        sys.exit(
            f"converted {n_written}/{n_records} resolved trajectories but found no "
            "instance solved by >=2 configs with differing tokens; need multi-config "
            "runs on shared tasks to form before/after pairs."
        )
    args.manifest.write_text(json.dumps(
        {"name": args.jsonl.stem, "entries": entries}, indent=2) + "\n")
    print(f"Converted {n_written}/{n_records} resolved trajectories; "
          f"{len(entries)} before/after pairs across {len(entries)} instances.")
    print(f"Manifest -> {args.manifest}")
    print(f"Next: python -m calibration.calibrate --dataset {args.manifest} "
          f"--out config/tas_weights.json --report config/calibration_report.md --prior default --l2 0.1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
