"""stdio MCP server exposing TraceRazor's machine surface to agent hosts.

The four tools shell out to the Rust `tracerazor` CLI (audit/import/verify) and
read the Claude Code session index; every tool returns JSON-serializable data.
Binary resolution is shared with the pip console launcher
(:mod:`tracerazor._launcher`) so a missing auditor raises one identical,
copy-pasteable recovery message everywhere.

The MCP SDK is an optional dependency (``pip install "tracerazor[mcp]"``). The
tool functions and ``--selftest`` inspection do not import it; only starting the
server (:func:`_build_server` / :func:`main`) does. Register with a host via:

    claude mcp add tracerazor -- tracerazor-mcp
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from tracerazor._launcher import find_binary, recovery_message


class BinaryNotFoundError(RuntimeError):
    """Raised when no auditor binary is available. Its message is the exact,
    copy-pasteable recovery text from the launcher."""


def _resolve_binary() -> str:
    """Return the auditor binary path or raise a teaching error.

    An explicitly-set ``TRACERAZOR_BIN`` that does not point at a file is a hard
    error (rather than silently searching elsewhere) so a misconfigured path is
    reported instead of masked by a stray PATH/checkout binary.
    """
    env = os.environ.get("TRACERAZOR_BIN")
    if env and not os.path.isfile(env):
        raise BinaryNotFoundError(recovery_message())
    binary = find_binary()
    if binary is None:
        raise BinaryNotFoundError(recovery_message())
    return binary


def _run(args: list[str]) -> subprocess.CompletedProcess:
    # stdin must be closed off: under an MCP stdio host the server's stdin is
    # the live protocol pipe, and a child that inherits it (e.g. the pip
    # console-script wrapper chain on Windows) blocks forever on a native read,
    # deadlocking the tool call.
    return subprocess.run(
        args,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _loads(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


# ── tools ───────────────────────────────────────────────────────────────────


def audit_trace(
    path: str,
    hermetic: bool = True,
    min_steps: int | None = None,
    threshold: int | None = None,
) -> dict:
    """Audit a trace file and return the parsed JSON report.

    Runs `tracerazor audit <path> --format json` (hermetic by default, so the
    score is a pure function of the trace, config, and version). Exit 0 and 1
    are both success — exit 1 only means an explicit `--threshold` gate failed,
    surfaced as `passed: false`. Exit 2 is returned as a structured error.
    """
    binary = _resolve_binary()
    args = [binary, "audit", str(path), "--format", "json"]
    if hermetic:
        args.append("--hermetic")
    if min_steps is not None:
        args += ["--min-steps", str(min_steps)]
    if threshold is not None:
        args += ["--threshold", str(threshold)]
    res = _run(args)
    if res.returncode == 2:
        return {
            "error": "audit failed",
            "exit_code": 2,
            "stderr": (res.stderr or "").strip(),
        }
    report = _loads(res.stdout)
    if not isinstance(report, dict):
        # No report on stdout — e.g. the trace is below --min-steps (a Notice
        # is written to stderr and the CLI exits 0 without auditing).
        return {
            "passed": res.returncode == 0,
            "audited": False,
            "message": (res.stderr or "").strip()
            or "no report produced (trace below --min-steps?)",
            "exit_code": res.returncode,
        }
    report["passed"] = res.returncode == 0
    return report


def convert_transcript(path: str, format: str = "auto") -> dict:
    """Normalize an external trace export into a TraceRazor trace (JSON).

    `.jsonl` inputs (and an explicit `claude-code` format) go through
    `tracerazor claude convert`; everything else through
    `tracerazor import <path> --from <format>`. Both print the trace JSON to
    stdout, which is parsed and returned.
    """
    binary = _resolve_binary()
    if format == "claude-code" or str(path).endswith(".jsonl"):
        args = [binary, "claude", "convert", str(path)]
    else:
        args = [binary, "import", str(path), "--from", format]
    res = _run(args)
    if res.returncode != 0:
        return {
            "error": "convert failed",
            "exit_code": res.returncode,
            "stderr": (res.stderr or "").strip(),
        }
    trace = _loads(res.stdout)
    if trace is None:
        return {
            "error": "converter produced no JSON on stdout",
            "exit_code": res.returncode,
            "stderr": (res.stderr or "").strip(),
        }
    return trace


def list_claude_sessions(cwd: str = ".") -> list:
    """Return the parsed `.tracerazor/claude-code/index.json` under `cwd`.

    This is the index written by the Claude Code SessionEnd hook. Returns an
    empty list when no sessions have been audited yet. Needs no auditor binary.
    """
    index_path = os.path.join(cwd, ".tracerazor", "claude-code", "index.json")
    if not os.path.isfile(index_path):
        return []
    try:
        with open(index_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def verify_report(report_path: str, trace_path: str | None = None) -> dict:
    """Re-verify a historical report against its trace / evidence bundle.

    Feature-detects a JSON verify mode: tries `verify ... --format json` first
    and returns its parsed verdict when present; otherwise falls back to the
    text mode and maps the exit code (0 = verified, 1 = tampered, 2 = error).
    """
    binary = _resolve_binary()
    base = [binary, "verify", str(report_path)]
    if trace_path is not None:
        base.append(str(trace_path))

    # Probe for the (optional) structured JSON verify mode.
    probe = _run(base + ["--format", "json"])
    if probe.returncode in (0, 1):
        parsed = _loads(probe.stdout)
        if isinstance(parsed, dict):
            return parsed

    # Fallback: exit-code + text semantics for CLIs without `--format json`.
    res = _run(base)
    status = {0: "verified", 1: "tampered"}.get(res.returncode, "error")
    return {
        "status": status,
        "exit_code": res.returncode,
        "stdout": (res.stdout or "").strip(),
        "stderr": (res.stderr or "").strip(),
    }


# Ordered (name, callable, description) — the single source of truth for both
# server registration and `--selftest`, so the advertised catalog can never
# drift from what is registered.
TOOL_SPECS = [
    (
        "audit_trace",
        audit_trace,
        "Audit a trace file and return the parsed JSON report (hermetic by "
        "default). Exit 1 => passed:false (threshold gate); exit 2 => error.",
    ),
    (
        "convert_transcript",
        convert_transcript,
        "Normalize an external trace export (LangSmith/Langfuse/Phoenix/OTel/"
        "Claude Code .jsonl/raw) into a TraceRazor trace JSON.",
    ),
    (
        "list_claude_sessions",
        list_claude_sessions,
        "List audited Claude Code sessions from .tracerazor/claude-code/"
        "index.json under the given cwd (empty list if none).",
    ),
    (
        "verify_report",
        verify_report,
        "Re-verify a report against its trace or evidence bundle; returns the "
        "verify verdict (feature-detects a JSON mode, falls back to exit code).",
    ),
]


def _tool_catalog() -> list[dict]:
    return [{"name": name, "description": desc} for name, _fn, desc in TOOL_SPECS]


def _build_server():
    """Construct the FastMCP server with all tools registered.

    Imports the MCP SDK lazily so the module (and `--selftest`) stay importable
    without it. Raises ImportError if the SDK is not installed.
    """
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("tracerazor")
    for name, fn, desc in TOOL_SPECS:
        server.tool(name=name, description=desc)(fn)
    return server


def _selftest() -> int:
    """Print the tool catalog as JSON without starting the server.

    Building the server first proves the SDK is importable and the tools
    register; on a missing SDK, emit the install hint and exit 1.
    """
    try:
        _build_server()
    except ImportError:
        print(
            json.dumps(
                {
                    "error": "the MCP SDK is not installed",
                    "install": 'pip install "tracerazor[mcp]"',
                }
            )
        )
        return 1
    print(json.dumps(_tool_catalog(), indent=2))
    return 0


def main() -> int:
    if "--selftest" in sys.argv[1:]:
        return _selftest()
    try:
        server = _build_server()
    except ImportError:
        sys.stderr.write(
            'tracerazor-mcp: the MCP SDK is not installed. '
            'Install it with: pip install "tracerazor[mcp]"\n'
        )
        return 1
    server.run()  # stdio transport (blocks until the host disconnects)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
