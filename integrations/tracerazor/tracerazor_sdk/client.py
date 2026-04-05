"""
TraceRazor client — CLI subprocess mode and HTTP mode.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceRazorReport:
    """Parsed result of a tracerazor audit."""

    trace_id: str
    agent_name: str
    framework: str
    total_steps: int
    total_tokens: int
    tas_score: float
    grade: str
    passes: bool
    threshold: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    savings: Dict[str, Any] = field(default_factory=dict)
    fixes: List[Dict] = field(default_factory=list)
    anomalies: List[Dict] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary."""
        saved = self.savings.get("tokens_saved", 0)
        pct = self.savings.get("reduction_pct", 0.0)
        return (
            f"TAS {self.tas_score:.1f}/100 [{self.grade}] | "
            f"{self.total_steps} steps, {self.total_tokens} tokens | "
            f"Saved {saved} tokens ({pct:.0f}%)"
        )

    def markdown(self) -> str:
        """Full markdown report (same as CLI output)."""
        return self.raw.get("report_markdown") or self._build_markdown()

    def _build_markdown(self) -> str:
        sep = "-" * 54
        s = self.metrics
        lines = [
            "TRACERAZOR REPORT",
            sep,
            f"Trace:   {self.trace_id}",
            f"Agent:   {self.agent_name}",
            f"Steps:   {self.total_steps}   Tokens: {self.total_tokens}",
            sep,
            f"TRACERAZOR SCORE:  {self.tas_score:.0f} / 100  [{self.grade.upper()}]",
            sep,
        ]
        for code in ("srr", "ldi", "tca", "rda", "isr", "tur", "cce", "dbo"):
            m = s.get(code, {})
            if m:
                status = "PASS" if m.get("pass") else "FAIL"
                lines.append(f"{code.upper():<6} {m.get('score', 0):.3f}   {status}")
        if self.savings:
            lines += [
                sep,
                "SAVINGS ESTIMATE",
                f"  Tokens saved:  {self.savings.get('tokens_saved', 0)}  "
                f"({self.savings.get('reduction_pct', 0):.1f}% reduction)",
                f"  Cost saved:    ${self.savings.get('cost_saved_per_run_usd', 0):.4f}/run",
                f"  At 50K/month:  ${self.savings.get('monthly_savings_usd', 0):.2f}/month",
            ]
        if self.fixes:
            lines += [sep, "AUTO-GENERATED FIXES"]
            for i, fix in enumerate(self.fixes, 1):
                lines.append(
                    f"  Fix {i}: [{fix.get('fix_type')}] → {fix.get('target')}\n"
                    f"    Patch: {fix.get('patch', '')[:120]}\n"
                    f"    Est. savings: {fix.get('estimated_token_savings', 0)} tokens/run"
                )
        if self.anomalies:
            lines += [sep, "ANOMALY ALERTS"]
            for a in self.anomalies:
                direction = "REGRESSION" if a.get("z_score", 0) < 0 else "IMPROVEMENT"
                lines.append(
                    f"  [{direction}] {a.get('metric')}: {a.get('value'):.1f} "
                    f"(z={a.get('z_score'):.1f})"
                )
        lines.append(sep)
        return "\n".join(lines)

    def assert_passes(self) -> None:
        """Raise AssertionError if TAS is below threshold. Use in CI/CD."""
        if not self.passes:
            raise AssertionError(
                f"TraceRazor: TAS {self.tas_score:.1f} is below "
                f"threshold {self.threshold}.\n\n{self.summary()}"
            )


class TraceRazorClient:
    """
    Submit a trace for analysis.

    CLI mode (default): calls the local tracerazor binary.
      - No server needed.
      - Requires the binary to be built or installed.

    HTTP mode: POSTs to a running tracerazor-server.
      - No binary needed on the machine running your agent.
      - Start the server once: ./tracerazor-server

    Args:
        bin_path:  Path to the tracerazor binary. Auto-detected if None.
                   Ignored when `server` is set.
        server:    Base URL of a running tracerazor-server, e.g.
                   "http://localhost:8080". When set, HTTP mode is used.
        threshold: Minimum TAS score for assert_passes() (default 70).
    """

    def __init__(
        self,
        bin_path: Optional[str] = None,
        server: Optional[str] = None,
        threshold: float = 70.0,
    ):
        self._server = server.rstrip("/") if server else None
        self._bin = None if self._server else (bin_path or self._find_binary())
        self._threshold = threshold

    def analyse(self, trace: Dict[str, Any]) -> TraceRazorReport:
        """Submit the trace and return a TraceRazorReport."""
        if self._server:
            return self._analyse_http(trace)
        return self._analyse_cli(trace)

    # ── CLI mode ──────────────────────────────────────────────────────────────

    def _analyse_cli(self, trace: Dict[str, Any]) -> TraceRazorReport:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(trace, f, indent=2)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [
                    self._bin,
                    "audit",
                    tmp_path,
                    "--format", "json",
                    "--threshold", str(self._threshold),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            # Exit code 1 = below threshold (still valid JSON output).
            if result.returncode not in (0, 1):
                raise RuntimeError(
                    f"tracerazor exited with code {result.returncode}:\n{result.stderr}"
                )
            data = json.loads(result.stdout)
            return self._parse_cli_report(data)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _parse_cli_report(self, data: Dict[str, Any]) -> TraceRazorReport:
        score = data.get("score", {})
        tas = score.get("score", 0.0)
        return TraceRazorReport(
            trace_id=data.get("trace_id", ""),
            agent_name=data.get("agent_name", ""),
            framework=data.get("framework", ""),
            total_steps=data.get("total_steps", 0),
            total_tokens=data.get("total_tokens", 0),
            tas_score=tas,
            grade=str(score.get("grade", "Unknown")),
            passes=tas >= self._threshold,
            threshold=self._threshold,
            metrics=score,
            savings=data.get("savings", {}),
            fixes=data.get("fixes", []),
            anomalies=data.get("anomalies", []),
            raw=data,
        )

    # ── HTTP mode ─────────────────────────────────────────────────────────────

    def _analyse_http(self, trace: Dict[str, Any]) -> TraceRazorReport:
        try:
            import requests
        except ImportError:
            raise ImportError(
                "HTTP mode requires the 'requests' library.\n"
                "Install with: pip install tracerazor[http]"
            )

        resp = requests.post(
            f"{self._server}/api/audit",
            json={"trace": trace},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return self._parse_http_report(data)

    def _parse_http_report(self, data: Dict[str, Any]) -> TraceRazorReport:
        tas = data.get("tas_score", 0.0)
        return TraceRazorReport(
            trace_id=data.get("trace_id", ""),
            agent_name=data.get("agent_name", ""),
            framework=data.get("framework", ""),
            total_steps=0,   # not in HTTP response top-level
            total_tokens=0,
            tas_score=tas,
            grade=data.get("grade", "Unknown"),
            passes=tas >= self._threshold,
            threshold=self._threshold,
            metrics={},
            savings={},
            fixes=[],
            anomalies=data.get("anomalies", []),
            raw=data,
        )

    # ── Binary discovery ──────────────────────────────────────────────────────

    @staticmethod
    def _find_binary() -> str:
        env_path = os.environ.get("TRACERAZOR_BIN")
        if env_path and os.path.isfile(env_path):
            return env_path

        found = shutil.which("tracerazor") or shutil.which("tracerazor.exe")
        if found:
            return found

        # Dev layout: integrations/tracerazor/tracerazor_sdk/ → ../../../target/release/
        here = os.path.dirname(os.path.abspath(__file__))
        for rel in [
            "../../../../target/release/tracerazor.exe",
            "../../../../target/release/tracerazor",
            "../../../../target/debug/tracerazor.exe",
            "../../../../target/debug/tracerazor",
        ]:
            candidate = os.path.normpath(os.path.join(here, rel))
            if os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError(
            "tracerazor binary not found.\n"
            "Options:\n"
            "  1. Set TRACERAZOR_BIN=/path/to/tracerazor\n"
            "  2. Build from source: cargo build --release\n"
            "  3. Use HTTP mode: Tracer(server='http://localhost:8080')"
        )
