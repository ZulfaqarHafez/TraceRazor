"""
TraceRazor CLI client.

Serialises a trace dict to a temp JSON file and invokes the tracerazor
CLI binary, then parses the resulting JSON report.
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
    """Parsed output from a tracerazor audit run."""

    trace_id: str
    agent_name: str
    framework: str
    total_steps: int
    total_tokens: int
    tas_score: float
    grade: str
    vae_score: float
    passes: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    diff: List[Dict] = field(default_factory=list)
    savings: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def markdown(self) -> str:
        """Re-generate a concise markdown summary from the parsed report."""
        sep = "-" * 54
        lines = [
            "TRACERAZOR REPORT",
            sep,
            f"Trace:   {self.trace_id}",
            f"Agent:   {self.agent_name}",
            f"Steps:   {self.total_steps}   Tokens: {self.total_tokens}",
            sep,
            f"TRACERAZOR SCORE: {self.tas_score:.1f} / 100  [{self.grade.upper()}]",
            f"VAE SCORE:        {self.vae_score:.2f}",
            sep,
        ]
        if self.savings:
            lines += [
                "SAVINGS ESTIMATE",
                f"  Tokens saved:  {self.savings.get('tokens_saved', 0)}  "
                f"({self.savings.get('reduction_pct', 0):.1f}% reduction)",
                f"  At 50K/month:  ${self.savings.get('monthly_savings_usd', 0):.2f}/month",
            ]
        return "\n".join(lines)

    def summary(self) -> str:
        return (
            f"TAS {self.tas_score:.1f}/100 [{self.grade}] | "
            f"VAE {self.vae_score:.2f} | "
            f"Saved {self.savings.get('reduction_pct', 0):.0f}% tokens"
        )


class TraceRazorClient:
    """
    Thin wrapper around the tracerazor CLI binary.

    Writes the trace to a temp file, invokes the binary, parses JSON output.
    """

    def __init__(self, bin_path: Optional[str] = None):
        self._bin = bin_path or self._find_binary()

    def analyse(
        self,
        trace: Dict[str, Any],
        semantic: bool = False,
        threshold: float = 70.0,
        cost_per_million: float = 3.0,
    ) -> TraceRazorReport:
        """
        Write the trace to a temp file and run tracerazor audit on it.

        Returns a TraceRazorReport with the full parsed result.
        Raises subprocess.CalledProcessError if the binary fails unexpectedly.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(trace, f, indent=2)
            tmp_path = f.name

        try:
            cmd = [
                self._bin,
                "audit",
                tmp_path,
                "--format", "json",
                "--threshold", str(threshold),
                "--cost-per-million", str(cost_per_million),
            ]
            if semantic:
                cmd.append("--semantic")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Exit code 1 = below threshold (still valid output). Other codes = error.
            if result.returncode not in (0, 1):
                raise RuntimeError(
                    f"tracerazor exited with code {result.returncode}:\n{result.stderr}"
                )

            report_json = json.loads(result.stdout)
            return self._parse_report(report_json, threshold)

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _parse_report(data: Dict[str, Any], threshold: float) -> TraceRazorReport:
        score = data.get("score", {})
        return TraceRazorReport(
            trace_id=data.get("trace_id", ""),
            agent_name=data.get("agent_name", ""),
            framework=data.get("framework", ""),
            total_steps=data.get("total_steps", 0),
            total_tokens=data.get("total_tokens", 0),
            tas_score=score.get("score", 0.0),
            grade=score.get("grade", "Unknown"),
            vae_score=score.get("vae", 0.0),
            passes=score.get("score", 0.0) >= threshold,
            metrics=score,
            diff=data.get("diff", []),
            savings=data.get("savings", {}),
            raw=data,
        )

    @staticmethod
    def _find_binary() -> str:
        """
        Locate the tracerazor binary.
        Search order:
          1. TRACERAZOR_BIN environment variable
          2. PATH (system-wide install)
          3. Relative paths from this file (dev repo layout)
        """
        env_path = os.environ.get("TRACERAZOR_BIN")
        if env_path and os.path.isfile(env_path):
            return env_path

        path_bin = shutil.which("tracerazor") or shutil.which("tracerazor.exe")
        if path_bin:
            return path_bin

        # Dev layout: integrations/langgraph/ → ../../target/release/
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
            "tracerazor binary not found. Set TRACERAZOR_BIN environment variable "
            "or add 'tracerazor' to PATH.\n"
            "Build with: cargo build --release"
        )
