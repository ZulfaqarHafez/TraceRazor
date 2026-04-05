"""
TraceRazor CLI client — shared with the LangGraph adapter.

Writes a trace dict to a temp file, calls the tracerazor binary,
and parses the JSON report.
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
    anomalies: List[Dict] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def markdown(self) -> str:
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
        if self.anomalies:
            lines.append("ANOMALIES")
            for a in self.anomalies:
                lines.append(
                    f"  [{a['metric']}] value={a['value']:.3f}  "
                    f"baseline={a['baseline_mean']:.3f}±{a['baseline_std']:.3f}  "
                    f"z={a['z_score']:+.2f}"
                )
        return "\n".join(lines)

    def summary(self) -> str:
        return (
            f"TAS {self.tas_score:.1f}/100 [{self.grade}] | "
            f"VAE {self.vae_score:.2f} | "
            f"Saved {self.savings.get('reduction_pct', 0):.0f}% tokens"
        )


class TraceRazorClient:
    """Thin wrapper around the tracerazor CLI binary."""

    def __init__(self, bin_path: Optional[str] = None):
        self._bin = bin_path or self._find_binary()

    def analyse(self, trace: Dict[str, Any], threshold: float = 70.0) -> TraceRazorReport:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(trace, f, indent=2)
            tmp_path = f.name

        try:
            cmd = [self._bin, "audit", tmp_path, "--format", "json", "--threshold", str(threshold)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode not in (0, 1):
                raise RuntimeError(
                    f"tracerazor exited with code {result.returncode}:\n{result.stderr}"
                )
            return self._parse(json.loads(result.stdout), threshold)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _parse(data: Dict[str, Any], threshold: float) -> TraceRazorReport:
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
            anomalies=data.get("anomalies", []),
            raw=data,
        )

    @staticmethod
    def _find_binary() -> str:
        env_path = os.environ.get("TRACERAZOR_BIN")
        if env_path and os.path.isfile(env_path):
            return env_path

        path_bin = shutil.which("tracerazor") or shutil.which("tracerazor.exe")
        if path_bin:
            return path_bin

        # Dev repo layout: integrations/crewai/ → ../../target/release/
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
            "tracerazor binary not found. Set TRACERAZOR_BIN or add 'tracerazor' to PATH.\n"
            "Build with: cargo build --release"
        )
