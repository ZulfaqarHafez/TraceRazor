"""
TraceRazorClient and TraceRazorReport.

TraceRazorClient submits a trace to the TraceRazor analyzer in one of two ways:

CLI mode (default): spawns the local tracerazor binary as a subprocess.
    No server required. Requires the binary to be on PATH or pointed to by
    the TRACERAZOR_BIN environment variable.

HTTP mode: POSTs to a running tracerazor-server.
    No binary on the agent machine required.
    Start the server with: ./tracerazor-server
    Then pass server="http://localhost:8080" to TraceRazorClient.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import AuditError, BelowMinStepsError, BinaryNotFoundError


@dataclass
class TraceRazorReport:
    """
    Parsed result of one tracerazor audit run.

    Attributes:
        trace_id:     Unique ID of the trace that was analysed.
        agent_name:   Name passed to Tracer() when the trace was recorded.
        framework:    Framework identifier (e.g. "openai", "langgraph").
        total_steps:  Number of steps in the trace.
        total_tokens: Total token count across all steps.
        tas_score:    Token Alignment Score, 0-100. Ordinal within one workload.
        grade:        Letter grade: Excellent / Good / Fair / Poor.
        passes:       True if the analyzer's pass/gate verdict passed.
        threshold:    The minimum score used for pass/fail.
        metrics:      Raw per-metric scores dict (srr, ldi, tca, ...).
        savings:      Estimated token and cost savings if fixes are applied.
        fixes:        Auto-generated fix patches.
        anomalies:    Regressions vs. the agent's historical baseline.
        raw:          Full JSON response from the analyzer.
    """

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
        """Return a one-line summary suitable for logging or CI output."""
        saved = self.savings.get("tokens_saved", 0)
        pct = self.savings.get("reduction_pct", 0.0)
        return (
            f"TAS {self.tas_score:.1f}/100 [{self.grade}] | "
            f"{self.total_steps} steps, {self.total_tokens} tokens | "
            f"Estimated {saved} tokens ({pct:.0f}%)"
        )

    def markdown(self) -> str:
        """Return the full markdown report (same format as the CLI output)."""
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
        for code in ("srr", "ldi", "tca", "rda", "isr", "tur", "cce", "dbo", "vdi", "shl", "ccr", "csd", "gar", "obs"):
            m = s.get(code, {})
            if m:
                status = "PASS" if m.get("pass") else "FAIL"
                lines.append(f"{code.upper():<6} {m.get('score', 0):.3f}   {status}")
        if self.savings:
            monthly_runs = self.savings.get("monthly_runs")
            monthly_assumed = self.savings.get("monthly_runs_assumed") is True
            monthly_label = (
                f"  At {'ASSUMED ' if monthly_assumed else ''}{int(monthly_runs):,}/month:  "
                if isinstance(monthly_runs, (int, float))
                else "  Monthly projection:  "
            )
            lines += [
                sep,
                "SAVINGS ESTIMATE",
                f"  Tokens saved:  {self.savings.get('tokens_saved', 0)}  "
                f"({self.savings.get('reduction_pct', 0):.1f}% reduction)",
                f"  Cost saved:    ${self.savings.get('cost_saved_per_run_usd', 0):.4f}/run",
                monthly_label
                + f"${self.savings.get('monthly_savings_usd', 0):.2f}/month (estimated)",
            ]
        if self.fixes:
            lines += [sep, "AUTO-GENERATED FIXES"]
            for i, fix in enumerate(self.fixes, 1):
                lines.append(
                    f"  Fix {i}: [{fix.get('fix_type')}] {fix.get('target')}\n"
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
        """Raise AssertionError if TAS is below threshold. Use in CI/CD pipelines."""
        if not self.passes:
            raise AssertionError(
                f"TraceRazor: TAS {self.tas_score:.1f} is below "
                f"threshold {self.threshold}.\n\n{self.summary()}"
            )


class TraceRazorClient:
    """
    Submit a trace for analysis and return a TraceRazorReport.

    Args:
        bin_path:  Path to the tracerazor binary. Auto-detected when None.
                   Ignored when server is set.
        server:    Base URL of a running tracerazor-server, e.g.
                   "http://localhost:8080". Activates HTTP mode.
        threshold: Explicit project-local TAS floor for the legacy
                   assert_passes() helper. Default 70 is retained for 1.x
                   compatibility and is not a universal quality threshold.
        hermetic:  Default CLI/server audit mode. True means score as a pure
                   function of trace, config, and version.
        min_steps: Optional audit floor to pass through to the CLI.
        api_token: Optional bearer token for HTTP mode. Defaults to
                   TRACERAZOR_API_TOKEN when set.
    """

    def __init__(
        self,
        bin_path: Optional[str] = None,
        server: Optional[str] = None,
        threshold: float = 70.0,
        min_steps: Optional[int] = None,
        hermetic: bool = True,
        weights: Optional[str | os.PathLike[str]] = None,
        enhanced: bool = False,
        store: Optional[bool] = None,
        api_token: Optional[str] = None,
        timeout_s: float = 60.0,
    ):
        self._server = server.rstrip("/") if server else None
        self._bin = None if self._server else bin_path
        self._threshold = threshold
        self._min_steps = min_steps
        self._hermetic = hermetic
        self._weights = os.fspath(weights) if weights is not None else None
        self._enhanced = enhanced
        self._store = store
        self._api_token = api_token if api_token is not None else os.environ.get("TRACERAZOR_API_TOKEN")
        self._timeout_s = timeout_s

    def analyse(
        self,
        trace: Dict[str, Any],
        *,
        min_steps: Optional[int] = None,
        hermetic: Optional[bool] = None,
        weights: Optional[str | os.PathLike[str]] = None,
        enhanced: Optional[bool] = None,
        store: Optional[bool] = None,
    ) -> TraceRazorReport:
        """Submit the trace dict and return a TraceRazorReport."""
        options = {
            "min_steps": self._min_steps if min_steps is None else min_steps,
            "hermetic": self._hermetic if hermetic is None else hermetic,
            "weights": self._weights if weights is None else os.fspath(weights),
            "enhanced": self._enhanced if enhanced is None else enhanced,
            "store": self._store if store is None else store,
        }
        if self._server:
            return self._analyse_http(trace, options)
        return self._analyse_cli(trace, options)

    def _analyse_cli(self, trace: Dict[str, Any], options: Dict[str, Any]) -> TraceRazorReport:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(trace, f, indent=2)
            tmp_path = f.name

        try:
            try:
                binary = self._bin or self._find_binary()
                cmd = [
                    binary,
                    "audit",
                    tmp_path,
                    "--format",
                    "json",
                    "--threshold",
                    str(self._threshold),
                ]
                if options["hermetic"]:
                    cmd.append("--hermetic")
                if options["min_steps"] is not None:
                    cmd += ["--min-steps", str(options["min_steps"])]
                if options["weights"] is not None:
                    cmd += ["--weights", str(options["weights"])]
                if options["enhanced"]:
                    cmd.append("--enhanced")
                if options["store"] is not None:
                    cmd += ["--store", "true" if options["store"] else "false"]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_s,
                )
            except FileNotFoundError as exc:
                raise BinaryNotFoundError(f"tracerazor binary not found: {self._bin}") from exc
            except subprocess.TimeoutExpired as exc:
                raise AuditError(
                    f"tracerazor audit timed out after {self._timeout_s:g} s. "
                    "The trace may be unusually large; try reducing it or increasing the timeout."
                ) from exc

            # Exit code 1 means below threshold but output is still valid JSON.
            if result.returncode not in (0, 1):
                raise AuditError(
                    f"tracerazor exited with code {result.returncode}:\n{result.stderr}"
                )
            stdout = result.stdout.strip()
            if not stdout:
                # CLI emitted a notice (e.g. too few steps) but no JSON.
                notice = result.stderr.strip() or "tracerazor returned no output"
                raise BelowMinStepsError(
                    f"tracerazor produced no JSON output.\n"
                    f"Note: traces need at least {options['min_steps'] or 5} steps to be analysed.\n"
                    f"Binary message: {notice}"
                )
            data = json.loads(stdout)
            if data.get("status") == "skipped" or data.get("audited") is False:
                raise BelowMinStepsError(str(data.get("message") or "trace below audit step floor"))
            return self._parse_cli_report(data, passed=result.returncode == 0)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _parse_cli_report(self, data: Dict[str, Any], *, passed: Optional[bool] = None) -> TraceRazorReport:
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
            passes=bool(data.get("passed", passed if passed is not None else tas >= self._threshold)),
            threshold=self._threshold,
            metrics=score,
            savings=data.get("savings", {}),
            fixes=data.get("fixes", []),
            anomalies=data.get("anomalies", []),
            raw=data,
        )

    def _analyse_http(self, trace: Dict[str, Any], options: Dict[str, Any]) -> TraceRazorReport:
        try:
            import requests
        except ImportError:
            raise ImportError(
                "HTTP mode requires the 'requests' library.\n"
                "Install with: pip install tracerazor[http]"
            )

        headers = {}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
        resp = requests.post(
            f"{self._server}/api/audit",
            json={"trace": trace, "hermetic": bool(options["hermetic"])},
            headers=headers,
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return self._parse_http_report(data)

    def _parse_http_report(self, data: Dict[str, Any]) -> TraceRazorReport:
        tas = data.get("tas_score", 0.0)
        savings: Dict[str, Any] = dict(data.get("savings") or {})
        if "tokens_saved" in data:
            savings["tokens_saved"] = data.get("tokens_saved", 0)
        metrics: Dict[str, Any] = {}
        if data.get("avs") is not None:
            metrics["avs"] = data["avs"]
        if "score" in data and isinstance(data["score"], dict):
            metrics.update(data["score"])
        if "manifest" in data:
            metrics["manifest"] = data["manifest"]
        return TraceRazorReport(
            trace_id=data.get("trace_id", ""),
            agent_name=data.get("agent_name", ""),
            framework=data.get("framework", ""),
            total_steps=data.get("total_steps", 0),
            total_tokens=data.get("total_tokens", 0),
            tas_score=tas,
            grade=data.get("grade", "Unknown"),
            passes=bool(data.get("passed", tas >= self._threshold)),
            threshold=self._threshold,
            metrics=metrics,
            savings=savings,
            fixes=data.get("fixes", []),
            anomalies=data.get("anomalies", []),
            raw=data,  # markdown() reads report_markdown from here
        )

    @staticmethod
    def _find_binary() -> str:
        # Platform wheels bundle the CLI inside the package; prefer it.
        here_pkg = os.path.dirname(os.path.abspath(__file__))
        for name in ("tracerazor", "tracerazor.exe"):
            bundled = os.path.join(here_pkg, "bin", name)
            if os.path.isfile(bundled) and os.access(bundled, os.X_OK):
                return bundled

        env_path = os.environ.get("TRACERAZOR_BIN")
        if env_path:
            if os.path.isfile(env_path):
                return env_path
            raise BinaryNotFoundError(
                "TRACERAZOR_BIN is set but does not point to a file:\n"
                f"  {env_path}\n"
                "Set it to an existing tracerazor binary or unset it to use auto-discovery."
            )

        found = shutil.which("tracerazor") or shutil.which("tracerazor.exe")
        if found:
            return found

        here = os.path.dirname(os.path.abspath(__file__))
        # This file lives at <repo>/tracerazor/_audit_client.py, so a source
        # checkout's `cargo build` output is exactly one level up.
        for rel in [
            "../target/release/tracerazor.exe",
            "../target/release/tracerazor",
            "../target/debug/tracerazor.exe",
            "../target/debug/tracerazor",
        ]:
            candidate = os.path.normpath(os.path.join(here, rel))
            if os.path.isfile(candidate):
                return candidate

        raise BinaryNotFoundError(
            "tracerazor binary not found.\n"
            "Options:\n"
            "  1. Set TRACERAZOR_BIN=/path/to/tracerazor\n"
            "  2. Build from source: cargo build --release -p tracerazor\n"
            "  3. Install a platform wheel (bundles the binary): pip install tracerazor\n"
            "  4. Use HTTP mode: TraceRazorClient(server='http://localhost:8080')"
        )
