"""Public trust diagnostics for TRICE/TraceRazor installs."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
PROJECT = "tracerazor"
GITHUB_REPO = "ZulfaqarHafez/tracerazor"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PROJECT}/json"
PIWHEELS_JSON_URL = f"https://www.piwheels.org/project/{PROJECT}/json"
CRATES_JSON_URL = f"https://crates.io/api/v1/crates/{PROJECT}"
GITHUB_RUNS_URL = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=30"

FetchJson = Callable[[str, float], dict[str, Any]]


@dataclass(frozen=True)
class DoctorOptions:
    offline: bool = False
    timeout_s: float = 5.0


def doctor_report(*, offline: bool = False, timeout_s: float = 5.0, fetch_json: FetchJson | None = None) -> dict[str, Any]:
    """Return a machine-readable public trust diagnostic report."""

    options = DoctorOptions(offline=offline, timeout_s=timeout_s)
    fetch = fetch_json or _fetch_json
    local_version = _local_version()
    checks = {
        "local_package": _check_local_package(local_version),
        "bundled_cli": _check_bundled_cli(),
        "schemas": _check_schemas(),
        "pypi": _check_pypi(local_version, options, fetch),
        "piwheels": _check_piwheels(local_version, options, fetch),
        "crates_io": _check_crates(options, fetch),
        "github_tag": _check_github_tag(local_version),
        "github_actions": _check_github_actions(options, fetch),
    }
    failed = [name for name, check in checks.items() if check.get("ok") is False]
    unknown = [name for name, check in checks.items() if check.get("ok") is None]
    return {
        "schema_version": "trice-doctor/v1",
        "package": PROJECT,
        "local_version": local_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "offline": offline,
        "checks": checks,
        "summary": {
            "ok": not failed,
            "failed": failed,
            "unknown": unknown,
        },
    }


def render_doctor_text(report: dict[str, Any]) -> str:
    lines = [
        f"TraceRazor TRICE doctor ({report.get('local_version') or 'unknown'})",
        f"offline: {str(bool(report.get('offline'))).lower()}",
        "",
    ]
    for name, check in report.get("checks", {}).items():
        marker = _marker(check.get("ok"))
        status = check.get("status") or ""
        detail = check.get("detail")
        line = f"{marker} {name}: {status}"
        if detail:
            line += f" - {detail}"
        lines.append(line)
    summary = report.get("summary", {})
    lines.extend(
        [
            "",
            f"summary: {'ok' if summary.get('ok') else 'needs-attention'}",
            f"failed: {', '.join(summary.get('failed') or []) or 'none'}",
            f"unknown: {', '.join(summary.get('unknown') or []) or 'none'}",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose TraceRazor/TRICE public trust signals.")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--offline", action="store_true", help="Skip public HTTP checks.")
    ap.add_argument("--timeout-s", type=float, default=5.0)
    args = ap.parse_args(argv)
    report = doctor_report(offline=args.offline, timeout_s=args.timeout_s)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_doctor_text(report))
    return 0 if report["summary"]["ok"] else 1


def _check_local_package(local_version: str | None) -> dict[str, Any]:
    if not local_version:
        return _check(False, "missing", "could not import tracerazor.__version__")
    return _check(True, "installed", f"version {local_version}")


def _check_bundled_cli() -> dict[str, Any]:
    try:
        import tracerazor
        from tracerazor._launcher import find_binary

        package_dir = Path(tracerazor.__file__).resolve().parent
    except Exception as exc:  # pragma: no cover - import failure is environment-specific.
        return _check(False, "missing", f"could not locate package: {exc}")
    binary = find_binary()
    if not binary:
        exe = "tracerazor.exe" if sys.platform.startswith("win") else "tracerazor"
        return _check(False, "missing", f"expected bundled binary at {package_dir / 'bin' / exe}")
    path = Path(binary).resolve()
    if package_dir / "bin" in path.parents:
        status = "bundled"
    elif (REPO / "target").resolve() in path.parents:
        status = "source-build"
    elif Path(str(shutil.which("tracerazor") or "")).resolve() == path:
        status = "on-path"
    else:
        status = "env"
    return _check(True, status, str(path))


def _check_schemas() -> dict[str, Any]:
    try:
        from .schemas import schema_path

        required = ["suite", "receipt", "adapter", "bundle", "evidence", "patch", "claim", "readiness"]
        missing = [name for name in required if not schema_path(name).is_file()]
    except Exception as exc:
        return _check(False, "error", str(exc))
    if missing:
        return _check(False, "missing", ", ".join(missing))
    return _check(True, "available", f"{len(required)} schemas")


def _check_pypi(local_version: str | None, options: DoctorOptions, fetch: FetchJson) -> dict[str, Any]:
    if options.offline:
        return _check(None, "skipped", "offline mode")
    data = fetch(PYPI_JSON_URL, options.timeout_s)
    if "__error__" in data:
        return _check(None, "unknown", data["__error__"], url=PYPI_JSON_URL)
    latest = str((data.get("info") or {}).get("version") or "")
    ok = bool(latest and local_version and latest == local_version)
    return _check(ok, "matched" if ok else "mismatch", f"latest={latest or 'unknown'} local={local_version}", url=PYPI_JSON_URL)


def _check_piwheels(local_version: str | None, options: DoctorOptions, fetch: FetchJson) -> dict[str, Any]:
    if options.offline:
        return _check(None, "skipped", "offline mode")
    data = fetch(PIWHEELS_JSON_URL, options.timeout_s)
    if "__error__" in data:
        return _check(None, "unknown", data["__error__"], url=PIWHEELS_JSON_URL)
    releases = data.get("releases") if isinstance(data.get("releases"), dict) else {}
    release = releases.get(local_version or "") if isinstance(releases, dict) else None
    files = []
    if isinstance(release, dict):
        raw_files = release.get("files")
        if isinstance(raw_files, dict):
            files = list(raw_files)
        elif isinstance(raw_files, list):
            files = [str(item.get("filename") or item) for item in raw_files]
    ok = bool(files)
    return _check(ok, "visible" if ok else "missing", f"files={len(files)} local={local_version}", url=PIWHEELS_JSON_URL)


def _check_crates(options: DoctorOptions, fetch: FetchJson) -> dict[str, Any]:
    if options.offline:
        return _check(None, "skipped", "offline mode")
    data = fetch(CRATES_JSON_URL, options.timeout_s)
    if data.get("__http_status") == 404:
        return _check(False, "missing", "crate tracerazor is not published", url=CRATES_JSON_URL)
    if "__error__" in data:
        return _check(None, "unknown", data["__error__"], url=CRATES_JSON_URL)
    crate = data.get("crate") if isinstance(data.get("crate"), dict) else {}
    newest = str(crate.get("newest_version") or "")
    return _check(bool(newest), "published" if newest else "missing", f"newest={newest or 'unknown'}", url=CRATES_JSON_URL)


def _check_github_tag(local_version: str | None) -> dict[str, Any]:
    if not local_version:
        return _check(None, "unknown", "local version unavailable")
    tag = f"v{local_version}"
    head = _git(["rev-parse", "HEAD"])
    if not head:
        return _check(None, "unavailable", "git metadata unavailable")
    tags = set((_git(["tag", "--points-at", "HEAD"]) or "").splitlines())
    remote = _git(["ls-remote", "origin", f"refs/tags/{tag}"])
    remote_sha = remote.split()[0] if remote else ""
    ok = bool(head and tag in tags and remote_sha == head)
    detail = f"head={head[:12] if head else 'unknown'} local_tag={tag in tags} remote_tag={bool(remote_sha)}"
    return _check(ok, "aligned" if ok else "pending", detail)


def _check_github_actions(options: DoctorOptions, fetch: FetchJson) -> dict[str, Any]:
    if options.offline:
        return _check(None, "skipped", "offline mode")
    data = fetch(GITHUB_RUNS_URL, options.timeout_s)
    if "__error__" in data:
        return _check(None, "unknown", data["__error__"], url=GITHUB_RUNS_URL)
    wanted = {"TraceRazor CI", "Agent Efficiency Gate", "Release"}
    latest: dict[str, dict[str, Any]] = {}
    for run in data.get("workflow_runs", []):
        if not isinstance(run, dict):
            continue
        name = str(run.get("name") or "")
        if name in wanted and name not in latest:
            latest[name] = {
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "html_url": run.get("html_url"),
            }
    missing = sorted(wanted - set(latest))
    failing = [name for name, run in latest.items() if run.get("conclusion") != "success"]
    ok = not missing and not failing
    detail = "; ".join(
        f"{name}={run.get('status')}/{run.get('conclusion')}" for name, run in sorted(latest.items())
    )
    if missing:
        detail = (detail + "; " if detail else "") + f"missing={','.join(missing)}"
    return _check(ok, "green" if ok else "not-green", detail, url=GITHUB_RUNS_URL)


def _fetch_json(url: str, timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "tracerazor-trice-doctor/1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=max(1.0, timeout_s)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            if isinstance(payload, dict):
                payload["__http_status"] = int(getattr(resp, "status", 200))
                return payload
            return {"__error__": "response was not a JSON object"}
    except urllib.error.HTTPError as exc:
        return {"__http_status": exc.code, "__error__": f"HTTP {exc.code}"}
    except Exception as exc:
        return {"__error__": str(exc)}


def _local_version() -> str | None:
    try:
        import tracerazor

        return str(tracerazor.__version__)
    except Exception:
        return None


def _git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _check(ok: bool | None, status: str, detail: str, *, url: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": ok, "status": status, "detail": detail}
    if url:
        payload["url"] = url
    return payload


def _marker(ok: Any) -> str:
    if ok is True:
        return "[ok]"
    if ok is False:
        return "[fail]"
    return "[unknown]"


if __name__ == "__main__":
    raise SystemExit(main())
