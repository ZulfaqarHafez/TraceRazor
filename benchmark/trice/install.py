"""Deterministic installability cards for built TraceRazor distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
import tomllib

from .evidence import canonical_json, sha256_file

INSTALL_CARD_SCHEMA_VERSION = "trice-install-card/v1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "docs" / "trice_install_card.json"
DEFAULT_DIST = REPO / "dist"


def build_install_card(
    *,
    dist_dir: str | Path = DEFAULT_DIST,
    wheel_path: str | Path | None = None,
    python_executable: str | Path = sys.executable,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Install a built wheel in a clean venv and return an installability card."""

    dist = Path(dist_dir).resolve()
    pyproject_path = REPO / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    expected_version = str((pyproject.get("project") or {}).get("version") or "")
    wheel = Path(wheel_path).resolve() if wheel_path is not None else _find_wheel(dist, expected_version)
    commands: dict[str, Any] = {}
    probe: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="trice-install-card-") as tmp:
        tmp_path = Path(tmp)
        venv_dir = tmp_path / "venv"
        py = _venv_python(venv_dir)
        scripts = _venv_scripts(venv_dir)
        commands["create_venv"] = _run([str(python_executable), "-m", "venv", str(venv_dir)], cwd=tmp_path, timeout_s=timeout_s, scrub_root=tmp_path)
        if commands["create_venv"]["ok"] and wheel is not None:
            commands["install_wheel"] = _run([str(py), "-m", "pip", "install", "--no-deps", str(wheel)], cwd=tmp_path, timeout_s=timeout_s, scrub_root=tmp_path)
        else:
            commands["install_wheel"] = _skipped("venv creation failed or wheel missing")
        if commands["install_wheel"]["ok"]:
            import_code = (
                "import json, tracerazor, tracerazor.trice as trice\n"
                "from benchmark.trice.schemas import load_schema\n"
                "payload = {\n"
                "  'version': tracerazor.__version__,\n"
                "  'install_schema_title': load_schema('install-card')['title'],\n"
                "  'crates_schema_title': load_schema('crates-card')['title'],\n"
                "  'research_schema_title': load_schema('research-card')['title'],\n"
                "  'build_install_card': callable(trice.build_install_card),\n"
                "  'verify_install_card_file': callable(trice.verify_install_card_file),\n"
                "  'build_crates_card': callable(trice.build_crates_card),\n"
                "  'build_research_card': callable(trice.build_research_card),\n"
                "  'verify_research_card_file': callable(trice.verify_research_card_file),\n"
                "}\n"
                "print(json.dumps(payload, sort_keys=True))\n"
            )
            commands["import_probe"] = _run([str(py), "-c", import_code], cwd=tmp_path, timeout_s=timeout_s, scrub_root=tmp_path)
            probe = _parse_probe(commands["import_probe"])
            commands["trice_console"] = _run([str(_console_script(scripts, "tracerazor-trice")), "schema", "install-card"], cwd=tmp_path, timeout_s=timeout_s, scrub_root=tmp_path)
            commands["rust_console"] = _run([str(_console_script(scripts, "tracerazor")), "--version"], cwd=tmp_path, timeout_s=timeout_s, scrub_root=tmp_path)
        else:
            commands["import_probe"] = _skipped("wheel install failed")
            commands["trice_console"] = _skipped("wheel install failed")
            commands["rust_console"] = _skipped("wheel install failed")

    checks = [
        _check("wheel_present", wheel is not None and wheel.is_file(), _display_path(wheel) if wheel else None, "built wheel exists"),
        _check("venv_created", commands["create_venv"]["ok"], commands["create_venv"]["exit_code"], "clean virtual environment can be created"),
        _check("wheel_installs", commands["install_wheel"]["ok"], commands["install_wheel"]["exit_code"], "wheel installs with pip --no-deps"),
        _check("version_matches", probe.get("version") == expected_version, probe.get("version"), expected_version),
        _check("schemas_importable", probe.get("install_schema_title") == "TRICE installability card" and probe.get("crates_schema_title") == "TRICE crates publish card" and probe.get("research_schema_title") == "TRICE research card", _probe_schema_summary(probe), "install, crates, and research schemas import from wheel"),
        _check("trice_api_importable", probe.get("build_install_card") is True and probe.get("verify_install_card_file") is True and probe.get("build_crates_card") is True and probe.get("build_research_card") is True and probe.get("verify_research_card_file") is True, _probe_api_summary(probe), "public tracerazor.trice install/crates/research APIs import"),
        _check("trice_console_works", commands["trice_console"]["ok"], commands["trice_console"]["exit_code"], "tracerazor-trice console script works after wheel install"),
        _check("rust_cli_bundled", commands["rust_console"]["ok"], _rust_cli_observed(commands["rust_console"]), "tracerazor console script can find a bundled Rust auditor binary"),
    ]
    card = {
        "schema_version": INSTALL_CARD_SCHEMA_VERSION,
        "scope": "TraceRazor wheel installability",
        "install_level": _install_level(checks),
        "install_score": _install_score(checks),
        "expected_version": expected_version,
        "python_executable": str(python_executable),
        "wheel": _artifact_row("wheel", wheel),
        "inputs": {
            "dist_dir": _dir_row(dist),
            "pyproject": _artifact_row("pyproject", pyproject_path),
            "wheel": _artifact_row("wheel", wheel),
        },
        "checks": checks,
        "commands": commands,
        "probe": probe,
        "research_basis": [
            "PyPA packaging guidance distinguishes building a distribution from installing and using it in a clean environment.",
            "Wheel artifacts are user-facing contracts: import surfaces, console scripts, and packaged data must be verified after installation, not only hashed before upload.",
            "TRICE separates Python/TRICE install readiness from full Rust CLI readiness so generic pure-Python wheels do not overclaim bundled-binary behavior.",
            "Research-card schemas and APIs must be importable from the wheel because the paper/release basis is now a public contract surface.",
        ],
        "next_actions": _next_actions(checks),
    }
    card["install_card_sha256"] = hashlib.sha256(canonical_json(_without_card_hash(card)).encode("utf-8")).hexdigest()
    return card


def verify_install_card_file(path: str | Path) -> dict[str, Any]:
    """Verify an install card self hash and bound artifact hashes."""

    card_path = Path(path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if card.get("schema_version") != INSTALL_CARD_SCHEMA_VERSION:
        errors.append(f"schema_version must be {INSTALL_CARD_SCHEMA_VERSION}")
    expected_hash = str(card.get("install_card_sha256") or "")
    actual_hash = hashlib.sha256(canonical_json(_without_card_hash(card)).encode("utf-8")).hexdigest()
    if expected_hash != actual_hash:
        errors.append("install_card_sha256 mismatch")
    checked_inputs = []
    for name, row in sorted((card.get("inputs") or {}).items()):
        if not isinstance(row, dict) or not row.get("path") or row.get("kind") == "directory":
            continue
        p = _resolve_bound_path(card_path, str(row["path"]))
        if not p.is_file():
            errors.append(f"input {name} missing: {row['path']}")
            continue
        if p.stat().st_size != int(row.get("bytes") or 0):
            errors.append(f"input {name} byte count mismatch")
        if sha256_file(p) != row.get("sha256"):
            errors.append(f"input {name} sha256 mismatch")
        else:
            checked_inputs.append(name)
    return {
        "ok": not errors,
        "schema_version": card.get("schema_version"),
        "install_level": card.get("install_level"),
        "install_score": card.get("install_score"),
        "install_card_sha256": expected_hash,
        "computed_install_card_sha256": actual_hash,
        "checked_inputs": checked_inputs,
        "errors": errors,
    }


def render_install_markdown(card: dict[str, Any]) -> str:
    lines = [
        "# TRICE Installability Card",
        "",
        f"- Scope: `{card['scope']}`",
        f"- Install level: `{card['install_level']}`",
        f"- Install score: **{card['install_score']}/100**",
        f"- Expected version: `{card['expected_version']}`",
        "",
        "## Checks",
        "",
        "| Check | Passed | Observed | Required |",
        "|---|---:|---|---|",
    ]
    for row in card["checks"]:
        lines.append(f"| {row['name']} | {'yes' if row['passed'] else 'no'} | {_md(row['observed'])} | {_md(row['required'])} |")
    lines.extend(["", "## Commands", "", "| Command | Exit | Status |", "|---|---:|---|"])
    for name, row in card["commands"].items():
        lines.append(f"| {name} | {row['exit_code']} | {'ok' if row['ok'] else row['status']} |")
    lines.extend(["", "## Next Actions", ""])
    for item in card["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Hash", "", f"- install card: `{card['install_card_sha256']}`", ""])
    return "\n".join(lines)


def render_install_tex(card: dict[str, Any]) -> str:
    rows = "\n".join(
        f"{_tex(row['name'])} & {'yes' if row['passed'] else 'no'} & {_tex(row['required'])} \\\\"
        for row in card["checks"]
    )
    return (
        "\\section{Installability Card}\n"
        f"Install level: \\texttt{{{_tex(card['install_level'])}}}; "
        f"score: {card['install_score']}/100; "
        f"expected version: \\texttt{{{_tex(card['expected_version'])}}}.\n\n"
        "\\begin{tabular}{lrl}\n"
        "Check & Passed & Required \\\\\n"
        "\\hline\n"
        f"{rows}\n"
        "\\end{tabular}\n"
    )


def render_install_svg(card: dict[str, Any]) -> str:
    stages = [
        ("wheel", _check_passed(card, "wheel_present")),
        ("install", _check_passed(card, "wheel_installs")),
        ("import", _check_passed(card, "version_matches") and _check_passed(card, "schemas_importable") and _check_passed(card, "trice_api_importable")),
        ("trice cli", _check_passed(card, "trice_console_works")),
        ("rust cli", _check_passed(card, "rust_cli_bundled")),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="300" viewBox="0 0 980 300">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Inter,Segoe UI,Arial" font-size="20" font-weight="700" fill="#111827">TRICE installability card</text>',
        f'<text x="28" y="58" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#4b5563">Score {card["install_score"]}/100 | level {card["install_level"]} | version {card["expected_version"]}</text>',
    ]
    x0, y = 36, 96
    for idx, (label, passed) in enumerate(stages):
        x = x0 + idx * 184
        fill = "#7c3aed" if passed else "#e5e7eb"
        text = "#ffffff" if passed else "#374151"
        parts.append(f'<rect x="{x}" y="{y}" width="138" height="58" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{x + 69}" y="{y + 36}" text-anchor="middle" font-family="Inter,Segoe UI,Arial" font-size="15" font-weight="700" fill="{text}">{label}</text>')
        if idx < len(stages) - 1:
            parts.append(f'<line x1="{x + 148}" y1="{y + 29}" x2="{x + 176}" y2="{y + 29}" stroke="#9ca3af" stroke-width="3"/>')
    parts.append(f'<text x="28" y="214" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">Generic wheels may be Python/TRICE-ready while full Rust CLI bundling remains a platform-wheel release task.</text>')
    parts.append(f'<text x="28" y="238" font-family="Inter,Segoe UI,Arial" font-size="12" fill="#6b7280">hash {card["install_card_sha256"][:16]}...</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_install_outputs(card: dict[str, Any], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = out.with_suffix(".md")
    tex = out.with_suffix(".tex")
    svg = out.with_suffix(".svg")
    md.write_text(render_install_markdown(card), encoding="utf-8")
    tex.write_text(render_install_tex(card), encoding="utf-8")
    svg.write_text(render_install_svg(card), encoding="utf-8")
    return {"json": str(out), "markdown": str(md), "tex": str(tex), "svg": str(svg)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a deterministic TraceRazor wheel installability card.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dist-dir", type=Path, default=DEFAULT_DIST)
    ap.add_argument("--wheel", type=Path, default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--format", choices=["json", "markdown", "tex"], default="json")
    args = ap.parse_args(argv)
    card = build_install_card(dist_dir=args.dist_dir, wheel_path=args.wheel, python_executable=args.python, timeout_s=args.timeout_s)
    outputs = write_install_outputs(card, args.out)
    if args.format == "markdown":
        print(render_install_markdown(card))
    elif args.format == "tex":
        print(render_install_tex(card))
    else:
        print(json.dumps({"install_card": card, "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if card["install_level"] != "install_unusable" else 1


def verify_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a deterministic TraceRazor installability card.")
    ap.add_argument("install_card", type=Path)
    args = ap.parse_args(argv)
    verdict = verify_install_card_file(args.install_card)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["ok"] else 1


def _find_wheel(dist: Path, version: str) -> Path | None:
    wheels = sorted(dist.glob(f"tracerazor-{version}-*.whl")) if version else sorted(dist.glob("tracerazor-*.whl"))
    return wheels[0] if wheels else None


def _run(cmd: list[str], *, cwd: Path, timeout_s: float, scrub_root: Path) -> dict[str, Any]:
    scrubbed_command = " ".join(_scrub(part, scrub_root) for part in cmd)
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except Exception as exc:
        return {
            "ok": False,
            "status": "error",
            "exit_code": -1,
            "command_sha256": hashlib.sha256(scrubbed_command.encode("utf-8")).hexdigest(),
            "detail": _scrub(str(exc), scrub_root),
        }
    stdout = _scrub(proc.stdout or "", scrub_root)
    stderr = _scrub(proc.stderr or "", scrub_root)
    return {
        "ok": proc.returncode == 0,
        "status": "ok" if proc.returncode == 0 else "failed",
        "exit_code": proc.returncode,
        "command_sha256": hashlib.sha256(scrubbed_command.encode("utf-8")).hexdigest(),
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stdout_excerpt": stdout[:500],
        "stderr_excerpt": stderr[:500],
    }


def _skipped(reason: str) -> dict[str, Any]:
    return {"ok": False, "status": "skipped", "exit_code": -1, "detail": reason}


def _parse_probe(row: dict[str, Any]) -> dict[str, Any]:
    if not row.get("ok"):
        return {}
    try:
        return json.loads(str(row.get("stdout_excerpt") or "{}").strip())
    except Exception:
        return {}


def _stdout_has(row: dict[str, Any], text: str) -> bool:
    return text in str(row.get("stdout_excerpt") or "")


def _check(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def _check_passed(card_or_checks: dict[str, Any] | list[dict[str, Any]], name: str) -> bool:
    checks = card_or_checks if isinstance(card_or_checks, list) else card_or_checks.get("checks", [])
    return any(row["name"] == name and row["passed"] for row in checks)


def _install_level(checks: list[dict[str, Any]]) -> str:
    passed = {row["name"] for row in checks if row["passed"]}
    required_python = {"wheel_present", "venv_created", "wheel_installs", "version_matches", "schemas_importable", "trice_api_importable", "trice_console_works"}
    if required_python | {"rust_cli_bundled"} <= passed:
        return "full_cli_install_ready"
    if required_python <= passed:
        return "python_trice_install_ready"
    return "install_unusable"


def _install_score(checks: list[dict[str, Any]]) -> int:
    weights = {
        "wheel_present": 10,
        "venv_created": 10,
        "wheel_installs": 20,
        "version_matches": 10,
        "schemas_importable": 15,
        "trice_api_importable": 15,
        "trice_console_works": 10,
        "rust_cli_bundled": 10,
    }
    return min(100, sum(weights.get(row["name"], 0) for row in checks if row["passed"]))


def _next_actions(checks: list[dict[str, Any]]) -> list[str]:
    if _install_level(checks) == "full_cli_install_ready":
        return ["Publish the install card with the release evidence bundle."]
    if _install_level(checks) == "python_trice_install_ready":
        return [
            "Keep Python/TRICE install claims allowed for the checked wheel.",
            "Keep full `tracerazor` CLI claims scoped to platform wheels or source builds until a bundled Rust binary is present.",
            "Generate platform-wheel install cards before claiming no-Rust-toolchain CLI install.",
        ]
    return ["Fix the built wheel before publishing: installation, imports, schemas, or console scripts failed."]


def _probe_schema_summary(probe: dict[str, Any]) -> dict[str, Any]:
    return {key: probe.get(key) for key in ("install_schema_title", "crates_schema_title", "research_schema_title")}


def _probe_api_summary(probe: dict[str, Any]) -> dict[str, Any]:
    return {key: probe.get(key) for key in ("build_install_card", "verify_install_card_file", "build_crates_card", "build_research_card", "verify_research_card_file")}


def _rust_cli_observed(row: dict[str, Any]) -> str:
    text = str(row.get("stdout_excerpt") or row.get("stderr_excerpt") or row.get("detail") or "")
    return f"exit={row.get('exit_code')}; {text[:180]}"


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")


def _venv_scripts(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if sys.platform.startswith("win") else "bin")


def _console_script(scripts_dir: Path, name: str) -> Path:
    exe = ".exe" if sys.platform.startswith("win") else ""
    return scripts_dir / f"{name}{exe}"


def _artifact_row(name: str, path: Path | None) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "file",
        "path": _display_path(path) if path else None,
        "present": bool(path and path.is_file()),
        "bytes": path.stat().st_size if path and path.is_file() else 0,
        "sha256": sha256_file(path) if path and path.is_file() else None,
    }


def _dir_row(path: Path) -> dict[str, Any]:
    return {"name": path.name, "kind": "directory", "path": _display_path(path), "present": path.is_dir()}


def _resolve_bound_path(card_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    repo_candidate = REPO / candidate
    if repo_candidate.exists():
        return repo_candidate
    return card_path.parent / candidate


def _without_card_hash(card: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out.pop("install_card_sha256", None)
    return out


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _scrub(text: str, scrub_root: Path) -> str:
    return text.replace(str(scrub_root), "<tmp>").replace(str(REPO), "<repo>")


def _md(value: Any) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return text.replace("|", "\\|")


def _tex(value: Any) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


if __name__ == "__main__":
    raise SystemExit(main())
