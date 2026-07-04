"""Console entry point: `tracerazor` from a pip install runs the Rust CLI.

Resolution order:
1. The binary bundled inside this package (platform wheels ship it at
   ``tracerazor/bin/``).
2. ``TRACERAZOR_BIN``.
3. ``tracerazor`` on PATH (avoiding this launcher itself).
4. A source checkout's ``target/{release,debug}`` next to the package.
"""
import os
import sys


def _bundled() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("tracerazor", "tracerazor.exe"):
        cand = os.path.join(here, "bin", name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _same_path(a: str, b: str) -> bool:
    """Case/normalisation-insensitive path equality.

    ``normcase`` folds case and separators on Windows (a no-op on POSIX), so a
    PATH hit that differs from ``sys.argv[0]`` only in case is still recognised
    as this launcher and rejected.
    """
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _is_own_console_script(path: str) -> bool:
    """True if ``path`` is this package's generated Python console script.

    pip/hatch console scripts are text launchers with a ``python`` shebang that
    import ``tracerazor``; execing one would recurse back into this launcher.
    We peek the head of the file for the shebang plus a ``tracerazor`` entry.
    (Windows ``.exe`` wrapper stubs are binary and won't match here — the
    ``_same_path`` self check covers the common Windows console-script case.)
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    text = head.decode("utf-8", "ignore")
    first = text.splitlines()[0] if text else ""
    if not (first.startswith("#!") and "python" in first.lower()):
        return False
    return "tracerazor" in text.lower()


def find_binary() -> str | None:
    bundled = _bundled()
    if bundled:
        return bundled
    env = os.environ.get("TRACERAZOR_BIN")
    if env and os.path.isfile(env):
        return env
    import shutil

    found = shutil.which("tracerazor") or shutil.which("tracerazor.exe")
    # `which` can resolve to this very console script (exact path, a
    # case-variant on Windows, or a same-package Python wrapper); reject all
    # three to avoid execing back into this launcher.
    if (
        found
        and not _same_path(found, sys.argv[0] if sys.argv else "")
        and not _is_own_console_script(found)
    ):
        return found
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in (
        "../target/release/tracerazor",
        "../target/release/tracerazor.exe",
        "../target/debug/tracerazor",
        "../target/debug/tracerazor.exe",
    ):
        cand = os.path.normpath(os.path.join(here, rel))
        if os.path.isfile(cand):
            return cand
    return None


def recovery_message() -> str:
    """Exact recovery steps for a missing auditor binary (shared with the MCP
    server so both surfaces teach identical commands). Kept <= 12 lines."""
    exe = "tracerazor.exe" if os.name == "nt" else "tracerazor"
    return (
        "tracerazor: no auditor binary found.\n"
        "Recover (pick one):\n"
        "  1. Install a platform wheel that bundles the binary:\n"
        "       pip install --force-reinstall tracerazor   (Linux/macOS/Windows wheels ship the CLI)\n"
        "  2. Point TRACERAZOR_BIN at an existing build:\n"
        f'       PowerShell: $env:TRACERAZOR_BIN = "C:\\path\\to\\{exe}"\n'
        "       bash:       export TRACERAZOR_BIN=/path/to/tracerazor\n"
        "  3. Build from source:\n"
        "       git clone https://github.com/ZulfaqarHafez/tracerazor\n"
        f"       cargo build --release -p tracerazor   # -> target/release/{exe}\n"
        "  4. tracerazor-trice (TRICE) works without this binary.\n"
    )


def main() -> int:
    binary = find_binary()
    if binary is None:
        sys.stderr.write(recovery_message())
        return 2
    argv = [binary] + sys.argv[1:]
    if os.name == "nt":
        import subprocess

        return subprocess.call(argv)
    os.execv(binary, argv)
    return 0  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
