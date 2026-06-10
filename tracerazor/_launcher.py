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


def find_binary() -> str | None:
    bundled = _bundled()
    if bundled:
        return bundled
    env = os.environ.get("TRACERAZOR_BIN")
    if env and os.path.isfile(env):
        return env
    import shutil

    found = shutil.which("tracerazor") or shutil.which("tracerazor.exe")
    # `which` can resolve to this very console script; reject that.
    if found and os.path.abspath(found) != os.path.abspath(sys.argv[0]):
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


def main() -> int:
    binary = find_binary()
    if binary is None:
        sys.stderr.write(
            "tracerazor: no auditor binary available.\n"
            "This looks like a pure-Python install (sdist). Options:\n"
            "  1. pip install a platform wheel (bundles the binary), or\n"
            "  2. cargo build --release -p tracerazor and set TRACERAZOR_BIN, or\n"
            "  3. use the HTTP server mode (tracerazor-server).\n"
        )
        return 2
    argv = [binary] + sys.argv[1:]
    if os.name == "nt":
        import subprocess

        return subprocess.call(argv)
    os.execv(binary, argv)
    return 0  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
