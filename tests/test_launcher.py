"""Unit tests for the pip console launcher (tracerazor._launcher).

Covers the self-detection fix (normcase / console-script rejection so `which`
never resolves back into this launcher) and the missing-binary recovery
message. Functions are invoked directly — nothing is exec'd.
"""

import os
import sys

from tracerazor import _launcher


def test_rejects_exact_self_path(monkeypatch):
    """A PATH hit equal to sys.argv[0] must not be returned."""
    monkeypatch.setattr(_launcher, "_bundled", lambda: None)
    monkeypatch.delenv("TRACERAZOR_BIN", raising=False)
    self_path = os.path.abspath("tracerazor")
    monkeypatch.setattr(sys, "argv", [self_path, "audit"])

    import shutil

    monkeypatch.setattr(
        shutil, "which", lambda n: self_path if n.startswith("tracerazor") else None
    )
    # Rejected as self -> falls through; whatever it returns, it is not the
    # self path (and is never that exact PATH hit).
    assert _launcher.find_binary() != self_path


def test_rejects_case_variant_self_on_windows(monkeypatch, tmp_path):
    """normcase folds Windows case, so a case-variant of argv[0] is still self.

    Under the old `abspath != abspath` check this candidate would be returned on
    Windows (paths differ only by case); the normcase check must reject it.
    """
    exe = tmp_path / "tracerazor.exe"
    exe.write_bytes(b"MZ\x90\x00")  # a real (binary) file, not a python script
    self_path = str(exe)
    which_path = self_path.upper()  # same file on Windows, different case

    monkeypatch.setattr(_launcher, "_bundled", lambda: None)
    monkeypatch.delenv("TRACERAZOR_BIN", raising=False)
    monkeypatch.setattr(sys, "argv", [self_path])

    import shutil

    monkeypatch.setattr(
        shutil, "which", lambda n: which_path if n.startswith("tracerazor") else None
    )

    result = _launcher.find_binary()
    # Only case-insensitive filesystems (Windows) treat the case-variant as
    # self; on POSIX it is a genuinely different path, so rejection there is not
    # expected.
    if os.name == "nt":
        assert result != which_path
        assert result is None or os.path.normcase(result) != os.path.normcase(which_path)


def test_rejects_own_python_console_script(monkeypatch, tmp_path):
    """A PATH hit that is this package's own python console script is rejected
    (execing it would recurse back into the launcher)."""
    script = tmp_path / "tracerazor"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "# -*- coding: utf-8 -*-\n"
        "import sys\n"
        "from tracerazor._launcher import main\n"
        "sys.exit(main())\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(_launcher, "_bundled", lambda: None)
    monkeypatch.delenv("TRACERAZOR_BIN", raising=False)
    monkeypatch.setattr(sys, "argv", ["/some/other/argv0"])

    import shutil

    monkeypatch.setattr(
        shutil, "which", lambda n: str(script) if n.startswith("tracerazor") else None
    )
    assert _launcher.find_binary() != str(script)


def test_is_own_console_script_detects_and_ignores(tmp_path):
    py = tmp_path / "wrapper"
    py.write_text("#!/usr/bin/python\nfrom tracerazor._launcher import main\n", "utf-8")
    assert _launcher._is_own_console_script(str(py)) is True

    real = tmp_path / "binary"
    real.write_bytes(b"MZ\x90\x00\x03")  # not a python shebang
    assert _launcher._is_own_console_script(str(real)) is False

    missing = tmp_path / "gone"
    assert _launcher._is_own_console_script(str(missing)) is False


def test_recovery_message_teaches_exact_commands():
    msg = _launcher.recovery_message()
    # Compact: at most 12 lines.
    assert msg.strip("\n").count("\n") + 1 <= 12
    # 1. platform wheel status, 2. TRACERAZOR_BIN, 3. build from source, 4. trice.
    lower = msg.lower()
    assert "wheel" in lower
    assert "$env:TRACERAZOR_BIN" in msg
    assert "export TRACERAZOR_BIN=" in msg
    assert "git clone https://github.com/ZulfaqarHafez/tracerazor" in msg
    assert "cargo build --release -p tracerazor" in msg
    assert "tracerazor-trice" in msg


def test_main_missing_binary_returns_2_and_teaches(monkeypatch, capsys):
    monkeypatch.setattr(_launcher, "find_binary", lambda: None)
    monkeypatch.setattr(sys, "argv", ["tracerazor", "audit"])
    rc = _launcher.main()
    assert rc == 2
    err = capsys.readouterr().err
    assert "TRACERAZOR_BIN" in err
    assert "cargo build --release -p tracerazor" in err
