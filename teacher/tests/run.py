"""Zero-dependency test runner (no pytest needed).

    python teacher/tests/run.py
"""
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODULES = ["teacher.tests.test_teacher", "teacher.tests.test_layers",
           "teacher.tests.test_online", "teacher.tests.test_harness",
           "teacher.tests.test_bandit"]


def main() -> int:
    total = passed = 0
    for modname in MODULES:
        mod = importlib.import_module(modname)
        fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
        for fn in fns:
            total += 1
            try:
                fn()
                passed += 1
                print(f"PASS {modname}.{fn.__name__}")
            except Exception:
                print(f"FAIL {modname}.{fn.__name__}")
                traceback.print_exc()
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
