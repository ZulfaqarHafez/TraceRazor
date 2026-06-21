"""Public TRICE command-line entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.trice.live import main as live_main
from benchmark.trice.schemas import load_schema, validate_patch_spec_file, validate_suite_manifest_file
from benchmark.trice.suite import main as suite_main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tracerazor-trice",
        description="TRICE deterministic live context-control tools.",
    )
    sub = ap.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run a deterministic TRICE live rollout.")
    run.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the live runner.")

    suite = sub.add_parser("suite", help="Run a manifest-driven deterministic TRICE live suite.")
    suite.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the suite runner.")

    verify = sub.add_parser("verify", help="Verify a TRICE evidence manifest.")
    verify.add_argument("manifest", type=Path)

    verify_suite = sub.add_parser("verify-suite", help="Deep-verify a TRICE suite evidence manifest.")
    verify_suite.add_argument("manifest", type=Path)

    schema = sub.add_parser("schema", help="Print a shipped TRICE JSON Schema.")
    schema.add_argument("name", choices=["patch", "manifest", "evidence", "suite"])

    validate = sub.add_parser("validate-patch", help="Validate a deterministic patch spec.")
    validate.add_argument("patch_spec", type=Path)

    validate_suite = sub.add_parser("validate-suite", help="Validate a TRICE suite manifest.")
    validate_suite.add_argument("suite_manifest", type=Path)

    args = ap.parse_args(argv)
    if args.command == "run":
        forwarded = list(args.args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        return live_main(forwarded)
    if args.command == "suite":
        forwarded = list(args.args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        return suite_main(forwarded)
    if args.command == "verify":
        return live_main(["--verify-manifest", str(args.manifest)])
    if args.command == "verify-suite":
        return suite_main(["ignored", "--verify-suite", str(args.manifest)])
    if args.command == "schema":
        print(json.dumps(load_schema(args.name), indent=2, sort_keys=True))
        return 0
    if args.command == "validate-patch":
        print(json.dumps(validate_patch_spec_file(args.patch_spec), indent=2, sort_keys=True))
        return 0
    if args.command == "validate-suite":
        print(json.dumps(validate_suite_manifest_file(args.suite_manifest), indent=2, sort_keys=True))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
