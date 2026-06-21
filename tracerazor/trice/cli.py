"""Public TRICE command-line entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.trice.bundle import main as bundle_main
from benchmark.trice.claim import main as claim_main, verify_claim_card_file
from benchmark.trice.doctor import main as doctor_main
from benchmark.trice.live import main as live_main
from benchmark.trice.receipt import validate_run_receipt_file
from benchmark.trice.schemas import load_schema, validate_adapter_profile_file, validate_patch_spec_file, validate_suite_manifest_file
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

    bundle = sub.add_parser("bundle", help="Export a portable TRICE evidence bundle zip.")
    bundle.add_argument("manifest", type=Path)
    bundle.add_argument("--result", type=Path, default=None)
    bundle.add_argument("--out", type=Path, default=None)

    verify_bundle = sub.add_parser("verify-bundle", help="Verify a portable TRICE evidence bundle zip.")
    verify_bundle.add_argument("bundle", type=Path)

    doctor = sub.add_parser("doctor", help="Diagnose local and public TraceRazor trust signals.")
    doctor.add_argument("--format", choices=["text", "json"], default="text")
    doctor.add_argument("--offline", action="store_true")
    doctor.add_argument("--timeout-s", type=float, default=5.0)

    claim = sub.add_parser("claim", help="Generate a deterministic TRICE claim card.")
    claim.add_argument("--suite-result", type=Path, required=True)
    claim.add_argument("--manifest", type=Path, default=None)
    claim.add_argument("--out", type=Path, default=Path("trice_claim_card.json"))
    claim.add_argument("--scope", default=None)
    claim.add_argument("--format", choices=["json", "markdown", "tex"], default="json")

    verify_claim = sub.add_parser("verify-claim", help="Verify a deterministic TRICE claim card.")
    verify_claim.add_argument("claim_card", type=Path)
    verify_claim.add_argument("--suite-result", type=Path, default=None)
    verify_claim.add_argument("--manifest", type=Path, default=None)

    schema = sub.add_parser("schema", help="Print a shipped TRICE JSON Schema.")
    schema.add_argument("name", choices=["patch", "manifest", "evidence", "suite", "bundle", "adapter", "adapter-profile", "receipt", "run-receipt", "claim", "claim-card", "readiness", "suite-readiness"])

    validate = sub.add_parser("validate-patch", help="Validate a deterministic patch spec.")
    validate.add_argument("patch_spec", type=Path)

    validate_adapter = sub.add_parser("validate-adapter", help="Validate a TRICE command adapter profile.")
    validate_adapter.add_argument("adapter_profile", type=Path)

    validate_receipt = sub.add_parser("validate-receipt", help="Validate a TRICE run receipt.")
    validate_receipt.add_argument("run_receipt", type=Path)

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
    if args.command == "bundle":
        forwarded = ["export", str(args.manifest)]
        if args.result is not None:
            forwarded += ["--result", str(args.result)]
        if args.out is not None:
            forwarded += ["--out", str(args.out)]
        return bundle_main(forwarded)
    if args.command == "verify-bundle":
        return bundle_main(["verify", str(args.bundle)])
    if args.command == "doctor":
        forwarded = ["--format", args.format, "--timeout-s", str(args.timeout_s)]
        if args.offline:
            forwarded.append("--offline")
        return doctor_main(forwarded)
    if args.command == "claim":
        forwarded = ["--suite-result", str(args.suite_result), "--out", str(args.out), "--format", args.format]
        if args.manifest is not None:
            forwarded += ["--manifest", str(args.manifest)]
        if args.scope is not None:
            forwarded += ["--scope", args.scope]
        return claim_main(forwarded)
    if args.command == "verify-claim":
        verdict = verify_claim_card_file(args.claim_card, suite_result_path=args.suite_result, manifest_path=args.manifest)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1
    if args.command == "schema":
        print(json.dumps(load_schema(args.name.replace("-", "_")), indent=2, sort_keys=True))
        return 0
    if args.command == "validate-patch":
        print(json.dumps(validate_patch_spec_file(args.patch_spec), indent=2, sort_keys=True))
        return 0
    if args.command == "validate-adapter":
        print(json.dumps(validate_adapter_profile_file(args.adapter_profile), indent=2, sort_keys=True))
        return 0
    if args.command == "validate-receipt":
        print(json.dumps(validate_run_receipt_file(args.run_receipt), indent=2, sort_keys=True))
        return 0
    if args.command == "validate-suite":
        print(json.dumps(validate_suite_manifest_file(args.suite_manifest), indent=2, sort_keys=True))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
