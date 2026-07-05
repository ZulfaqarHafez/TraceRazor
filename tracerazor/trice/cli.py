"""Public TRICE command-line entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tracerazor._trice.artifact import main as artifact_main, verify_artifact_card_file
from tracerazor._trice.bundle import main as bundle_main
from tracerazor._trice.claim import main as claim_main, verify_claim_card_file
from tracerazor._trice.contract import main as contract_main, verify_contract_card_file
from tracerazor._trice.crates import main as crates_main, verify_crates_card_file
from tracerazor._trice.design import main as design_main, verify_design_card_file
from tracerazor._trice.doctor import main as doctor_main
from tracerazor._trice.evidence import verify_manifest
from tracerazor._trice.integrity import main as integrity_main, verify_integrity_card_file
from tracerazor._trice.install import main as install_main, verify_install_card_file
from tracerazor._trice.live import main as live_main
from tracerazor._trice.protocol import main as protocol_main, verify_protocol_lock_file
from tracerazor._trice.receipt import validate_run_receipt_file
from tracerazor._trice.release import main as release_main, verify_release_card_file
from tracerazor._trice.release_evidence import main as release_evidence_main, verify_release_evidence_file
from tracerazor._trice.reproduction import main as reproduction_main, verify_reproduction_card_file
from tracerazor._trice.research import main as research_main, verify_research_card_file
from tracerazor._trice.schemas import load_schema, validate_adapter_profile_file, validate_patch_spec_file, validate_suite_manifest_file
from tracerazor._trice.suite import main as suite_main


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
    verify.add_argument("--result", type=Path, default=None)

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

    artifact = sub.add_parser("artifact", help="Generate a deterministic public artifact review card.")
    artifact.add_argument("--out", type=Path, default=Path("trice_artifact_card.json"))
    artifact.add_argument("--readiness", type=Path, default=Path("docs/trice_suite_readiness.json"))
    artifact.add_argument("--protocol", type=Path, default=Path("docs/trice_protocol_lock.json"))
    artifact.add_argument("--design", type=Path, default=Path("docs/trice_design_card.json"))
    artifact.add_argument("--reproduction", type=Path, default=Path("docs/trice_reproduction_card.json"))
    artifact.add_argument("--contract", type=Path, default=Path("docs/trice_contract_card.json"))
    artifact.add_argument("--install", type=Path, default=Path("docs/trice_install_card.json"))
    artifact.add_argument("--research", type=Path, default=Path("docs/trice_research_card.json"))
    artifact.add_argument("--claim", type=Path, default=Path("docs/trice_claim_card.json"))
    artifact.add_argument("--bundle", type=Path, default=Path("benchmark/trice/results/v2-broad-smoke/trice_broad_smoke_evidence.trice.zip"))
    artifact.add_argument("--paper-manifest", type=Path, default=Path("paper/trice_v3_research_manifest.json"))
    artifact.add_argument("--paper-result", type=Path, default=Path("benchmark/trice/results/v2-smoke/trice_v2_live_results.json"))
    artifact.add_argument("--paper-tex", type=Path, default=Path("paper/trice_v3_research_paper.tex"))
    artifact.add_argument("--paper-pdf", type=Path, default=Path("paper/trice_v3_research_paper.pdf"))
    artifact.add_argument("--readme", type=Path, default=Path("README.md"))
    artifact.add_argument("--library-doc", type=Path, default=Path("docs/trice_library.md"))
    artifact.add_argument("--format", choices=["json", "markdown", "tex"], default="json")

    verify_artifact = sub.add_parser("verify-artifact", help="Verify a deterministic public artifact review card.")
    verify_artifact.add_argument("artifact_card", type=Path)

    protocol = sub.add_parser("protocol", help="Generate a deterministic TRICE protocol lock.")
    protocol.add_argument("--manifest", type=Path, required=True)
    protocol.add_argument("--out", type=Path, default=Path("trice_protocol_lock.json"))
    protocol.add_argument("--protocol-id", default=None)
    protocol.add_argument("--format", choices=["json", "markdown", "tex"], default="json")

    verify_protocol = sub.add_parser("verify-protocol", help="Verify a deterministic TRICE protocol lock.")
    verify_protocol.add_argument("protocol_lock", type=Path)
    verify_protocol.add_argument("--manifest", type=Path, default=None)

    design = sub.add_parser("design", help="Generate a deterministic TRICE statistical design card.")
    design.add_argument("--protocol", type=Path, default=Path("docs/trice_protocol_lock.json"))
    design.add_argument("--suite-result", type=Path, default=Path("benchmark/trice/results/v2-broad-smoke/trice_suite_results.json"))
    design.add_argument("--out", type=Path, default=Path("trice_design_card.json"))
    design.add_argument("--format", choices=["json", "markdown", "tex"], default="json")

    verify_design = sub.add_parser("verify-design", help="Verify a deterministic TRICE statistical design card.")
    verify_design.add_argument("design_card", type=Path)
    verify_design.add_argument("--protocol", type=Path, default=None)
    verify_design.add_argument("--suite-result", type=Path, default=None)

    reproduction = sub.add_parser("reproduction", help="Generate a deterministic TRICE reproduction card.")
    reproduction.add_argument("--out", type=Path, default=Path("trice_reproduction_card.json"))
    reproduction.add_argument("--format", choices=["json", "markdown", "tex"], default="json")

    verify_reproduction = sub.add_parser("verify-reproduction", help="Verify a deterministic TRICE reproduction card.")
    verify_reproduction.add_argument("reproduction_card", type=Path)

    release = sub.add_parser("release", help="Generate a deterministic TRICE release card.")
    release.add_argument("--out", type=Path, default=Path("trice_release_card.json"))
    release.add_argument("--install", type=Path, default=Path("docs/trice_install_card.json"))
    release.add_argument("--offline", action="store_true")
    release.add_argument("--timeout-s", type=float, default=10.0)
    release.add_argument("--format", choices=["json", "markdown", "tex"], default="json")

    verify_release = sub.add_parser("verify-release", help="Verify a deterministic TRICE release card.")
    verify_release.add_argument("release_card", type=Path)

    release_evidence = sub.add_parser("release-evidence", help="Generate deterministic release checksums, SBOMs, and provenance.")
    release_evidence.add_argument("--out", type=Path, default=Path("trice_release_evidence.json"))
    release_evidence.add_argument("--dist-dir", type=Path, default=Path("dist"))
    release_evidence.add_argument("--cli-binary", type=Path, default=None)
    release_evidence.add_argument("--format", choices=["json", "markdown", "tex"], default="json")

    verify_release_evidence = sub.add_parser("verify-release-evidence", help="Verify deterministic release checksums, SBOMs, and provenance.")
    verify_release_evidence.add_argument("release_evidence", type=Path)

    integrity = sub.add_parser("integrity", help="Generate a deterministic TRICE proof-graph integrity card.")
    integrity.add_argument("--out", type=Path, default=Path("trice_integrity_card.json"))
    integrity.add_argument("--release-evidence", type=Path, default=Path("docs/trice_release_evidence.json"))
    integrity.add_argument("--release", type=Path, default=Path("docs/trice_release_card.json"))
    integrity.add_argument("--crates", type=Path, default=Path("docs/trice_crates_card.json"))
    integrity.add_argument("--install", type=Path, default=Path("docs/trice_install_card.json"))
    integrity.add_argument("--research", type=Path, default=Path("docs/trice_research_card.json"))
    integrity.add_argument("--format", choices=["json", "markdown", "tex"], default="json")

    verify_integrity = sub.add_parser("verify-integrity", help="Verify a deterministic TRICE proof-graph integrity card.")
    verify_integrity.add_argument("integrity_card", type=Path)

    research = sub.add_parser("research", help="Generate a deterministic TRICE research-basis card.")
    research.add_argument("--ledger", type=Path, default=Path("docs/trice_research_ledger.md"))
    research.add_argument("--out", type=Path, default=Path("trice_research_card.json"))
    research.add_argument("--min-sources", type=int, default=150)
    research.add_argument("--format", choices=["json", "markdown", "tex"], default="json")

    verify_research = sub.add_parser("verify-research", help="Verify a deterministic TRICE research-basis card.")
    verify_research.add_argument("research_card", type=Path)

    crates = sub.add_parser("crates", help="Generate a deterministic crates.io staged publish card.")
    crates.add_argument("--out", type=Path, default=Path("trice_crates_card.json"))
    crates.add_argument("--offline", action="store_true")
    crates.add_argument("--timeout-s", type=float, default=5.0)
    crates.add_argument("--format", choices=["json", "markdown", "tex"], default="json")

    verify_crates = sub.add_parser("verify-crates", help="Verify a deterministic crates.io staged publish card.")
    verify_crates.add_argument("crates_card", type=Path)

    install = sub.add_parser("install", help="Generate a deterministic clean-wheel installability card.")
    install.add_argument("--out", type=Path, default=Path("trice_install_card.json"))
    install.add_argument("--dist-dir", type=Path, default=Path("dist"))
    install.add_argument("--wheel", type=Path, default=None)
    install.add_argument("--python", default=None)
    install.add_argument("--timeout-s", type=float, default=120.0)
    install.add_argument("--format", choices=["json", "markdown", "tex"], default="json")

    verify_install = sub.add_parser("verify-install", help="Verify a deterministic clean-wheel installability card.")
    verify_install.add_argument("install_card", type=Path)

    contract = sub.add_parser("contract", help="Generate a deterministic public API/CLI/schema contract card.")
    contract.add_argument("--out", type=Path, default=Path("trice_contract_card.json"))
    contract.add_argument("--scope", default=None)
    contract.add_argument("--format", choices=["json", "markdown", "tex"], default="json")

    verify_contract = sub.add_parser("verify-contract", help="Verify a deterministic public contract card.")
    verify_contract.add_argument("contract_card", type=Path)

    schema = sub.add_parser("schema", help="Print a shipped TRICE JSON Schema.")
    schema.add_argument("name", choices=["patch", "manifest", "evidence", "suite", "bundle", "adapter", "adapter-profile", "receipt", "run-receipt", "claim", "claim-card", "readiness", "suite-readiness", "artifact", "artifact-card", "protocol", "protocol-lock", "design", "design-card", "reproduction", "reproduction-card", "release", "release-card", "release-evidence", "release-evidence-card", "integrity", "integrity-card", "research", "research-card", "crates", "crates-card", "install", "install-card", "contract", "contract-card"])

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
        verdict = verify_manifest(args.manifest, args.result)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1
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
    if args.command == "artifact":
        forwarded = [
            "--out",
            str(args.out),
            "--readiness",
            str(args.readiness),
            "--protocol",
            str(args.protocol),
            "--design",
            str(args.design),
            "--reproduction",
            str(args.reproduction),
            "--contract",
            str(args.contract),
            "--install",
            str(args.install),
            "--research",
            str(args.research),
            "--claim",
            str(args.claim),
            "--bundle",
            str(args.bundle),
            "--paper-manifest",
            str(args.paper_manifest),
            "--paper-result",
            str(args.paper_result),
            "--paper-tex",
            str(args.paper_tex),
            "--paper-pdf",
            str(args.paper_pdf),
            "--readme",
            str(args.readme),
            "--library-doc",
            str(args.library_doc),
            "--format",
            args.format,
        ]
        return artifact_main(forwarded)
    if args.command == "verify-artifact":
        verdict = verify_artifact_card_file(args.artifact_card)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1
    if args.command == "protocol":
        forwarded = ["--manifest", str(args.manifest), "--out", str(args.out), "--format", args.format]
        if args.protocol_id is not None:
            forwarded += ["--protocol-id", args.protocol_id]
        return protocol_main(forwarded)
    if args.command == "verify-protocol":
        verdict = verify_protocol_lock_file(args.protocol_lock, manifest_path=args.manifest)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1
    if args.command == "design":
        return design_main(["--protocol", str(args.protocol), "--suite-result", str(args.suite_result), "--out", str(args.out), "--format", args.format])
    if args.command == "verify-design":
        verdict = verify_design_card_file(args.design_card, protocol_path=args.protocol, suite_result_path=args.suite_result)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1
    if args.command == "reproduction":
        return reproduction_main(["--out", str(args.out), "--format", args.format])
    if args.command == "verify-reproduction":
        verdict = verify_reproduction_card_file(args.reproduction_card)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1
    if args.command == "release":
        forwarded = ["--out", str(args.out), "--install", str(args.install), "--timeout-s", str(args.timeout_s), "--format", args.format]
        if args.offline:
            forwarded.append("--offline")
        return release_main(forwarded)
    if args.command == "verify-release":
        verdict = verify_release_card_file(args.release_card)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1
    if args.command == "release-evidence":
        forwarded = ["--out", str(args.out), "--dist-dir", str(args.dist_dir), "--format", args.format]
        if args.cli_binary is not None:
            forwarded += ["--cli-binary", str(args.cli_binary)]
        return release_evidence_main(forwarded)
    if args.command == "verify-release-evidence":
        verdict = verify_release_evidence_file(args.release_evidence)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1
    if args.command == "integrity":
        return integrity_main(["--out", str(args.out), "--release", str(args.release), "--release-evidence", str(args.release_evidence), "--crates", str(args.crates), "--install", str(args.install), "--research", str(args.research), "--format", args.format])
    if args.command == "verify-integrity":
        verdict = verify_integrity_card_file(args.integrity_card)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1
    if args.command == "research":
        return research_main(["--ledger", str(args.ledger), "--out", str(args.out), "--min-sources", str(args.min_sources), "--format", args.format])
    if args.command == "verify-research":
        verdict = verify_research_card_file(args.research_card)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1
    if args.command == "crates":
        forwarded = ["--out", str(args.out), "--timeout-s", str(args.timeout_s), "--format", args.format]
        if args.offline:
            forwarded.append("--offline")
        return crates_main(forwarded)
    if args.command == "verify-crates":
        verdict = verify_crates_card_file(args.crates_card)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1
    if args.command == "install":
        forwarded = ["--out", str(args.out), "--dist-dir", str(args.dist_dir), "--timeout-s", str(args.timeout_s), "--format", args.format]
        if args.wheel is not None:
            forwarded += ["--wheel", str(args.wheel)]
        if args.python is not None:
            forwarded += ["--python", str(args.python)]
        return install_main(forwarded)
    if args.command == "verify-install":
        verdict = verify_install_card_file(args.install_card)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0 if verdict["ok"] else 1
    if args.command == "contract":
        forwarded = ["--out", str(args.out), "--format", args.format]
        if args.scope is not None:
            forwarded += ["--scope", args.scope]
        return contract_main(forwarded)
    if args.command == "verify-contract":
        verdict = verify_contract_card_file(args.contract_card)
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
