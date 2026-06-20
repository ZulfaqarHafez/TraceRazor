"""
Regenerates key published figures and checks them against docs/figures_manifest.json.
Usage:
  python -m benchmark.regen_manifest           # check mode (fails if drifted)
  python -m benchmark.regen_manifest --update  # update the manifest
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

MANIFEST_PATH = Path("docs/figures_manifest.json")
TOLERANCE = 0.02  # 2% tolerance


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text())


def check_figure(name, recomputed, committed, tol=TOLERANCE):
    if committed == 0:
        return recomputed == 0, f"{name}: committed=0, recomputed={recomputed}"
    pct_diff = abs(recomputed - committed) / abs(committed)
    ok = pct_diff <= tol
    status = "OK" if ok else f"DRIFT {pct_diff:.1%}"
    return ok, f"{name}: committed={committed}, recomputed={recomputed} [{status}]"


def regen_sample_trace_tas():
    """Recompute TAS for the bundled sample trace."""
    binary = shutil.which("tracerazor")
    if not binary:
        return None, "tracerazor binary not in PATH — skipping sample TAS check"
    result = subprocess.run(
        [binary, "audit", "traces/support-agent-run-2847.json", "--format", "json"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        report = json.loads(result.stdout)
        return report.get("score"), None
    return None, f"audit failed: {result.stderr[:200]}"


def main():
    update_mode = "--update" in sys.argv
    manifest = load_manifest()
    figures = manifest["figures"]
    results = []
    all_ok = True

    # Check sample trace TAS
    tas, err = regen_sample_trace_tas()
    if tas is not None:
        ok, msg = check_figure("sample_trace_tas", tas, figures["sample_trace_tas"])
        results.append(msg)
        if not ok:
            all_ok = False
    else:
        results.append(f"sample_trace_tas: SKIP ({err})")

    # Print results
    for r in results:
        print(r)

    if update_mode:
        print("\n[--update mode] Updating manifest with recomputed values...")
        if tas is not None:
            figures["sample_trace_tas"] = tas
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
        print(f"Updated {MANIFEST_PATH}")
    elif not all_ok:
        print("\nERROR: Some figures have drifted. Run with --update to refresh, or fix the code.")
        sys.exit(1)
    else:
        print("\nAll checked figures match committed manifest.")


if __name__ == "__main__":
    main()
