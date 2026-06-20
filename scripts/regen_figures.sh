#!/usr/bin/env bash
# Regenerate all published figures. Run from project root.
set -e
echo "=== TraceRazor: Regenerating published figures ==="

echo "--- metric_effectiveness ---"
python -m benchmark.metric_effectiveness 2>&1 | tee /tmp/tr_metric_eff.txt || echo "SKIP: benchmark not runnable in this environment"

echo "--- HuggingFace audit stats ---"
python -m benchmark.hf_audit_stats 2>&1 | tee /tmp/tr_hf_audit.txt || echo "SKIP: hf audit stats not available"

echo "--- Sample trace TAS (requires built binary) ---"
if command -v tracerazor &>/dev/null; then
  tracerazor audit traces/support-agent-run-2847.json --format json | python -c "import sys,json; r=json.load(sys.stdin); print(f'TAS={r[\"score\"]}')"
else
  echo "SKIP: tracerazor binary not in PATH"
fi

echo "=== Done. Review /tmp/tr_*.txt for any drifted figures. ==="
