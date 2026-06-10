#!/usr/bin/env bash
# TraceRazor action core logic (ship-plan 4.2).
#
# JSON-first contract: the score is parsed from the audit's JSON report; a
# malformed or missing report is a hard step failure (exit 2) — never a
# silent score of 0. Audit exit 1 (threshold gate) and compare exit 1
# (regression gate) propagate as exit 1.
#
# Inputs via environment:
#   TR_BIN                (required) path to the tracerazor binary
#   TRACE_FILE            (required) trace JSON to audit
#   THRESHOLD             minimum TAS; empty = no gate
#   REPORT_FORMAT         "markdown" (default) or "json" — format of the
#                         `report` output; scoring always uses JSON
#   BASELINE_TRACE        optional baseline for `compare` regression gating
#   REGRESSION_THRESHOLD  per-metric % drop that fails the gate (default 10)
#   OUT_DIR               where report.json / report.md / comment.md land
#   GITHUB_OUTPUT, GITHUB_STEP_SUMMARY  honoured when set
set -euo pipefail

: "${TR_BIN:?TR_BIN is required}"
: "${TRACE_FILE:?TRACE_FILE is required}"
THRESHOLD="${THRESHOLD:-}"
REPORT_FORMAT="${REPORT_FORMAT:-markdown}"
BASELINE_TRACE="${BASELINE_TRACE:-}"
REGRESSION_THRESHOLD="${REGRESSION_THRESHOLD:-10}"
OUT_DIR="${OUT_DIR:-.}"
mkdir -p "$OUT_DIR"

fail() {
  echo "::error::$1" >&2
  exit 2
}

[ -x "$TR_BIN" ] || fail "tracerazor binary not found or not executable: $TR_BIN"
[ -f "$TRACE_FILE" ] || fail "trace file not found: $TRACE_FILE"
command -v python3 >/dev/null 2>&1 || fail "python3 is required to parse the report JSON"

# ── Audit (hermetic; JSON is the source of truth) ────────────────────────────
AUDIT_ARGS=(audit "$TRACE_FILE" --format json --hermetic)
if [ -n "$THRESHOLD" ]; then
  AUDIT_ARGS+=(--threshold "$THRESHOLD")
fi

set +e
"$TR_BIN" "${AUDIT_ARGS[@]}" > "$OUT_DIR/report.json" 2> "$OUT_DIR/audit.stderr"
AUDIT_EXIT=$?
set -e

if [ "$AUDIT_EXIT" -ge 2 ]; then
  cat "$OUT_DIR/audit.stderr" >&2 || true
  fail "tracerazor audit errored (exit $AUDIT_EXIT) — bad input or IO; refusing to report a score"
fi
if [ ! -s "$OUT_DIR/report.json" ]; then
  cat "$OUT_DIR/audit.stderr" >&2 || true
  fail "audit produced no report (trace below the --min-steps floor?); refusing to report a score"
fi

# Parse the JSON report. Any parse failure is a hard error.
PARSED=$(python3 - "$OUT_DIR/report.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d["score"]["score"])
print(d["score"]["grade"])
print(d["savings"]["tokens_saved"])
print(d.get("summary_oneliner", ""))
PY
) || fail "audit output is not a valid report JSON — refusing to report a score"
TAS=$(sed -n 1p <<<"$PARSED")
GRADE=$(sed -n 2p <<<"$PARSED")
TOKENS_SAVED=$(sed -n 3p <<<"$PARSED")
ONELINER=$(sed -n 4p <<<"$PARSED")

PASSES=$([ "$AUDIT_EXIT" -eq 0 ] && echo "true" || echo "false")

# ── Optional regression gate against a baseline trace ────────────────────────
REGRESSION_DETECTED="false"
COMPARE_SUMMARY=""
if [ -n "$BASELINE_TRACE" ]; then
  [ -f "$BASELINE_TRACE" ] || fail "baseline trace not found: $BASELINE_TRACE"
  set +e
  "$TR_BIN" compare "$BASELINE_TRACE" "$TRACE_FILE" \
    --format json --regression-threshold "$REGRESSION_THRESHOLD" \
    > "$OUT_DIR/compare.json" 2> "$OUT_DIR/compare.stderr"
  COMPARE_EXIT=$?
  set -e
  if [ "$COMPARE_EXIT" -ge 2 ]; then
    cat "$OUT_DIR/compare.stderr" >&2 || true
    fail "tracerazor compare errored (exit $COMPARE_EXIT)"
  fi
  [ "$COMPARE_EXIT" -eq 1 ] && REGRESSION_DETECTED="true"
  COMPARE_SUMMARY=$(python3 - "$OUT_DIR/compare.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
delta = d["delta"]
lines = [f"**TAS delta:** {delta['tas']:+.1f}  ·  **Token delta:** {delta['tokens']:+d}", ""]
regs = d.get("regressions", [])
if regs:
    lines.append("| Regressed metric | Drop |")
    lines.append("|---|---|")
    for r in regs:
        lines.append(f"| {r['metric']} | {abs(r['delta']) * 100:.1f}% |")
else:
    lines.append("No per-metric regressions beyond the threshold.")
print("\n".join(lines))
PY
) || fail "compare output is not valid JSON"
fi

# ── Human-readable report for the `report` output ────────────────────────────
if [ "$REPORT_FORMAT" = "markdown" ]; then
  "$TR_BIN" audit "$TRACE_FILE" --format markdown --hermetic \
    > "$OUT_DIR/report.md" 2>/dev/null \
    || fail "markdown report run failed"
  REPORT_PATH="$OUT_DIR/report.md"
else
  REPORT_PATH="$OUT_DIR/report.json"
fi

# ── PR comment body (sticky marker on line 1) ─────────────────────────────────
GATE_LINE="no threshold gate configured"
if [ -n "$THRESHOLD" ]; then
  if [ "$PASSES" = "true" ]; then
    GATE_LINE="threshold ${THRESHOLD}: ✅ pass"
  else
    GATE_LINE="threshold ${THRESHOLD}: ❌ **FAIL**"
  fi
fi
{
  echo '<!-- tracerazor-report -->'
  echo '## TraceRazor Efficiency Report'
  echo
  echo "**TAS:** ${TAS} [${GRADE}] — ${GATE_LINE}"
  echo
  if [ -n "$ONELINER" ]; then
    echo "> ${ONELINER}"
    echo
  fi
  if [ -n "$COMPARE_SUMMARY" ]; then
    echo '### Regression check vs baseline'
    echo
    if [ "$REGRESSION_DETECTED" = "true" ]; then
      echo "❌ **Regression beyond ${REGRESSION_THRESHOLD}% detected.**"
    else
      echo "✅ No regression beyond ${REGRESSION_THRESHOLD}%."
    fi
    echo
    echo "$COMPARE_SUMMARY"
    echo
  fi
  echo '<details><summary>Full report</summary>'
  echo
  echo '```'
  cat "$REPORT_PATH"
  echo '```'
  echo
  echo '</details>'
} > "$OUT_DIR/comment.md"

# ── Outputs ───────────────────────────────────────────────────────────────────
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "tas-score=$TAS"
    echo "grade=$GRADE"
    echo "passes=$PASSES"
    echo "tokens-saved=$TOKENS_SAVED"
    echo "regression-detected=$REGRESSION_DETECTED"
    echo "report-json-path=$OUT_DIR/report.json"
    DELIM="EOF_${RANDOM}${RANDOM}"
    echo "report<<$DELIM"
    cat "$REPORT_PATH"
    echo "$DELIM"
  } >> "$GITHUB_OUTPUT"
fi
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "$OUT_DIR/comment.md" >> "$GITHUB_STEP_SUMMARY"
fi

echo "TAS: $TAS [$GRADE]  passes=$PASSES  regression-detected=$REGRESSION_DETECTED"

# ── Gate ──────────────────────────────────────────────────────────────────────
if [ "$AUDIT_EXIT" -ne 0 ]; then
  echo "::error::TAS $TAS is below threshold $THRESHOLD"
  exit 1
fi
if [ "$REGRESSION_DETECTED" = "true" ]; then
  echo "::error::per-metric regression beyond ${REGRESSION_THRESHOLD}% vs ${BASELINE_TRACE}"
  exit 1
fi
