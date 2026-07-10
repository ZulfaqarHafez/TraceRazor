#!/bin/sh
# Select policy without overriding an explicit operator decision. A committed
# project policy wins over the trusted image fallback when /workspace is mounted.
set -eu

if [ -z "${TRACERAZOR_POLICY:-}" ]; then
  if [ -f /workspace/tracerazor.toml ]; then
    TRACERAZOR_POLICY=/workspace/tracerazor.toml
  else
    TRACERAZOR_POLICY="${TRACERAZOR_IMAGE_ROOT:?}/tracerazor.toml"
  fi
  export TRACERAZOR_POLICY
fi

exec tracerazor "$@"
