#!/usr/bin/env bash
# Resolve the tracerazor binary for the action (ship-plan 4.2).
#
# Priority: BINARY_PATH input > prebuilt release download. Building from
# source is a separate, explicit opt-in step in action.yml — the default
# path must work from any repo with no Rust toolchain.
#
# Inputs via environment:
#   BINARY_PATH    explicit path to an existing binary ("" = none)
#   VERSION        release tag (e.g. "v0.4.0") or "latest"
#   RELEASE_REPO   owner/repo hosting the release assets
#   DEST_DIR       where the downloaded binary is unpacked
#
# Output: writes `path=<binary>` to $GITHUB_OUTPUT (or stdout when unset).
set -euo pipefail

BINARY_PATH="${BINARY_PATH:-}"
VERSION="${VERSION:-latest}"
RELEASE_REPO="${RELEASE_REPO:-ZulfaqarHafez/TraceRazor}"
DEST_DIR="${DEST_DIR:-${RUNNER_TEMP:-/tmp}/tracerazor-bin}"

emit() {
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "path=$1" >> "$GITHUB_OUTPUT"
  fi
  echo "$1"
}

if [ -n "$BINARY_PATH" ]; then
  if [ ! -x "$BINARY_PATH" ]; then
    echo "::error::binary-path was set but is not an executable file: $BINARY_PATH" >&2
    exit 2
  fi
  emit "$BINARY_PATH"
  exit 0
fi

# Archive layout defaults match the Unix release assets; the Windows arm
# overrides them (release.yml ships a .zip holding tracerazor.exe).
ARCHIVE_EXT="tar.gz"
BIN_NAME="tracerazor"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)   TRIPLE="x86_64-unknown-linux-gnu" ;;
  Linux-aarch64)  TRIPLE="aarch64-unknown-linux-gnu" ;;
  Darwin-arm64)   TRIPLE="aarch64-apple-darwin" ;;
  Darwin-x86_64)  TRIPLE="x86_64-apple-darwin" ;;
  # Git-bash / MSYS / Cygwin shells on a Windows runner. `uname -m` is x86_64;
  # the only Windows asset release.yml builds is x86_64-pc-windows-msvc.
  MINGW*|MSYS*|CYGWIN*|Windows_NT*)
    TRIPLE="x86_64-pc-windows-msvc"
    ARCHIVE_EXT="zip"
    BIN_NAME="tracerazor.exe"
    ;;
  *)
    echo "::error::unsupported runner platform $(uname -s)-$(uname -m); pass binary-path or build-from-source: true" >&2
    exit 2
    ;;
esac

ASSET="tracerazor-${TRIPLE}.${ARCHIVE_EXT}"
if [ "$VERSION" = "latest" ]; then
  URL="https://github.com/${RELEASE_REPO}/releases/latest/download/${ASSET}"
else
  URL="https://github.com/${RELEASE_REPO}/releases/download/${VERSION}/${ASSET}"
fi

mkdir -p "$DEST_DIR"
echo "Downloading ${URL}" >&2
if ! curl -fsSL --retry 3 --retry-delay 2 -o "$DEST_DIR/$ASSET" "$URL"; then
  echo "::error::could not download prebuilt binary from ${URL}. Either the release has no asset for ${TRIPLE}, or the tag is wrong. Alternatives: pass binary-path, or set build-from-source: true (requires a Rust toolchain and the TraceRazor sources)." >&2
  exit 2
fi
# The Windows asset is a .zip; extract with unzip when present, else bsdtar
# (`tar -xf` reads zips on Windows runners). Unix assets stay gzip tarballs.
if [ "$ARCHIVE_EXT" = "zip" ]; then
  if command -v unzip >/dev/null 2>&1; then
    unzip -o "$DEST_DIR/$ASSET" -d "$DEST_DIR" >/dev/null
  else
    tar -xf "$DEST_DIR/$ASSET" -C "$DEST_DIR"
  fi
else
  tar -xzf "$DEST_DIR/$ASSET" -C "$DEST_DIR"
fi
BIN="$DEST_DIR/$BIN_NAME"
if [ ! -x "$BIN" ]; then
  echo "::error::downloaded archive did not contain an executable '${BIN_NAME}' binary" >&2
  exit 2
fi
emit "$BIN"
