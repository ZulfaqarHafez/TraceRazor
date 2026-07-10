#!/usr/bin/env bash
# Smoke the exact OCI image digest that will be promoted as a TraceRazor release.
# The container receives no network, capabilities, or writable root filesystem.
set -euo pipefail

IMAGE_REF="${1:?usage: smoke_agent_image.sh <image-ref-at-digest> <linux/amd64|linux/arm64>}"
PLATFORM="${2:?usage: smoke_agent_image.sh <image-ref-at-digest> <linux/amd64|linux/arm64>}"

case "$PLATFORM" in
  linux/amd64|linux/arm64) ;;
  *) echo "unsupported smoke platform: $PLATFORM" >&2; exit 2 ;;
esac

if [[ "$IMAGE_REF" != *@sha256:* ]]; then
  echo "image must be pinned by digest, got: $IMAGE_REF" >&2
  exit 2
fi

tmp_dir=$(mktemp -d)
probe_container=""
cleanup() {
  if [[ -n "$probe_container" ]]; then
    docker rm -f "$probe_container" >/dev/null 2>&1 || true
  fi
  # Docker's classic image store cannot retain two platform variants under
  # the same manifest-list digest. Each invocation must release its pulled
  # platform before the next architecture smoke uses the identical digest.
  docker image rm --force "$IMAGE_REF" >/dev/null 2>&1 || true
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

docker pull --platform "$PLATFORM" "$IMAGE_REF"

probe_container=$(docker create --platform "$PLATFORM" "$IMAGE_REF")
configured_user=$(docker inspect --format '{{.Config.User}}' "$probe_container")
if [[ "$configured_user" != "10001:10001" ]]; then
  echo "agent image must default to uid/gid 10001:10001, got: $configured_user" >&2
  exit 1
fi
version_label=$(docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.version" }}' "$probe_container")
revision_label=$(docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "$probe_container")
if [[ ! "$version_label" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "agent image version label is not a release version: $version_label" >&2
  exit 1
fi
if [[ ! "$revision_label" =~ ^[0-9a-f]{40}$ ]]; then
  echo "agent image revision label is not a full Git commit: $revision_label" >&2
  exit 1
fi
docker rm "$probe_container" >/dev/null
probe_container=""

run_flags=(
  --rm
  --platform "$PLATFORM"
  --network none
  --read-only
  --cap-drop ALL
  --security-opt no-new-privileges
  --pids-limit 128
  --tmpfs /tmp:rw,noexec,nosuid,size=64m
)

# The default command must remain a read-only doctor operation.
docker run "${run_flags[@]}" "$IMAGE_REF" > "$tmp_dir/doctor.json"
TRACERAZOR_EXPECTED_VERSION="$version_label" python3 - "$tmp_dir/doctor.json" <<'PY'
import json
import os
import pathlib
import sys

doctor = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert doctor["schema_version"] == 1
assert doctor["command"] == "doctor"
assert doctor["ok"] is True
assert doctor["version"] == os.environ["TRACERAZOR_EXPECTED_VERSION"].removeprefix("v")
assert doctor["image_root"] == "/opt/tracerazor-image"
policy = doctor["image_policy"]
assert policy["exists"] is True
assert policy["valid"] is True
assert policy["mode"] == "coach"
assert policy["capture"] == "auto"
assert policy["privacy"] == "local-redacted"
assert policy["persist_raw_content"] is False
assert policy["enforcement_enabled"] is False
PY

# A committed project policy must override the image fallback without changing
# the baked image policy or requiring a writable layer.
mkdir -p "$tmp_dir/project"
cat > "$tmp_dir/project/tracerazor.toml" <<'TOML'
schema_version = 1
mode = "passive"
capture = "auto"
hermetic = true
privacy = "local-redacted"
persist_raw_content = false
artifact_dir = ".tracerazor/runs"
min_steps = 5

[quality]
verifier = ""

[enforcement]
enabled = false
TOML
docker run "${run_flags[@]}" \
  --mount "type=bind,src=$tmp_dir/project,dst=/workspace,readonly" \
  "$IMAGE_REF" > "$tmp_dir/project-doctor.json"
python3 - "$tmp_dir/project-doctor.json" <<'PY'
import json
import pathlib
import sys

doctor = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert doctor["ok"] is True
assert doctor["policy"]["path"] == "/workspace/tracerazor.toml"
assert doctor["policy"]["mode"] == "passive"
assert doctor["image_policy"]["mode"] == "coach"
PY

# Provisioning must be ownership-recorded and healthy before agents inherit it.
docker run "${run_flags[@]}" "$IMAGE_REF" \
  agent status --host generic --scope image --format json > "$tmp_dir/status.json"
python3 - "$tmp_dir/status.json" <<'PY'
import json
import pathlib
import sys

status = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert status["schema_version"] == 1
assert status["command"] == "status"
assert status["ok"] is True
assert status["installed"] is True
assert status["healthy"] is True
assert status["host"] == "generic"
assert status["scope"] == "image"
assert len(status["installations"]) == 1
PY

docker run "${run_flags[@]}" --entrypoint /bin/sh "$IMAGE_REF" -ec '
  test "$(id -u):$(id -g)" = "10001:10001"
  test -s /opt/tracerazor-image/install-receipt.json
  test -s /opt/tracerazor-image/status-receipt.json
  test -s /opt/tracerazor-image/tracerazor.toml
  test -s /opt/tracerazor-image/.tracerazor/agent-install.json
'

docker run "${run_flags[@]}" --entrypoint python "$IMAGE_REF" -c '
import datetime
import importlib.metadata
import json
import os
import pathlib

expected = datetime.datetime.fromtimestamp(
    int(os.environ["SOURCE_DATE_EPOCH"]),
    tz=datetime.timezone.utc,
).isoformat()

def installed_at(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "installed_at":
                yield child
            else:
                yield from installed_at(child)
    elif isinstance(value, list):
        for child in value:
            yield from installed_at(child)

for name in (
    ".tracerazor/agent-install.json",
    "install-receipt.json",
    "status-receipt.json",
):
    payload = json.loads(
        pathlib.Path("/opt/tracerazor-image", name).read_text(encoding="utf-8")
    )
    values = list(installed_at(payload))
    assert values and set(values) == {expected}, (name, values, expected)
assert importlib.metadata.version("mcp") == "1.28.1"
'

docker run "${run_flags[@]}" --entrypoint tracerazor-mcp "$IMAGE_REF" \
  --selftest > "$tmp_dir/mcp.json"
python3 - "$tmp_dir/mcp.json" <<'PY'
import json
import pathlib
import sys

selftest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert isinstance(selftest, list)
names = {tool["name"] for tool in selftest}
assert len(names) >= 13
assert {"doctor", "audit_trace", "audit_current_run", "verify_evidence"} <= names
PY

# Audit the sample from package resources without a checkout or writable store.
docker run "${run_flags[@]}" --entrypoint /bin/sh "$IMAGE_REF" -ec '
  trace=$(python -c "from importlib.resources import files; print(files(\"tracerazor\").joinpath(\"agent_assets/traces/support-agent-run-2847.json\"))")
  exec tracerazor audit "$trace" --hermetic --format json
' > "$tmp_dir/report.json"
python3 - "$tmp_dir/report.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["schema_version"] == "tracerazor-report/v1"
assert report["trace_id"] == "support-agent-run-2847"
assert report["total_steps"] >= 5
PY

echo "TraceRazor agent image smoke passed for $PLATFORM at $IMAGE_REF"
