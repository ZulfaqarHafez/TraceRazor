from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent


def test_release_is_evidence_gated_platform_only_and_immutable():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert 'python-version: ["3.10", "3.12"]' in workflow
    assert (
        'needs: [tag, quality, wheels, binaries, release-evidence, agent-image]'
        in workflow
    )
    assert 'needs: [tag, quality, release-evidence]' in workflow
    assert "python -m build --sdist" not in workflow
    assert "rm -rf tracerazor/bin" not in workflow
    assert "--wheel \"$LINUX_WHEEL\"" in workflow
    assert "release-assets/SHA256SUMS" in workflow
    assert "--clobber" not in workflow
    assert "refusing to replace changed asset" in workflow
    assert '"$TAG" != "v$PY_VERSION"' in workflow
    assert 'commit: ${{ steps.resolve.outputs.commit }}' in workflow
    assert 'git rev-list -n 1 "$TAG"' in workflow
    assert "Build native CLI for release-readiness tests" in workflow


def test_action_has_no_default_absolute_gate_and_verifies_release_checksum():
    action = (ROOT / ".github" / "actions" / "tracerazor" / "action.yml").read_text(
        encoding="utf-8"
    )
    resolver = (
        ROOT / ".github" / "actions" / "tracerazor" / "resolve-binary.sh"
    ).read_text(encoding="utf-8")
    threshold = action.split("  threshold:\n", 1)[1].split("  baseline-trace:\n", 1)[0]
    version = action.split("  version:\n", 1)[1].split("  release-repo:\n", 1)[0]
    assert 'default: ""' in threshold
    assert 'default: "v1.1.0"' in version
    assert "SHA256SUMS" in resolver
    assert "checksum mismatch" in resolver
    assert 'VERSION="${VERSION:-}"' in resolver
    assert "version is required" in resolver


def test_platform_wheel_contains_agent_assets_and_sample_trace():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for required in (
        '"schemas/tracerazor_event.schema.json"',
        '"traces/support-agent-run-2847.json"',
        '".agents/skills/tracerazor"',
        '"plugins/tracerazor"',
        '"extensions/claude-code/tracerazor"',
        '"extensions/gemini-cli/tracerazor"',
    ):
        assert required in pyproject


def test_platform_builds_remove_foreign_host_binaries_before_packaging():
    unix_builder = (ROOT / "scripts" / "build_platform_wheel.sh").read_text(
        encoding="utf-8"
    )
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert "rm -f tracerazor/bin/tracerazor tracerazor/bin/tracerazor.exe" in unix_builder
    assert (
        "Remove-Item tracerazor/bin/tracerazor,tracerazor/bin/tracerazor.exe"
        in release
    )
    assert 'GLIBC_BASELINE="${TRACERAZOR_EXPECTED_GLIBC:-}"' in unix_builder
    assert "BASH_REMATCH[1]" in unix_builder
    assert "BASH_REMATCH[2]" in unix_builder
    assert 'tracerazor/bin/tracerazor "$GLIBC_BASELINE"' in unix_builder
    assert "${TRACERAZOR_EXPECTED_GLIBC:?}" not in unix_builder


def test_agent_image_uses_supported_rust_and_non_root_runtime():
    dockerfile = (ROOT / "Dockerfile.agent").read_text(encoding="utf-8")

    assert (
        "FROM rust:1.88-bookworm@sha256:"
        "af306cfa71d987911a781c37b59d7d67d934f49684058f96cf72079c3626bfe0"
        " AS native-builder"
    ) in dockerfile
    python_base = (
        "python:3.12-slim-bookworm@sha256:"
        "8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b"
    )
    assert dockerfile.count(python_base) == 2
    assert "USER 10001:10001" in dockerfile
    assert "ARG TRACERAZOR_VERSION=dev" in dockerfile
    assert 'org.opencontainers.image.version="${TRACERAZOR_VERSION}"' in dockerfile
    assert 'org.opencontainers.image.revision="${VCS_REF}"' in dockerfile
    assert "ARG SOURCE_DATE_EPOCH=315532800" in dockerfile
    assert "requirements/agent-image-build.lock" in dockerfile
    assert "requirements/agent-image-runtime.lock" in dockerfile
    assert "--no-deps" in dockerfile
    assert "python -m pip check" in dockerfile
    assert "normalize-agent-image-receipts.py" in dockerfile
    assert "groupadd --gid 10001 tracerazor" in dockerfile
    assert "useradd --uid 10001 --gid 10001" in dockerfile
    assert "HOME=/home/tracerazor" in dockerfile
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile
    assert "COPY scripts/agent_image_entrypoint.sh" in dockerfile
    assert 'ENTRYPOINT ["tracerazor-agent-entrypoint"]' in dockerfile
    assert (
        "tracerazor agent install --host generic --scope image --mode coach --format json"
        in dockerfile
    )
    assert "/opt/tracerazor-image/install-receipt.json" in dockerfile
    assert "/opt/tracerazor-image/status-receipt.json" in dockerfile


def test_agent_image_dependency_locks_are_exact_and_mcp_cannot_drift():
    runtime = (
        ROOT / "requirements" / "agent-image-runtime.lock"
    ).read_text(encoding="utf-8")
    build = (
        ROOT / "requirements" / "agent-image-build.lock"
    ).read_text(encoding="utf-8")

    for lock in (runtime, build):
        requirements = [
            line.strip()
            for line in lock.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert requirements
        assert all(line.count("==") == 1 for line in requirements)
        assert not any(">=" in line or "~=" in line for line in requirements)
    assert "mcp==1.28.1" in runtime.splitlines()


def test_image_receipt_timestamp_normalization_uses_source_epoch(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"record": {"installed_at": "wall-clock"}}),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = "1700000000"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "normalize_agent_image_receipts.py"),
            str(receipt),
        ],
        env=env,
        check=True,
    )

    normalized = json.loads(receipt.read_text(encoding="utf-8"))
    assert normalized["record"]["installed_at"] == "2023-11-14T22:13:20+00:00"


def test_agent_image_smoke_is_digest_pinned_offline_and_read_only():
    smoke = (ROOT / "scripts" / "smoke_agent_image.sh").read_text(encoding="utf-8")
    entrypoint = (ROOT / "scripts" / "agent_image_entrypoint.sh").read_text(
        encoding="utf-8"
    )

    assert 'if [[ "$IMAGE_REF" != *@sha256:* ]]' in smoke
    assert 'docker image rm --force "$IMAGE_REF"' in smoke
    assert '--network none' in smoke
    assert '--read-only' in smoke
    assert '--cap-drop ALL' in smoke
    assert '--security-opt no-new-privileges' in smoke
    assert '"10001:10001"' in smoke
    assert 'dst=/workspace,readonly' in smoke
    assert 'doctor["policy"]["mode"] == "passive"' in smoke
    assert 'doctor["image_policy"]["mode"] == "coach"' in smoke
    assert 'importlib.metadata.version("mcp") == "1.28.1"' in smoke
    assert 'set(values) == {expected}' in smoke
    assert "agent status --host generic --scope image" in smoke
    assert "--entrypoint tracerazor-mcp" in smoke
    assert 'tracerazor audit "$trace" --hermetic --format json' in smoke
    assert '${TRACERAZOR_POLICY:-}' in entrypoint
    assert "/workspace/tracerazor.toml" in entrypoint
    assert '${TRACERAZOR_IMAGE_ROOT:?}/tracerazor.toml' in entrypoint
    assert 'exec tracerazor "$@"' in entrypoint


def test_agent_image_release_smokes_both_architectures_before_signed_promotion():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    image_job = workflow.split("  agent-image:\n", 1)[1].split(
        "\n  # Generate deterministic release evidence", 1
    )[0]

    assert "platforms: linux/amd64,linux/arm64" in image_job
    assert "build-${{ needs.tag.outputs.commit }}" in image_job
    assert "VCS_REF=${{ needs.tag.outputs.commit }}" in image_job
    assert "provenance: mode=max" in image_job
    assert "sbom: true" in image_job
    assert 'smoke_agent_image.sh "$IMAGE_NAME@$IMAGE_DIGEST" linux/amd64' in image_job
    assert 'smoke_agent_image.sh "$IMAGE_NAME@$IMAGE_DIGEST" linux/arm64' in image_job
    assert "uses: actions/attest@v4" in image_job
    assert "subject-digest: ${{ steps.push.outputs.digest }}" in image_job
    assert "push-to-registry: true" in image_job
    assert 'gh attestation verify "oci://$IMAGE_NAME@$IMAGE_DIGEST"' in image_job
    assert "--bundle-from-oci" in image_job
    assert '--signer-workflow "$GITHUB_REPOSITORY/.github/workflows/release.yml"' in image_job
    assert "--deny-self-hosted-runners" in image_job
    assert "refusing to replace changed image" in image_job
    assert '"schema_version": "tracerazor-agent-image-release/v1"' in image_job
    assert '"immutable_ref": f"{image}@{digest}"' in image_job
    assert '"source_revision": os.environ["TAG_COMMIT"]' in image_job
    assert "name: agent-image-release-receipt" in image_job
    assert "SOURCE_DATE_EPOCH=${{ steps.source-date.outputs.epoch }}" in image_job
    assert "Prove GHCR is publicly readable at the exact digest" in image_job
    assert "docker logout ghcr.io" in image_job
    assert 'docker pull --quiet --platform linux/amd64 "$IMAGE_NAME@$IMAGE_DIGEST"' in image_job
    assert 'case "$http_code" in' in image_job
    assert "404)" in image_job
    assert "refusing to treat it as absent" in image_job

    smoke_position = image_job.index("Verify manifest and smoke the exact digest")
    attest_position = image_job.index("Sign image provenance")
    receipt_position = image_job.index("Write deterministic image release receipt")
    upload_position = image_job.index("name: agent-image-release-receipt")
    public_position = image_job.index("Prove GHCR is publicly readable")
    promote_position = image_job.index("Promote tested digest")
    assert (
        smoke_position
        < attest_position
        < receipt_position
        < upload_position
        < public_position
        < promote_position
    )
    assert image_job.rstrip().endswith('test "$latest_digest" = "$IMAGE_DIGEST"')

    publish = workflow.split("  publish-release:\n", 1)[1]
    assert "Bind the image receipt into the release checksum manifest" in publish
    assert "artifacts/**/agent-image-release.json" in publish
    assert "-name 'agent-image-release.json'" in publish


def test_release_keeps_all_five_native_platform_wheels():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    wheels = workflow.split("  wheels:\n", 1)[1].split("\n  # Standalone CLI", 1)[0]

    for artifact in (
        "linux-x64",
        "linux-arm64",
        "macos-arm64",
        "macos-x64",
        "windows-x64",
    ):
        assert f"artifact: {artifact}" in wheels
    assert wheels.count("          - os:") == 5
    assert "macos-13" not in workflow
    assert workflow.count("          - os: macos-15-intel") == 2
    assert 'glibc: "2.35"' in wheels
    assert "wheel-platform: manylinux_2_35_x86_64" in wheels
    assert 'glibc: "2.39"' in wheels
    assert "wheel-platform: manylinux_2_39_aarch64" in wheels
    assert "TRACERAZOR_EXPECTED_GLIBC" in wheels
    assert "Clean-room smoke on declared platform baseline" in wheels


def test_trice_integrity_uses_the_linux_release_wheel_baseline():
    workflow = (ROOT / ".github" / "workflows" / "tracerazor.yml").read_text(
        encoding="utf-8"
    )
    integrity = workflow.split("  trice-integrity:\n", 1)[1].split(
        "\n  #", 1
    )[0]

    assert "runs-on: ubuntu-22.04" in integrity
    assert 'TRACERAZOR_EXPECTED_GLIBC: "2.35"' in integrity
    assert (
        "TRACERAZOR_EXPECTED_WHEEL_PLATFORM: manylinux_2_35_x86_64"
        in integrity
    )
    for card in ("crates", "install", "research"):
        assert f"--out /tmp/trice_{card}_card.json" in integrity
        assert f"--{card} /tmp/trice_{card}_card.json" in integrity
        assert f"--out docs/trice_{card}_card.json" not in integrity
    assert "git diff --exit-code" in integrity


def test_proof_bound_worktree_bytes_match_the_git_index():
    if not (ROOT / ".git").exists():
        return

    paths = [
        "README.md",
        "pyproject.toml",
        "Cargo.toml",
        "crates/tracerazor-cli/Cargo.toml",
        "crates/tracerazor-core/Cargo.toml",
        "crates/tracerazor-ingest/Cargo.toml",
        "crates/tracerazor-semantic/Cargo.toml",
        "crates/tracerazor-server/Cargo.toml",
        "crates/tracerazor-store/Cargo.toml",
        "docs/public_trust_matrix.md",
        "docs/release_checklist.md",
        "benchmark/trice/results/v2-smoke/trice_v2_live_results.json",
    ]
    paths.extend(
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "examples").glob("trice_*"))
        if path.is_file()
    )
    paths.extend(
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "benchmark" / "live" / "tasks").rglob("*"))
        if path.is_file()
    )

    mismatches = []
    for relative in paths:
        indexed = subprocess.run(
            ["git", "show", f":{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if (ROOT / relative).read_bytes() != indexed:
            mismatches.append(relative)
    assert mismatches == []


def test_release_fails_on_existing_pypi_files_and_evidence_uses_downloaded_cli():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert "skip-existing: true" not in workflow

    evidence = workflow.split("  release-evidence:\n", 1)[1].split(
        "\n  # Attach wheels", 1
    )[0]
    assert "Extract downloaded standalone CLI for evidence subject" in evidence
    assert "tracerazor-x86_64-unknown-linux-gnu.tar.gz" in evidence
    assert 'tar -xzf "$archive" -C release-cli' in evidence
    assert "--cli-binary release-cli/tracerazor" in evidence
    assert "cargo build --release -p tracerazor" not in evidence


def test_docker_context_excludes_foreign_binaries_but_keeps_agent_assets():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "tracerazor/bin/tracerazor\n" in dockerignore
    assert "tracerazor/bin/tracerazor.exe\n" in dockerignore
    assert "!README.md" in dockerignore
    assert "!skills/tracerazor/SKILL.md" in dockerignore
    assert "!plugins/tracerazor/skills/tracerazor/SKILL.md" in dockerignore
