from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from pathlib import Path
import zipfile

from scripts.verify_wheel_licenses import verify_wheel
from tracerazor._trice.release_evidence import _cargo_sbom


REPO = Path(__file__).resolve().parent.parent


def test_python_metadata_uses_pep639_license_fields() -> None:
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE", "THIRD_PARTY_NOTICES.md"]
    assert "License :: OSI Approved :: MIT License" not in project["classifiers"]
    native_notices = (REPO / "crates" / "tracerazor-py" / "THIRD_PARTY_NOTICES.md").read_text(
        encoding="utf-8"
    )
    root_notices = (REPO / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert native_notices.split() == root_notices.split()


def test_product_distributions_exclude_external_research_corpora() -> None:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    excludes = pyproject["tool"]["hatch"]["build"]["exclude"]
    assert "/traces/external/**" in excludes
    assert "/benchmark/data/_agentinstruct_hf_sample.py" in excludes

    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY traces/ ./traces/" not in dockerfile
    assert "COPY traces/support-agent-run-2847.json ./traces/support-agent-run-2847.json" in dockerfile


def test_release_surfaces_carry_project_and_third_party_notices() -> None:
    workflow = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "cp LICENSE THIRD_PARTY_NOTICES.md standalone-package/" in workflow
    assert "Copy-Item LICENSE,THIRD_PARTY_NOTICES.md standalone-package" in workflow
    assert "COPY LICENSE THIRD_PARTY_NOTICES.md ./licenses/" in dockerfile
    assert "python scripts/verify_wheel_licenses.py dist" in workflow

    for bundle in (
        REPO / "plugins" / "tracerazor",
        REPO / "extensions" / "claude-code" / "tracerazor",
        REPO / "extensions" / "gemini-cli" / "tracerazor",
    ):
        assert (bundle / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


def test_manual_python_publisher_is_wheel_only() -> None:
    publish = (REPO / "publish.sh").read_text(encoding="utf-8")
    assert "python -m build --sdist" not in publish
    assert "twine upload dist/*.whl" in publish


def test_crate_archives_have_a_cross_platform_license_copy() -> None:
    for license_path in sorted((REPO / "crates").glob("tracerazor-*/LICENSE-MIT")):
        assert not license_path.is_symlink()
        assert license_path.read_text(encoding="utf-8").startswith("MIT License")
    assert len(list((REPO / "crates").glob("tracerazor-*/LICENSE-MIT"))) == 6


def test_cargo_sbom_contains_resolved_license_expressions() -> None:
    sbom = _cargo_sbom(REPO / "Cargo.lock")
    assert sbom["component_count"] > 0
    assert sbom["licensed_component_count"] == sbom["component_count"]
    assert all(component.get("licenses") for component in sbom["bom"]["components"])


def test_generated_artwork_has_durable_provenance() -> None:
    assert (REPO / "docs" / "assets" / "tracerazor-hero.webp").is_file()
    provenance = (REPO / "docs" / "assets" / "README.md").read_text(encoding="utf-8")
    assert "project-original artwork" in provenance
    assert "MIT license" in provenance


def test_wheel_verifier_checks_metadata_and_substantive_legal_files(tmp_path) -> None:
    wheel = tmp_path / "tracerazor-1.1.0-py3-none-any.whl"
    mit = (REPO / "LICENSE").read_text(encoding="utf-8")
    notices = (REPO / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    files = {
        "tracerazor-1.1.0.dist-info/METADATA": "Metadata-Version: 2.4\nName: tracerazor\nVersion: 1.1.0\n",
        "tracerazor-1.1.0.dist-info/licenses/LICENSE": mit,
        "tracerazor-1.1.0.dist-info/licenses/THIRD_PARTY_NOTICES.md": notices,
        "tracerazor/agent_assets/plugins/tracerazor/LICENSE": mit,
        "tracerazor/agent_assets/extensions/claude-code/tracerazor/LICENSE": mit,
        "tracerazor/agent_assets/extensions/gemini-cli/tracerazor/LICENSE": mit,
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)

    assert "METADATA License-Expression: MIT" in verify_wheel(wheel)

    files["tracerazor-1.1.0.dist-info/METADATA"] += (
        "License-Expression: MIT\n"
        "License-File: LICENSE\n"
        "License-File: THIRD_PARTY_NOTICES.md\n"
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)

    assert verify_wheel(wheel) == []
