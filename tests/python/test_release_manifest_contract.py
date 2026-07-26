from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
RELEASE_MEASURE = ROOT / "tests" / "release_measure"
sys.path.insert(0, str(RELEASE_MEASURE))

import artifact_01_release_composition as artifact_gate  # noqa: E402
from release_manifest import load_release_manifest, render_release_matrix  # noqa: E402


def test_manifest_derives_bundle_count_and_generated_document() -> None:
    manifest = load_release_manifest(ROOT)

    assert manifest.standard_artifact_count == sum(
        len(distribution.wheels) + 1 for distribution in manifest.standard_distributions
    )
    assert (ROOT / "docs" / "releases" / "release-artifact-matrix.md").read_text(
        encoding="utf-8"
    ) == render_release_matrix(manifest)


def test_workflow_artifact_drift_names_distribution_and_platform() -> None:
    workflow = (ROOT / ".github" / "workflows" / "build_wheels.yml").read_text(
        encoding="utf-8"
    )
    mutated = workflow.replace(
        "artifact: cibw-cuda-windows-wheels",
        "artifact: stale-cuda-windows-wheels",
        1,
    )
    assert mutated != workflow

    with pytest.raises(
        AssertionError,
        match=r"gafime-cuda/win_amd64 build artifact .* "
        r"build_cuda_payload_wheels",
    ):
        artifact_gate._assert_release_manifest_workflow(mutated)


def test_optional_dependency_drift_names_distribution_and_extra() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mutated = copy.deepcopy(project)
    mutated["project"]["optional-dependencies"]["metal"] = [
        "gafime-metal==0; platform_system == 'Darwin'"
    ]

    with pytest.raises(
        AssertionError,
        match=r"release manifest gafime-metal extra 'metal' expects",
    ):
        artifact_gate._assert_release_manifest_pyproject(
            mutated, project["project"]["version"]
        )
