from __future__ import annotations

import copy
from fnmatch import fnmatchcase
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
from release_version import validate_project_versions  # noqa: E402


def test_manifest_derives_bundle_count_and_generated_document() -> None:
    manifest = load_release_manifest(ROOT)

    assert manifest.standard_artifact_count == sum(
        distribution.artifact_count
        for distribution in manifest.standard_distributions
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
        match=r"gafime-cuda/win_amd64 artifact .* "
        r"is absent from build_cuda_payload_wheels",
    ):
        artifact_gate._assert_release_manifest_workflow(mutated)


def test_cuda_linux_validation_accepts_auditwheel_multi_platform_tag() -> None:
    manifest = load_release_manifest(ROOT)
    cuda_linux = manifest.distribution("gafime-cuda").wheels[0]
    repaired_wheel = (
        "gafime_cuda-1.0.0b2-cp310-cp310-"
        "manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
    )

    assert fnmatchcase(repaired_wheel, cuda_linux.filename_pattern("3.10"))


def test_optional_dependency_drift_names_distribution_and_extra() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mutated = copy.deepcopy(project)
    mutated["project"]["optional-dependencies"]["metal"] = [
        "gafime-metal==0; platform_system == 'Darwin'"
    ]

    with pytest.raises(AssertionError, match=r"release manifest backend extras"):
        artifact_gate._assert_release_manifest_pyproject(
            mutated, project["project"]["version"]
        )


def test_release_tag_uses_semver_while_artifacts_use_pep440() -> None:
    release = validate_project_versions(ROOT)

    artifact_gate._assert_release_tag(
        ROOT,
        release,
        f"refs/tags/{release.tag}",
        None,
        False,
    )
    with pytest.raises(AssertionError, match="canonical SemVer tag|must equal"):
        artifact_gate._assert_release_tag(
            ROOT,
            release,
            f"refs/tags/v{release.pep440}",
            None,
            False,
        )


def test_rocm_policy_report_aggregates_every_cpython_wheel_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python_tags = ("cp310", "cp311", "cp312", "cp313", "cp314")
    artifacts = [
        artifact_gate.Artifact(
            path=tmp_path
            / f"gafime_rocm-1.0.0b2-{tag}-{tag}-linux_x86_64.whl",
            kind="wheel",
            distribution="gafime-rocm",
            version="1.0.0b2",
            metadata=None,
            members=frozenset(),
        )
        for tag in reversed(python_tags)
    ]

    def fake_wheel_report(
        artifact: artifact_gate.Artifact, _root: Path
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact": artifact.path.name,
            "wheel_bytes": 100,
            "wheel_uncompressed_bytes": 200,
            "native_payload_uncompressed_bytes": 150,
            "policy_sha256": "0" * 64,
            "wheel_policy": "system",
            "rocm_version": "7.2.3",
            "platform_tag": "linux_x86_64",
            "userspace_bundled": False,
            "required_sonames": ["libamdhip64.so.7"],
        }

    monkeypatch.setattr(
        artifact_gate, "_assert_rocm_system_wheel", fake_wheel_report
    )
    report = artifact_gate._rocm_system_policy_report(artifacts, ROOT)

    assert report["schema_version"] == 2
    assert report["wheel_count"] == len(python_tags)
    assert [wheel["artifact"] for wheel in report["wheels"]] == [
        f"gafime_rocm-1.0.0b2-{tag}-{tag}-linux_x86_64.whl"
        for tag in python_tags
    ]
