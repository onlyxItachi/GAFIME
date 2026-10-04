from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "release_measure"))

import artifact_01_release_composition as artifact_gate  # noqa: E402


@pytest.fixture
def rocm_system_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[artifact_gate.Artifact, dict[str, tuple[str, ...]]]:
    policy = json.loads(artifact_gate.ROCM_SYSTEM_POLICY.read_text(encoding="utf-8"))
    path = tmp_path / "gafime_rocm-1.0.0-cp311-cp311-linux_x86_64.whl"
    native_member = "gafime_rocm/libgafime_rocm.so"
    # Synthetic archive: exercise policy validation without compiling/loading HIP.
    payload = (
        b"GAFIME_ROCM_BUILD_INFO:arch="
        + ",".join(policy["gfx_targets"]).encode("ascii")
        + b";wave_mi_mask=3;"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(native_member, payload)
    artifact = artifact_gate.Artifact(
        path=path,
        kind="wheel",
        distribution="gafime-rocm",
        version="1.0.0",
        metadata=None,
        members=frozenset({native_member}),
        platforms=frozenset({"linux_x86_64"}),
        build_policy=policy,
    )
    dynamic = {
        "NEEDED": (
            "ld-linux-x86-64.so.2",
            "libamdhip64.so.7",
            "libc.so.6",
            "libgcc_s.so.1",
            "libm.so.6",
            "libstdc++.so.6",
        ),
        "SONAME": (),
        "RPATH": (),
        "RUNPATH": (),
    }
    monkeypatch.setattr(artifact_gate, "_readelf_dynamic", lambda _path: dynamic)
    return artifact, dynamic


@pytest.mark.parametrize("thread_dependency", [(), ("libpthread.so.0",)])
def test_rocm_system_wheel_accepts_optional_host_pthread(
    rocm_system_wheel, thread_dependency: tuple[str, ...]
) -> None:
    artifact, dynamic = rocm_system_wheel
    dynamic["NEEDED"] += thread_dependency

    report = artifact_gate._assert_rocm_system_wheel(artifact, ROOT)

    assert report["required_sonames"] == ["libamdhip64.so.7"]
    assert report["userspace_bundled"] is False


@pytest.mark.parametrize(
    "undeclared",
    ["libpthread.so.1", "libpthread-private.so.0", "libhsa-runtime64.so.1"],
)
def test_rocm_system_wheel_still_rejects_undeclared_dependencies(
    rocm_system_wheel, undeclared: str
) -> None:
    artifact, dynamic = rocm_system_wheel
    dynamic["NEEDED"] += ("libpthread.so.0", undeclared)

    with pytest.raises(AssertionError, match="undeclared direct dependencies"):
        artifact_gate._assert_rocm_system_wheel(artifact, ROOT)


@pytest.mark.parametrize("search_tag", ["RPATH", "RUNPATH"])
def test_rocm_system_wheel_pthread_does_not_allow_private_search_paths(
    rocm_system_wheel, search_tag: str
) -> None:
    artifact, dynamic = rocm_system_wheel
    dynamic["NEEDED"] += ("libpthread.so.0",)
    dynamic[search_tag] = ("$ORIGIN/.libs",)

    with pytest.raises(AssertionError, match="embeds a runtime search path"):
        artifact_gate._assert_rocm_system_wheel(artifact, ROOT)


@pytest.mark.parametrize(
    "hip_dependencies", [(), ("libamdhip64.so.6",), ("libamdhip64.so.7",) * 2]
)
def test_rocm_system_wheel_pthread_preserves_exact_hip_soname(
    rocm_system_wheel, hip_dependencies: tuple[str, ...]
) -> None:
    artifact, dynamic = rocm_system_wheel
    dynamic["NEEDED"] = ("libc.so.6", "libpthread.so.0") + hip_dependencies

    with pytest.raises(AssertionError, match="runtime dependency differs from policy"):
        artifact_gate._assert_rocm_system_wheel(artifact, ROOT)


@pytest.mark.parametrize(
    "vendored_member",
    ["gafime_rocm.libs/libpthread.so.0", "gafime_rocm/libamdhip64.so.7"],
)
def test_rocm_system_wheel_pthread_does_not_allow_vendoring(
    rocm_system_wheel, vendored_member: str
) -> None:
    artifact, dynamic = rocm_system_wheel
    dynamic["NEEDED"] += ("libpthread.so.0",)
    with zipfile.ZipFile(artifact.path, "a") as archive:
        archive.writestr(vendored_member, b"not permitted")

    with pytest.raises(AssertionError, match="vendored ROCm userspace"):
        artifact_gate._assert_rocm_system_wheel(artifact, ROOT)
