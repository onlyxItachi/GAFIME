from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
import types

import pytest

from gafime import _payloads as payloads
from gafime import v1_adapter


VERSION = "1.0.0a0"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clear_payload_environment(monkeypatch):
    names = (
        payloads.CUDA_LIBRARY_ENV,
        payloads.ROCM_LIBRARY_ENV,
        payloads.METAL_LIBRARY_ENV,
        payloads.METAL_METALLIB_ENV,
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    yield
    for name in names:
        payloads.os.environ.pop(name, None)


def write_payload_distribution(
    site: Path,
    *,
    distribution: str,
    package: str,
    version: str = VERSION,
    libraries: tuple[str, ...] = (),
) -> Path:
    package_dir = site / package
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    for library in libraries:
        (package_dir / library).write_bytes(b"payload")
    dist_info = site / f"{distribution.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    importlib.invalidate_caches()
    return package_dir


def test_explicit_payload_environment_wins_over_installed_package(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    write_payload_distribution(
        site,
        distribution="gafime-cuda",
        package="gafime_cuda",
        libraries=("libgafime_cuda.so",),
    )
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("linux", "x86_64"))
    monkeypatch.setenv(payloads.CUDA_LIBRARY_ENV, "/explicit/libgafime_cuda.so")

    assert payloads.discover_payloads("cuda") == {}
    assert payloads.os.environ[payloads.CUDA_LIBRARY_ENV] == "/explicit/libgafime_cuda.so"


def test_discovers_exactly_one_matching_installed_payload(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    package_dir = write_payload_distribution(
        site,
        distribution="gafime-cuda",
        package="gafime_cuda",
        libraries=("libgafime_cuda.so",),
    )
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("linux", "x86_64"))

    discovered = payloads.discover_payloads("cuda")

    expected = (package_dir / "libgafime_cuda.so").resolve()
    assert discovered == {"cuda": expected}
    assert Path(payloads.os.environ[payloads.CUDA_LIBRARY_ENV]) == expected


def test_missing_payload_leaves_environment_unset(monkeypatch):
    monkeypatch.setattr(payloads.metadata, "distributions", lambda: ())
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("linux", "x86_64"))

    assert payloads.discover_payloads("cuda") == {}
    assert payloads.CUDA_LIBRARY_ENV not in payloads.os.environ


def test_importable_source_namespace_without_distribution_metadata_is_ignored(
    tmp_path, monkeypatch
):
    site = tmp_path / "site"
    (site / "gafime_rocm").mkdir(parents=True)
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(payloads.metadata, "distributions", lambda: ())
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("linux", "x86_64"))

    assert payloads.discover_payloads("rocm") == {}
    assert payloads.ROCM_LIBRARY_ENV not in payloads.os.environ


def test_rejects_payload_without_its_expected_library(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    write_payload_distribution(
        site,
        distribution="gafime-cuda",
        package="gafime_cuda",
    )
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("linux", "x86_64"))

    with pytest.raises(payloads.PayloadDiscoveryError, match="no native library"):
        payloads.discover_payloads("cuda")


def test_rejects_payload_with_multiple_native_library_candidates(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    write_payload_distribution(
        site,
        distribution="gafime-cuda",
        package="gafime_cuda",
        libraries=("libgafime_cuda.so", "gafime_cuda.so"),
    )
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("linux", "x86_64"))

    with pytest.raises(payloads.PayloadDiscoveryError, match="multiple native payload libraries"):
        payloads.discover_payloads("cuda")


def test_rejects_multiple_installed_payload_distributions(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.syspath_prepend(str(first))
    monkeypatch.syspath_prepend(str(second))
    for site in (first, second):
        write_payload_distribution(
            site,
            distribution="gafime-cuda",
            package="gafime_cuda",
            libraries=("libgafime_cuda.so",),
        )
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("linux", "x86_64"))

    with pytest.raises(payloads.PayloadDiscoveryError, match="multiple installed gafime-cuda"):
        payloads.discover_payloads("cuda")


def test_rejects_payload_version_mismatch(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    write_payload_distribution(
        site,
        distribution="gafime-rocm",
        package="gafime_rocm",
        version="9.9.9",
        libraries=("libgafime_rocm.so",),
    )
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("linux", "x86_64"))

    with pytest.raises(payloads.PayloadDiscoveryError, match="does not match gafime version"):
        payloads.discover_payloads("rocm")


@pytest.mark.parametrize(
    ("backend", "platform_name", "machine", "expected"),
    [
        ("cuda", "linux", "x86_64", True),
        ("cuda", "darwin", "arm64", False),
        ("rocm", "win32", "AMD64", True),
        ("rocm", "linux", "aarch64", False),
        ("metal", "darwin", "arm64", True),
        ("metal", "linux", "x86_64", False),
    ],
)
def test_payload_platform_filtering(backend, platform_name, machine, expected):
    assert payloads._platform_supports(backend, platform_name, machine) is expected


def test_discovers_paired_bundled_metal_artifacts(tmp_path, monkeypatch):
    package_dir = tmp_path / "gafime"
    metal_dir = package_dir / "_metal"
    metal_dir.mkdir(parents=True)
    library = metal_dir / "libgafime_metal_v1.dylib"
    metallib = metal_dir / "gafime_metal_v1.metallib"
    library.write_bytes(b"dylib")
    metallib.write_bytes(b"metallib")
    monkeypatch.setattr(payloads, "_base_package_dir", lambda: package_dir)
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("darwin", "arm64"))

    discovered = payloads.discover_payloads("metal")

    assert discovered == {"metal": library.resolve()}
    assert Path(payloads.os.environ[payloads.METAL_LIBRARY_ENV]) == library.resolve()
    assert Path(payloads.os.environ[payloads.METAL_METALLIB_ENV]) == metallib.resolve()


def test_rejects_unpaired_bundled_metal_artifact(tmp_path, monkeypatch):
    package_dir = tmp_path / "gafime"
    metal_dir = package_dir / "_metal"
    metal_dir.mkdir(parents=True)
    (metal_dir / "libgafime_metal_v1.dylib").write_bytes(b"dylib")
    monkeypatch.setattr(payloads, "_base_package_dir", lambda: package_dir)
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("darwin", "arm64"))

    with pytest.raises(payloads.PayloadDiscoveryError, match="incomplete"):
        payloads.discover_payloads("metal")


def test_adapter_discovers_payloads_before_importing_native_boundary(monkeypatch):
    events: list[tuple[str, str]] = []
    boundary = types.ModuleType("gafime_fake_boundary")
    boundary.compile_continuous = lambda *args, **kwargs: None

    monkeypatch.setattr(
        v1_adapter,
        "discover_payloads",
        lambda backend: events.append(("discover", str(backend))),
    )

    def import_boundary(name: str):
        events.append(("import", name))
        return boundary

    monkeypatch.setattr(v1_adapter.importlib, "import_module", import_boundary)

    assert v1_adapter._load_boundary_for_backend("cuda") is boundary
    assert events == [("discover", "cuda"), ("import", "gafime.gafime_py")]


@pytest.mark.parametrize(
    ("backend", "sources"),
    [
        (
            "cuda",
            (
                "kernels.cu",
                "rt_kernels.cu",
                "launcher.cu",
                "rt_launcher.cu",
            ),
        ),
        ("rocm", ("kernels.hip", "launcher.hip")),
    ],
)
def test_staged_payload_uses_release_optimization_and_complete_sources(
    tmp_path, backend, sources
):
    output = tmp_path / f"gafime-{backend}"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "stage_gpu_payload.py"),
            backend,
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )

    setup_source = (output / "setup.py").read_text(encoding="utf-8")
    assert '"-O3"' in setup_source
    assert "py_limited_api=True" in setup_source
    assert '"py_limited_api": "cp310"' in setup_source
    if backend == "cuda":
        assert '"-rdc=true"' in setup_source
    else:
        assert '"-print-file-name=libstdc++.so"' in setup_source
    source_root = output / "src" / backend
    for name in sources:
        assert (source_root / name).is_file()


def test_payload_workflow_uses_proven_manylinux_rocm_and_stable_abi_wheels():
    workflow = (ROOT / ".github" / "workflows" / "build_wheels.yml").read_text(
        encoding="utf-8"
    )

    assert workflow.count('CIBW_BUILD: "cp310-*"') >= 2
    assert "https://repo.radeon.com/rocm/el8/7.2.3/main" in workflow
    assert "hip-devel7.2.3 rocm-device-libs7.2.3 libstdc++-devel" in workflow
    assert "auditwheel repair --plat manylinux_2_28_x86_64" in workflow
    assert "gafime_rocm-*-cp310-abi3-*.whl" in workflow
    assert "gafime_cuda-*-cp310-abi3-*.whl" in workflow
    assert "ubuntu/noble" not in workflow

    cuda_publish = workflow.split("\n  publish_pypi_cuda:\n", 1)[1].split(
        "\n  publish_pypi_rocm:\n", 1
    )[0]
    rocm_publish = workflow.split("\n  publish_pypi_rocm:\n", 1)[1]
    core_publish = workflow.split("\n  publish_pypi_core:\n", 1)[1].split(
        "\n  publish_pypi_cuda:\n", 1
    )[0]
    assert "build_rocm_linux_payload_wheels" not in cuda_publish
    assert "validate_rocm_payload_wheels" not in cuda_publish
    assert "build_cuda_payload_wheels" not in rocm_publish
    assert "validate_cuda_payload_wheels" not in rocm_publish
    assert "build_cuda_payload_wheels" not in core_publish
    assert "build_rocm_linux_payload_wheels" not in core_publish
