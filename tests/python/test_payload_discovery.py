from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from gafime import _payloads as payloads
from gafime import v1_adapter


VERSION = "1.0.0b1"
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
    build_policy: dict[str, object] | None = None,
) -> Path:
    package_dir = site / package
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    for library in libraries:
        (package_dir / library).write_bytes(b"payload")
    if build_policy is not None:
        (package_dir / "build_policy.json").write_text(
            json.dumps(build_policy), encoding="utf-8"
        )
    dist_info = site / f"{distribution.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    importlib.invalidate_caches()
    return package_dir


def test_explicit_payload_environment_wins_over_installed_package(
    tmp_path, monkeypatch
):
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
    assert (
        payloads.os.environ[payloads.CUDA_LIBRARY_ENV] == "/explicit/libgafime_cuda.so"
    )


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


def test_discovers_rt_cuda_payload_under_distinct_identity(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    package_dir = write_payload_distribution(
        site,
        distribution="gafime-cuda-rt",
        package="gafime_cuda_rt",
        libraries=("libgafime_cuda_rt.so",),
    )
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("linux", "x86_64"))

    discovered = payloads.discover_payloads("cuda")

    expected = (package_dir / "libgafime_cuda_rt.so").resolve()
    assert discovered == {"cuda": expected}
    assert Path(payloads.os.environ[payloads.CUDA_LIBRARY_ENV]) == expected


def test_rejects_ambiguous_cuda_variant_installation(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    write_payload_distribution(
        site,
        distribution="gafime-cuda",
        package="gafime_cuda",
        libraries=("libgafime_cuda.so",),
    )
    write_payload_distribution(
        site,
        distribution="gafime-cuda-rt",
        package="gafime_cuda_rt",
        libraries=("libgafime_cuda_rt.so",),
    )
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("linux", "x86_64"))

    with pytest.raises(
        payloads.PayloadDiscoveryError,
        match="multiple installed cuda payload variants.*GAFIME_CUDA_V1_LIB",
    ):
        payloads.discover_payloads("cuda")

    assert payloads.CUDA_LIBRARY_ENV not in payloads.os.environ


def test_explicit_cuda_environment_resolves_dual_variant_installation(
    tmp_path, monkeypatch
):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    write_payload_distribution(
        site,
        distribution="gafime-cuda",
        package="gafime_cuda",
        libraries=("libgafime_cuda.so",),
    )
    write_payload_distribution(
        site,
        distribution="gafime-cuda-rt",
        package="gafime_cuda_rt",
        libraries=("libgafime_cuda_rt.so",),
    )
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("linux", "x86_64"))
    explicit = "/explicit/libgafime_cuda_rt.so"
    monkeypatch.setenv(payloads.CUDA_LIBRARY_ENV, explicit)

    assert payloads.discover_payloads("cuda") == {}
    assert payloads.os.environ[payloads.CUDA_LIBRARY_ENV] == explicit


def test_missing_payload_leaves_environment_unset(monkeypatch):
    monkeypatch.setattr(payloads.metadata, "distributions", lambda: ())
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("linux", "x86_64"))

    assert payloads.discover_payloads("cuda") == {}
    assert payloads.CUDA_LIBRARY_ENV not in payloads.os.environ


def test_reads_installed_payload_policy_without_loading_library(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    expected = {
        "backend": "rocm",
        "wheel_policy": "system",
        "mixed_runtime_coexistence": "host-managed-single-runtime",
    }
    write_payload_distribution(
        site,
        distribution="gafime-rocm",
        package="gafime_rocm",
        libraries=("libgafime_rocm.so",),
        build_policy=expected,
    )
    was_imported = "gafime_rocm" in sys.modules

    policy, detail = payloads.installed_payload_build_policy("rocm")

    assert policy == expected
    assert "gafime-rocm 1.0.0b1" in detail
    assert payloads.ROCM_LIBRARY_ENV not in payloads.os.environ
    assert ("gafime_rocm" in sys.modules) is was_imported


def test_does_not_attribute_installed_policy_to_external_library(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    write_payload_distribution(
        site,
        distribution="gafime-rocm",
        package="gafime_rocm",
        libraries=("libgafime_rocm.so",),
        build_policy={"wheel_policy": "system"},
    )
    monkeypatch.setenv(payloads.ROCM_LIBRARY_ENV, "/external/libgafime_rocm.so")

    policy, detail = payloads.installed_payload_build_policy("rocm")

    assert policy is None
    assert "external library" in detail


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

    with pytest.raises(
        payloads.PayloadDiscoveryError, match="multiple native payload libraries"
    ):
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

    with pytest.raises(
        payloads.PayloadDiscoveryError, match="multiple installed gafime-cuda"
    ):
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

    with pytest.raises(
        payloads.PayloadDiscoveryError, match="does not match gafime version"
    ):
        payloads.discover_payloads("rocm")


@pytest.mark.parametrize(
    ("backend", "platform_name", "machine", "expected"),
    [
        ("cuda", "linux", "x86_64", True),
        ("cuda", "darwin", "arm64", False),
        ("rocm", "win32", "AMD64", False),
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


def test_discovers_separate_metal_distribution(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    package_dir = write_payload_distribution(
        site,
        distribution="gafime-metal",
        package="gafime_metal",
        libraries=(
            "libgafime_metal_v1.dylib",
            "gafime_metal_v1.metallib",
        ),
    )
    empty_base = tmp_path / "base" / "gafime"
    empty_base.mkdir(parents=True)
    monkeypatch.setattr(payloads, "_base_package_dir", lambda: empty_base)
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("darwin", "arm64"))

    discovered = payloads.discover_payloads("metal")

    library = (package_dir / "libgafime_metal_v1.dylib").resolve()
    metallib = (package_dir / "gafime_metal_v1.metallib").resolve()
    assert discovered == {"metal": library}
    assert Path(payloads.os.environ[payloads.METAL_LIBRARY_ENV]) == library
    assert Path(payloads.os.environ[payloads.METAL_METALLIB_ENV]) == metallib


def test_rejects_unpaired_separate_metal_distribution(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    write_payload_distribution(
        site,
        distribution="gafime-metal",
        package="gafime_metal",
        libraries=("libgafime_metal_v1.dylib",),
    )
    empty_base = tmp_path / "base" / "gafime"
    empty_base.mkdir(parents=True)
    monkeypatch.setattr(payloads, "_base_package_dir", lambda: empty_base)
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("darwin", "arm64"))

    with pytest.raises(payloads.PayloadDiscoveryError, match="missing paired metallib"):
        payloads.discover_payloads("metal")


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
    policy_args = ["--rocm-wheel-policy", "system"] if backend == "rocm" else []
    subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "stage_gpu_payload.py"),
            backend,
            str(output),
            *policy_args,
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
        assert 'CUDA_LANGUAGE_STANDARD = "c++20"' in setup_source
        assert setup_source.count('f"--std={CUDA_LANGUAGE_STANDARD}"') == 2
    else:
        assert '"-print-file-name=libstdc++.so"' in setup_source
        assert 'ROCM_WHEEL_POLICY = "system"' in setup_source
        assert 'DIST_NAME = "gafime-rocm"' in setup_source
        assert "wheel policy supports " in setup_source
        assert '[patchelf, "--remove-rpath"' in setup_source
    source_root = output / "src" / backend
    for name in sources:
        assert (source_root / name).is_file()


def test_staged_rocm_requires_explicit_reviewed_wheel_policy(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "stage_gpu_payload.py"),
            "rocm",
            str(tmp_path / "gafime-rocm"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "requires explicit --rocm-wheel-policy system|bundled" in result.stderr


@pytest.mark.parametrize("policy", ("amd-wheels", "unknown"))
def test_staged_rocm_rejects_unimplemented_wheel_policies(tmp_path, policy):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "stage_gpu_payload.py"),
            "rocm",
            str(tmp_path / "gafime-rocm"),
            "--rocm-wheel-policy",
            policy,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert f"ROCm wheel policy '{policy}' is not implemented" in result.stderr


@pytest.mark.parametrize(
    ("policy", "distribution", "package", "manifest", "conflict"),
    [
        (
            "system",
            "gafime-rocm",
            "gafime_rocm",
            "rocm_7_2_3_system_policy.json",
            "bundled",
        ),
        (
            "bundled",
            "gafime-rocm-bundled",
            "gafime_rocm_bundled",
            "rocm_7_2_3_bundled_policy.json",
            "system",
        ),
    ],
)
def test_staged_rocm_policy_is_immutable_and_matches_manifest(
    tmp_path, policy, distribution, package, manifest, conflict
):
    output = tmp_path / distribution
    subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "stage_gpu_payload.py"),
            "rocm",
            str(output),
            "--rocm-wheel-policy",
            policy,
        ],
        cwd=ROOT,
        check=True,
    )

    expected = json.loads(
        (ROOT / ".github" / "scripts" / manifest).read_text(encoding="utf-8")
    )
    actual = json.loads(
        (output / package / "build_policy.json").read_text(encoding="utf-8")
    )
    pyproject = (output / "pyproject.toml").read_text(encoding="utf-8")
    setup_source = (output / "setup.py").read_text(encoding="utf-8")

    assert actual == expected
    assert f'name = "{distribution}"' in pyproject
    assert f'ROCM_WHEEL_POLICY = "{policy}"' in setup_source
    assert "restage the payload instead" in setup_source

    conflicting_environment = os.environ.copy()
    conflicting_environment["GAFIME_ROCM_WHEEL_POLICY"] = conflict
    conflict_result = subprocess.run(
        [sys.executable, "setup.py", "build_ext"],
        cwd=output,
        env=conflicting_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert conflict_result.returncode != 0
    assert f"immutable wheel policy '{policy}', not '{conflict}'" in (
        conflict_result.stdout + conflict_result.stderr
    )


def test_rocm_policy_environment_does_not_change_cuda_staging(tmp_path, monkeypatch):
    monkeypatch.setenv("GAFIME_ROCM_WHEEL_POLICY", "bundled")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "stage_gpu_payload.py"),
            "cuda",
            str(tmp_path / "gafime-cuda"),
            "--cuda-rt",
            "off",
        ],
        cwd=ROOT,
        check=True,
    )


@pytest.mark.parametrize(
    ("rt_mode", "distribution", "package", "native_library", "other_package"),
    [
        ("off", "gafime-cuda", "gafime_cuda", "libgafime_cuda.so", "gafime_cuda_rt"),
        (
            "on",
            "gafime-cuda-rt",
            "gafime_cuda_rt",
            "libgafime_cuda_rt.so",
            "gafime_cuda",
        ),
    ],
)
def test_staged_cuda_variants_have_noncolliding_distribution_identity(
    tmp_path, rt_mode, distribution, package, native_library, other_package
):
    output = tmp_path / distribution
    optix_digest = "a" * 64
    cuda_fixture_image = "docker.io/nvidia/cuda:13.3.0-devel@sha256:" + "b" * 64
    wheel_builder_image = "quay.io/pypa/manylinux@sha256:" + "c" * 64
    cuda_rpm_base_url = "https://developer.download.nvidia.com/cuda"
    rpm_manifest = tmp_path / "cuda-rpms.sha256"
    rpm_manifest.write_text(
        f"{'d' * 64}  cuda-nvcc-13-3-13.3.73-1.x86_64.rpm\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(ROOT / ".github" / "scripts" / "stage_gpu_payload.py"),
        "cuda",
        str(output),
        "--cuda-rt",
        rt_mode,
    ]
    if rt_mode == "on":
        command.extend(
            [
                "--optix-sdk-archive-sha256",
                optix_digest,
                "--cuda-fixture-image",
                cuda_fixture_image,
                "--wheel-builder-image",
                wheel_builder_image,
                "--cuda-rpm-base-url",
                cuda_rpm_base_url,
                "--cuda-rpm-manifest",
                str(rpm_manifest),
            ]
        )
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )

    pyproject = (output / "pyproject.toml").read_text(encoding="utf-8")
    setup_source = (output / "setup.py").read_text(encoding="utf-8")
    package_source = (output / package / "__init__.py").read_text(encoding="utf-8")
    policy = json.loads(
        (output / package / "build_policy.json").read_text(encoding="utf-8")
    )

    assert f'name = "{distribution}"' in pyproject
    assert f'DIST_NAME = "{distribution}"' in setup_source
    assert f'PACKAGE_NAME = "{package}"' in setup_source
    assert native_library in package_source
    assert (output / package).is_dir()
    assert not (output / other_package).exists()
    assert policy["optix_rt"] == rt_mode
    provenance_path = output / package / "build_provenance.json"
    if rt_mode == "on":
        assert json.loads(provenance_path.read_text(encoding="utf-8")) == {
            "cuda_fixture_image": cuda_fixture_image,
            "cuda_rpm_base_url": cuda_rpm_base_url,
            "cuda_toolkit_rpms": [
                {
                    "filename": "cuda-nvcc-13-3-13.3.73-1.x86_64.rpm",
                    "sha256": "d" * 64,
                }
            ],
            "optix_sdk_archive_sha256": optix_digest,
            "wheel_builder_image": wheel_builder_image,
        }
    else:
        assert not provenance_path.exists()


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        ([], "--optix-sdk-archive-sha256 is required"),
        (
            [
                "--optix-sdk-archive-sha256",
                "not-a-digest",
                "--cuda-fixture-image",
                "docker.io/nvidia/cuda:13.3.0-devel@sha256:" + "b" * 64,
            ],
            "must be exactly 64 hex digits",
        ),
        (
            [
                "--optix-sdk-archive-sha256",
                "a" * 64,
                "--cuda-fixture-image",
                "nvidia/cuda:13.3.0-devel",
            ],
            "must end with @sha256:",
        ),
    ],
)
def test_staged_cuda_rt_requires_pinned_provenance(tmp_path, extra_args, message):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "stage_gpu_payload.py"),
            "cuda",
            str(tmp_path / "gafime-cuda-rt"),
            "--cuda-rt",
            "on",
            *extra_args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_staged_cuda_rt_requires_wheel_builder_identity(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "stage_gpu_payload.py"),
            "cuda",
            str(tmp_path / "gafime-cuda-rt"),
            "--cuda-rt",
            "on",
            "--optix-sdk-archive-sha256",
            "a" * 64,
            "--cuda-fixture-image",
            "docker.io/nvidia/cuda@sha256:" + "b" * 64,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--wheel-builder-image is required" in result.stderr


def test_staged_cuda_rt_rejects_unhashed_cuda_rpm_manifest(tmp_path):
    rpm_manifest = tmp_path / "cuda-rpms.sha256"
    rpm_manifest.write_text("not-a-digest  cuda-nvcc.rpm\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "stage_gpu_payload.py"),
            "cuda",
            str(tmp_path / "gafime-cuda-rt"),
            "--cuda-rt",
            "on",
            "--optix-sdk-archive-sha256",
            "a" * 64,
            "--cuda-fixture-image",
            "docker.io/nvidia/cuda@sha256:" + "b" * 64,
            "--wheel-builder-image",
            "quay.io/pypa/manylinux@sha256:" + "c" * 64,
            "--cuda-rpm-base-url",
            "https://developer.download.nvidia.com/cuda",
            "--cuda-rpm-manifest",
            str(rpm_manifest),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "invalid CUDA RPM SHA-256" in result.stderr


def test_payload_workflow_tests_stable_abi_and_separates_system_rocm():
    workflow = (ROOT / ".github" / "workflows" / "build_wheels.yml").read_text(
        encoding="utf-8"
    )

    stable_abi_matrix = 'CIBW_BUILD: "cp310-* cp311-* cp312-* cp313-* cp314-*"'
    assert workflow.count(stable_abi_matrix) >= 5
    rocm_build = workflow.split(
        "  build_rocm_linux_payload_wheels:", maxsplit=1
    )[1].split("\n  validate_wheels:", maxsplit=1)[0]
    assert 'CIBW_BUILD: "cp310-*"' in rocm_build
    assert "cp311-*" not in rocm_build
    rocm_validation = workflow.split(
        "  validate_rocm_payload_wheels:", maxsplit=1
    )[1].split("\n  build_sdist:", maxsplit=1)[0]
    for python_tag in (
        "cp310-cp310",
        "cp311-cp311",
        "cp312-cp312",
        "cp313-cp313",
        "cp314-cp314",
    ):
        assert python_tag in rocm_validation
    assert "https://repo.radeon.com/rocm/el8/7.2.3/main" in workflow
    for package in (
        "hip-devel7.2.3-7.2.53211.70203-90.el8.x86_64",
        "rocm-device-libs7.2.3-1.0.0.70203-90.el8.x86_64",
        "libstdc++-devel-8.5.0-28.el8_10.alma.1.x86_64",
    ):
        assert package in workflow
    assert (
        "2de99e2354646a90d9903e2a669fc4e36b02c1bbff7075c481e12d7edab2c88b"
        in workflow
    )
    assert 'CIBW_REPAIR_WHEEL_COMMAND_LINUX: "cp {wheel} {dest_dir}/"' in workflow
    assert "gafime_rocm-*-cp310-abi3-linux_x86_64.whl" in workflow
    assert "gafime_cuda-*-cp310-abi3-*.whl" in workflow
    assert "gafime_metal-*.whl" in workflow
    assert "gafime_cuda_rt-*-cp310-abi3-*.whl" in workflow
    assert "ubuntu/noble" not in workflow
    assert workflow.count("if: ${{ !cancelled() }}") == 3
    assert "pull_request:" in workflow
    assert "pull_request_target:" not in workflow
    assert "GAFIME_OPTIX_SDK_ARCHIVE_SHA256" in workflow
    assert "sha256sum --check --strict" in workflow
    assert (
        "docker.io/nvidia/cuda:13.3.0-devel-ubuntu24.04@sha256:"
        "69e9e39eb8fe2cda271654a0f5eac2f1bb946b2fb9c460eb19c7c3c155f4e64e" in workflow
    )
    assert (
        "quay.io/pypa/manylinux_2_28_x86_64@sha256:"
        "a61875a2f84cab7df8de222ff12cabc08ff86eb4ad402ac90ba7bdaed9600cca" in workflow
    )
    assert "cuda_13_3_rpms.sha256" in workflow
    assert "/project/payload-src/gafime-cuda-rt/.cuda-rpms/*.rpm" in workflow
    assert "/project/payload-src/gafime-cuda-rt/.optix-sdk/include" in workflow
    assert workflow.count("retag_wheel_build.py") == 1
    assert workflow.index("retag_wheel_build.py") < workflow.index("--write-checksums")
    assert workflow.count("--rocm-wheel-policy system") >= 2
    assert "GAFIME_ROCM_WHEEL_POLICY=system" in workflow
    assert "gafime_rocm-*.whl" not in workflow.split(
        "\n  publish_pypi_rocm:\n", 1
    )[1].split("\n  publish_pypi_metal:\n", 1)[0]
    assert "--write-rocm-report wheelhouse/rocm-wheel-policy-report.json" in workflow
    assert "./wheelhouse/rocm-wheel-policy-report.json" in workflow

    rt_job = workflow.split("\n  build_cuda_rt_linux_payload:\n", 1)[1].split(
        "\n  build_rocm_linux_payload_wheels:\n", 1
    )[0]
    assert (
        "if: ${{ github.event_name == 'workflow_dispatch' && "
        "inputs.build_cuda_rt_payload == true }}" in rt_job
    )
    assert "secrets.GAFIME_OPTIX_SDK_ARCHIVE_URL" in rt_job
    assert "secrets.GAFIME_OPTIX_SDK_ARCHIVE_SHA256" in rt_job
    assert "/opt/rh/gcc-toolset-14/root/usr/bin/g++" in rt_job
    assert "rpm -Uvh --nodeps" in rt_job
    assert "dnf install" not in rt_job

    release_preflight = workflow.split("\n  release_preflight:\n", 1)[1].split(
        "\n  release:\n", 1
    )[0]
    assert "build_cuda_rt_linux_payload" not in release_preflight

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
    for publish_job in (cuda_publish, rocm_publish):
        assert (
            "if: (github.event_name == 'push' && "
            "startsWith(github.ref, 'refs/tags/v')) ||" in publish_job
        )
        assert "github.event_name == 'workflow_dispatch'" in publish_job
    assert "always()" in core_publish
    for dependency in ("publish_pypi_cuda", "publish_pypi_rocm"):
        assert f"- {dependency}" in core_publish
        assert f"needs.{dependency}.result == 'success'" in core_publish
    for publish_job in (core_publish, cuda_publish, rocm_publish):
        assert "check_pypi_artifact_collisions.py" in publish_job
        assert (
            "skip-existing: ${{ github.event_name == 'workflow_dispatch' && "
            "inputs.allow_matching_existing_pypi_files == true }}" in publish_job
        )
    assert "skip-existing: true" not in workflow

    release_job = workflow.split("\n  release:\n", 1)[1].split(
        "\n  publish_pypi_core:\n", 1
    )[0]
    assert "always()" in release_job
    for dependency in (
        "publish_pypi_cuda",
        "publish_pypi_rocm",
        "publish_pypi_core",
    ):
        assert f"- {dependency}" in release_job
        assert f"needs.{dependency}.result == 'success'" in release_job
    assert "prerelease: ${{" in release_job
    assert "inputs.publish_github_release == true" in release_job
    assert release_job.count("inputs.publish_pypi_") >= 3
    assert "startsWith(github.ref, 'refs/tags/v')" in release_job
    assert (
        "PUBLISH_REQUESTED: ${{ (github.event_name == 'push' && "
        "startsWith(github.ref, 'refs/tags/v')) ||" in release_preflight
    )
    assert "inputs.check_pypi_collisions == true" in release_preflight
    assert "--artifacts dist" in release_preflight


def test_metal_staging_uses_lipo_input_before_verify_command():
    source = (ROOT / ".github" / "scripts" / "stage_metal_payload.py").read_text(
        encoding="utf-8"
    )

    assert 'run(["lipo", str(library), "-verify_arch", "arm64"])' in source
    assert 'os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0")' in source
    assert 'f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}"' in source


def test_native_platform_workflow_references_current_installed_contracts():
    workflow = (
        ROOT / ".github" / "workflows" / "native_platform_validation.yml"
    ).read_text(encoding="utf-8")

    assert "tests/release_measure/installed_wheel_smoke.py" in workflow
    assert "tests/release_measure/contract_05_capability_surface.py" in workflow
    assert "tests/release_measure/installed_payload_smoke.py" in workflow
    assert workflow.count("gafime-native-platform-venv") == 5
    assert 'MACOSX_DEPLOYMENT_TARGET: "11.0"' in workflow
    assert "pull_request:" in workflow
    assert "pull_request_target:" not in workflow
    assert "secrets." not in workflow
    assert ".platform-validation-venv" not in workflow
    assert "pip install --no-deps wheelhouse" not in workflow
    for removed in (
        "tests/test_v045_native_spine.py",
        "tests/benchmark_distribution_wheel.py",
        "tests/platform_extreme_validation.py",
        "tests/metal_hardcore_benchmark.py",
    ):
        assert removed not in workflow
