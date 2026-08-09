from __future__ import annotations

import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import types

import pytest

from gafime import _payloads as payloads
from gafime import v1_adapter


VERSION = "1.0.0b2"
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



def test_legacy_distributed_rt_identity_is_not_discovered(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    write_payload_distribution(
        site,
        distribution="gafime-cuda-rt",
        package="gafime_cuda_rt",
        libraries=("libgafime_cuda_rt.so",),
    )
    monkeypatch.setattr(payloads, "_current_platform", lambda: ("linux", "x86_64"))

    assert payloads.discover_payloads("cuda") == {}
    assert payloads.CUDA_LIBRARY_ENV not in payloads.os.environ


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
    assert "gafime-rocm 1.0.0b2" in detail
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
                "cuda_api.hpp",
                "cuda_internal.hpp",
                "kernels.cuh",
                "precision_kernels.cu",
                "precision_kernels.cuh",
                "precision_launcher.cu",
            ),
        ),
        ("rocm", ("kernels.hip", "kernels.hpp", "launcher.hip", "rocm_api.hpp")),
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
    assert "py_limited_api" not in setup_source
    assert "Py_LIMITED_API" not in setup_source
    if backend == "cuda":
        assert '"-rdc=true"' in setup_source
        assert 'CUDA_LANGUAGE_STANDARD = "c++20"' in setup_source
        assert 'f"--std={CUDA_LANGUAGE_STANDARD}"' in setup_source
        assert "GAFIME_CUDA_DISTRIBUTION_NO_RT" not in setup_source
    else:
        assert '"-print-file-name=libstdc++.so"' in setup_source
        assert 'ROCM_WHEEL_POLICY = "system"' in setup_source
        assert 'DIST_NAME = "gafime-rocm"' in setup_source
        assert "wheel policy supports " in setup_source
        assert '[patchelf, "--remove-rpath"' in setup_source
    package_name = f"gafime_{backend}"
    package_source_path = output / package_name / "__init__.py"
    package_source = package_source_path.read_text(encoding="utf-8")
    namespace = {"__file__": str(package_source_path)}
    exec(compile(package_source, str(package_source_path), "exec"), namespace)
    assert namespace["package_dir"]() == package_source_path.parent
    assert [path.name for path in namespace["library_candidates"]()] == [
        f"{package_name}.dll",
        f"lib{package_name}.so",
        f"{package_name}.so",
        f"{package_name}.pyd",
    ]
    source_root = output / "src" / backend
    for name in sources:
        assert (source_root / name).is_file()
    assert not any(path.name.startswith("rt_") for path in source_root.iterdir())


def test_staged_rocm_defaults_to_pinned_system_policy(tmp_path):
    output = tmp_path / "gafime-rocm"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "stage_gpu_payload.py"),
            "rocm",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    policy = json.loads(
        (output / "gafime_rocm" / "build_policy.json").read_text(encoding="utf-8")
    )
    assert policy["wheel_policy"] == "system"
    assert policy["userspace_bundled"] is False


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
    assert "invalid choice" in result.stderr



def test_staged_rocm_policy_is_immutable_and_matches_system_manifest(
    tmp_path, monkeypatch
):
    output = tmp_path / "gafime-rocm"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "stage_gpu_payload.py"),
            "rocm",
            str(output),
            "--rocm-wheel-policy",
            "system",
        ],
        cwd=ROOT,
        check=True,
    )

    expected = json.loads(
        (
            ROOT / ".github" / "scripts" / "rocm_7_2_3_system_policy.json"
        ).read_text(encoding="utf-8")
    )
    actual = json.loads(
        (output / "gafime_rocm" / "build_policy.json").read_text(encoding="utf-8")
    )
    pyproject = (output / "pyproject.toml").read_text(encoding="utf-8")
    setup_source = (output / "setup.py").read_text(encoding="utf-8")

    assert actual == expected
    assert 'name = "gafime-rocm"' in pyproject
    assert 'dependencies = ["gafime==1.0.0b2"]' in pyproject
    assert 'ROCM_WHEEL_POLICY = "system"' in setup_source
    assert "restage the payload instead" in setup_source
    assert not (
        ROOT / ".github" / "scripts" / "rocm_7_2_3_bundled_policy.json"
    ).exists()

    monkeypatch.setenv("GAFIME_ROCM_WHEEL_POLICY", "bundled")
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext"],
        cwd=output,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "immutable wheel policy 'system', not 'bundled'" in (
        result.stdout + result.stderr
    )


def test_staged_cuda_has_one_distribution_identity_and_no_rt_sources(tmp_path):
    output = tmp_path / "gafime-cuda"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "stage_gpu_payload.py"),
            "cuda",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )

    pyproject = (output / "pyproject.toml").read_text(encoding="utf-8")
    setup_source = (output / "setup.py").read_text(encoding="utf-8")
    policy = json.loads(
        (output / "gafime_cuda" / "build_policy.json").read_text(encoding="utf-8")
    )
    staged_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }

    assert 'name = "gafime-cuda"' in pyproject
    assert 'dependencies = ["gafime==1.0.0b2"]' in pyproject
    assert "abi3" not in setup_source
    assert '"-cudart",\n            "shared"' in setup_source
    assert '"-cudart",\n            "static"' not in setup_source
    assert policy["cuda_runtime"] == "system"
    assert policy["cuda_runtime_libraries"] == {
        "linux": "libcudart.so.13",
        "windows": "nvcudart_hybrid64.dll",
    }
    assert policy["optix_rt"] == "off"
    assert policy["rt_sources_included"] is False
    assert "GAFIME_CUDA_DISTRIBUTION_NO_RT" not in setup_source
    assert not any(Path(name).name.startswith("rt_") for name in staged_files)
    assert not any("optix" in name.lower() for name in staged_files)
    distributed_code = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"README.md", "build_policy.json"}
    )
    for forbidden_identity in (
        "GafimeDecisionPathTerm",
        "GafimeDecisionPathBatch",
        "GafimeDecisionPathScoreBatch",
        "gafime_gpu_decision_path_membership",
        "gafime_gpu_decision_path_score",
        "gafime_gpu_decision_path_release_device_state",
        "GAFIME_GPU_DEVICE_FLAG_OPTIX_RT",
        "GAFIME_DECISION_PATH_FLAG_REQUIRE_RT",
        "GAFIME_CUDA_ENABLE_OPTIX_RT",
        "GAFIME_CUDA_RT_BUILD_MODE",
        "GAFIME_CUDA_DISTRIBUTION_NO_RT",
        "features_are_rt_representable",
        "tune_rt_kernels_for_device",
    ):
        assert forbidden_identity not in distributed_code
    assert not (output / "gafime_cuda" / "build_provenance.json").exists()


def test_payload_workflows_use_per_cpython_frozen_core_first_publication():
    build = (ROOT / ".github" / "workflows" / "build_wheels.yml").read_text(
        encoding="utf-8"
    )
    publish = (ROOT / ".github" / "workflows" / "publish_release.yml").read_text(
        encoding="utf-8"
    )
    release_manifest = json.loads(
        (ROOT / ".github" / "release-artifacts.json").read_text(encoding="utf-8")
    )
    versions = release_manifest["python"]["supported_versions"]
    selector = " ".join(f"cp{version.replace('.', '')}-*" for version in versions)

    assert release_manifest["python"]["abi_policy"] == "per-cpython"
    assert f'CIBW_BUILD: "{selector}"' in build
    assert "abi3" not in build
    assert "gh-action-pypi-publish" not in build
    assert "softprops/action-gh-release" not in build
    assert "refs/tags/" not in build
    assert "release_bundle.py create" in build
    assert "--scope full-release" in build
    assert "Bind built and authoritative source identities" in build
    assert "git rev-parse HEAD" in build
    assert "github.event.pull_request.head.sha" in build
    assert "--built-source-sha" in build
    assert "--authoritative-source-sha" in build
    assert (
        "auditwheel repair --plat manylinux_2_28_x86_64 "
        "--exclude libcudart.so.13"
    ) in build
    assert (
        'delvewheel repair --exclude "cudart64_13.dll;nvcudart_hybrid64.dll"'
        in build
    )
    assert "cudart_static.lib" not in build
    assert not re.search(r"(?m)^\s+target\s*$", build)
    for forbidden in (
        "gafime-cuda-rt",
        "gafime_cuda_rt",
        "gafime-rocm-bundled",
        "gafime_rocm_bundled",
        "GAFIME_OPTIX",
        "OPTIX_SDK",
    ):
        assert forbidden not in build
        assert forbidden not in publish

    assert "workflow_dispatch:" in publish
    assert "\n  push:" not in publish
    for forbidden in (
        "python -m cibuildwheel",
        "python -m build",
        "maturin build",
        "auditwheel",
        "delvewheel",
        "retag_wheel",
    ):
        assert forbidden not in publish
    preflight = publish.split("\n  publication_preflight:\n", 1)[1].split(
        "\n  publish_pypi_core:\n", 1
    )[0]
    core = publish.split("\n  publish_pypi_core:\n", 1)[1].split(
        "\n  publish_pypi_cuda:\n", 1
    )[0]
    cuda = publish.split("\n  publish_pypi_cuda:\n", 1)[1].split(
        "\n  publish_pypi_rocm:\n", 1
    )[0]
    rocm = publish.split("\n  publish_pypi_rocm:\n", 1)[1].split(
        "\n  verify_public_core_and_cuda:\n", 1
    )[0]
    github_release = publish.split("\n  publish_github_release:\n", 1)[1]
    for job in (preflight, core, cuda, rocm, github_release):
        assert "cibuildwheel" not in job
    assert "needs: publication_preflight" in core
    assert "publish_pypi_core" in cuda
    assert "publish_pypi_core" in rocm
    assert "gafime_rocm-*.tar.gz" in rocm
    assert "gafime_rocm-*.whl" not in rocm
    assert publish.count("release_bundle.py verify") >= 5
    assert "verify_public_core_and_cuda" in publish
    assert "verify_public_windows_arm_core" in publish
    assert "verify_public_rocm_install" in publish
    assert "Publish GitHub Release after public installation" in publish
    windows_arm_builder = build.split("\n  build_arm_windows_wheels:\n", 1)[1].split(
        "\n  build_cuda_payload_wheels:\n", 1
    )[0]
    windows_arm_validator = build.split("\n  validate_windows_arm_wheel:\n", 1)[
        1
    ].split("\n  validate_cuda_payload_wheels:\n", 1)[0]
    public_windows_arm = publish.split("\n  verify_public_windows_arm_core:\n", 1)[
        1
    ].split("\n  verify_public_rocm_install:\n", 1)[0]
    for job in (windows_arm_validator, public_windows_arm):
        assert "cp310-win_arm64" in job
        assert 'python-version: "3.11"' in job
        assert "cibuildwheel==3.4.1" in job
        assert "provision_windows_arm64_python.py" in job
        assert "--venv " in job
        assert "$env:TARGET_PYTHON" in job
    assert "CIBW_BUILD: ${{ env.CIBW_BUILD }}" in windows_arm_builder
    assert "--identifier cp310-win_arm64" in windows_arm_builder
    assert "CIBW_TEST_COMMAND:" in windows_arm_builder
    assert "python .github/scripts/stage_metal_payload.py" in build
    assert "gafime_metal-*.whl" not in build + publish


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
