from __future__ import annotations

import argparse
import json
import re
import shutil
import textwrap
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


CUDA_SETUP = r"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent
DIST_NAME = "{dist_name}"
PACKAGE_NAME = "{package_name}"
CUDA_LANGUAGE_STANDARD = "c++20"
CUDA_RT_BUILD_MODE = "{cuda_rt_mode}"
CUDA_ARCHITECTURES = ("75", "80", "86", "89", "90", "100", "120")
CUDA_TUNING_POLICY = "runtime-device-class"
RUNTIME_ARCHITECTURE_DISPATCH = True
PER_ARCHITECTURE_TUNING = False


def _find_nvcc() -> str | None:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    exe_name = "nvcc.exe" if sys.platform == "win32" else "nvcc"
    for env_name in ("CUDA_PATH", "CUDA_HOME"):
        cuda_root = os.environ.get(env_name)
        if not cuda_root:
            continue
        candidate = Path(cuda_root) / "bin" / exe_name
        if candidate.exists():
            return str(candidate)
    return None


def _cuda_rt_build_mode() -> str:
    requested = os.environ.get("GAFIME_CUDA_RT_BUILD_MODE")
    if requested is not None and requested.strip().lower() != CUDA_RT_BUILD_MODE:
        raise RuntimeError(
            f"this staged {{DIST_NAME}} source has immutable RT policy "
            f"{{CUDA_RT_BUILD_MODE!r}}; restage with --cuda-rt to select another policy"
        )
    return CUDA_RT_BUILD_MODE


def _optix_include_dir() -> Path:
    direct = os.environ.get("GAFIME_OPTIX_INCLUDE_DIR") or os.environ.get("OPTIX_INCLUDE_DIR")
    root = os.environ.get("OPTIX_ROOT") or os.environ.get("OPTIX_SDK_ROOT")
    candidate = Path(direct) if direct else (Path(root) / "include" if root else None)
    if candidate is None or not (candidate / "optix.h").is_file():
        raise RuntimeError(
            "GAFIME_CUDA_RT_BUILD_MODE=on requires GAFIME_OPTIX_INCLUDE_DIR, "
            "OPTIX_INCLUDE_DIR, or OPTIX_ROOT/OPTIX_SDK_ROOT with optix.h."
        )
    return candidate


def _write_optix_ptx_header(ptx: Path, header: Path) -> None:
    source = ptx.read_text(encoding="utf-8")
    header.write_text(
        "#ifndef GAFIME_RT_OPTIX_PTX_HPP\n"
        "#define GAFIME_RT_OPTIX_PTX_HPP\n\n"
        "#include <cstddef>\n\n"
        "namespace gafime_cuda_v1 {{\n"
        "static constexpr const char kRtOptixPtx[] = R\"GAFIME_PTX(\n"
        f"{{source}}\n"
        ")GAFIME_PTX\";\n"
        "static constexpr std::size_t kRtOptixPtxSize = sizeof(kRtOptixPtx) - 1u;\n"
        "}}  // namespace gafime_cuda_v1\n\n"
        "#endif  // GAFIME_RT_OPTIX_PTX_HPP\n",
        encoding="utf-8",
    )


class CudaPayloadBuildExt(build_ext):
    def run(self):
        package_dir = Path(self.build_lib) / PACKAGE_NAME
        package_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = package_dir
        self.build_cuda_backend()
        super().run()

    def build_cuda_backend(self) -> None:
        machine = platform.machine().lower()
        if machine in {{"aarch64", "arm64"}} or machine.startswith("arm"):
            raise RuntimeError(f"{{DIST_NAME}} does not support ARM target {{platform.machine()}}.")
        nvcc = _find_nvcc()
        if not nvcc:
            raise RuntimeError(
                f"nvcc was not found. Install CUDA Toolkit 13.2+ to build {{DIST_NAME}} "
                "or set CUDA_PATH/CUDA_HOME to the toolkit root."
            )

        src_dir = ROOT / "src"
        cuda_sources = [
            src_dir / "cuda" / "kernels.cu",
            src_dir / "cuda" / "rt_kernels.cu",
            src_dir / "cuda" / "launcher.cu",
            src_dir / "cuda" / "rt_launcher.cu",
        ]
        if sys.platform == "win32":
            output_file = self.output_dir / f"{{PACKAGE_NAME}}.dll"
            compiler_flags = ["/MD"]
        else:
            output_file = self.output_dir / f"lib{{PACKAGE_NAME}}.so"
            compiler_flags = ["-fPIC"]

        gencode_flags = [
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_100,code=sm_100",
            "-gencode=arch=compute_120,code=sm_120",
            "-gencode=arch=compute_120,code=compute_120",
        ]
        rt_mode = _cuda_rt_build_mode()
        rt_flags: list[str] = []
        if rt_mode == "on":
            optix_include = _optix_include_dir()
            ptx_arch = os.environ.get("GAFIME_CUDA_OPTIX_PTX_ARCH", "compute_75")
            if not re.fullmatch(r"compute_[0-9]+", ptx_arch):
                raise RuntimeError("GAFIME_CUDA_OPTIX_PTX_ARCH must be a compute_<SM> target.")
            generated = Path(self.build_temp) / "gafime_cuda_optix"
            generated.mkdir(parents=True, exist_ok=True)
            ptx = generated / "gafime_cuda_decision_path.ptx"
            header = generated / "gafime_rt_optix_ptx.hpp"
            ptx_result = subprocess.run(
                [
                    nvcc,
                    f"--std={{CUDA_LANGUAGE_STANDARD}}",
                    "-O3",
                    f"--gpu-architecture={{ptx_arch}}",
                    "-I",
                    str(optix_include),
                    "-DGAFIME_CUDA_RT_OPTIX_DEVICE",
                    "--ptx",
                    str(src_dir / "cuda" / "rt_kernels.cu"),
                    "-o",
                    str(ptx),
                ],
                capture_output=True,
                text=True,
            )
            if ptx_result.returncode != 0:
                raise RuntimeError(
                    "CUDA OptiX PTX build failed\\n"
                    f"STDOUT:\\n{{ptx_result.stdout}}\\nSTDERR:\\n{{ptx_result.stderr}}"
                )
            _write_optix_ptx_header(ptx, header)
            rt_flags = [
                "-DGAFIME_CUDA_ENABLE_OPTIX_RT=1",
                "-I",
                str(optix_include),
                "-I",
                str(generated),
                "-lcuda",
            ]
        cmd = [
            nvcc,
            *gencode_flags,
            f"--std={{CUDA_LANGUAGE_STANDARD}}",
            "-O3",
            "-rdc=true",
            "--shared",
            "-DGAFIME_GPU_BUILDING_DLL",
            "-cudart",
            "static",
            "-Xcompiler",
            ",".join(compiler_flags),
            "-I",
            str(src_dir / "common"),
            "-I",
            str(src_dir / "cuda"),
            "-o",
            str(output_file),
            *rt_flags,
            *(str(source) for source in cuda_sources),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"CUDA build failed\nSTDOUT:\n{{result.stdout}}\nSTDERR:\n{{result.stderr}}")


setup(
    packages=[PACKAGE_NAME],
    package_data={{
        PACKAGE_NAME: [
            "*.so",
            "*.dll",
            "*.pyd",
            "build_policy.json",
            "build_provenance.json",
        ]
    }},
    include_package_data=False,
    ext_modules=[
        Extension(
            f"{{PACKAGE_NAME}}._native",
            sources=[str(ROOT / "gafime" / "_dummy.c")],
            py_limited_api=True,
        )
    ],
    cmdclass={{"build_ext": CudaPayloadBuildExt}},
    options={{"bdist_wheel": {{"py_limited_api": "cp310"}}}},
)
"""


ROCM_SETUP = r"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent


def _linux_cxx_runtime_link_flags() -> list[str]:
    if sys.platform != "linux":
        return []
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        return []
    result = subprocess.run(
        [compiler, "-print-file-name=libstdc++.so"],
        capture_output=True,
        text=True,
        check=False,
    )
    library = Path(result.stdout.strip())
    if result.returncode == 0 and library.is_file():
        return ["-L", str(library.parent)]
    return []


class RocmPayloadBuildExt(build_ext):
    def run(self):
        package_dir = Path(self.build_lib) / "gafime_rocm"
        package_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = package_dir
        self.build_rocm_backend()
        super().run()

    def build_rocm_backend(self) -> None:
        if sys.platform not in {{"linux", "win32"}}:
            raise RuntimeError("gafime-rocm currently supports Linux/Windows x86_64 targets.")
        machine = platform.machine().lower()
        if machine in {{"aarch64", "arm64"}} or machine.startswith("arm"):
            raise RuntimeError(f"gafime-rocm does not support ARM target {{platform.machine()}}.")

        hipcc = shutil.which("hipcc")
        if not hipcc:
            raise RuntimeError("hipcc was not found. Install ROCm/HIP to build gafime-rocm.")

        src_dir = ROOT / "src"
        rocm_sources = [
            src_dir / "rocm" / "kernels.hip",
            src_dir / "rocm" / "launcher.hip",
        ]
        output_file = self.output_dir / ("gafime_rocm.dll" if sys.platform == "win32" else "libgafime_rocm.so")
        arch_env = os.environ.get("GAFIME_ROCM_ARCHS")
        if arch_env:
            arch_mode = arch_env.strip().lower().replace("_", "-")
            if arch_mode in {{"release", "package", "wheel", "release-wheel"}}:
                archs = self._windows_release_rocm_archs() if sys.platform == "win32" else self._linux_release_rocm_archs()
            elif arch_mode in {{"linux-release", "linux-wheel"}}:
                archs = self._linux_release_rocm_archs()
            elif arch_mode in {{"windows-release", "windows-wheel"}}:
                archs = self._windows_release_rocm_archs()
            else:
                archs = [arch.strip() for arch in arch_env.replace(";", ",").replace(" ", ",").split(",") if arch.strip()]
        else:
            archs = self._detect_rocm_archs()
        if not archs:
            raise RuntimeError(
                "Unable to detect ROCm/HIP offload architecture. "
                "Set GAFIME_ROCM_ARCHS explicitly, for example GAFIME_ROCM_ARCHS=<rocm-offload-target>."
            )
        arch_flags = [f"--offload-arch={{arch}}" for arch in archs]
        runtime_link_flags = _linux_cxx_runtime_link_flags()

        cmd = [
            hipcc,
            *arch_flags,
            "--std=c++23",
            "-O3",
            "--shared",
            "-DGAFIME_GPU_BUILDING_DLL",
            "-I",
            str(src_dir / "common"),
            "-I",
            str(src_dir / "rocm"),
            *runtime_link_flags,
            "-o",
            str(output_file),
            *(str(source) for source in rocm_sources),
        ]
        if sys.platform != "win32":
            cmd.insert(cmd.index("--shared"), "-fPIC")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ROCm/HIP build failed\nSTDOUT:\n{{result.stdout}}\nSTDERR:\n{{result.stderr}}")

    @staticmethod
    def _linux_release_rocm_archs() -> list[str]:
        # ROCm does not provide a single NVIDIA-PTX-like forward-compatible
        # code object for every AMD GPU. Release wheels therefore carry a
        # package-policy target set covering current ROCm 7.x client, APU, and
        # datacenter families instead of baking in one developer machine target.
        return [
            "gfx90a",
            "gfx942",
            "gfx950",
            "gfx1030",
            "gfx1031",
            "gfx1032",
            "gfx1100",
            "gfx1101",
            "gfx1102",
            "gfx1150",
            "gfx1151",
            "gfx1200",
            "gfx1201",
        ]

    @staticmethod
    def _windows_release_rocm_archs() -> list[str]:
        # Windows HIP SDK publishes a narrower officially supported set than
        # Linux ROCm. Keep this list aligned with AMD's Windows HIP SDK support
        # table so CI does not feed datacenter/Linux-only code objects to the
        # Windows installer toolchain.
        return [
            "gfx1030",
            "gfx1031",
            "gfx1032",
            "gfx1100",
            "gfx1101",
            "gfx1102",
            "gfx1151",
            "gfx1200",
            "gfx1201",
        ]

    @staticmethod
    def _detect_rocm_archs() -> list[str]:
        enumerator = shutil.which("rocm_agent_enumerator")
        if enumerator:
            try:
                result = subprocess.run([enumerator], capture_output=True, text=True, check=False, timeout=10)
                archs: list[str] = []
                for line in result.stdout.splitlines():
                    arch = line.strip()
                    if arch.startswith("gfx") and arch not in archs:
                        archs.append(arch)
                if archs:
                    return archs
            except Exception:
                pass
        return []


setup(
    packages=["gafime_rocm"],
    package_data={{"gafime_rocm": ["*.so", "*.dll", "*.pyd"]}},
    include_package_data=False,
    ext_modules=[
        Extension(
            "gafime_rocm._native",
            sources=[str(ROOT / "gafime" / "_dummy.c")],
            py_limited_api=True,
        )
    ],
    cmdclass={{"build_ext": RocmPayloadBuildExt}},
    options={{"bdist_wheel": {{"py_limited_api": "cp310"}}}},
)
"""


def project_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def _cuda_rt_provenance(
    cuda_rt_mode: str,
    optix_sdk_archive_sha256: str | None,
    cuda_fixture_image: str | None,
    wheel_builder_image: str | None,
    cuda_rpm_base_url: str | None,
    cuda_rpm_manifest: Path | None,
) -> dict[str, object] | None:
    provenance_values = (
        optix_sdk_archive_sha256,
        cuda_fixture_image,
        wheel_builder_image,
        cuda_rpm_base_url,
        cuda_rpm_manifest,
    )
    if cuda_rt_mode == "off":
        if any(value is not None for value in provenance_values):
            raise ValueError(
                "OptiX SDK and CUDA build provenance apply only with --cuda-rt on"
            )
        return None

    if not optix_sdk_archive_sha256:
        raise ValueError("--optix-sdk-archive-sha256 is required with --cuda-rt on")
    digest = optix_sdk_archive_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("--optix-sdk-archive-sha256 must be exactly 64 hex digits")

    def pinned_image(value: str | None, option: str) -> str:
        if not value:
            raise ValueError(f"{option} is required with --cuda-rt on")
        image = value.strip()
        if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image):
            raise ValueError(
                f"{option} must end with @sha256:<64 lowercase hex digits>"
            )
        return image

    fixture_image = pinned_image(cuda_fixture_image, "--cuda-fixture-image")
    builder_image = pinned_image(wheel_builder_image, "--wheel-builder-image")

    if not cuda_rpm_base_url:
        raise ValueError("--cuda-rpm-base-url is required with --cuda-rt on")
    rpm_base_url = cuda_rpm_base_url.strip().rstrip("/")
    if not re.fullmatch(r"https://[^\s]+", rpm_base_url):
        raise ValueError("--cuda-rpm-base-url must be an HTTPS URL")

    if cuda_rpm_manifest is None:
        raise ValueError("--cuda-rpm-manifest is required with --cuda-rt on")
    rpm_manifest_path = cuda_rpm_manifest.resolve()
    if not rpm_manifest_path.is_file():
        raise ValueError(f"CUDA RPM manifest does not exist: {cuda_rpm_manifest}")
    rpm_entries: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for line_number, raw_line in enumerate(
        rpm_manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(
                f"invalid CUDA RPM manifest line {line_number}: expected SHA-256 and filename"
            )
        rpm_digest, filename = fields
        if not re.fullmatch(r"[0-9a-f]{64}", rpm_digest):
            raise ValueError(f"invalid CUDA RPM SHA-256 on manifest line {line_number}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*\.rpm", filename):
            raise ValueError(
                f"invalid CUDA RPM filename on manifest line {line_number}"
            )
        if filename in seen_names:
            raise ValueError(f"duplicate CUDA RPM filename in manifest: {filename}")
        seen_names.add(filename)
        rpm_entries.append({"filename": filename, "sha256": rpm_digest})
    if not rpm_entries:
        raise ValueError("CUDA RPM manifest must contain at least one package")

    return {
        "cuda_fixture_image": fixture_image,
        "cuda_rpm_base_url": rpm_base_url,
        "cuda_toolkit_rpms": rpm_entries,
        "optix_sdk_archive_sha256": digest,
        "wheel_builder_image": builder_image,
    }


def stage_payload(
    kind: str,
    output: Path,
    cuda_rt_mode: str = "off",
    optix_sdk_archive_sha256: str | None = None,
    cuda_fixture_image: str | None = None,
    wheel_builder_image: str | None = None,
    cuda_rpm_base_url: str | None = None,
    cuda_rpm_manifest: Path | None = None,
) -> None:
    if kind != "cuda" and cuda_rt_mode != "off":
        raise ValueError("--cuda-rt applies only to the CUDA payload")
    if kind != "cuda" and any(
        value is not None
        for value in (
            optix_sdk_archive_sha256,
            cuda_fixture_image,
            wheel_builder_image,
            cuda_rpm_base_url,
            cuda_rpm_manifest,
        )
    ):
        raise ValueError("CUDA provenance options apply only to the CUDA payload")
    provenance = (
        _cuda_rt_provenance(
            cuda_rt_mode,
            optix_sdk_archive_sha256,
            cuda_fixture_image,
            wheel_builder_image,
            cuda_rpm_base_url,
            cuda_rpm_manifest,
        )
        if kind == "cuda"
        else None
    )
    version = project_version()
    cuda_rt = kind == "cuda" and cuda_rt_mode == "on"
    package_name = "gafime_cuda_rt" if cuda_rt else f"gafime_{kind}"
    dist_name = "gafime-cuda-rt" if cuda_rt else f"gafime-{kind}"
    gpu_src_root = REPO_ROOT / "src"
    source_subdir = "cuda" if kind == "cuda" else "rocm"
    source_names = (
        [
            "cuda_api.hpp",
            "kernels.cuh",
            "kernels.cu",
            "rt_kernels.cuh",
            "rt_kernels.cu",
            "rt_launcher.cuh",
            "rt_launcher.cu",
            "launcher.cu",
        ]
        if kind == "cuda"
        else ["rocm_api.hpp", "kernels.hpp", "kernels.hip", "launcher.hip"]
    )
    setup_template = CUDA_SETUP if kind == "cuda" else ROCM_SETUP

    if output.exists():
        shutil.rmtree(output)
    (output / package_name).mkdir(parents=True)
    (output / "gafime").mkdir(parents=True)
    (output / "src" / source_subdir).mkdir(parents=True)
    (output / "src" / "common").mkdir(parents=True)

    write_text(
        output / "gafime" / "_dummy.c",
        """
    #define Py_LIMITED_API 0x030A0000
    #include <Python.h>

    static struct PyModuleDef gafime_gpu_payload_module = {
        PyModuleDef_HEAD_INIT,
        "_native",
        NULL,
        -1,
        NULL,
    };

    PyMODINIT_FUNC PyInit__native(void) {
        return PyModule_Create(&gafime_gpu_payload_module);
    }
    """,
    )
    for source_name in source_names:
        shutil.copy2(
            gpu_src_root / source_subdir / source_name,
            output / "src" / source_subdir / source_name,
        )
    shutil.copy2(
        gpu_src_root / "common" / "gafime_gpu_abi.hpp",
        output / "src" / "common" / "gafime_gpu_abi.hpp",
    )
    shutil.copy2(
        gpu_src_root / "common" / "gpu_abi_impl.hpp",
        output / "src" / "common" / "gpu_abi_impl.hpp",
    )

    description = (
        "NVIDIA CUDA and OptiX RT runtime payload for GAFIME"
        if kind == "cuda" and cuda_rt_mode == "on"
        else "NVIDIA CUDA runtime payload for GAFIME (OptiX RT disabled)"
        if kind == "cuda"
        else "AMD ROCm/HIP runtime payload for GAFIME"
    )
    write_text(
        output / "pyproject.toml",
        f"""
    [build-system]
    requires = ["setuptools>=77", "wheel"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "{dist_name}"
    version = "{version}"
    description = "{description}"
    readme = "README.md"
    license = "Apache-2.0"
    license-files = ["LICENSE"]
    requires-python = ">=3.10"
    dependencies = ["gafime=={version}"]
    """,
    )
    write_text(
        output / "MANIFEST.in",
        f"""
    include README.md
    include LICENSE
    recursive-include {package_name} *
    recursive-include gafime _dummy.c
    recursive-include src/common *.hpp
    recursive-include src/{source_subdir} *
    global-exclude *.py[cod]
    global-exclude __pycache__
    """,
    )
    rt_policy_text = (
        "This separately selected variant enables OptiX RT and requires an "
        "OptiX SDK include directory at build time. It is not part of the "
        "standard PyPI release bundle."
        if kind == "cuda" and cuda_rt_mode == "on"
        else "The standard CUDA variant keeps OptiX RT disabled and requires "
        "no OptiX SDK headers or OptiX build cost."
        if kind == "cuda"
        else ""
    )
    write_text(
        output / "README.md",
        f"""
    # {dist_name}

    Vendor GPU runtime payload for GAFIME {version}.

    This package is generated from the GAFIME source tree during CI and carries
    only the {kind.upper()} native runtime payload. Install the base package
    with `gafime`; use this package only for the matching GPU runtime.

    {rt_policy_text}
    """,
    )
    shutil.copy2(REPO_ROOT / "LICENSE", output / "LICENSE")
    if kind == "cuda":
        write_text(
            output / package_name / "build_policy.json",
            json.dumps(
                {
                    "cuda_architectures": ["75", "80", "86", "89", "90", "100", "120"],
                    "cuda_tuning_policy": "runtime-device-class",
                    "cuda_tuning_sm": None,
                    "optix_rt": cuda_rt_mode,
                    "per_architecture_tuning": False,
                    "runtime_architecture_dispatch": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        if provenance is not None:
            write_text(
                output / package_name / "build_provenance.json",
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            )
    write_text(
        output / package_name / "__init__.py",
        f"""
    from __future__ import annotations

    from pathlib import Path

    __version__ = "{version}"


    def package_dir() -> Path:
        return Path(__file__).resolve().parent


    def library_candidates() -> list[Path]:
        base = package_dir()
        return [
            base / "{package_name}.dll",
            base / "lib{package_name}.so",
            base / "{package_name}.so",
            base / "{package_name}.pyd",
        ]
    """,
    )
    write_text(
        output / "setup.py",
        setup_template.format(
            version=version,
            cuda_rt_mode=cuda_rt_mode,
            dist_name=dist_name,
            package_name=package_name,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("cuda", "rocm"))
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--cuda-rt",
        choices=("off", "on"),
        default="off",
        help="Stage an immutable OptiX RT policy; standard release staging uses off.",
    )
    parser.add_argument(
        "--optix-sdk-archive-sha256",
        help="Expected OptiX SDK archive SHA-256; required with --cuda-rt on.",
    )
    parser.add_argument(
        "--cuda-fixture-image",
        help="Digest-pinned CUDA lifecycle-fixture image; required with --cuda-rt on.",
    )
    parser.add_argument(
        "--wheel-builder-image",
        help="Digest-pinned manylinux wheel-builder image; required with --cuda-rt on.",
    )
    parser.add_argument(
        "--cuda-rpm-base-url",
        help="HTTPS repository base URL for pinned CUDA RPMs; required with --cuda-rt on.",
    )
    parser.add_argument(
        "--cuda-rpm-manifest",
        type=Path,
        help="SHA-256 manifest for CUDA RPM inputs; required with --cuda-rt on.",
    )
    args = parser.parse_args()
    try:
        stage_payload(
            args.kind,
            args.output,
            args.cuda_rt,
            args.optix_sdk_archive_sha256,
            args.cuda_fixture_image,
            args.wheel_builder_image,
            args.cuda_rpm_base_url,
            args.cuda_rpm_manifest,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
