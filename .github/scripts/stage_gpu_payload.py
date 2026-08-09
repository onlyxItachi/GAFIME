from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
ROCM_SYSTEM_POLICY_PATH = (
    REPO_ROOT / ".github" / "scripts" / "rocm_7_2_3_system_policy.json"
)
ROCM_RELEASE_ARCHITECTURES = (
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
)


CUDA_SETUP = r"""
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
DIST_NAME = "{dist_name}"
PACKAGE_NAME = "{package_name}"
CUDA_LANGUAGE_STANDARD = "c++20"
CUDA_ARCHITECTURES = ("75", "80", "86", "89", "90", "100", "120")
CUDA_TUNING_POLICY = "runtime-device-class"
PRECISION_ABI_VERSION = "1.1"
PRECISION_PROFILES = ("fp32", "mixed", "fp64")
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
            src_dir / "cuda" / "precision_kernels.cu",
            src_dir / "cuda" / "precision_launcher.cu",
        ]
        if sys.platform == "win32":
            output_file = self.output_dir / f"{{PACKAGE_NAME}}.dll"
            compiler_flags = ["/MD"]
        else:
            output_file = self.output_dir / f"lib{{PACKAGE_NAME}}.so"
            compiler_flags = [
                "-fPIC",
                "-fvisibility=hidden",
                "-fvisibility-inlines-hidden",
            ]

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
        cmd = [
            nvcc,
            *gencode_flags,
            f"--std={{CUDA_LANGUAGE_STANDARD}}",
            "-O3",
            "-rdc=true",
            "--shared",
            "-DGAFIME_GPU_BUILDING_DLL",
            "-DGAFIME_GPU_MI_ACCUMULATION_FP64=0",
            "-cudart",
            "shared",
            "-Xcompiler",
            ",".join(compiler_flags),
            *(
                [
                    "-Xlinker",
                    f"--version-script={{src_dir / 'common' / 'gafime_gpu_exports.map'}}",
                ]
                if sys.platform != "win32"
                else []
            ),
            "-I",
            str(src_dir / "common"),
            "-I",
            str(src_dir / "cuda"),
            "-o",
            str(output_file),
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
        ]
    }},
    include_package_data=False,
    ext_modules=[
        Extension(
            f"{{PACKAGE_NAME}}._native",
            sources=[str(ROOT / "gafime" / "_dummy.c")],
        )
    ],
    cmdclass={{"build_ext": CudaPayloadBuildExt}},
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

ROOT = Path(__file__).resolve().parent
DIST_NAME = "{dist_name}"
PACKAGE_NAME = "{package_name}"
ROCM_WHEEL_POLICY = "{rocm_wheel_policy}"
PRECISION_ABI_VERSION = "1.1"
PRECISION_PROFILES = ("fp32", "mixed", "fp64")


def _rocm_wheel_policy() -> str:
    requested = os.environ.get("GAFIME_ROCM_WHEEL_POLICY")
    if requested is not None and requested.strip().lower() != ROCM_WHEEL_POLICY:
        raise RuntimeError(
            f"this staged {{DIST_NAME}} source has immutable wheel policy "
            f"{{ROCM_WHEEL_POLICY!r}}, not {{requested!r}}; restage the payload instead"
        )
    return ROCM_WHEEL_POLICY


_rocm_wheel_policy()

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


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
        package_dir = Path(self.build_lib) / PACKAGE_NAME
        package_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = package_dir
        self.build_rocm_backend()
        super().run()

    def build_rocm_backend(self) -> None:
        _rocm_wheel_policy()
        if sys.platform != "linux":
            raise RuntimeError(
                f"the {{ROCM_WHEEL_POLICY}} {{DIST_NAME}} wheel policy supports "
                "Linux x86_64 only"
            )
        machine = platform.machine().lower()
        if machine in {{"aarch64", "arm64"}} or machine.startswith("arm"):
            raise RuntimeError(
                f"{{DIST_NAME}} does not support ARM target {{platform.machine()}}."
            )

        hipcc = shutil.which("hipcc")
        if not hipcc:
            raise RuntimeError(
                f"hipcc was not found. Install ROCm/HIP to build {{DIST_NAME}}."
            )

        src_dir = ROOT / "src"
        rocm_sources = [
            src_dir / "rocm" / "kernels.hip",
            src_dir / "rocm" / "launcher.hip",
        ]
        output_file = self.output_dir / f"lib{{PACKAGE_NAME}}.so"
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
            "-DGAFIME_GPU_MI_ACCUMULATION_FP64=0",
            "-DGAFIME_HIP_PRECISION_PROFILE_MASK=7",
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
            shared_index = cmd.index("--shared")
            cmd[shared_index:shared_index] = [
                "-fPIC",
                "-fvisibility=hidden",
                "-fvisibility-inlines-hidden",
                f"-Wl,--version-script={{src_dir / 'common' / 'gafime_gpu_exports.map'}}",
            ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ROCm/HIP build failed\nSTDOUT:\n{{result.stdout}}\nSTDERR:\n{{result.stderr}}")
        if ROCM_WHEEL_POLICY == "system":
            patchelf = shutil.which("patchelf")
            if patchelf is None:
                raise RuntimeError(
                    "the system ROCm wheel policy requires patchelf to remove "
                    "build-host ROCm search paths"
                )
            subprocess.run([patchelf, "--remove-rpath", str(output_file)], check=True)

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
    packages=[PACKAGE_NAME],
    package_data={{
        PACKAGE_NAME: ["*.so", "*.dll", "*.pyd", "build_policy.json"]
    }},
    include_package_data=False,
    ext_modules=[
        Extension(
            f"{{PACKAGE_NAME}}._native",
            sources=[str(ROOT / "gafime" / "_dummy.c")],
        )
    ],
    cmdclass={{"build_ext": RocmPayloadBuildExt}},
)
"""


def project_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def stage_payload(
    kind: str,
    output: Path,
    rocm_wheel_policy: str = "system",
) -> None:
    if kind == "cuda" and rocm_wheel_policy != "system":
        raise ValueError("--rocm-wheel-policy applies only to the ROCm payload")
    rocm_policy = None
    if kind == "rocm":
        rocm_wheel_policy = rocm_wheel_policy.strip().lower()
        if rocm_wheel_policy != "system":
            raise ValueError(
                f"ROCm wheel policy {rocm_wheel_policy!r} is not implemented; "
                "distributed ROCm payloads require the system runtime"
            )
        rocm_policy = json.loads(ROCM_SYSTEM_POLICY_PATH.read_text(encoding="utf-8"))
        if rocm_policy.get("wheel_policy") != rocm_wheel_policy:
            raise ValueError("checked-in ROCm wheel policy does not match the request")
        if rocm_policy.get("gfx_targets") != list(ROCM_RELEASE_ARCHITECTURES):
            raise ValueError(
                "checked-in ROCm wheel policy targets do not match staged release targets"
            )
    version = project_version()
    package_name = f"gafime_{kind}"
    dist_name = f"gafime-{kind}"
    if (
        rocm_policy is not None
        and rocm_policy.get("distribution_identity") != dist_name
    ):
        raise ValueError(
            "checked-in ROCm policy distribution identity does not match staged source"
        )
    gpu_src_root = REPO_ROOT / "src"
    source_subdir = "cuda" if kind == "cuda" else "rocm"
    source_names = (
        [
            "cuda_api.hpp",
            "cuda_internal.hpp",
            "kernels.cuh",
            "precision_kernels.cuh",
            "precision_kernels.cu",
            "precision_launcher.cu",
        ]
        if kind == "cuda"
        else [
            "rocm_api.hpp",
            "kernels.hpp",
            "kernels.hip",
            "launcher.hip",
            "precision.hpp",
        ]
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
    common_source_names = (
        "covariance_policy.hpp",
        "gafime_gpu_abi.hpp",
        "gafime_gpu_internal_abi.hpp",
        "gafime_gpu_exports.map",
        "gpu_abi_impl.hpp",
    )
    for source_name in common_source_names:
        shutil.copy2(
            gpu_src_root / "common" / source_name,
            output / "src" / "common" / source_name,
        )

    description = (
        "NVIDIA CUDA system-runtime payload for GAFIME"
        if kind == "cuda"
        else "AMD ROCm/HIP system-runtime payload for GAFIME"
    )
    build_requirements = (
        '["setuptools>=77", "wheel", "patchelf>=0.17"]'
        if kind == "rocm"
        else '["setuptools>=77", "wheel"]'
    )
    write_text(
        output / "pyproject.toml",
        f"""
    [build-system]
    requires = {build_requirements}
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
    recursive-include src/common *.hpp *.map
    recursive-include src/{source_subdir} *
    global-exclude *.py[cod]
    global-exclude __pycache__
    """,
    )
    runtime_policy_text = (
        "This distribution contains only GAFIME's CUDA binaries. It excludes "
        "OptiX/RT sources and requires a system CUDA runtime."
        if kind == "cuda"
        else "This distribution contains only GAFIME's ROCm binaries and "
        "requires a system ROCm runtime."
    )
    write_text(
        output / "README.md",
        f"""
    # {dist_name}

    Vendor GPU runtime payload for GAFIME {version}.

    This package is generated from the GAFIME source tree during CI and carries
    only the {kind.upper()} native runtime payload. Install the base package
    with `gafime`; use this package only for the matching GPU runtime.

    {runtime_policy_text}
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
                    "cuda_runtime": "system",
                    "cuda_runtime_libraries": {
                        "linux": "libcudart.so.13",
                        "windows": "nvcudart_hybrid64.dll",
                    },
                    "optix_rt": "off",
                    "precision_abi_version": "1.1",
                    "precision_profiles": ["fp32", "mixed", "fp64"],
                    "rt_sources_included": False,
                    "per_architecture_tuning": False,
                    "runtime_architecture_dispatch": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    else:
        write_text(
            output / package_name / "build_policy.json",
            json.dumps(rocm_policy, indent=2, sort_keys=True) + "\n",
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
            dist_name=dist_name,
            package_name=package_name,
            rocm_wheel_policy=rocm_wheel_policy,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("cuda", "rocm"))
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--rocm-wheel-policy",
        choices=("system",),
        default="system",
        help="ROCm distributions always require the system runtime.",
    )
    args = parser.parse_args()
    try:
        stage_payload(
            args.kind,
            args.output,
            args.rocm_wheel_policy,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
