from __future__ import annotations

import argparse
import shutil
import textwrap
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


CUDA_SETUP = r'''
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


VERSION = "{version}"
ROOT = Path(__file__).resolve().parent


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
        package_dir = Path(self.build_lib) / "gafime_cuda"
        package_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = package_dir
        self.build_cuda_backend()
        super().run()

    def build_cuda_backend(self) -> None:
        machine = platform.machine().lower()
        if machine in {{"aarch64", "arm64"}} or machine.startswith("arm"):
            raise RuntimeError(f"gafime-cuda does not support ARM target {{platform.machine()}}.")
        nvcc = _find_nvcc()
        if not nvcc:
            raise RuntimeError(
                "nvcc was not found. Install CUDA Toolkit 13.2+ to build gafime-cuda "
                "or set CUDA_PATH/CUDA_HOME to the toolkit root."
            )

        src_dir = ROOT / "src"
        cuda_source = src_dir / "cuda" / "kernels.cu"
        if sys.platform == "win32":
            output_file = self.output_dir / "gafime_cuda.dll"
            compiler_flags = ["/MD", "/O2"]
        else:
            output_file = self.output_dir / "libgafime_cuda.so"
            compiler_flags = ["-fPIC", "-O3"]

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
            "-O3",
            "--shared",
            "-DGAFIME_BUILDING_DLL",
            "-cudart",
            "static",
            "-Xcompiler",
            ",".join(compiler_flags),
            "-I",
            str(src_dir / "common"),
            "-o",
            str(output_file),
            str(cuda_source),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"CUDA build failed\nSTDOUT:\n{{result.stdout}}\nSTDERR:\n{{result.stderr}}")


setup(
    name="gafime-cuda",
    version=VERSION,
    description="NVIDIA CUDA runtime payload for GAFIME",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=["gafime_cuda"],
    package_data={{"gafime_cuda": ["*.so", "*.dll", "*.pyd"]}},
    include_package_data=False,
    install_requires=[f"gafime=={{VERSION}}"],
    python_requires=">=3.10",
    ext_modules=[Extension("gafime_cuda._native", sources=[str(ROOT / "gafime" / "_dummy.c")])],
    cmdclass={{"build_ext": CudaPayloadBuildExt}},
)
'''


ROCM_SETUP = r'''
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


VERSION = "{version}"
ROOT = Path(__file__).resolve().parent


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
        rocm_source = src_dir / "rocm" / "kernels.hip"
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

        cmd = [
            hipcc,
            *arch_flags,
            "-O3",
            "--shared",
            "-Wno-unused-result",
            "-DGAFIME_BUILDING_DLL",
            "-I",
            str(src_dir / "common"),
            "-o",
            str(output_file),
            str(rocm_source),
        ]
        if sys.platform != "win32":
            cmd.insert(cmd.index("-Wno-unused-result"), "-fPIC")
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
    name="gafime-rocm",
    version=VERSION,
    description="AMD ROCm/HIP runtime payload for GAFIME",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=["gafime_rocm"],
    package_data={{"gafime_rocm": ["*.so", "*.dll", "*.pyd"]}},
    include_package_data=False,
    install_requires=[f"gafime=={{VERSION}}"],
    python_requires=">=3.10",
    ext_modules=[Extension("gafime_rocm._native", sources=[str(ROOT / "gafime" / "_dummy.c")])],
    cmdclass={{"build_ext": RocmPayloadBuildExt}},
)
'''


def project_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def stage_payload(kind: str, output: Path) -> None:
    version = project_version()
    package_name = f"gafime_{kind}"
    dist_name = f"gafime-{kind}"
    source_subdir = "cuda" if kind == "cuda" else "rocm"
    source_name = "kernels.cu" if kind == "cuda" else "kernels.hip"
    setup_template = CUDA_SETUP if kind == "cuda" else ROCM_SETUP

    if output.exists():
        shutil.rmtree(output)
    (output / package_name).mkdir(parents=True)
    (output / "gafime").mkdir(parents=True)
    (output / "src" / source_subdir).mkdir(parents=True)
    (output / "src" / "common").mkdir(parents=True)

    shutil.copy2(REPO_ROOT / "gafime" / "_dummy.c", output / "gafime" / "_dummy.c")
    shutil.copy2(REPO_ROOT / "src" / source_subdir / source_name, output / "src" / source_subdir / source_name)
    shutil.copy2(REPO_ROOT / "src" / "common" / "interfaces.h", output / "src" / "common" / "interfaces.h")

    write_text(output / "pyproject.toml", """
    [build-system]
    requires = ["setuptools>=77", "wheel"]
    build-backend = "setuptools.build_meta"
    """)
    write_text(output / "MANIFEST.in", f"""
    include README.md
    recursive-include {package_name} *
    recursive-include gafime _dummy.c
    recursive-include src/common interfaces.h
    recursive-include src/{source_subdir} {source_name}
    global-exclude *.py[cod]
    global-exclude __pycache__
    """)
    write_text(output / "README.md", f"""
    # {dist_name}

    Vendor GPU runtime payload for GAFIME {version}.

    This package is generated from the GAFIME source tree during CI and carries
    only the {kind.upper()} native runtime payload. Install the base package
    with `gafime`; use this package only for the matching GPU runtime.
    """)
    write_text(output / package_name / "__init__.py", f"""
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
    """)
    write_text(output / "setup.py", setup_template.format(version=version))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("cuda", "rocm"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage_payload(args.kind, args.output)


if __name__ == "__main__":
    main()
