from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


VERSION = "0.4.7"
PACKAGE_ROOT = Path(__file__).resolve().parent


def _source_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "src" / "cuda" / "kernels.cu").exists():
        return repo_root
    return Path(__file__).resolve().parent


ROOT = _source_root()


class CudaPayloadBuildExt(build_ext):
    def run(self):
        package_dir = Path(self.build_lib) / "gafime_cuda"
        package_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = package_dir
        self.build_cuda_backend()
        super().run()

    def build_cuda_backend(self) -> None:
        machine = platform.machine().lower()
        if machine in {"aarch64", "arm64"} or machine.startswith("arm"):
            raise RuntimeError(f"gafime-cuda does not support ARM target {platform.machine()}.")

        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("nvcc was not found. Install CUDA Toolkit 13.2+ to build gafime-cuda.")

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
            raise RuntimeError(f"CUDA build failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


setup(
    name="gafime-cuda",
    version=VERSION,
    description="NVIDIA CUDA runtime payload for GAFIME",
    long_description=(PACKAGE_ROOT / "README.md").read_text(encoding="utf-8") if (PACKAGE_ROOT / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    packages=["gafime_cuda"],
    package_dir={"": str(ROOT)},
    package_data={"gafime_cuda": ["*.so", "*.dll", "*.pyd"]},
    include_package_data=False,
    install_requires=[f"gafime=={VERSION}"],
    python_requires=">=3.10",
    ext_modules=[Extension("gafime_cuda._native", sources=[str(ROOT / "gafime" / "_dummy.c")])],
    cmdclass={"build_ext": CudaPayloadBuildExt},
)
