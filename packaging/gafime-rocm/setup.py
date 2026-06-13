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
    if (repo_root / "src" / "rocm" / "kernels.hip").exists():
        return repo_root
    return Path(__file__).resolve().parent


ROOT = _source_root()


class RocmPayloadBuildExt(build_ext):
    def run(self):
        package_dir = Path(self.build_lib) / "gafime_rocm"
        package_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = package_dir
        self.build_rocm_backend()
        super().run()

    def build_rocm_backend(self) -> None:
        if sys.platform not in {"linux", "win32"}:
            raise RuntimeError("gafime-rocm currently supports Linux/Windows x86_64 targets.")
        machine = platform.machine().lower()
        if machine in {"aarch64", "arm64"} or machine.startswith("arm"):
            raise RuntimeError(f"gafime-rocm does not support ARM target {platform.machine()}.")

        hipcc = shutil.which("hipcc")
        if not hipcc:
            raise RuntimeError("hipcc was not found. Install ROCm/HIP to build gafime-rocm.")

        src_dir = ROOT / "src"
        rocm_source = src_dir / "rocm" / "kernels.hip"
        output_file = self.output_dir / ("gafime_rocm.dll" if sys.platform == "win32" else "libgafime_rocm.so")
        arch_env = os.environ.get("GAFIME_ROCM_ARCHS")
        if arch_env:
            archs = [arch.strip() for arch in arch_env.replace(";", ",").replace(" ", ",").split(",") if arch.strip()]
        else:
            archs = self._detect_rocm_archs()
        if not archs:
            raise RuntimeError(
                "Unable to detect ROCm/HIP offload architecture. "
                "Set GAFIME_ROCM_ARCHS explicitly, for example GAFIME_ROCM_ARCHS=<gfx-target>."
            )
        arch_flags = [f"--offload-arch={arch}" for arch in archs]

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
            raise RuntimeError(f"ROCm/HIP build failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

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
    long_description=(PACKAGE_ROOT / "README.md").read_text(encoding="utf-8") if (PACKAGE_ROOT / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    packages=["gafime_rocm"],
    package_dir={"": str(ROOT)},
    package_data={"gafime_rocm": ["*.so", "*.dll", "*.pyd"]},
    include_package_data=False,
    install_requires=[f"gafime=={VERSION}"],
    python_requires=">=3.10",
    ext_modules=[Extension("gafime_rocm._native", sources=[str(ROOT / "gafime" / "_dummy.c")])],
    cmdclass={"build_ext": RocmPayloadBuildExt},
)
