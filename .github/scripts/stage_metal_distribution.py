from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import textwrap

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "gafime_metal"
DIST_NAME = "gafime-metal"
METAL_LIBRARY = "libgafime_metal_v1.dylib"
METALLIB = "gafime_metal_v1.metallib"


SETUP = r"""
from __future__ import annotations

import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "gafime_metal"
METAL_LIBRARY = "libgafime_metal_v1.dylib"
METALLIB = "gafime_metal_v1.metallib"


class MetalPayloadBuildExt(build_ext):
    def run(self) -> None:
        if sys.platform != "darwin" or platform.machine().lower() not in {
            "arm64",
            "aarch64",
        }:
            raise RuntimeError("gafime-metal supports macOS arm64 only")

        deployment_target = os.environ.get(
            "MACOSX_DEPLOYMENT_TARGET", "11.0"
        ).strip()
        if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", deployment_target):
            raise RuntimeError(
                "MACOSX_DEPLOYMENT_TARGET must be a numeric macOS version"
            )

        package_dir = Path(self.build_lib) / PACKAGE_NAME
        package_dir.mkdir(parents=True, exist_ok=True)
        build_dir = Path(self.build_temp).resolve() / "gafime-metal"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)

        configure = [
            "cmake",
            "-S",
            str(ROOT / "src" / "metal"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_OSX_ARCHITECTURES=arm64",
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={package_dir.resolve()}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={package_dir.resolve()}",
        ]
        subprocess.run(configure, check=True)
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                "gafime_metal_v1",
            ],
            check=True,
        )

        library = package_dir / METAL_LIBRARY
        metallib_source = build_dir / METALLIB
        metallib = package_dir / METALLIB
        if not library.is_file() or library.stat().st_size == 0:
            raise RuntimeError(f"Metal build did not produce {library}")
        if not metallib_source.is_file() or metallib_source.stat().st_size == 0:
            raise RuntimeError(f"Metal build did not produce {metallib_source}")
        shutil.copy2(metallib_source, metallib)
        subprocess.run(["lipo", str(library), "-verify_arch", "arm64"], check=True)
        super().run()


setup(
    packages=[PACKAGE_NAME],
    package_data={
        PACKAGE_NAME: [
            "*.dylib",
            "*.metallib",
            "build_policy.json",
        ]
    },
    include_package_data=False,
    ext_modules=[
        Extension(
            f"{PACKAGE_NAME}._native",
            sources=[str(ROOT / "gafime" / "_dummy.c")],
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": MetalPayloadBuildExt},
    options={
        "bdist_wheel": {
            "py_limited_api": "cp310",
            "plat_name": "macosx_11_0_arm64",
        }
    },
)
"""


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def _project_version() -> str:
    project = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return str(project["project"]["version"])


def stage_metal_distribution(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        shutil.rmtree(output)

    version = _project_version()
    (output / PACKAGE_NAME).mkdir(parents=True)
    (output / "gafime").mkdir(parents=True)
    (output / "src" / "metal").mkdir(parents=True)
    (output / "src" / "common").mkdir(parents=True)

    for source in (
        "CMakeLists.txt",
        "launcher.mm",
        "metal_api.hpp",
        "shader.metal",
    ):
        shutil.copy2(REPO_ROOT / "src" / "metal" / source, output / "src" / "metal")
    for source in (
        "covariance_policy.hpp",
        "gafime_gpu_abi.hpp",
        "gpu_abi_impl.hpp",
    ):
        shutil.copy2(REPO_ROOT / "src" / "common" / source, output / "src" / "common")
    shutil.copy2(REPO_ROOT / "LICENSE", output / "LICENSE")

    _write_text(
        output / "gafime" / "_dummy.c",
        """
        #define Py_LIMITED_API 0x030A0000
        #include <Python.h>

        static struct PyModuleDef gafime_metal_payload_module = {
            PyModuleDef_HEAD_INIT,
            "_native",
            NULL,
            -1,
            NULL,
        };

        PyMODINIT_FUNC PyInit__native(void) {
            return PyModule_Create(&gafime_metal_payload_module);
        }
        """,
    )
    _write_text(
        output / "pyproject.toml",
        f"""
        [build-system]
        requires = ["setuptools>=77", "wheel"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "{DIST_NAME}"
        version = "{version}"
        description = "Apple Metal runtime payload for GAFIME"
        readme = "README.md"
        license = "Apache-2.0"
        license-files = ["LICENSE"]
        requires-python = ">=3.10"
        dependencies = ["gafime=={version}"]
        """,
    )
    _write_text(
        output / "MANIFEST.in",
        f"""
        include LICENSE
        include README.md
        recursive-include {PACKAGE_NAME} *
        recursive-include gafime _dummy.c
        recursive-include src/common *.hpp
        recursive-include src/metal *
        global-exclude *.py[cod]
        global-exclude __pycache__
        """,
    )
    _write_text(
        output / "README.md",
        f"""
        # {DIST_NAME}

        Apple Metal runtime payload for GAFIME {version}.

        Install `gafime` and this exact-version payload on Apple Silicon macOS.
        The base distribution remains vendor-payload-free; this package owns the
        paired Metal dylib and metallib used by the public `backend="metal"` path.
        """,
    )
    policy = {
        "backend": "metal",
        "deployment_target": "11.0",
        "distribution_identity": DIST_NAME,
        "library": METAL_LIBRARY,
        "metallib": METALLIB,
        "platform": "macosx_11_0_arm64",
        "schema_version": 1,
    }
    _write_text(
        output / PACKAGE_NAME / "build_policy.json",
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
    )
    _write_text(
        output / PACKAGE_NAME / "__init__.py",
        f"""
        from __future__ import annotations

        from pathlib import Path

        __version__ = "{version}"


        def package_dir() -> Path:
            return Path(__file__).resolve().parent


        def library_candidates() -> list[Path]:
            return [package_dir() / "{METAL_LIBRARY}"]


        def metallib_candidates() -> list[Path]:
            return [package_dir() / "{METALLIB}"]
        """,
    )
    _write_text(output / "setup.py", SETUP)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage_metal_distribution(args.output)


if __name__ == "__main__":
    main()
