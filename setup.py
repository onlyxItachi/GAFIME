"""
GAFIME Build System - Base Python/Core/Rust/Metal Package

This setup.py builds the base ``gafime`` package:
- Python API and backend resolver
- C++ Core backend
- Rust helper/subfunctions
- Metal backend on macOS arm64

CUDA and ROCm/HIP runtime payloads are separate distributions:
- ``gafime-cuda``
- ``gafime-rocm``

Usage:
    python setup.py build_ext --inplace
    
Requirements:
    - C++ compiler with OpenMP support (for CPU backend)
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


VERSION = "0.4.7"


BASE_PACKAGE_DATA = [
    "_native*.so",
    "_native*.pyd",
    "gafime_core*.so",
    "gafime_core*.pyd",
    "gafime_cpu.so",
    "gafime_cpu.pyd",
    "gafime_metal.dylib",
    "gafime_kernels.metallib",
]

VENDOR_PAYLOAD_PATTERNS = [
    "libgafime_cuda.so",
    "gafime_cuda.so",
    "gafime_cuda.dll",
    "gafime_cuda.pyd",
    "libgafime_rocm.so",
    "gafime_rocm.so",
    "gafime_rocm.dll",
    "gafime_rocm.pyd",
]


def _remove_vendor_payload_artifacts(directory: Path) -> None:
    if not directory.exists():
        return
    for pattern in VENDOR_PAYLOAD_PATTERNS:
        for file in directory.rglob(pattern):
            try:
                file.unlink()
            except OSError:
                pass


class NativeBuildExt(build_ext):
    """Custom build command for the base package native backends."""
    
    def run(self):
        # Clean source tree's gafime/ to prevent cross-contamination across builds
        src_gafime = Path(__file__).parent / "gafime"
        if src_gafime.exists():
            for ext in ["*.so", "*.dll", "*.dylib", "*.metallib", "*.pyd", "*.air"]:
                for f in src_gafime.rglob(ext):
                    try:
                        f.unlink()
                    except OSError:
                        pass
        
        # Decide output directory based on editable mode vs isolated build
        if self.inplace:
            self.output_dir = src_gafime
        else:
            self.output_dir = Path(self.build_lib) / "gafime"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _remove_vendor_payload_artifacts(self.output_dir)
        
        # We manually build base backends and drop the .so/.dll/.dylib
        # artifacts directly into the targeted python package folder
        self.build_metal_backend()
        self.build_cpp_core()
        self.build_rust_backend()
        
        # We must call super().run() so setuptools correctly identifies this as a non-pure python wheel
        super().run()
        _remove_vendor_payload_artifacts(self.output_dir)
        
        # Print summary of what was built
        self._print_build_summary()
        
    def build_metal_backend(self):
        """Build Metal backend for Apple Silicon."""
        print("\n" + "=" * 60)
        print("Building Metal Backend")
        print("=" * 60)
        
        if sys.platform != "darwin" or platform.machine() != "arm64":
            print(">> Skipping Metal backend (requires macOS arm64)")
            return
        
        src_dir = Path(__file__).parent / "src"
        output_dir = self.output_dir
        metal_dir = src_dir / "metal"
        
        xcrun = shutil.which("xcrun")
        if not xcrun:
            print("!  xcrun not found")
            return
            
        air_file = output_dir / "gafime_kernels.air"
        metallib_file = output_dir / "gafime_kernels.metallib"
        dylib_file = output_dir / "gafime_metal.dylib"
        
        cmd_air = [xcrun, "metal", "-std=metal3.0", "-O3", "-c", str(metal_dir / "gafime_kernels.metal"), "-o", str(air_file)]
        cmd_lib = [xcrun, "metallib", str(air_file), "-o", str(metallib_file)]
        cmd_dylib = [
            shutil.which("clang++"), "-std=c++23", "-O3", "-shared", "-fPIC", "-fobjc-arc",
            "-framework", "Metal", "-framework", "Foundation",
            f"-I{metal_dir}", f"-I{src_dir / 'common'}",
            "-o", str(dylib_file), str(metal_dir / "metal_backend.mm"),
        ]
        
        subprocess.run(cmd_air, check=True)
        subprocess.run(cmd_lib, check=True)
        air_file.unlink(missing_ok=True)
        subprocess.run(cmd_dylib, check=True)
        print(f"[OK] Metal backend built: {dylib_file.name}")

    def build_cpp_core(self):
        """Build C++ pybind11 Core backend using CMake."""
        print("\n" + "=" * 60)
        print("Building C++ Core (gafime_core)")
        print("=" * 60)
        
        src_dir = Path(__file__).parent / "gafime_core"
        build_dir = src_dir / "build"
        output_dir = self.output_dir
        
        cmake = shutil.which("cmake")
        if not cmake or not src_dir.exists():
            if os.environ.get("STRICT_CPU", "0") == "1":
                print("! STRICT_CPU is set but cmake or gafime_core source not found!")
                sys.exit(1)
            print("!  cmake or source not found")
            return
            
        # Prevent older python architecture binaries from bleeding into new wheels
        shutil.rmtree(build_dir, ignore_errors=True)
        build_dir.mkdir(exist_ok=True)
        
        pybind_cmd = [sys.executable, "-m", "pybind11", "--cmakedir"]
        pybind_dir = subprocess.check_output(pybind_cmd).decode('utf-8').strip()
        
        import sysconfig
        cmake_cmd = [
            cmake, "..",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGAFIME_CORE_ENABLE_OPENMP=ON",
            "-DGAFIME_CORE_USE_FETCHCONTENT=OFF",
            f"-DGAFIME_CORE_USE_DOUBLE_PRECISION={os.environ.get('GAFIME_CORE_USE_DOUBLE_PRECISION', 'OFF')}",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython3_INCLUDE_DIR={sysconfig.get_path('include')}",
            f"-Dpybind11_DIR={pybind_dir}"
        ]
        subprocess.run(cmake_cmd, cwd=build_dir, check=True)
        subprocess.run([cmake, "--build", ".", "--config", "Release"], cwd=build_dir, check=True)
        
        # Copy pybind artifact (.so / .pyd) to gafime/
        for ext in ["*.so", "*.pyd", "*.dylib"]:
            for file in build_dir.rglob(ext):
                if "gafime_core" in file.name:
                    shutil.copy(file, output_dir / file.name)
        print("[OK] C++ Core built")

    def build_rust_backend(self):
        """Build Rust PyO3 Extension."""
        print("\n" + "=" * 60)
        print("Building Rust Backend (gafime_cpu)")
        print("=" * 60)
        
        rust_dir = Path(__file__).parent / "src" / "cpu" / "gafime_cpu"
        output_dir = self.output_dir
        
        cargo = shutil.which("cargo")
        if not cargo or not rust_dir.exists():
            if os.environ.get("STRICT_CPU", "0") == "1":
                print("! STRICT_CPU is set but cargo or Rust source not found!")
                sys.exit(1)
            print("!  cargo not found")
            return
            
        env = os.environ.copy()
        # Bypass PyO3's strict version check for Python versions newer than the crate (e.g. Python 3.14 alpha/beta)
        env["PYO3_USE_ABI3_FORWARD_COMPATIBILITY"] = "1"
        if sys.platform == "darwin":
            # PyO3 on macOS requires these linker flags when built directly via cargo cdylib
            env["RUSTFLAGS"] = env.get("RUSTFLAGS", "") + " -C link-arg=-undefined -C link-arg=dynamic_lookup"
            
        subprocess.run([cargo, "build", "--release", "--manifest-path", str(rust_dir / "Cargo.toml")], env=env, check=True)
        
        # Find the compiled binary in target/release/
        target_dir = rust_dir / "target" / "release"
        found = False
        for ext in ["*.so", "*.dll", "*.dylib"]:
            for file in target_dir.glob(ext):
                # PyO3 requires specific extension based on OS
                target_name = "gafime_cpu.so"
                if sys.platform == "win32":
                    target_name = "gafime_cpu.pyd"
                    
                shutil.copy(file, output_dir / target_name)
                found = True
                break
        if found:
            print("[OK] Rust Core built")
        else:
            print("[ERROR] Rust binary not found in target/release/")
            sys.exit(1)

    def _print_build_summary(self):
        """Print summary of built artifacts for CI visibility."""
        print("\n" + "=" * 60)
        print("BUILD SUMMARY")
        print("=" * 60)
        
        expected = {
            "Metal": ["gafime_metal.dylib"],
            "Core (pybind11)": ["gafime_core"],
            "Rust (PyO3)": ["gafime_cpu.so", "gafime_cpu.pyd"],
        }
        
        found_any = False
        for name, patterns in expected.items():
            found = []
            for p in patterns:
                if p == "gafime_core":
                    found.extend(self.output_dir.glob("*gafime_core*"))
                else:
                    candidate = self.output_dir / p
                    if candidate.exists():
                        found.append(candidate)
            if found:
                for f in found:
                    size_kb = f.stat().st_size / 1024
                    print(f"  [OK] {name:20s} -> {f.name} ({size_kb:.0f} KB)")
                found_any = True
            else:
                print(f"  [--] {name:20s} -> NOT BUILT")
        
        if not found_any:
            print("\n  WARNING: No native backends were built!")
        print("=" * 60 + "\n")


setup(
    name="gafime",
    version=VERSION,
    description="GPU Accelerated Feature Interaction Mining Engine",
    author="Hamza",
    packages=find_packages(include=["gafime", "gafime.*"], exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "polars>=0.20",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
        "sklearn": [
            "scikit-learn>=1.0",
        ],
        "cuda": [
            f"gafime-cuda=={VERSION}",
        ],
        "rocm": [
            f"gafime-rocm=={VERSION}; platform_system == 'Linux' and platform_machine == 'x86_64'",
        ],
        "bench": [
            "pandas>=2.0",
            "scipy>=1.10",
            "scikit-learn>=1.0",
            "xgboost>=2.0",
            "lightgbm>=4.0",
            "catboost>=1.2",
            "build",
            "twine",
        ],
    },
    # Including an Extension tells cibuildwheel this is a native C/C++/Rust package,
    # forcing it to output a platform-specific .whl (e.g. macos_14_arm64) instead of py3-none-any.
    # We include a dummy C file so older/newer setuptools don't optimize out the extension!
    ext_modules=[Extension("gafime._native", sources=["gafime/_dummy.c"])],
    package_data={
        "gafime": BASE_PACKAGE_DATA,
    },
    exclude_package_data={
        "gafime": VENDOR_PAYLOAD_PATTERNS,
    },
    include_package_data=False,
    entry_points={
        "console_scripts": [
            "gafime=gafime.cli:main",
        ],
    },
    cmdclass={
        "build_ext": NativeBuildExt,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
