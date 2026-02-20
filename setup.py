"""
GAFIME Build System - Native CUDA/CPU Backend Compilation

This setup.py handles compilation of the native backends:
- CUDA backend: Uses nvcc for RTX 4060 (SM89, Ada Lovelace)
- CPU backend: Uses system compiler with OpenMP

Usage:
    python setup.py build_ext --inplace
    
Requirements:
    - NVIDIA CUDA Toolkit (for CUDA backend)
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


class NativeBuildExt(build_ext):
    """Custom build command for all native backends."""
    
    def run(self):
        # Decide output directory based on editable mode vs isolated build
        if self.inplace:
            self.output_dir = Path(__file__).parent / "gafime"
        else:
            self.output_dir = Path(self.build_lib) / "gafime"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # We manually build all backends and drop the .so/.dll/.dylib 
        # artifacts directly into the targeted python package folder
        self.build_cuda_backend()
        self.build_metal_backend()
        self.build_cpu_backend()
        self.build_cpp_core()
        self.build_rust_backend()
        
        # Don't call super().run() as we handle the extensions manually
        
    def build_cuda_backend(self):
        """Build CUDA backend using nvcc."""
        print("\n" + "=" * 60)
        print("Building CUDA Backend")
        print("=" * 60)
        
        nvcc = shutil.which("nvcc")
        if not nvcc:
            print("!  nvcc not found - skipping CUDA backend")
            return
        
        src_dir = Path(__file__).parent / "src"
        output_dir = self.output_dir
        cuda_source = src_dir / "cuda" / "kernels.cu"
        
        if sys.platform == "win32":
            output_file = output_dir / "gafime_cuda.dll"
            compiler_flags = ["/MD", "/O2"]
        else:
            output_file = output_dir / "libgafime_cuda.so"
            compiler_flags = ["-fPIC", "-O3"]
        
        # Dynamically determine supported architectures based on nvcc version
        gencode_flags = [
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
        ]
        
        try:
            version_out = subprocess.check_output([nvcc, "--version"]).decode("utf-8")
            if "release 13." in version_out or "release 14." in version_out:
                gencode_flags.extend([
                    "-gencode=arch=compute_100,code=sm_100",
                    "-gencode=arch=compute_120,code=sm_120",
                    "-gencode=arch=compute_120,code=compute_120",
                ])
            else:
                gencode_flags.append("-gencode=arch=compute_90,code=compute_90")
        except Exception as e:
            print(f"! Could not query nvcc version: {e}")
            gencode_flags.append("-gencode=arch=compute_90,code=compute_90")
        
        cmd = [
            nvcc, *gencode_flags, "-O3", "--shared",
            "-cudart", "static",  # Statically link libcudart so users don't need CUDA toolkit to run wheels!
            "-Xcompiler", ",".join(compiler_flags),
            "-I", str(src_dir / "common"),
            "-o", str(output_file), str(cuda_source),
        ]
        
        print(f"[BUILD] Compiling CUDA: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] CUDA build failed:\n{result.stderr}")
            sys.exit(1)
        print(f"[OK] CUDA backend built: {output_file.name}")

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
            shutil.which("clang++"), "-std=c++17", "-O3", "-shared", "-fPIC", "-fobjc-arc",
            "-framework", "Metal", "-framework", "Foundation",
            f"-I{metal_dir}", f"-I{src_dir / 'common'}",
            "-o", str(dylib_file), str(metal_dir / "metal_backend.mm"),
        ]
        
        subprocess.run(cmd_air, check=True)
        subprocess.run(cmd_lib, check=True)
        air_file.unlink(missing_ok=True)
        subprocess.run(cmd_dylib, check=True)
        print(f"[OK] Metal backend built: {dylib_file.name}")

    def build_cpu_backend(self):
        """Build CPU OpenMP backend."""
        print("\n" + "=" * 60)
        print("Building CPU Backend")
        print("=" * 60)
        
        src_dir = Path(__file__).parent / "src"
        output_dir = self.output_dir
        cpu_source = src_dir / "cpu" / "cpu_backend.cpp"
        
        if sys.platform == "win32":
            compiler = shutil.which("cl")
            if compiler:
                output_file = output_dir / "gafime_cpu.dll"
                cmd = [compiler, "/O2", "/EHsc", "/openmp", "/LD", f"/I{src_dir / 'common'}", f"/Fe:{output_file}", str(cpu_source)]
            else:
                print("!  No MSVC compiler found")
                return
        else:
            compiler = shutil.which("g++") or shutil.which("clang++")
            output_file = output_dir / "libgafime_cpu.so"
            flags = ["-O3", "-shared", "-fPIC"]
            if sys.platform != "darwin":
                flags.append("-fopenmp")
            cmd = [compiler, *flags, f"-I{src_dir / 'common'}", "-o", str(output_file), str(cpu_source)]
            
        subprocess.run(cmd, check=True)
        print(f"[OK] CPU backend built: {output_file.name}")

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
            print("!  cmake or source not found")
            return
            
        build_dir.mkdir(exist_ok=True)
        
        # Injecting pybind11 via pip to ensure it resolves inside CI
        subprocess.run([sys.executable, "-m", "pip", "install", "pybind11"], check=False)
        
        pybind_cmd = [sys.executable, "-m", "pybind11", "--cmakedir"]
        pybind_dir = subprocess.check_output(pybind_cmd).decode('utf-8').strip()
        
        cmake_cmd = [
            cmake, "..",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGAFIME_CORE_ENABLE_OPENMP=ON",
            "-DGAFIME_CORE_USE_FETCHCONTENT=OFF",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            f"-DPython3_EXECUTABLE={sys.executable}",
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
            print("!  cargo not found")
            return
            
        env = os.environ.copy()
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


setup(
    name="gafime",
    version="0.2.0",
    description="GPU Accelerated Feature Interaction Mining Engine",
    author="Hamza",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "polars>=0.20",
    ],
    # Including an Extension tells cibuildwheel this is a native C/C++/Rust package,
    # forcing it to output a platform-specific .whl (e.g. macos_14_arm64) instead of py3-none-any.
    # We include a dummy C file so older/newer setuptools don't optimize out the extension!
    ext_modules=[Extension("gafime._native", sources=["gafime/_dummy.c"])],
    package_data={
        "gafime": ["*.so", "*.dll", "*.dylib", "*.metallib", "*.pyd"],
    },
    include_package_data=True,
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
