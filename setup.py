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
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext


class NativeBuildExt(build_ext):
    """Custom build command for CUDA, Metal, and CPU backends."""
    
    def run(self):
        self.build_cuda_backend()
        self.build_metal_backend()
        self.build_cpu_backend()
        super().run()
    
    def build_cuda_backend(self):
        """Build CUDA backend using nvcc."""
        print("\n" + "=" * 60)
        print("Building CUDA Backend")
        print("=" * 60)
        
        # Check for nvcc
        nvcc = shutil.which("nvcc")
        if not nvcc:
            print("⚠️  nvcc not found - skipping CUDA backend")
            print("   Install CUDA Toolkit to enable GPU acceleration")
            return
        
        src_dir = Path(__file__).parent / "src"
        output_dir = Path(__file__).parent
        
        cuda_source = src_dir / "cuda" / "kernels.cu"
        if not cuda_source.exists():
            print(f"⚠️  CUDA source not found: {cuda_source}")
            return
        
        # Determine output file extension
        if sys.platform == "win32":
            output_file = output_dir / "gafime_cuda.dll"
            compiler_flags = ["/MD", "/O2"]
        else:
            output_file = output_dir / "libgafime_cuda.so"
            compiler_flags = ["-fPIC", "-O3"]
        
        # Multi-architecture nvcc: fat binary for all supported GPUs
        # Pascal (GTX 10xx), Volta (V100), Turing (RTX 20xx),
        # Ampere (RTX 30xx/A100), Ada (RTX 40xx), Hopper (H100)
        gencode_flags = [
            "-gencode=arch=compute_60,code=sm_60",  # Pascal
            "-gencode=arch=compute_70,code=sm_70",  # Volta
            "-gencode=arch=compute_75,code=sm_75",  # Turing
            "-gencode=arch=compute_80,code=sm_80",  # Ampere
            "-gencode=arch=compute_86,code=sm_86",  # Ampere (GA10x)
            "-gencode=arch=compute_89,code=sm_89",  # Ada Lovelace
            "-gencode=arch=compute_90,code=sm_90",  # Hopper
            "-gencode=arch=compute_90,code=compute_90",  # PTX for future GPUs
        ]
        
        cmd = [
            nvcc,
            *gencode_flags,
            "-O3",                    # Optimization level
            "--shared",               # Build shared library
            "-Xcompiler", ",".join(compiler_flags),
            "-I", str(src_dir / "common"),
            "-o", str(output_file),
            str(cuda_source),
        ]
        
        archs = "sm_60/70/75/80/86/89/90"
        print(f"📦 Building: {cuda_source.name}")
        print(f"   Targets: {archs} (Pascal through Hopper)")
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ CUDA backend built: {output_file.name}")
            else:
                print(f"❌ CUDA build failed:")
                print(result.stderr)
        except Exception as e:
            print(f"❌ CUDA build error: {e}")
    
    def build_metal_backend(self):
        """Build Metal backend for Apple Silicon (macOS arm64 only)."""
        print("\n" + "=" * 60)
        print("Building Metal Backend")
        print("=" * 60)
        
        # Only build on macOS Apple Silicon
        if sys.platform != "darwin" or platform.machine() != "arm64":
            print("⏭️  Skipping Metal backend (requires macOS on Apple Silicon)")
            return
        
        src_dir = Path(__file__).parent / "src"
        output_dir = Path(__file__).parent
        metal_dir = src_dir / "metal"
        
        metal_source = metal_dir / "gafime_kernels.metal"
        mm_source = metal_dir / "metal_backend.mm"
        
        if not metal_source.exists() or not mm_source.exists():
            print(f"⚠️  Metal source not found in {metal_dir}")
            return
        
        # Check for xcrun (Xcode command-line tools)
        xcrun = shutil.which("xcrun")
        if not xcrun:
            print("⚠️  xcrun not found - install Xcode Command Line Tools")
            return
        
        # Step 1: Compile .metal → .air (intermediate)
        air_file = output_dir / "gafime_kernels.air"
        cmd_air = [
            xcrun, "metal",
            "-std=metal3.0",
            "-O3",
            "-o", str(air_file),
            "-c", str(metal_source),
        ]
        
        print(f"📦 Compiling Metal shaders: {metal_source.name}")
        print(f"   Command: {' '.join(cmd_air)}")
        
        try:
            result = subprocess.run(cmd_air, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Metal shader compile failed:")
                print(result.stderr)
                return
        except Exception as e:
            print(f"❌ Metal shader compile error: {e}")
            return
        
        # Step 2: Link .air → .metallib (GPU binary)
        metallib_file = output_dir / "gafime_kernels.metallib"
        cmd_lib = [
            xcrun, "metallib",
            str(air_file),
            "-o", str(metallib_file),
        ]
        
        print(f"📦 Linking Metal library: {metallib_file.name}")
        
        try:
            result = subprocess.run(cmd_lib, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Metal library link failed:")
                print(result.stderr)
                return
            # Clean up intermediate .air file
            air_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"❌ Metal library link error: {e}")
            return
        
        # Step 3: Compile Objective-C++ wrapper → .dylib
        dylib_file = output_dir / "gafime_metal.dylib"
        compiler = shutil.which("clang++")
        if not compiler:
            print("⚠️  clang++ not found")
            return
        
        cmd_dylib = [
            compiler,
            "-std=c++17",
            "-O3",
            "-shared",
            "-fPIC",
            "-fobjc-arc",
            "-framework", "Metal",
            "-framework", "Foundation",
            f"-I{metal_dir}",
            f"-I{src_dir / 'common'}",
            "-o", str(dylib_file),
            str(mm_source),
        ]
        
        print(f"📦 Building Metal wrapper: {mm_source.name}")
        print(f"   Command: {' '.join(cmd_dylib)}")
        
        try:
            result = subprocess.run(cmd_dylib, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Metal backend built: {dylib_file.name} + {metallib_file.name}")
            else:
                print(f"❌ Metal wrapper build failed:")
                print(result.stderr)
        except Exception as e:
            print(f"❌ Metal wrapper build error: {e}")
    
    def build_cpu_backend(self):
        """Build CPU backend with OpenMP."""
        print("\n" + "=" * 60)
        print("Building CPU Backend")
        print("=" * 60)
        
        src_dir = Path(__file__).parent / "src"
        output_dir = Path(__file__).parent
        
        cpu_source = src_dir / "cpu" / "cpu_backend.cpp"
        if not cpu_source.exists():
            print(f"⚠️  CPU source not found: {cpu_source}")
            return
        
        # Determine compiler and flags
        if sys.platform == "win32":
            # Try MSVC
            compiler = shutil.which("cl")
            if compiler:
                output_file = output_dir / "gafime_cpu.dll"
                cmd = [
                    compiler,
                    "/O2", "/EHsc", "/openmp", "/LD",
                    f"/I{src_dir / 'common'}",
                    f"/Fe:{output_file}",
                    str(cpu_source),
                ]
            else:
                # Try MinGW
                compiler = shutil.which("g++")
                if not compiler:
                    print("⚠️  No C++ compiler found (cl or g++)")
                    return
                output_file = output_dir / "gafime_cpu.dll"
                cmd = [
                    compiler,
                    "-O3", "-fopenmp", "-shared", "-fPIC",
                    f"-I{src_dir / 'common'}",
                    "-o", str(output_file),
                    str(cpu_source),
                ]
        else:
            compiler = shutil.which("g++") or shutil.which("clang++")
            if not compiler:
                print("⚠️  No C++ compiler found")
                return
            output_file = output_dir / "libgafime_cpu.so"
            cmd = [
                compiler,
                "-O3", "-fopenmp", "-shared", "-fPIC",
                f"-I{src_dir / 'common'}",
                "-o", str(output_file),
                str(cpu_source),
            ]
        
        print(f"📦 Building: {cpu_source.name}")
        print(f"   Compiler: {compiler}")
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ CPU backend built: {output_file.name}")
            else:
                print(f"❌ CPU build failed:")
                print(result.stderr)
        except Exception as e:
            print(f"❌ CPU build error: {e}")


setup(
    name="gafime",
    version="0.2.0",
    description="Go Ahead! Find It - Mutual Explanations (Feature Interaction Mining)",
    author="Hamza",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "polars>=0.20",  # Required for TimeSeriesPreprocessor
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    cmdclass={
        "build_ext": NativeBuildExt,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
