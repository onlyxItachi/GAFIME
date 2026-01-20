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
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext


class NativeBuildExt(build_ext):
    """Custom build command for CUDA and CPU backends."""
    
    def run(self):
        self.build_cuda_backend()
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
        
        # nvcc command for RTX 4060 (Ada Lovelace, SM89)
        cmd = [
            nvcc,
            "-arch=sm_89",           # RTX 4060 architecture
            "-O3",                    # Optimization level
            "--shared",               # Build shared library
            "-Xcompiler", ",".join(compiler_flags),
            "-I", str(src_dir / "common"),
            "-o", str(output_file),
            str(cuda_source),
        ]
        
        print(f"📦 Building: {cuda_source.name}")
        print(f"   Target: sm_89 (RTX 4060 Ada Lovelace)")
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
