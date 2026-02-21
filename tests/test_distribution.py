import os
import sys
import platform
import ctypes
from pathlib import Path
import gafime

def iter_libs():
    gafime_dir = Path(gafime.__file__).parent
    for file in gafime_dir.iterdir():
        if file.suffix in [".so", ".dll", ".dylib", ".pyd"]:
            yield file

def test_distribution_payload():
    gafime_dir = Path(gafime.__file__).parent
    print(f"Testing installed package at: {gafime_dir}")
    libs = list(iter_libs())
    print(f"Found native libraries: {[l.name for l in libs]}")
    
    cpu_lib = [l for l in libs if "gafime_cpu" in l.name]
    core_lib = [l for l in libs if "gafime_core" in l.name]
    
    assert cpu_lib, "gafime_cpu native library is missing!"
    assert core_lib, "gafime_core native library is missing!"
    
    # Check STRICT_CUDA
    if os.environ.get("STRICT_CUDA") == "1":
        if sys.platform in ["win32", "linux"]:
            cuda_lib = [l for l in libs if "gafime_cuda" in l.name]
            assert cuda_lib, "STRICT_CUDA=1 but gafime_cuda native library is missing!"
            
            # Try to load it to ensure dependencies (like cudart) are resolved
            try:
                ctypes.CDLL(str(cuda_lib[0]))
                print("Successfully loaded CUDA library.")
            except Exception as e:
                raise AssertionError(f"Failed to load CUDA library: {e}")
                
    # macOS Metal payload check
    if sys.platform == "darwin" and platform.machine() == "arm64":
        metal_lib = [l for l in libs if "gafime_metal" in l.name]
        metallib = list(gafime_dir.glob("*.metallib"))
        assert metal_lib, "gafime_metal dynamic library missing on macOS arm64"
        assert metallib, "gafime_kernels.metallib missing on macOS arm64"

if __name__ == "__main__":
    test_distribution_payload()
    print("Payload check passed!")
