---
name: troubleshoot-backend
description: Diagnose why a GAFIME compute backend is not loading or falling back to NumPy. Use when the user reports that GAFIME is slow, GPU is not being used, CUDA/Metal backend failed to load, gets warnings about falling back to NumPy, DLL or shared library loading errors, or says things like "why is GAFIME using CPU", "CUDA not working", "backend not found", "gafime is slow", or "GPU not detected".
---

# Backend Troubleshooting

Diagnose and fix GAFIME backend loading issues.

## Instructions

1. Run the diagnostic script:

   ```bash
   python .claude/skills/troubleshoot-backend/scripts/diagnose_backends.py
   ```

2. The script tests each backend individually and reports:
   - Whether each backend's shared library (.dll/.so/.dylib) exists on disk
   - Whether it loads successfully via ctypes
   - The exact error message if it fails
   - CUDA driver vs toolkit version compatibility
   - Library search path issues

3. Based on the output, provide targeted fixes:

   **CUDA backend not loading:**
   - Missing DLL/SO: Reinstall GAFIME with `pip install --force-reinstall gafime`
   - CUDA driver too old: User needs to update their NVIDIA driver
   - Architecture mismatch: The wheel was built for a different GPU architecture

   **Metal backend not loading:**
   - Not on Apple Silicon: Metal only works on macOS arm64
   - Missing .metallib: Reinstall GAFIME

   **C++ core not loading:**
   - Missing pybind11 module: Rebuild with `pip install -e .`
   - OpenMP runtime missing: Install `libomp` (macOS) or `libgomp` (Linux)

   **All backends failing:**
   - GAFIME not installed properly: `pip install gafime`
   - Virtual environment issues: Check which Python is being used

4. After diagnosis, suggest the specific fix command.

## Example

**User says:** "GAFIME keeps saying NumPy fallback, but I have an RTX 4060"

**Actions:** Run `diagnose_backends.py`, find CUDA DLL loading error.

**Result:** "The CUDA backend found `gafime_cuda.dll` but it failed to load because `cudart64_12.dll` is missing from your PATH. Since GAFIME wheels ship with static CUDA runtime, this suggests a corrupted install. Fix: `pip install --force-reinstall gafime`"
