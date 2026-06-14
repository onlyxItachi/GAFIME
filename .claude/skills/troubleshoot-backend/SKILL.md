---
name: troubleshoot-backend
description: Diagnose why a GAFIME native compute backend is not loading. Use when the user reports that GAFIME is slow, GPU is not being used, CUDA/Metal backend failed to load, DLL or shared library loading errors, or says things like "why is GAFIME using CPU", "CUDA not working", "backend not found", "gafime is slow", or "GPU not detected".
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
   - Whether vendor payload distributions (`gafime-cuda`, `gafime-rocm`) are installed
   - Whether it loads successfully via ctypes
   - The exact error message if it fails
   - CUDA driver vs toolkit version compatibility
   - Library search path issues
   - Whether CUDA/Metal expose the discrete soft batch API
   - Whether CUDA exposes `gafime_discrete_selection_adaptive_cuda`
   - Whether Rust exposes `BatchScheduler.create_batches`
   - Whether `from gafime import subfunctions` exposes the Rust helper alias

3. Based on the output, provide targeted fixes:

   **CUDA backend not loading:**
   - Missing CUDA payload: install `pip install "gafime[cuda]"`
   - Missing DLL/SO after payload install: reinstall with `pip install --force-reinstall "gafime[cuda]"`
   - CUDA driver too old: User needs to update their NVIDIA driver
   - Architecture mismatch: The wheel was built for a different GPU architecture
   - macOS request: CUDA is not a macOS backend; use `backend="metal"` or `backend="core"`
   - ARM Linux/Windows request: current ARM wheels use C++ Core, not CUDA
   - Discrete hard mode on CUDA: use `discrete_mode="soft"` or the C++ Core backend
   - Unsupported GPU report metric: use `backend="core"` when a vendor payload does not report the requested metric natively

   **Metal backend not loading:**
   - Not on Apple Silicon: Metal only works on macOS arm64
   - Missing .metallib: Reinstall GAFIME
   - Non-macOS request: Metal is invalid outside macOS; use `backend="cuda"` or `backend="core"`

   **Rust subfunctions not loading:**
   - Rebuild native extensions with `python setup.py build_ext --inplace`
   - User-facing import should be `from gafime import subfunctions`
   - Direct `import gafime_cpu` is an implementation detail

   **C++ core not loading:**
   - Missing pybind11 module: Rebuild with `pip install -e .`
   - OpenMP runtime missing: Install `libomp` (macOS) or `libgomp` (Linux)

   **ROCm backend not loading:**
   - Missing ROCm payload: install `pip install "gafime[rocm]"`
   - macOS request: ROCm/HIP is not a macOS backend; use `backend="metal"` or `backend="core"`
   - ARM Linux/Windows request: current ARM wheels use C++ Core, not ROCm/HIP
   - Missing `/dev/kfd` or render device permissions on Linux: fix ROCm device access or use `backend="core"`

   **All backends failing:**
   - GAFIME not installed properly: `pip install gafime`
   - Virtual environment issues: Check which Python is being used

4. After diagnosis, suggest the specific fix command.

5. For v0.4.x discrete feature engineering, remember:
   - `backend="auto"` is platform-aware: macOS uses Metal then Core; x86
     Linux/Windows uses installed vendor payloads then Core; ARM Linux/Windows uses Core.
   - Vendor GPU payloads are explicit: `gafime[cuda]` for NVIDIA and `gafime[rocm]` for AMD ROCm/HIP.
   - `backend="gpu"` is deprecated because it is ambiguous across platforms.
   - GPU backends support soft/vectorized discrete functions only.
   - GPU hard mode raises `GPU feature engineering with discrete hard mode is not supported!`
   - C++ Core can evaluate hard mode.
   - CUDA selector math requires `gafime_discrete_selection_adaptive_cuda`.
   - CUDA continuous scoring uses `gafime_cuda_matrix_compute_batch` with global feature-index batches.
   - CUDA time-series scoring still uses `gafime_bucket_compute_batch` for compact transform batches.
   - If NCU shows poor occupancy in `gafime_discrete_selection_adaptive_kernel<96>`,
     first check register count and loop unrolling. The native-spine fix keeps the
     final MI target-bin loops at `#pragma unroll 1`, reducing local `<96>`
     profiles from 215 to 56 registers/thread.
   - If register count is healthy but skewed target-bin workloads are still
     slow, look for shared-memory atomic contention on the soft-binary
     histograms rather than assuming an L2 cache-locality failure.

## Example

**User says:** "GAFIME is using CPU, but I have an RTX 4060"

**Actions:** Run `diagnose_backends.py`, find CUDA DLL loading error.

**Result:** "The CUDA backend found `gafime_cuda.dll` but it failed to load because `cudart64_12.dll` is missing from your PATH. Since GAFIME wheels ship with static CUDA runtime, this suggests a corrupted CUDA payload install. Fix: `pip install --force-reinstall \"gafime[cuda]\"`"
