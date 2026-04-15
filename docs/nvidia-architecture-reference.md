# NVIDIA GPU Architecture Reference for GAFIME

> **Research date:** July 2025
> **Scope:** Turing (sm_75) → Blackwell Ultra — gencode flags, tensor cores, CUDA toolkit compatibility
> **Context:** GAFIME compiles fused CUDA kernels (map-reduce, Pearson correlation, dot products)

---

## 1. Architecture Table (Turing → Blackwell Ultra)

| Architecture | Compute Capability | Key GPUs | Year | Tensor Core Gen | Status |
|---|---|---|---|---|---|
| **Turing** | sm_75 (7.5) | RTX 2060–2080 Ti, GTX 1650–1660, T4, Quadro RTX | 2018 | 2nd Gen | Current (minimum supported) |
| **Ampere** | sm_80 (8.0) | A100, A30, A10, RTX A6000 | 2020 | 3rd Gen | Current |
| **Ampere** | sm_86 (8.6) | RTX 3060–3090, A40, A16, RTX A2000–A5000 | 2020 | 3rd Gen | Current |
| **Ampere** | sm_87 (8.7) | Jetson AGX Orin, Orin NX, Orin Nano | 2022 | 3rd Gen | Current (embedded only) |
| **Ada Lovelace** | sm_89 (8.9) | RTX 4060–4090, L4, L40, RTX 6000 Ada | 2022 | 4th Gen | Current |
| **Hopper** | sm_90 (9.0) | H100 (PCIe/SXM) | 2022 | 4th Gen | Current |
| **Hopper** | sm_90a (9.0a) | H200, GH200 Grace Hopper | 2023 | 4th Gen | Current (arch-specific) |
| **Blackwell DC** | sm_100 (10.0) | B100, B200 | 2024 | 5th Gen | Current |
| **Blackwell DC** | sm_100a (10.0a) | B200 (arch-specific features) | 2024 | 5th Gen | Current (arch-specific) |
| **Blackwell Consumer** | sm_120 (12.0) | RTX 5060–5090 | 2025 | 5th Gen | Current |
| **Blackwell Workstation** | sm_121 (12.1) | DGX Spark, RTX PRO Server | 2025 | 5th Gen | Current |
| **Blackwell Ultra DC** | sm_103a (10.3a) | B300, GB300 NVL72 | 2025 | 5th Gen | Current (arch-specific) |

**Key notes:**
- `sm_100` (datacenter) and `sm_120` (consumer) are **NOT binary-compatible** despite both being "Blackwell". Separate gencode targets required.
- The `a` suffix (e.g., `sm_90a`, `sm_100a`, `sm_103a`) produces architecture-specific code that runs **only** on that exact SM variant. Not forward/backward compatible.
- The `f` suffix (e.g., `sm_100f`) produces family-specific code that runs across a major architecture's range of minor versions (new in CUDA 12.9+).
- Pre-Turing architectures (Volta sm_70/sm_72, Pascal sm_60/61/62, etc.) should be considered **DEPRECATED** for GAFIME.

---

## 2. Tensor Core Generations & Data Type Support

| Architecture | TC Gen | FP16 | BF16 | TF32 | FP8 | FP4 | INT8 | INT4 | FP64 |
|---|---|---|---|---|---|---|---|---|---|
| Turing (sm_75) | 2nd | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Ampere (sm_80/86/87) | 3rd | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅* |
| Ada Lovelace (sm_89) | 4th | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| Hopper (sm_90/90a) | 4th | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Blackwell (sm_100/120) | 5th | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Blackwell Ultra (sm_103a) | 5th | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

\* FP64 tensor core support on Ampere is A100-only.

### Key capabilities per generation:

**2nd Gen (Turing):**
- FP16 mixed-precision (FP16 input → FP32 accumulator)
- 4×4×4 matrix fragments
- WMMA API introduced

**3rd Gen (Ampere):**
- TF32: Drop-in FP32 acceleration for training (19-bit, same range as FP32)
- BF16: Brain floating-point for AI training
- Sparsity support (2:4 structured sparsity, 2× throughput)
- Larger fragments: 16×16×16 for FP16, 8×8×4 for TF32

**4th Gen (Ada Lovelace / Hopper):**
- FP8 (E4M3 + E5M2 formats): Ultra-fast transformer inference/training
- Hopper adds warp-group-level MMA (wgmma) and TMA (tensor memory accelerator)
- Ada is consumer-oriented; Hopper is datacenter with FP64 TC

**5th Gen (Blackwell / Blackwell Ultra):**
- NVFP4: New 4-bit format (~1.8× smaller than FP8, massive inference throughput)
- FP6: Intermediate precision between FP8 and FP4
- Enhanced attention hardware (10.7 TeraExponentials/s for Softmax)
- Tensor Memory (TMEM): 256 KB/SM dedicated tensor core scratchpad
- B300 Ultra: 15 PFLOPS FP4 dense per GPU

---

## 3. Recommended gencode Flags

### For CUDA 13.x (current CI: CUDA 13.1)

```python
gencode_flags = [
    # Turing — minimum supported (RTX 20xx, T4, GTX 16xx)
    "-gencode=arch=compute_75,code=sm_75",
    # Ampere — datacenter (A100, A30)
    "-gencode=arch=compute_80,code=sm_80",
    # Ampere — consumer (RTX 3060–3090, A40)
    "-gencode=arch=compute_86,code=sm_86",
    # Ampere — Jetson Orin (embedded, optional — remove if not targeting edge)
    # "-gencode=arch=compute_87,code=sm_87",
    # Ada Lovelace (RTX 4060–4090, L4, L40)
    "-gencode=arch=compute_89,code=sm_89",
    # Hopper — datacenter (H100)
    "-gencode=arch=compute_90,code=sm_90",
    # Blackwell — datacenter (B100, B200)
    "-gencode=arch=compute_100,code=sm_100",
    # Blackwell — consumer (RTX 5060–5090)
    "-gencode=arch=compute_120,code=sm_120",
    # PTX fallback for forward compatibility (future architectures)
    "-gencode=arch=compute_120,code=compute_120",
]
```

### What changed vs. current setup.py:

| Change | Rationale |
|---|---|
| **Added** `sm_75` (Turing) | Currently missing — drops support for RTX 20xx and T4 users |
| **Added** `sm_100` (Blackwell DC) | Supports B100/B200 datacenter GPUs |
| **Added** `sm_120` (Blackwell consumer) | Supports RTX 5060–5090 |
| **Changed PTX fallback** from `compute_90` → `compute_120` | PTX should be highest supported for maximum forward compat |
| **Omitted** `sm_87` (Jetson Orin) | Embedded-only; add if GAFIME targets edge devices |
| **Omitted** `sm_90a`, `sm_100a`, `sm_103a` | Arch-specific; only needed for hardware-exclusive features |
| **Omitted** `sm_121` (DGX Spark) | Niche workstation; covered by sm_120 PTX fallback |

### Notes on arch-specific (`a`) and family (`f`) suffixes:
- `sm_90a` (H200/GH200): Only add if using Hopper-specific features like wgmma instructions
- `sm_100a` (B200 arch-specific): Only for datacenter code using B200-only hardware features
- `sm_103a` (B300 Ultra): Only for Blackwell Ultra-specific features; requires CUDA 12.9+
- `sm_100f` / `sm_120f`: Family-level targets (CUDA 12.9+ only); covers all variants within a Blackwell sub-family
- **For GAFIME's use case (fused reduction kernels), standard SM targets are sufficient.**

---

## 4. CUDA Toolkit Compatibility

| Architecture | Minimum CUDA | GAFIME CI (13.1) | Notes |
|---|---|---|---|
| Turing (sm_75) | CUDA 10.0 | ✅ | Long-supported |
| Ampere (sm_80/86) | CUDA 11.0 | ✅ | |
| Ampere (sm_87) | CUDA 11.4 | ✅ | Jetson Orin |
| Ada Lovelace (sm_89) | CUDA 11.8 | ✅ | |
| Hopper (sm_90) | CUDA 12.0 | ✅ | |
| Hopper (sm_90a) | CUDA 12.0 | ✅ | |
| Blackwell DC (sm_100) | CUDA 12.8 | ✅ | |
| Blackwell Consumer (sm_120) | CUDA 12.8 | ✅ | |
| Blackwell Ultra (sm_103a) | CUDA 12.9 | ✅ | Likely; sm_103a may need 12.9+ |

### Verdict: CUDA 13.1 is sufficient ✅

CUDA 13.1 (currently used in CI) supports **all architectures** from Turing through Blackwell, including:
- All standard SM targets (sm_75 through sm_120)
- Architecture-specific targets (sm_90a, sm_100a) if needed
- Family targets (sm_100f, sm_120f)

**No CUDA toolkit upgrade is required.** CUDA 13.1 is ahead of the minimum for every target.

### Minimum driver versions:
- Blackwell consumer (RTX 50xx): Driver ≥ 566.03
- Blackwell datacenter (B100/B200): Driver ≥ 560.x (datacenter branch)

---

## 5. Tensor Core Integration Notes for GAFIME

### Can tensor cores accelerate Pearson correlation / dot products?

**Yes — and it's practical for GAFIME's batch computation pattern.**

Pearson correlation decomposes into dot products and reductions:

```
r = (n·Σxy - Σx·Σy) / sqrt((n·Σx² - (Σx)²) · (n·Σy² - (Σy)²))
```

All numerically intensive terms (Σxy, Σx², Σy²) are dot products. When computing
an all-pairs correlation matrix across N features, this becomes a GEMM:

```
C = X · Xᵀ     (where X is an N×M matrix of features)
```

This is the **ideal workload for tensor cores** — it's literally a matrix multiply.

### What would a tensor core path look like?

**Option A: WMMA API (sm_75+, widest compatibility)**
```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Declare fragments for 16×16×16 FP16 GEMM
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

// Load from shared memory, compute, store
fill_fragment(c_frag, 0.0f);
load_matrix_sync(a_frag, shmem_a, 16);
load_matrix_sync(b_frag, shmem_b, 16);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(shmem_c, c_frag, 16, mem_row_major);
```

**Compatibility:** Works on Turing and all later architectures. FP16 input → FP32 accumulator.

**Option B: TF32 via WMMA (sm_80+, no precision loss for FP32 data)**
```cpp
// TF32 path — input is float, hardware truncates mantissa
fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
// ... same pattern, but FP32 data goes in, TF32 precision applied automatically
```

**Advantage:** No need to convert data to FP16. TF32 has same range as FP32
but ~10-bit mantissa. For correlation coefficients (range [-1, 1]), this is
more than sufficient precision.

**Option C: cuBLAS (pragmatic approach, any architecture)**
```python
# In Python/C++, just call cuBLAS GEMM — it auto-uses tensor cores
cublasSgemm(handle, ..., X, X_T, C)  # FP32 GEMM, TC-accelerated via TF32 on Ampere+
```

cuBLAS automatically leverages tensor cores when available. This is the
**lowest-effort, highest-payoff** approach.

### Is this practical for feature interaction scoring?

**Analysis for GAFIME's use case:**

| Criterion | Assessment |
|---|---|
| Workload shape | ✅ Batch all-pairs = GEMM — perfect for TC |
| Data volume | ✅ Typically N=1000+ features × M=10K+ samples — large enough for TC gains |
| Precision needs | ✅ Correlation coefficients are inherently noisy; TF32/FP16 precision is fine |
| Implementation effort | 🟡 WMMA: moderate. cuBLAS GEMM: trivial. |
| Expected speedup | ✅ 2–8× over CUDA core-only for the GEMM portion |

**Recommendation for GAFIME:**

1. **Immediate (low effort):** For the correlation matrix kernel, compute `X·Xᵀ` via
   cuBLAS GEMM. This auto-uses tensor cores on Ampere+ with zero kernel changes.

2. **Medium term:** Add a WMMA-based fused kernel that computes the correlation
   matrix in one pass (GEMM + mean subtraction + normalization fused). This avoids
   the memory round-trips of separate GEMM + reduction kernels.

3. **Futureproofing stub:** Add `#ifdef` paths for TF32 (sm_80+) and FP8 (sm_89+)
   that can be enabled per-architecture. The WMMA API is forward-compatible, so
   code written for sm_75 WMMA will work on Blackwell without changes.

```cpp
// Example: architecture-adaptive tensor core dispatch
#if __CUDA_ARCH__ >= 800
    // TF32 path — Ampere and later
    // Use nvcuda::wmma with tf32 precision
#elif __CUDA_ARCH__ >= 750
    // FP16 path — Turing
    // Use nvcuda::wmma with half precision
#else
    // CUDA core fallback
#endif
```

**Bottom line:** For GAFIME's fused map-reduce kernels doing Pearson correlation
across feature pairs, tensor cores are **practical and recommended**, not just a
futureproofing stub. The all-pairs correlation pattern maps directly to GEMM,
which is the workload tensor cores were designed for.

---

## References

- [NVIDIA CUDA Compute Capability List](https://developer.nvidia.com/cuda-gpus)
- [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)
- [Blackwell Tuning Guide (CUDA 13.2)](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [Matching CUDA arch and gencode](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
- [NVIDIA Blackwell Ultra Architecture Blog](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- [CUTLASS SM100 Blackwell Architecture](https://deepwiki.com/NVIDIA/cutlass/7.2-sm100-blackwell-architecture)
- [PyTorch Build Guide for SM120](https://github.com/bajegani/pytorch-build-blackwell-sm120)
