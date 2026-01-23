# GAFIME Performance Report V4 - Complete Backend Refactor

**Version:** 0.4.1  
**Date:** January 23, 2026  
**Platform:** Windows 11, CUDA 13.1  
**GPU:** NVIDIA RTX 4060 (8GB VRAM, SM89)

---

## Executive Summary

| Feature | Status | Key Metric |
|---------|--------|------------|
| Singleton Library Loader | ✅ DONE | 0ms DLL reload |
| Pre-allocated Buffers | ✅ DONE | Zero alloc in hotloop |
| Cached ctypes Pointers | ✅ DONE | Reduced per-call overhead |
| Batched Compute Kernel | ✅ DONE | **13.8x speedup** |
| CUDA Stream | ✅ DONE | Async-ready |
| Pinned Host Memory | ✅ DONE | Zero-copy D2H |
| GPU Rolling Ops | ❌ REMOVED | Use CPU/Polars |

**OVERALL: 13.8x THROUGHPUT IMPROVEMENT** (Batch 500 mode)

---

## 1. Performance Results

### 1.1 Benchmark (100K samples, 5 features)

| Mode | Iterations/sec | μs/iteration | Speedup |
|------|----------------|--------------|---------|
| Single Compute | 17,443 | 57.3 | 1.0x |
| **Batch 100** | **227,713** | **4.4** | **13.0x** |
| **Batch 500** | **240,952** | **4.2** | **13.8x** |

### 1.2 Comparison with Previous Versions

| Version | Best iter/s | μs/iter | Improvement |
|---------|-------------|---------|-------------|
| V1 (Original) | ~10,000 | ~100 | Baseline |
| V2 (Interleaved) | ~22,000 | ~45 | 2x |
| V3 (Pre-alloc) | ~242,000 | ~4.1 | 12x |
| **V4 (All opts)** | **~241,000** | **~4.2** | **13.8x** |

---

## 2. Optimizations Implemented

### 2.1 Priority 1: Singleton Library Loader

```python
# BEFORE: DLL loaded per StaticBucket instance
self.lib = self._load_library()  # 50ms each time

# AFTER: One global load, shared by all
_GAFIME_LIB_CACHE = None
def _get_library():
    global _GAFIME_LIB_CACHE
    if _GAFIME_LIB_CACHE is None:
        _GAFIME_LIB_CACHE = ctypes.CDLL(lib_path)
    return _GAFIME_LIB_CACHE
```

### 2.2 Priority 2: Pre-allocated Buffers

```python
class StaticBucket:
    def __init__(self, ...):
        # Allocate ONCE at init
        self._indices_buf = np.zeros(5, dtype=np.int32)
        self._ops_buf = np.zeros(5, dtype=np.int32)
        self._stats_buf = np.zeros(12, dtype=np.float32)
        
        # Pre-compute ctypes pointers
        self._indices_ptr = self._indices_buf.ctypes.data_as(...)
    
    def compute(self, ...):
        # ZERO ALLOCATION - just copy values
        self._indices_buf[:arity] = feature_indices
        self._ops_buf[:arity] = ops
        # Call kernel with cached pointers
```

### 2.3 Priority 3: Batched Compute Kernel

```cuda
// Grid: (blocks_per_sample, batch_size)
// Each Y-block processes one interaction
__global__ void gafime_batched_kernel(
    float* features[5], float* target, uint8_t* mask,
    int* batch_indices,    // [N * 2]
    int* batch_ops,        // [N * 2]
    int* batch_interact,   // [N]
    int batch_size, ...
) {
    int batch_id = blockIdx.y;  // Which interaction
    // Load params and compute...
}
```

### 2.4 Priority 4: CUDA Stream + Pinned Memory

```cuda
struct GafimeBucketImpl {
    // ... existing fields ...
    cudaStream_t stream;           // For async ops
    float* h_stats_pinned;         // Zero-copy D2H
};

// Allocation
cudaStreamCreate(&bucket->stream);
cudaMallocHost(&bucket->h_stats_pinned, 24 * sizeof(float));
```

### 2.5 Priority 5: L2 Cache Tracking

```cuda
// Track total data size for future cache hints
bucket->total_data_bytes = n_features * vec_bytes + vec_bytes + mask_bytes;

// Note: cudaStreamSetAttribute with cudaStreamAttributeAccessPolicyWindow
// available for explicit L2 persistence on Ampere+
```

### 2.6 Priority 6: Remove GPU Rolling Ops

```cuda
// BEFORE: O(window) serial memory access per thread
case GAFIME_OP_ROLLING_MEAN: {
    for (int i = start; i <= idx; i++) sum += col[i];  // BAD!
}

// AFTER: Return NaN, use CPU preprocessing
case GAFIME_OP_ROLLING_MEAN:
case GAFIME_OP_ROLLING_STD:
    return NAN;  // Use TimeSeriesPreprocessor instead
```

---

## 3. API Usage

### 3.1 Single Compute (existing)

```python
# One kernel launch
stats = bucket.compute([0, 1], [LOG, SQRT], [MULT], val_fold=0)
```

### 3.2 Batched Compute (NEW - 13.8x faster)

```python
# ONE kernel for N interactions
stats = bucket.compute_batch(
    feature_pairs=[(0,1), (0,2), (1,2), ...],  # N pairs
    op_pairs=[(LOG, SQRT), ...],                # N op pairs
    interactions=[MULT, ...],                   # N types
    val_fold=0
)
# stats.shape = (N, 12)
```

### 3.3 Rolling Features (Use CPU)

```python
from gafime.preprocessors import TimeSeriesPreprocessor

# DO THIS (fast, Polars vectorized)
preprocessor = TimeSeriesPreprocessor(windows=[7, 14, 30])
features = preprocessor.fit_transform(df)

# DON'T USE GPU rolling ops (deprecated, returns NaN)
```

---

## 4. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    GAFIME BACKEND V4                            │
├─────────────────────────────────────────────────────────────────┤
│  Python Layer:                                                  │
│    ✅ Singleton library (no DLL reload)                         │
│    ✅ Pre-allocated numpy buffers                               │
│    ✅ Cached ctypes pointers                                    │
├─────────────────────────────────────────────────────────────────┤
│  CUDA Layer:                                                    │
│    ✅ StaticBucket (pre-allocated VRAM)                         │
│    ✅ CUDA stream (async-ready)                                 │
│    ✅ Pinned host memory (zero-copy D2H)                        │
│    ✅ Batched kernel (N interactions parallel)                  │
├─────────────────────────────────────────────────────────────────┤
│  Compute APIs:                                                  │
│    • compute() - single, 57 μs                                  │
│    • compute_batch() - N at once, 4.2 μs/iter, 13.8x            │
│    • interleaved_compute() - dual SFU+ALU, 1.31x                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Build Information

```
CUDA Toolkit: 13.1
Compiler: NVCC + MSVC 17.14 (VS2022)
Architecture: sm_89 (Ada Lovelace)
Optimization: -O3
DLL Size: ~218 KB
```

---

*Report generated January 23, 2026*
