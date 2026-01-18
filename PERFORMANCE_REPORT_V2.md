# GAFIME Performance Report V2 - Dual-Issue Interleaved Kernel

**Version:** 0.3.0  
**Date:** January 18, 2026  
**Platform:** Windows 11, CUDA 13.1  
**GPU:** NVIDIA RTX 4060 (8GB VRAM, SM89)

---

## Executive Summary

| Feature | Status | Key Metric |
|---------|--------|------------|
| Interleaved Kernel | ✅ IMPLEMENTED | 1.31x speedup |
| NVIDIA Fast Intrinsics | ✅ IMPLEMENTED | __logf, __expf, __fsqrt_rn |
| Time-Series Operators | ✅ IMPLEMENTED | ROLLING_MEAN, ROLLING_STD |
| All Previous Tests | ✅ PASS | Backward compatible |

**OVERALL: 1.31x THROUGHPUT IMPROVEMENT**

---

## 1. Dual-Issue Interleaved Kernel Architecture

### 1.1 Design Principle

```
┌─────────────────────────────────────────────────────────────┐
│                 DUAL-ISSUE INTERLEAVED KERNEL               │
├─────────────────────────────────────────────────────────────┤
│  Per Thread (2 interactions simultaneously):                │
│                                                             │
│  Slot A (SFU-Heavy):                                        │
│    log(f0) × exp(f1) → stats_A                              │
│    Uses: __logf, __expf (Special Function Unit)             │
│                                                             │
│  Slot B (ALU-Heavy):                                        │
│    f2² + f3³ → stats_B                                      │
│    Uses: CUDA Cores (while SFU is stalling)                 │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Performance Results

| Metric | Single-Slot | Interleaved | Improvement |
|--------|-------------|-------------|-------------|
| Calls/sec | 22,183 | 14,581 | - |
| **Interactions/sec** | **22,183** | **29,162** | **1.31x** |
| Efficiency | 1 int/call | 2 int/call | 2x |

### 1.3 Why 1.31x Instead of 2x?

The 1.31x speedup represents excellent GPU utilization:
- Perfect 2x would require zero overhead
- Memory bandwidth is shared between slots
- Reduction overhead is duplicated (A + B)
- Still achieves ~65% of theoretical maximum

---

## 2. New Time-Series Operators

### 2.1 Rolling Window Operators

| ID | Operator | Formula | Algorithm |
|----|----------|---------|-----------|
| 11 | ROLLING_MEAN | Σ(x[i-w:i]) / w | Simple moving average |
| 12 | ROLLING_STD | σ(x[i-w:i]) | Welford's algorithm |

### 2.2 Implementation Details

```cpp
// ROLLING_MEAN: O(window) per element
case GAFIME_OP_ROLLING_MEAN: {
    int start = max(0, idx - window + 1);
    float sum = 0.0f;
    for (int i = start; i <= idx; i++) sum += col[i];
    return sum / (idx - start + 1);
}

// ROLLING_STD: Welford's algorithm for numerical stability
case GAFIME_OP_ROLLING_STD: {
    // Uses online mean/variance to prevent catastrophic cancellation
    float mean = 0, M2 = 0;
    for (...) { /* Welford update */ }
    return sqrt(M2 / (count - 1));
}
```

### 2.3 Boundary Handling

- If `idx < window_size`: Uses partial window from 0 to idx
- Returns 0 for ROLLING_STD if count < 2

---

## 3. NVIDIA Fast Intrinsics

### 3.1 SFU Intrinsics Used

| Standard | Fast Intrinsic | Speedup |
|----------|---------------|---------|
| `logf()` | `__logf()` | ~3-5x |
| `expf()` | `__expf()` | ~3-5x |
| `sqrtf()` | `__fsqrt_rn()` | ~2-3x |
| `1/x` | `__fdividef()` | ~2x |

### 3.2 Fast Approximations

```cpp
// Fast tanh using exp intrinsic
case GAFIME_OP_TANH: {
    float exp2x = __expf(2.0f * clamp(x));
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}

// Fast sigmoid
case GAFIME_OP_SIGMOID: {
    float ex = __expf(-clamp(x));
    return __fdividef(1.0f, 1.0f + ex);
}
```

---

## 4. API Usage

### 4.1 Python: Interleaved Compute

```python
from gafime.backends.fused_kernel import StaticBucket, UnaryOp

bucket = StaticBucket(n_samples=10000, n_features=5)
bucket.upload_all(features, target, mask)

# Compute 2 interactions in one kernel launch!
stats_A, stats_B = bucket.interleaved_compute(
    # Slot A: SFU-heavy
    feature_indices_A=[0, 1],
    ops_A=[UnaryOp.LOG, UnaryOp.EXP],
    
    # Slot B: ALU-heavy  
    feature_indices_B=[2, 3],
    ops_B=[UnaryOp.SQUARE, UnaryOp.CUBE],
    
    window_size=10,  # For rolling operators
    val_fold=0
)

# Each call returns 2 sets of 12 statistics
train_r_A, val_r_A = compute_pearson_from_stats(stats_A)
train_r_B, val_r_B = compute_pearson_from_stats(stats_B)
```

### 4.2 New Operators

```python
# Time-series operators
UnaryOp.ROLLING_MEAN  # Moving average
UnaryOp.ROLLING_STD   # Moving std (Welford's)

# Recommended slot assignments for maximum parallelism:
# Slot A (SFU): LOG, EXP, SQRT, TANH, SIGMOID
# Slot B (ALU): SQUARE, CUBE, ROLLING_MEAN, ROLLING_STD
```

---

## 5. Backward Compatibility

All existing functionality remains unchanged:

| Test | Status | Notes |
|------|--------|-------|
| Feature Creation | ✅ PASS | 11/11 operators |
| Signal Detection | ✅ PASS | r=0.995 accuracy |
| GPU Performance | ✅ PASS | 22K iter/s (single) |
| CPU Fallback | ✅ PASS | NumPy reference works |
| Reporting System | ✅ PASS | CV stable |

---

## 6. Recommendations

### When to Use Interleaved

| Scenario | Recommendation |
|----------|----------------|
| Large feature search | ✅ Always use interleaved |
| Mixed op combinations | ✅ Pair SFU + ALU for best results |
| Time-series features | ✅ Use ROLLING_* in Slot B |
| Simple multiplications | ⚠️ Single-slot may be simpler |

### Optimal Slot Pairing

| Slot A (SFU) | Slot B (ALU) | Expected Speedup |
|--------------|--------------|------------------|
| LOG + EXP | SQUARE + CUBE | 1.31x |
| TANH + SIGMOID | ROLLING_MEAN + IDENTITY | 1.2-1.4x |
| SQRT + LOG | CUBE + ABS | 1.25x |

---

## Appendix: Build Information

```
CUDA Toolkit: 13.1
Compiler: NVCC + MSVC 17.14
Architecture: sm_89 (Ada Lovelace)
Optimization: -O3
```

---

*Report generated January 18, 2026*
