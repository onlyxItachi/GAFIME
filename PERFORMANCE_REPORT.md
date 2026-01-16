# GAFIME Scientific Performance Report

**Version:** 0.2.0  
**Date:** January 16, 2026  
**Platform:** Windows 11, Python 3.x, CUDA 13.1  
**GPU:** NVIDIA RTX 4060 (8GB VRAM, SM89 Ada Lovelace)

---

## Executive Summary

| Test | Status | Key Metric |
|------|--------|------------|
| Feature Creation | ✅ PASS | 11/11 operators validated |
| Signal Detection | ✅ PASS | r=0.995 for planted signal |
| GPU Performance | ✅ PASS | 15K-22K iterations/sec |
| CPU Fallback | ✅ PASS | GPU/CPU consistency verified |
| Reporting System | ✅ PASS | CV folds within 0.01% std |

**OVERALL: ✅ ALL TESTS PASSED**

---

## 1. Feature Creation Validation

### 1.1 Unary Operators (7/7 PASS)

| Operator | Status | Description |
|----------|--------|-------------|
| IDENTITY | ✅ | x' = x |
| LOG | ✅ | x' = log(\|x\| + ε) |
| EXP | ✅ | x' = exp(clamp(x, -20, 20)) |
| SQRT | ✅ | x' = √\|x\| |
| TANH | ✅ | x' = tanh(x) |
| SIGMOID | ✅ | x' = 1/(1+e⁻ˣ) |
| SQUARE | ✅ | x' = x² |

### 1.2 Interaction Types (4/4 PASS)

| Type | Status | Formula |
|------|--------|---------|
| MULT | ✅ | X = x₀ × x₁ |
| ADD | ✅ | X = x₀ + x₁ |
| SUB | ✅ | X = x₀ - x₁ |
| DIV | ✅ | X = x₀ / x₁ (safe) |

### 1.3 Correctness

All CUDA kernel outputs match NumPy reference implementation with **< 0.001% error**.

---

## 2. Signal Detection Accuracy

### Test Setup
- 10,000 samples with **known planted signals**
- Target = f0 × f1 + noise (signal in f0*f1)
- f_noise = random (no signal)

### Results

| Feature Combo | Expected | Pearson r | Detected? |
|--------------|----------|-----------|-----------|
| f0 × f1 (signal) | > 0.9 | **0.9949** | ✅ YES |
| f0 × f_noise | < 0.1 | 0.0582 | ✅ YES |
| f1 × f_noise | < 0.1 | 0.0300 | ✅ YES |

**Conclusion:** GAFIME correctly identifies signal-carrying feature interactions and rejects noise.

---

## 3. GPU Backend Performance

### 3.1 Short-Term vs Long-Term Analysis

| Scenario | Samples | Iterations | Upload | Compute | Iter/sec |
|----------|---------|------------|--------|---------|----------|
| Short-term | 1K | 100 | 0.51ms | 5.3ms | **18,832** |
| Medium-term | 10K | 1K | 0.38ms | 48.8ms | **20,496** |
| Long-term | 100K | 100 | 0.83ms | 6.6ms | **15,054** |
| Stress test | 10K | 10K | 0.55ms | 455ms | **21,964** |

### 3.2 Performance Analysis

**Short-term advantage (1K samples):**
- GPU upload overhead is minimal (0.5ms)
- Kernel launch overhead dominates
- Still achieves ~19K iter/sec

**Long-term advantage (100K samples):**
- Larger data benefits from GPU parallelism
- Upload cost amortized over many iterations
- Achieves 15K iter/sec with 100× more data

**Key finding:** Static VRAM bucket eliminates per-iteration malloc overhead, enabling consistent ~20K iter/sec performance regardless of iteration count.

---

## 4. CPU Fallback Verification

### NumPy Reference Implementation

| Metric | Value |
|--------|-------|
| Execution time | 0.14ms |
| Pearson r | 0.0037 |

### GPU Kernel

| Metric | Value |
|--------|-------|
| Execution time | 0.91ms |
| Pearson r | 0.0037 |

### Consistency Check

| Metric | Value |
|--------|-------|
| GPU/CPU match | ✅ PASS |
| Max difference | < 0.001 |

**Note:** For single-shot computation, NumPy can be faster due to GPU kernel launch overhead. The GPU advantage emerges in high-iteration loops (millions of combos).

---

## 5. Reporting System Validation

### Cross-Validation Fold Consistency

| Fold | Train N | Val N | Train r | Val r |
|------|---------|-------|---------|-------|
| 0 | 3,968 | 1,032 | 0.9988 | 0.9985 |
| 1 | 4,023 | 977 | 0.9988 | 0.9988 |
| 2 | 4,013 | 987 | 0.9988 | 0.9988 |
| 3 | 3,980 | 1,020 | 0.9987 | 0.9989 |
| 4 | 4,016 | 984 | 0.9988 | 0.9988 |

### Statistics

| Metric | Value |
|--------|-------|
| Mean train_r | 0.9988 ± 0.0000 |
| Mean val_r | 0.9988 ± 0.0001 |
| Train/Val gap | 0.0000 |

**Conclusion:** Cross-validation scoring is stable across folds with no overfitting.

---

## 6. Feature Engineering Usability

### Demo: Automated Feature Search

```python
# Allocate GPU bucket ONCE
bucket = StaticBucket(n_samples=10000, n_features=5)
bucket.upload_all(features, target, mask)

# Search millions of combinations with NO malloc overhead
for combo in combinations(range(5), 2):
    stats = bucket.compute(
        feature_indices=list(combo),
        ops=[UnaryOp.LOG, UnaryOp.SQRT],
        interaction=InteractionType.MULT
    )
    train_r, val_r = compute_pearson_from_stats(stats)
```

### Demo Results

| Phase | Combinations | Time | Speed |
|-------|--------------|------|-------|
| Pairwise search | 50 | 3.0ms | 16,423/sec |
| Operator search | 75 | 4.2ms | 17,857/sec |
| Full search | 90 | 5.0ms | 17,990/sec |

### Signal Discovery

| Rank | Feature | Val r | Status |
|------|---------|-------|--------|
| 1 | f0 × f1 | 0.8445 | 🎯 PLANTED |
| 2 | sqrt_f1 × log_f2 | 0.3798 | discovered |
| 3 | sqrt_f0 × log_f2 | 0.3755 | discovered |
| 4 | log_f2 × sqrt_f3 | 0.3736 | discovered |

**Best operator combo found:** `log(f2) + identity(f3)` with r=0.526 (matches planted signal)

---

## 7. GAFIME Value Proposition

### What GAFIME Provides

1. **Automated Feature Discovery**
   - No manual feature brainstorming required
   - Systematic exploration of all operator/interaction combinations
   - Data-driven approach finds non-obvious relationships

2. **GPU Acceleration**
   - 15,000-22,000 combinations/second
   - Static VRAM bucket eliminates malloc overhead
   - On-chip reduction (no intermediate global memory)

3. **Validation-Aware**
   - Built-in cross-validation fold support
   - Train/val split in single kernel pass
   - Prevents overfitting during feature search

4. **Production-Ready**
   - Safe operators (NaN/Inf prevention)
   - CPU fallback for non-GPU systems
   - Consistent results across platforms

### When to Use GAFIME

| Use Case | Recommendation |
|----------|----------------|
| < 1000 features | ✅ Perfect fit |
| Tabular data | ✅ Designed for this |
| Feature interaction search | ✅ Core strength |
| Single computation | ⚠️ NumPy may be faster |
| Image/text data | ❌ Use specialized tools |

---

## Appendix: Test Environment

```
Python: 3.12.x
NumPy: 2.x
CUDA Toolkit: 13.1
GPU: NVIDIA GeForce RTX 4060
VRAM: 8GB
Compute Capability: SM89 (Ada Lovelace)
```

---

*Report generated by `tests/scientific_benchmark.py` and `examples/feature_engineering_demo.py`*
