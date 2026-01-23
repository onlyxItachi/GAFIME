# GAFIME AI Codewriter Guide

> **For AI Coding Assistants**: This guide explains the GAFIME architecture to enable you to effectively extend, modify, and work with the codebase.

---

## Overview

**GAFIME** is a GPU-Accelerated Feature Interaction Mining Engine(as capital letters). It automatically discovers feature combinations that correlate with a target variable.

**Key Design Principles:**
1. **Separation of Concerns**: Time-series preprocessing (Polars) is separate from interaction mining (CUDA kernels)
2. **Static Memory Management**: Pre-allocate GPU memory once, reuse for millions of iterations  
3. **Pluggable Backends**: CUDA → C++ → NumPy fallback chain
4. **Validation-Aware**: Built-in train/val split, stability analysis, permutation testing

---

## Architecture Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │                  User Code                       │
                    │  GafimeEngine.analyze(X, y, feature_names)      │
                    └─────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────────┐
                    │              GafimeEngine                        │
                    │  ┌──────────────────────────────────────────┐  │
                    │  │ 1. coerce_inputs() → X_array, y_array    │  │
                    │  │ 2. resolve_backend() → CUDA/CPU/NumPy    │  │
                    │  │ 3. plan_unary() → single-feature combos  │  │
                    │  │ 4. _score_combos() → metric scores       │  │
                    │  │ 5. select_top_features() → top N         │  │
                    │  │ 6. plan_higher_order() → k-way combos    │  │
                    │  │ 7. StabilityAnalyzer.assess()            │  │
                    │  │ 8. PermutationTester.test()              │  │
                    │  │ 9. _make_decision() → signal detected?   │  │
                    │  └──────────────────────────────────────────┘  │
                    └─────────────────────────────────────────────────┘
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          │                              │                              │
    ┌─────▼─────┐               ┌────────▼────────┐           ┌────────▼────────┐
    │  Backend  │               │   MetricSuite   │           │    Reporting    │
    │ ┌───────┐ │               │ pearson, spearman│          │ DiagnosticReport│
    │ │ CUDA  │ │               │ mutual_info, r2 │           │ InteractionResult│
    │ │ C++   │ │               └─────────────────┘           │ Decision        │
    │ │ NumPy │ │                                              └─────────────────┘
    │ └───────┘ │
    └───────────┘
```

---

## Key Classes Reference

### Core Classes

| Class | File | Purpose |
|-------|------|---------|
| `GafimeEngine` | `engine.py` | Main orchestrator - the entry point for analysis |
| `EngineConfig` | `config.py` | Immutable configuration (backend, metrics, thresholds) |
| `ComputeBudget` | `config.py` | Controls search space (max_comb_size, top_features) |
| `GafimeStreamer` | `io.py` | VRAM-aware data streaming from disk |

### Backend Classes

| Class | File | Purpose |
|-------|------|---------|
| `Backend` | `backends/base.py` | Base class - NumPy fallback implementation |
| `NativeCudaBackend` | `backends/native_cuda_backend.py` | GPU acceleration via ctypes → DLL |
| `CoreBackend` | `backends/core_backend.py` | C++ OpenMP backend |
| `StaticBucket` | `backends/fused_kernel.py` | Zero-malloc CUDA memory bucket |
| `FusedKernelWrapper` | `backends/fused_kernel.py` | Python wrapper for fused CUDA kernels |

### Preprocessors

| Class | File | Purpose |
|-------|------|---------|
| `TimeSeriesPreprocessor` | `preprocessors/time_series.py` | Creates lags, differentials, calculus features |
| `TimeSeriesConfig` | `preprocessors/time_series.py` | Configuration for window sizes, feature columns |

### Optimizer (Advanced)

| Class | File | Purpose |
|-------|------|---------|
| `GafimeOrchestrator` | `optimizer/orchestrator.py` | Time-budgeted search with auto strategy |
| `EnsembleSearchEngine` | `optimizer/ensemble_search.py` | Multi-scout parallel search |
| `TimeAdaptiveOptimizer` | `optimizer/adaptive.py` | Dynamic budget allocation |

### Validation & Reporting

| Class | File | Purpose |
|-------|------|---------|
| `StabilityAnalyzer` | `validation/stability.py` | Measures metric variance across repeats |
| `PermutationTester` | `validation/permutation.py` | Computes p-values via null distribution |
| `MetricSuite` | `metrics/base.py` | Computes pearson, spearman, MI, R2 |
| `DiagnosticReport` | `reporting/report.py` | Final output with all results |

---

## Data Flow

### 1. Input Processing
```python
# User provides raw NumPy arrays
X_raw = np.array(...)  # shape: (n_samples, n_features)
y_raw = np.array(...)  # shape: (n_samples,)

# Engine coerces to correct types
X_array, y_array, names = coerce_inputs(X_raw, y_raw, feature_names)
# -> X_array: float64, C-contiguous
# -> y_array: float64, C-contiguous
# -> names: List[str] (auto-generated if None)
```

### 2. Backend Selection
```python
# Automatic backend resolution (priority order)
backend, warnings = resolve_backend(config, X_array, y_array)
# 1. Try NativeCudaBackend (GPU)
# 2. Try CoreBackend (C++ OpenMP)  
# 3. Fall back to Backend (NumPy)
```

### 3. Combination Planning
```python
# Phase 1: Score all single features
unary_combos = [(0,), (1,), (2,), ...]  # Each feature alone
unary_scores = backend.score_combos(X, y, unary_combos, metric_suite)

# Phase 2: Select top performers
top_features = select_top_features(feature_scores, top_n=50)

# Phase 3: Higher-order combinations
higher_combos = [(0,1), (0,2), (1,2), (0,1,2), ...]  # Pairwise, triplets, etc.
```

### 4. GPU Kernel Execution (StaticBucket)
```python
# Pre-allocate GPU memory ONCE
bucket = StaticBucket(n_samples=10000, n_features=5)
bucket.upload_all(features=[f0,f1,f2,f3,f4], target=y, mask=fold_mask)

# Execute millions of iterations - NO malloc/free!
for combo in combos:
    for ops in operator_configs:
        stats = bucket.compute(
            feature_indices=[0, 1, 2],
            ops=[UnaryOp.LOG, UnaryOp.SQRT, UnaryOp.IDENTITY],
            interaction_types=[InteractionType.MULT, InteractionType.ADD],  # A*B+C
            val_fold=0
        )
        train_r, val_r = compute_pearson_from_stats(stats)
```

### 5. Validation
```python
# Stability: repeat N times, measure variance
stability_results = StabilityAnalyzer.assess(X, y, combos, num_repeats=3)

# Permutation: shuffle target M times, compute p-values
perm_results = PermutationTester.test(X, y, combos, num_permutations=25)
```

### 6. Decision
```python
# Signal detected if: p < 0.05 AND std < 0.10 AND strength > 0
decision = Decision(signal_detected=True, message="Learnable signal detected")
```

---

## Extension Points

### Adding a New Metric

1. Add to `gafime/metrics/cpu_metrics.py`:
```python
def my_custom_metric(x: np.ndarray, y: np.ndarray, xp=np) -> float:
    # Your metric implementation
    return float(result)
```

2. Register in `gafime/metrics/base.py`:
```python
SUPPORTED_METRICS = ("pearson", "spearman", "mutual_info", "r2", "my_metric")

# In MetricSuite.score():
elif name == "my_metric":
    results[name] = self.ops.my_custom_metric(x, y, xp=self.xp)
```

### Adding a New Unary Operator

1. Add constant in `src/common/interfaces.h`:
```c
#define GAFIME_OP_MY_OP  13  // x' = my_transform(x)
```

2. Implement in `src/cuda/kernels.cu`:
```cuda
case GAFIME_OP_MY_OP:
    return my_transform(x);
```

3. Add to Python in `gafime/backends/fused_kernel.py`:
```python
class UnaryOp:
    MY_OP = 13
    _names = {..., 13: "my_op"}
```

### Adding a New Interaction Type

1. Add constant in `src/common/interfaces.h`:
```c
#define GAFIME_INTERACT_MY_INTERACT  6
```

2. Implement in `src/cuda/kernels.cu` (in `combine()` function):
```cuda
case GAFIME_INTERACT_MY_INTERACT:
    return my_combine(a, b);
```

3. Update Python:
```python
class InteractionType:
    MY_INTERACT = 6
```

---

## Common Patterns

### Basic Usage
```python
from gafime import GafimeEngine, EngineConfig, ComputeBudget

config = EngineConfig(
    budget=ComputeBudget(max_comb_size=2, top_features_for_higher_k=50),
    backend="auto"
)

engine = GafimeEngine(config)
report = engine.analyze(X, y, feature_names=names)

print(f"Signal detected: {report.decision.signal_detected}")
for interaction in report.interactions[:5]:
    print(f"  {interaction.feature_names}: {interaction.metrics}")
```

### Time-Series Workflow
```python
from gafime.preprocessors import TimeSeriesPreprocessor
import polars as pl

# 1. Create time series features
tsp = TimeSeriesPreprocessor(
    group_col='customer_id',
    time_col='date',
    windows=[7, 30, 90, 180, 360],
    enable_calculus=True  # Velocity, acceleration, momentum
)

# 2. Transform and aggregate
df_features = tsp.aggregate_to_entity(raw_df, target_df, 'churn')

# 3. Convert to NumPy and mine interactions
X = df_features.drop(['customer_id', 'churn']).to_numpy()
y = df_features['churn'].to_numpy()

report = GafimeEngine().analyze(X, y)
```

### Advanced: Direct Kernel Access
```python
from gafime.backends.fused_kernel import StaticBucket, UnaryOp, InteractionType

bucket = StaticBucket(n_samples=10000, n_features=3)
bucket.upload_all(features=[A, B, C], target=y, mask=fold_mask)

# Mixed interaction: (log(A) * sqrt(B)) + C
stats = bucket.compute(
    feature_indices=[0, 1, 2],
    ops=[UnaryOp.LOG, UnaryOp.SQRT, UnaryOp.IDENTITY],
    interaction_types=[InteractionType.MULT, InteractionType.ADD],
    val_fold=0
)
```

### Time-Budgeted Search
```python
from gafime.optimizer import gafime_search

# Run for 60 seconds
results = gafime_search(X, y, target_time=60.0, top_k=100)

for candidate in results[:10]:
    print(f"{candidate.signature}: {candidate.val_correlation:.4f}")
```

---

## File Structure Quick Reference

```
gafime/
├── __init__.py           # Exports: GafimeEngine, EngineConfig, ComputeBudget, GafimeStreamer
├── engine.py             # GafimeEngine class - main entry point
├── config.py             # EngineConfig, ComputeBudget dataclasses
├── io.py                 # GafimeStreamer - VRAM-aware data loading
│
├── backends/
│   ├── __init__.py       # resolve_backend() - auto-selects CUDA/CPU/NumPy
│   ├── base.py           # Backend base class (NumPy fallback)
│   ├── native_cuda_backend.py  # NativeCudaBackend (GPU via ctypes)
│   ├── core_backend.py   # CoreBackend (C++ OpenMP)
│   └── fused_kernel.py   # StaticBucket, FusedKernelWrapper, UnaryOp, InteractionType
│
├── preprocessors/
│   └── time_series.py    # TimeSeriesPreprocessor, create_calculus_features()
│
├── metrics/
│   ├── base.py           # MetricSuite class
│   └── cpu_metrics.py    # pearson_corr, spearman_corr, mutual_info, linear_r2
│
├── planning/
│   └── combinations.py   # plan_unary, plan_higher_order, select_top_features
│
├── validation/
│   ├── stability.py      # StabilityAnalyzer
│   └── permutation.py    # PermutationTester
│
├── optimizer/
│   ├── orchestrator.py   # GafimeOrchestrator, gafime_search()
│   ├── ensemble_search.py # EnsembleSearchEngine, Scout, VotingSystem
│   └── adaptive.py       # TimeAdaptiveOptimizer
│
├── reporting/
│   └── report.py         # DiagnosticReport, InteractionResult, Decision
│
└── utils/
    ├── arrays.py         # coerce_inputs, build_interaction_vector
    └── safety.py         # validate_budget
```

---

## Build Commands

```bash
# Build CUDA + CPU backends
cd C:\Users\Hamza\Desktop\GAFIME
C:\AI_KERNELS\ds_gpu\Scripts\python.exe setup.py build_ext --inplace

# Direct nvcc build (requires VS2022 Developer environment)
nvcc -arch=sm_89 -O3 --shared -Xcompiler "/MD,/O2" -DGAFIME_BUILDING_DLL -I src/common -o gafime_cuda.dll src/cuda/kernels.cu
```

---

## Important Constants

### Unary Operators (`src/common/interfaces.h`)
| ID | Name | Operation | Hardware |
|----|------|-----------|----------|
| 0 | IDENTITY | x' = x | ALU |
| 1 | LOG | x' = log(abs(x) + eps) | SFU |
| 2 | EXP | x' = exp(clamp(x)) | SFU |
| 3 | SQRT | x' = sqrt(abs(x)) | SFU |
| 4 | TANH | x' = tanh(x) | SFU |
| 5 | SIGMOID | x' = 1/(1+exp(-x)) | SFU |
| 6 | SQUARE | x' = x^2 | ALU |
| 7 | NEGATE | x' = -x | ALU |
| 8 | ABS | x' = abs(x) | ALU |
| 9 | INVERSE | x' = 1/x | ALU |
| 10 | CUBE | x' = x^3 | ALU |
| 11 | ROLLING_MEAN | x' = mean(window) | Memory |
| 12 | ROLLING_STD | x' = std(window) | Memory |

### Interaction Types
| ID | Name | Operation |
|----|------|-----------|
| 0 | MULT | a * b |
| 1 | ADD | a + b |
| 2 | SUB | a - b |
| 3 | DIV | a / b (safe) |
| 4 | MAX | max(a, b) |
| 5 | MIN | min(a, b) |

---

## Tips for AI Coding Assistants

1. **Always use absolute paths** when modifying files
2. **Rebuild DLL** after modifying `kernels.cu` or `interfaces.h`
3. **Check backend resolution** if CUDA isn't being used
4. **StaticBucket is for hot loops** - allocate once, compute millions of times
5. **interaction_types is an array** of (arity-1) elements for mixed operations
6. **TimeSeriesPreprocessor** is separate from GAFIME core - use it for time-series data
7. **Test with `tests/test_perpair_interactions.py`** after kernel changes
