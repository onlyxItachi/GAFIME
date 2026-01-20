# GAFIME Project Structure

**GPU-Accelerated Feature Interaction Mining Engine**

A high-performance feature engineering framework using CUDA for brute-force 
feature interaction discovery with cross-validation.

---

## Root Directory

```
GAFIME/
├── gafime/                  # Python package (main library)
├── src/                     # Native code (CUDA/C++)
├── tests/                   # Test suite
├── examples/                # Usage examples
├── gafime_core/             # Core algorithm definitions
│
├── gafime_cuda.dll          # Compiled CUDA backend (Windows)
├── gafime_cpu.dll           # Compiled CPU backend (Windows)
│
├── setup.py                 # Package installation
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── LICENSE                  # Open source license
└── GAFIME_TimeSeries_Limitation_Report.txt  # Time series gap analysis
```

---

## gafime/ (Python Package)

### Core Modules

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `engine.py` | Main GAFIME engine orchestration |
| `config.py` | Configuration dataclasses |
| `io.py` | Data loading/saving utilities |

### gafime/backends/

**Compute backends for feature interaction calculation**

| File | Purpose |
|------|---------|
| `__init__.py` | Backend selector and exports |
| `base.py` | Abstract backend interface |
| `fused_kernel.py` | **Core: StaticBucket, UnaryOp, InteractionType, compute_pearson_from_stats** |
| `native_cuda_backend.py` | CUDA DLL interface |
| `core_backend.py` | Core algorithm bindings |

**Key Classes:**
- `StaticBucket` - GPU memory container for features
- `UnaryOp` - Feature transformations (LOG, SQRT, SQUARE, etc.)
- `InteractionType` - Binary operations (MULT, ADD, SUB, DIV, MAX, MIN)
- `compute_pearson_from_stats()` - Correlation calculation

### gafime/optimizer/

**Search algorithms for feature discovery**

| File | Purpose |
|------|---------|
| `__init__.py` | Exports |
| `ensemble_search.py` | **EnsembleSearchEngine** - main search orchestrator |
| `adaptive.py` | Adaptive search strategies |
| `orchestrator.py` | Multi-phase search coordination |

**Key Classes:**
- `EnsembleSearchEngine` - Scouting, voting, verification pipeline
- `SearchConfig` - Search parameters
- `FeatureRecipe` - Discovered feature specification

### gafime/preprocessors/

**Data preprocessing for different data types**

| File | Purpose |
|------|---------|
| `__init__.py` | Exports |
| `time_series.py` | **TimeSeriesPreprocessor** - lags, differentials, rolling windows |

**Key Classes:**
- `TimeSeriesPreprocessor` - Polars-based time series feature generation
- `TimeSeriesConfig` - Configuration for windows, lags, differentials

### gafime/metrics/

| File | Purpose |
|------|---------|
| `base.py` | Metric interface |
| `cpu_metrics.py` | CPU-based metric calculations |

### gafime/validation/

| File | Purpose |
|------|---------|
| `permutation.py` | Permutation importance |
| `stability.py` | Feature stability analysis |

### gafime/planning/

| File | Purpose |
|------|---------|
| Planning and execution control |

### gafime/reporting/

| File | Purpose |
|------|---------|
| Result formatting and exports |

### gafime/utils/

| File | Purpose |
|------|---------|
| Utility functions |

---

## src/ (Native Code)

### src/cuda/

| File | Purpose |
|------|---------|
| `kernels.cu` | **CUDA kernels** - GPU feature computation, correlation, statistics |

### src/cpu/

| File | Purpose |
|------|---------|
| CPU fallback implementation |

### src/common/

| File | Purpose |
|------|---------|
| Shared code between CPU/CUDA |

---

## Key Operations

### UnaryOp (Feature Transformations)
```
IDENTITY    - f(x) = x
LOG         - f(x) = log(|x| + ε)
EXP         - f(x) = exp(x)
SQRT        - f(x) = √|x|
TANH        - f(x) = tanh(x)
SIGMOID     - f(x) = 1/(1+e^(-x))
SQUARE      - f(x) = x²
NEGATE      - f(x) = -x
ABS         - f(x) = |x|
INVERSE     - f(x) = 1/x
CUBE        - f(x) = x³
ROLLING_MEAN - Moving average
ROLLING_STD  - Moving standard deviation
```

### InteractionType (Binary Operations)
```
MULT - a × b
ADD  - a + b
SUB  - a - b
DIV  - a / b
MAX  - max(a, b)
MIN  - min(a, b)
```

---

## Usage Flow

```
1. Load data → gafime.io

2. Preprocess (optional) 
   → TimeSeriesPreprocessor for time series data
   
3. Create backend
   → StaticBucket(n_samples, n_features)
   → Upload features, target, mask
   
4. Search
   → EnsembleSearchEngine.search()
   → Returns ranked feature recipes
   
5. Apply features
   → Apply discovered recipes to train/test
   
6. Train model
   → XGBoost, LightGBM, CatBoost, etc.
```

---

## Performance Characteristics

- **GPU Memory**: StaticBucket holds features in VRAM
- **Compute**: Fused kernels minimize memory traffic
- **Parallelism**: All feature pairs evaluated in parallel
- **Validation**: Built-in cross-validation during search
