# GAFIME Usage Guide

Welcome to the advanced technical reference for GAFIME.
This guide details how to control the `GafimeEngine` underneath the hood using `EngineConfig` and `ComputeBudget`.

## The Engine Configuration

When you instantiate GAFIME, you can pass a configuration object to strictly define its boundaries, random deterministic states, and validation thresholds.

```python
from gafime import GafimeEngine, EngineConfig, ComputeBudget

config = EngineConfig(
    budget=ComputeBudget(
        max_comb_size=2,                # Maximum interaction depth (1 = unary, 2 = pairs, 3 = trios)
        max_combinations_per_k=5000,    # Max combinations to search at each depth dimension
        top_features_for_higher_k=50,   # How many of the best unary features pass to the pairwise step
        keep_in_vram=True,              # Keeps data pinned to GPU VRAM for the entire analysis
        vram_budget_mb=6144             # Defines the maximum VRAM we allocate (e.g. 6GB on an RTX 4060)
    ),
    metric_names=("pearson", "spearman", "mutual_info", "r2"), # Metrics to evaluate interactions against
    num_repeats=3,                      # Number of cross-validation-like repeated stability tests
    stability_std_threshold=0.10,       # Maximum allowed standard deviation across repeated metric sweeps
    permutation_tests=25,               # How many random target shuffles to perform for significance testing
    permutation_p_threshold=0.05,       # Maximum p-value allowed to consider a signal "real"
    backend="auto"                      # Auto-discovers the fastest hardware (CUDA > Metal > C++ Core > NumPy)
)

engine = GafimeEngine(config=config)
```

Available backends are `"auto"`, `"cuda"`, `"gpu"`, `"metal"`, `"cpu"`, `"numpy"`, `"core"`, and `"cpp"`.

## Discrete Function Search (v0.4.0)

GAFIME can also search discrete function representations inside the normal
engine flow. These candidates are planned, scored, validated, and reported
alongside unary and higher-order continuous interactions.

```python
from gafime import ComputeBudget, EngineConfig, GafimeEngine

config = EngineConfig(
    metric_names=("pearson",),          # Also controls discrete candidates
    enable_discrete_functions=True,
    discrete_mode="soft",              # "hard" is CPU/NumPy only
    discrete_ranking="split_aware",    # Default internal ranking for discrete candidates
    discrete_threshold_source="quantile",
    discrete_gate_sharpness=12.0,
    budget=ComputeBudget(
        max_discrete_candidates=100_000,
        max_thresholds_per_feature=9,
        max_intervals_per_feature=12,
        max_feature_pairs_for_rectangles=500,
        top_k_features_for_discrete=50,
    ),
)

report = GafimeEngine(config).analyze(X, y, feature_names=feature_names)
```

Implemented families:

* `discrete_function_soft_threshold`
* `discrete_function_soft_interval`
* `discrete_function_value_gated_threshold`
* `discrete_function_soft_rectangle`
* `discrete_function_value_in_soft_rectangle`

Discrete candidates do not have a separate metric selector in v0.4.0. They
honor `EngineConfig.metric_names` exactly for report scoring. Their default
ordering uses `discrete_ranking="split_aware"` so split/interval/rectangle
candidates are not ranked by Pearson alone. Use `discrete_ranking="metric"` to
rank by the selected report metrics, or `"none"` to preserve planning order.

### Backend Rules

CUDA and Metal GPU paths use soft, vectorized discrete approximations only. If
a GPU backend is selected with `discrete_mode="hard"`, the engine raises:

```text
GPU feature engineering with discrete hard mode is not supported!
```

CPU and NumPy can evaluate hard mode with host-side vectorized comparisons.
Thresholds are quantile-generated in v0.4.0. Tree-inspired and learnable
thresholds are future work, not engine release behavior.

### Rust Helper Alias

Rust helper/orchestration APIs are exposed as:

```python
from gafime import subfunctions

scheduler = subfunctions.BatchScheduler(max_blocks=1024)
```

Prefer `subfunctions` in docs and examples. Direct `gafime_cpu` imports are an
implementation detail.

## Available Evaluation Metrics

The `EngineConfig` accepts a `metric_names` tuple. You can use any combination:

* **`pearson`**: Classical linear correlation. Great for continuous features vs continuous targets.
* **`spearman`**: Rank correlation. Perfect when you suspect monotonic (but non-linear) relationships.
* **`mutual_info`**: Mutual information. Useful for capturing non-linear dependency between a feature (or interaction) and the target.
* **`r2`**: R-squared variance explanation for regression-style signal strength.

## Arithmetic Operators

The base `GafimeEngine` scores the combinations it plans internally via `engine.analyze(X, y)`. If you want to control how selected pairwise interactions are materialized (for example `multiply`, `add`, `subtract`, or `divide`), use the Scikit-Learn wrapper instead:

```python
import numpy as np
from gafime import GafimeSelector

X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]], dtype=float)
y = np.array([0.2, 0.5, 0.9], dtype=float)

selector = GafimeSelector(operator="multiply", metric="pearson", backend="auto")
selector.fit(X, y)
```

* **`multiply`**: $X_1 \times X_2$ (Most common).
* **`add`**: $X_1 + X_2$.
* **`subtract`**: $X_1 - X_2$.
* **`divide`**: $X_1 \div X_2$ (Protected against division by zero via epsilon addition).

## Using the Underlying Report

The engine's `analyze` function returns a `DiagnosticReport` dataclass which is rich with information, allowing you to debug exactly why a pipeline thought a feature was interesting:

```python
import numpy as np
from gafime import GafimeEngine, EngineConfig

X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]], dtype=float)
y = np.array([0.2, 0.5, 0.9], dtype=float)

engine = GafimeEngine(config=EngineConfig())
report = engine.analyze(X, y)

print(f"Signal Detected: {report.decision.signal_detected}")

# View the raw stability variance of the top interaction
print(report.stability[0].metrics_std)

# View the exact p-value against the random noise threshold!
print(report.permutations[0].p_values)
```
