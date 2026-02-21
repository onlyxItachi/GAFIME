# GAFIME Usage Guide

Welcome to the advanced technical reference for GAFIME.
This guide details how to control the `FeatureInteractionEngine` underneath the hood using `EngineConfig` and `ComputeBudget`.

## The Engine Configuration

When you instantiate GAFIME, you can pass a configuration object to strictly define its boundaries, random deterministic states, and validation thresholds.

```python
from gafime.config import EngineConfig, ComputeBudget
from gafime.engine import FeatureInteractionEngine

config = EngineConfig(
    budget=ComputeBudget(
        max_comb_size=2,                # Maximum interaction depth (1 = unary, 2 = pairs, 3 = trios)
        max_combinations_per_k=5000,    # Max combinations to search at each depth dimension
        top_features_for_higher_k=50,   # How many of the best unary features pass to the pairwise step
        keep_in_vram=True,              # Keeps data pinned to GPU VRAM for the entire analysis
        vram_budget_mb=6144             # Defines the maximum VRAM we allocate (e.g. 6GB on an RTX 4060)
    ),
    metric_names=("pearson", "spearman", "iv"), # Metrics to evaluate interactons against
    num_repeats=3,                      # Number of cross-validation-like repeated stability tests
    stability_std_threshold=0.10,       # Maximum allowed standard deviation across repeated metric sweeps
    permutation_tests=25,               # How many random target shuffles to perform for significance testing
    permutation_p_threshold=0.05,       # Maximum p-value allowed to consider a signal "real"
    backend="auto"                      # Auto-discovers the fastest hardware (CUDA > Metal > CPU Rust)
)

engine = FeatureInteractionEngine(config=config, verbose=True)
```

## Available Evaluation Metrics

The `EngineConfig` accepts a `metric_names` tuple. You can use any combination:

* **`pearson`**: Classical linear correlation. Great for continuous features vs continuous targets.
* **`spearman`**: Rank correlation. Perfect when you suspect monotonic (but non-linear) relationships.
* **`iv`**: Information Value. The gold standard in banking/finance for evaluating categorical feature strength against a binary target.
* **`r2`**: R-Squared variance explanation.

## Arithmetic Operators

When using the raw engine or the Scikit-Learn wrapper, you can specify how the interaction is mathematically generated before it is scored:

```python
engine.evaluate_interactions(X, y, operator="multiply")
```

* **`multiply`**: $X_1 \times X_2$ (Most common).
* **`add`**: $X_1 + X_2$.
* **`subtract`**: $X_1 - X_2$.
* **`divide`**: $X_1 \div X_2$ (Protected against division by zero via epsilon addition).

## Using the Underlying Report

The engine's `analyze` function returns a `DiagnosticReport` dataclass which is rich with information, allowing you to debug exactly why a pipeline thought a feature was interesting:

```python
report = engine.analyze(X, y)

print(f"Signal Detected: {report.decision.signal_detected}")

# View the raw stability variance of the top interaction
print(report.stability[0].metrics_std)

# View the exact p-value against the random noise threshold!
print(report.permutations[0].p_values)
```
