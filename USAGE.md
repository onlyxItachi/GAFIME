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
    num_repeats=3,                      # Selected-candidate bootstrap repeat count
    stability_std_threshold=0.10,       # Maximum conditional bootstrap metric std
    permutation_tests=25,               # How many random target shuffles to perform for significance testing
    significance_top_n=50,              # Maximum selected interactions evaluated/reported for significance
    permutation_p_threshold=0.05,       # Maximum p-value allowed to consider a signal "real"
    mi_bins=96,                         # Adaptive maximum bins for mutual information
    backend="auto"                      # Uses the v1 resolver for Rust CPU or configured GPU payloads
)

engine = GafimeEngine(config=config)
```

`mi_bins` is a ceiling. GAFIME selects the largest sample-safe histogram shape
from `2, 4, 8, 12, 16, 24, 32, 48, 64, 96` using the v0.4.1 density rule
`8 * bins^2 <= n_samples`; Metal is capped at 48, and 2 is the minimum fallback
for very small samples. Setting a value between templates rounds the ceiling
down, never up to 96.

`significance_top_n` is separate from
`budget.top_features_for_higher_k`. The former bounds significance/report
selection; the latter controls the unary shortlist used to generate
higher-order candidates.

Current v1 permutation significance uses family-wise Westfall-Young maxT.
Published v0.4.7 and `v0.5.0-legacy` used candidate-wise tests with one shared
Python RNG stream, so their stochastic p-values and final significance
decisions are not numerically interchangeable with current reports. Fixed-seed
one-shot, resident, and explicit-compiled runs within current v1 remain exact
parity contracts.

GPU maxT stays on the requested backend. CUDA uses its compact native
permutation ABI for target-independent families. Adaptive families, ROCm, Metal,
and older CUDA payloads repeat device screening and obtain complete-family maxima
through bounded device ranking; the observed target is restored before the
resident artifact can be reused.

Available public backend names are `"auto"`, `"core"`, `"cpu"`, `"cuda"`,
`"rocm"`, `"hip"`, and `"metal"`. `backend="gpu"` is rejected because it is
ambiguous in v1; request a vendor backend explicitly.

For v0.4.7 migration, the first ten positional `EngineConfig` arguments remain
stable through `device_id`. Generated-family options are keyword-only. In
particular, positional argument 11 was formerly `enable_discrete_functions`;
v1 accepts only the exactly compatible disabled value (`False`) with a
`DeprecationWarning`. An enabled or otherwise ambiguous legacy value is rejected
with migration guidance instead of being treated as
`enable_time_series_functions`.

## Compile Artifacts

`gafime.compile` creates an in-memory compiled artifact that owns the coerced
native data, compact scenario plan, backend session, and optional export handles.
Repeated `artifact.analyze()` calls reuse resident backend state where the
selected backend supports it.

For continuous eager analysis, `GAFIME_V1_ANALYZE_CACHE_SIZE=0` selects the
stateless one-shot boundary. A positive capacity enables the content-keyed eager
resident LRU when `keep_in_vram=True`. The LRU and its capacity are local to the
calling thread because the native PyO3 artifacts are thread-affine. Explicit
compiled artifacts do not use that LRU: they own their matrix and plan until
`close()`, and must be analyzed, updated, and closed on their creation thread. See
[docs/eager-resident-compiled-execution.md](docs/eager-resident-compiled-execution.md)
for lifetime, performance, and correctness details.

The current extension transports validated continuous inputs as contiguous
little-endian fp32 bytes. Representable NaN and infinity values are accepted;
finite values outside the fp32 range are rejected on every execution path.
Integer seeds retain all Python integer words for candidate planning. With
`random_seed=None`, each `analyze()` call receives a fresh stochastic stream,
including replay through an explicit compiled artifact.

```python
import gafime
from gafime import CompileFlags, ComputeBudget, EngineConfig

config = EngineConfig(
    backend="auto",
    metric_names=("pearson", "r2"),
    budget=ComputeBudget(
        max_comb_size=3,
        max_combinations_per_k=5000,
        top_features_for_higher_k=50,
    ),
    permutation_tests=0,
    num_repeats=1,
)

artifact = gafime.compile(
    X,
    y,
    feature_names=feature_names,
    config=config,
    flags=CompileFlags(plan=True, export=True),
)
try:
    report = artifact.analyze()
    print(artifact.backend)
    print(report.interactions.top_k(5, metric_name="pearson"))
    arrow_capsules = artifact.export_arrow()
finally:
    artifact.close()
```

`CompileFlags(export=True)` enables Arrow C Data export for the compact native
result table. Backend graph capture/replay is owned by the selected native
payload and must not change result semantics or introduce silent fallback.

`artifact.flags` remains an exact `CompileFlags` view for v0.5 compatibility.
The legacy `artifact.exports` property remains available with a
`DeprecationWarning`; it exposes the owning native artifact as
`feature_matrix_handle` and the last native report as `result_table_handle`.
The current boundary has no independent candidate-table handle, so
`candidate_table_handle` is `None`. New code should use `export_arrow()`.

## Decision Paths and Time-Series Candidates

GAFIME reports continuous interactions by default. Optional decision-path or
time-series family generation is enabled through `EngineConfig`:

```python
config = EngineConfig(
    metric_names=("pearson", "r2"),
    enable_decision_path_functions=True,
    permutation_tests=0,
    decision_path_max_depth=2,
    decision_path_max_paths=32,
)

ts_config = EngineConfig(
    metric_names=("pearson", "r2"),
    enable_time_series_functions=True,
    time_series_lags=(1, 2, 4, 8),
    time_series_windows=(4, 8, 16),
    budget=ComputeBudget(max_comb_size=2, max_time_series_candidates=1000),
)
```

Decision-path bootstrap stability remains available through `num_repeats`, but
permutation significance is not yet supported because every permuted target
must rediscover its own paths. Since `permutation_tests` defaults to `25`, a
decision-path configuration must currently set `permutation_tests=0`; positive
values fail closed with `V1UnsupportedError` before backend execution.

The v0.4 discrete candidate family is no longer part of the current engine API.
Tree-like threshold and region structure now belongs to the native
`decision_path` family.

The top-level `DecisionPathCandidate` data object remains available for code
that stores or exchanges v0.5 path descriptors. Current analysis results are
still returned as `InteractionResult` records; the compatibility object does
not add a separate execution path.

## Compatibility Utilities

`GafimeSelector`, `GafimeStreamer`, and `generate_tutorial` are available from
the top-level package as in v0.4.7. `GafimeStreamer` preserves CSV/Parquet batch
iteration through Polars, while `generate_tutorial(path)` writes a notebook
using the current v1 API.

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

`metrics_std` is computed by bootstrapping an already-selected candidate on the
same rows that selected it. It measures variability conditional on selection,
not out-of-sample or out-of-fold performance, and it does not correct selection
bias. Use an untouched holdout or nested cross-validation before treating a
selected interaction as generalizable.
