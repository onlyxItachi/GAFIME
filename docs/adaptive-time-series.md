# Adaptive time-series selection and significance

Time-series source selection is target-dependent. Rust first scores the
eligible original unary features on the requested backend, retains the strongest
`top_k_features_for_time_series` sources, and expands them in ranked order. The
ordinary feature-prefix budget and seed-ordered unary candidate cap apply before
source selection. `max_time_series_candidates` truncates the generated columns
in that order; downstream continuous candidate caps and screening still apply.
Even retaining every source does not make selection non-adaptive when a later
cap makes its order observable.

## Permutations and bootstrap

Each maxT permutation uses the established deterministic target shuffle and
repeats source selection, expansion, ordered truncation, and downstream
screening. Its null maximum covers the complete resulting candidate family,
not only reported rows or the observed generated columns. Pearson and Spearman
use two-sided extrema; MI and R2 use their existing one-sided rules. Non-finite
scores keep the existing maxT convention, including p=1 for invalid observed
scores. The plus-one p-value uses the result lane's dtype.

Core scores the null family through its existing native prepared execution.
CUDA, ROCm, and Metal score and reduce it on the selected device through bounded
native ranking (`top_k=1`, both directions for signed metrics); missing device
ranking fails closed. Host work owns target shuffling and generated-column
materialization, not GPU null-family scoring. Metal remains fp32-only. No public
API or ABI is added, and fp32/mixed/fp64 storage and arithmetic contracts remain
unchanged.

Bootstrap continues to describe the observed materialized candidate signals.
It resamples their rows with the established candidate-identity stream; it does
not manufacture a new temporal sequence by computing lags after resampling.

## Compiled lifecycle

An artifact retains privately owned original typed inputs, original feature
names, and temporal settings as well as its prepared expanded matrix.
`update_target` prepares a fresh selection/expansion/native execution state,
then commits it with refreshed public generated names and plan metadata. A
failed preparation does not commit the new discovery inputs. Reseeding repeats
this preparation because the seed-ordered unary cap can change the eligible
sources. `random_seed=None` therefore refreshes the family and its public names
before each analysis. After a native commit, the wrapper invalidates old report,
export, and plan state before reading new metadata. A metadata read/conversion
failure retires the wrapper and attempts native closure. A cleanup failure
does not reopen the wrapper or restore stale reports. Ordinary repeated analysis
with a fixed seed reuses the resident observed expansion. No GIL,
thread-affinity, or scheduling changes are part of this behavior.

The existing decision-path metadata conversion now retires its artifact on an
internal dtype-invariant error after discovery has advanced. Restoring advanced
discovery beside old execution, or leaving an open artifact without discovery,
would be inconsistent. The regression injects invalid internal metadata; it
does not establish that supported user input can produce such a variant.

## Focused qualification

`tests/python/test_v1_adaptive_time_series.py` separates an independent scalar
Pearson/maxT oracle from public eager/compiled parity. It covers all three Core
precision profiles, top-k=2 with changing sources, generated-column truncation,
the all-sources-but-truncated case, capped seeded source eligibility, injected
`random_seed=None` reseeds, bootstrap preservation, and invalid-score maxT.

Installed-package invocation:

```bash
GAFIME_TEST_INSTALLED_PACKAGE=1 python -m pytest -q tests/python/test_v1_adaptive_time_series.py
```

`GAFIME_TEMPORAL_TEST_BACKEND=cuda`, `rocm`, or `metal` explicitly selects a
separately provisioned native lane. These are correctness tests, not comparative
performance evidence. Metal's mixed/fp64 cases are explicit unsupported skips,
not successful executions. CPU tests do not qualify device execution.
