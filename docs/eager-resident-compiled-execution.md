# Eager, Resident, and Compiled Continuous Execution

Date: 2026-07-19

GAFIME has three continuous-analysis paths. They share one candidate-planning
and scoring contract, but they do not share Python lifetime or native residency.
Keeping those paths explicit prevents a disabled cache from paying for retained
state and prevents compiled replay from repeating validation or descriptor
uploads that belong at compile time.

## Path Ownership

| path | selection | retained state | intended use |
|---|---|---|---|
| one-shot eager | `GAFIME_V1_ANALYZE_CACHE_SIZE=0`, `keep_in_vram=False`, or a continuous call that bypasses the resident LRU | process-wide immutable payload DSO only | independent calls and small mutable Python inputs |
| resident eager LRU | continuous `GafimeEngine.analyze` with `keep_in_vram=True` and cache capacity above zero | up to the configured number of native artifacts, keyed by configuration, feature names, shape, and fp32 feature content | repeated calls where avoiding matrix upload is worth content hashing |
| explicit compiled | `gafime.compile(...)` or `GafimeEngine.compile(...)` | one caller-owned native matrix, compact plan, backend session, and optional graph/export state | repeated analysis, target replacement, graph replay, and deterministic lifetime control |

The shipped Python boundary converts validated input directly to contiguous
little-endian fp32 bytes and calls the Rust one-shot buffer entrypoint. It does
not construct or retain a Python compiled artifact. Older or custom native
boundaries without the buffer entrypoint retain the nested-row compatibility
fallback. CUDA, ROCm, and Metal payload libraries are process-cached because a
loaded immutable DSO is code, not user matrix or result state.

The resident eager cache is content-aware. It does not assume a mutable list or
NumPy array stayed unchanged merely because its Python object identity stayed
the same. That correctness rule requires an fp32 content scan on every lookup.
For small list inputs, one-shot eager can therefore be faster even though it
recreates native matrix state.

Only the resident path computes those content digests. One-shot and explicit
compile still perform fp32 validation and contiguous conversion, but do not hash
the input they will not look up. Changing `GAFIME_V1_ANALYZE_CACHE_SIZE` to zero
closes and removes existing resident-cache artifacts before the next analysis,
so disabled mode does not leave hidden matrix residency behind.

Explicit compilation avoids that lookup tradeoff. The caller establishes the
artifact lifetime once, and `update_target()` is the only mutation operation.
`close()` immediately drops the backend matrix, compact plans, significance
matrix, cached report handles, and optional graph state. A previously returned
Python report remains readable because it owns its report view independently.

## Compiled Replay Contract

A prepared plan is fully validated when the artifact is built. General
`execute_plan` callers still validate arbitrary plans on every call; compiled
replay uses a trusted internal execute method and does not repeat structural
validation.

Current payloads advertise
`GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL`. Rust requests
`GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL` transiently only for a prepared
execution against a payload that advertises that capability. CUDA and ROCm may
then retain uploaded combo and metric descriptors while their host pointers and
lengths identify the same prepared plan. Metal may retain the corresponding
combo, metric, chunk, and launch-info buffers. A matrix upload or
`update_target()` ends that immutable epoch on every backend, because target
screening may rebuild the higher-order candidate plan.

This flag is an internal optimization contract. It does not change the public
ABI layout, candidate equations, metric definitions, rank order, or result
table format. An older same-ABI payload does not advertise the capability, so
the host strips the optional launch hint and the payload continues to upload
descriptors for every execution.

## Shared Mathematical Invariants

All three paths use the same Rust candidate planner:

- unary candidates are capped with the legacy CPython MT19937 shuffle;
- unary metric strength selects the stable top-feature shortlist;
- the same continued RNG stream shuffles that shortlist before arity 2..5
  planning;
- each arity has its own `max_combinations_per_k` cap;
- seeded tuple order and CPU/GPU tie behavior match the published legacy paths;
- a target update rescans unary candidates and rebuilds the screened plan;
- observed and null MI use the same estimator and backend-specific adaptive-bin
  ceiling;
- all Python integer seed words participate in planning, values through `u64`
  preserve the legacy significance stream, and wider values are deterministically
  reduced for the bounded native significance ABI;
- `random_seed=None` reseeds every explicit-artifact analysis without rebuilding
  or reuploading its matrix;
- representable NaN and infinity values reach the native math unchanged, while a
  finite value that overflows fp32 is rejected consistently on every path;
- maxT exceedance uses the exact `permuted >= observed` relation with no hidden
  epsilon.

`significance_top_n` is independent of
`budget.top_features_for_higher_k`. The first bounds the reported/significance
selection; the second controls which unary features can enter higher-order
candidate generation. Changing one must not silently change the other.

Stability work is also bounded without changing its equations. Repeats of one
skip bootstrap construction. For repeated runs, each sampled feature column is
materialized once per repeat and reused across candidates while preserving the
legacy f64 mean and multiplication order. The cache is capped at 256 MiB across
concurrent repeats; larger shapes use the original scratch path, with bitwise
equivalence covered by a Rust test.

## Legacy Regression Evidence

The bounded A/B gate used OpenML `cpu_act` (8,192 rows, 21 features), Python-list
input, Pearson plus R2, seed 7, arity 1..3, and a 12-feature higher-order
shortlist. This produces 307 candidates and 2,514,944 candidate-row evaluations.
Every comparison required the exact legacy candidate-identity SHA-256, maximum
metric drift at most `5e-6`, and end-to-end speedup at least `1.0x`.

| backend | cache-disabled vs v0.4.7 | compiled vs v0.4.7 | cache-disabled vs `v0.5.0-legacy` | compiled vs `v0.5.0-legacy` | max abs drift |
|---|---:|---:|---:|---:|---:|
| Core | `2.718x` | `6.278x` | `3.607x` | `8.785x` | `2.18e-6` |
| CUDA, RTX 4060 `sm_89` | `1.997x` | `5.856x` | `1.543x` | `4.964x` | `3.28e-7` |
| ROCm, `gfx1150` | `3.140x` | `7.142x` | `1.469x` | `3.347x` | `3.35e-7` |

The `v0.5.0-legacy` tag still declares package version `0.4.7`; its baseline is
identified by tag commit `88c1ef8`, not by package metadata alone. These are
short local regression checks captured before the final compatibility-only
corrections, not publication-quality throughput claims. Their durable value is
the exact candidate identity, bounded numeric drift, and failure-on-regression
thresholds in
`tests/release_measure/perf_08_v047_distribution_ab.py`.

On this screened list workload, the resident eager LRU remained faster than
both legacy baselines but was slower than one-shot eager because it must coerce
and hash mutable input on every lookup. Callers that know they will replay the
same data should use an explicit compiled artifact; GAFIME does not replace a
content check with unsafe object-identity caching.

## CUDA and Windows Language Policy

CUDA 13.3 supports Visual Studio 2026 / MSVC 195x as a Windows host toolchain,
but NVIDIA's Windows table lists CUDA dialects only through C++20 and the CUDA
C++23 language table marks Microsoft Visual Studio unsupported. The repository
therefore keeps CUDA at C++20, including the wheel staging script and OptiX PTX
build. ROCm and Metal remain C++23 where their toolchains support it.

Visual Studio 2026 and MSVC 14.51 improve host C++23 support, but that does not
override NVIDIA's CUDA-host combination matrix. Windows CUDA CI should move to
C++23 only after NVIDIA documents that combination as supported and the native
payload plus standalone ABI smoke pass on the actual toolchain.

References:

- [CUDA 13.3 Windows compiler support](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [CUDA C++ language support](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html)
- [CUDA 13.3 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [Visual Studio 2026 release notes](https://learn.microsoft.com/en-us/visualstudio/releases/2026/release-notes)
