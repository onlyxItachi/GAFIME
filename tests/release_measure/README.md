# GAFIME v1 Release Measurement Suite

These scripts validate the v1 runtime contract from the top-level Python API
down through Rust orchestration, Rust CPU kernels, and optional GPU C ABI
payloads. Targeted gates also preserve public-result compatibility with the
published legacy distributions and host compatibility with older same-ABI GPU
payloads; the removed legacy Python/C++ runtime itself is not built in-tree.

## How To Run

Use a built editable install or put the thin Python package first on
`PYTHONPATH`:

```bash
export PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure
PY=/home/hamza-usta/GAFIME/.venv-release/bin/python

$PY tests/release_measure/contract_00_policy_files.py
$PY tests/release_measure/contract_01_top_level_numpy_parity.py
$PY tests/release_measure/contract_02_feature_generation_reference.py
$PY tests/release_measure/contract_06_release_facing_artifacts.py
$PY tests/release_measure/artifact_01_release_composition.py --scope source-tree
$PY tests/release_measure/v1_architecture_gate.py
```

`backend_01_availability_smoke.py` and
`backend_03_e2e_smoke_per_backend.py` accept a comma-separated
`GAFIME_BACKENDS` selection. They report missing optional payloads as skips only
when the matching `GAFIME_*_V1_LIB` variable is unset and the request reaches
that missing-payload boundary. A configured payload error or a selection that
completes no backend exits nonzero.

When `PYTHONPATH` points at this checkout, rebuild and copy the current native
extension first (`cargo build --release -p gafime-py`, then install
`target/release/libgafime_py.so` under the matching CPython extension filename in
`python/gafime/`). The full
ordered sequence is in `docs/cuda-template-kernel-hardening.md`.

When native GPU payloads are available:

```bash
export GAFIME_CUDA_V1_LIB="$PWD/build/cuda-template-hardening-both/libgafime_cuda_v1.so"
export GAFIME_CUDA_RT_V1_LIB="$PWD/build/cuda-template-hardening-both/libgafime_cuda_v1_rt.so"
export GAFIME_ROCM_V1_LIB="$PWD/build/rocm-template-hardening-default/libgafime_rocm_v1.so"
export GAFIME_CUDA_ABI_SMOKE=/tmp/cuda_v1_abi_smoke
export GAFIME_CUDA_RT_ABI_SMOKE=/tmp/cuda_v1_abi_smoke_rt
export GAFIME_ROCM_ABI_SMOKE=/tmp/rocm_v1_abi_smoke
python3 tests/release_measure/v1_architecture_gate.py --include-gpu
python3 tests/release_measure/installed_payload_smoke.py \
  --backend cuda --source-root "$PWD" --execute-profiles
python3 tests/release_measure/installed_payload_smoke.py \
  --backend rocm --source-root "$PWD" --execute-profiles
```

Build both smoke binaries from the current tree before that command; the exact
payload and smoke build sequence is in
`docs/cuda-template-kernel-hardening.md`.

Generated CUDA/HIP artifacts can also be inspected without reserving a GPU:

```bash
python3 tests/release_measure/gpu_static_kernel_report.py \
  --cuda-lib build/cuda-template-hardening-both/libgafime_cuda_v1.so \
  --hip-lib build/rocm-template-hardening-default/libgafime_rocm_v1.so \
  --hip-target gfx1150 \
  --require-template-matrix \
  --require-precision-profiles \
  --require-topk-split \
  --require-no-spills
```

`--require-template-matrix` accepts either exact arity instantiations or the
smaller shared runtime-arity device body for every required metric/bin route.
The latter is coupled to the independent ABI CTests that physically execute
arities 1 through 5; restoring duplicate CUDA kernels merely to satisfy a name
census is forbidden.

The macOS workflow runs the Metal behavioral gates with the built payload:

```bash
export GAFIME_METAL_V1_LIB=build/metal-cmake/libgafime_metal_v1.dylib
export GAFIME_METAL_V1_METALLIB=build/metal-cmake/gafime_metal_v1.metallib
export GAFIME_METAL_PARITY_TOLERANCE=0.00005
cargo test -p gafime-gpu-sys \
  metal_device_topk_covers_split_directions_ties_and_large_k_when_available \
  -- --nocapture
cargo test -p gafime-gpu-sys \
  metal_continuous_metrics_match_cpu_on_high_dynamic_and_nonfinite_inputs_when_available \
  -- --nocapture
```

Telemetry helpers write schema `gafime.telemetry.v0.5.0-rc1` records until the
next telemetry schema bump. That schema name is historical; the runtime being
measured here is v1.

## Active Scripts

### contract

| script | validates | needs |
|---|---|---|
| `contract_00_policy_files.py` | contract docs, agent docs, compiler/safety policy text | CPU |
| `contract_01_top_level_numpy_parity.py` | top-level API bit parity against NumPy reference for base metrics | CPU |
| `contract_02_feature_generation_reference.py` | continuous, compile, time-series, decision-path, and dataload reference checks | CPU |
| `contract_03_family_metric_backend_surface.py` | all configured backends across continuous, time-series, decision-path, and all metric ids | CPU/GPU |
| `contract_04_adaptive_mi_quantization.py` | adaptive MI template resolution and ranking stability against a large-sample reference | CPU |
| `contract_06_release_facing_artifacts.py` | current README/release runbook links, documented CLI parsing, support-skill API guidance, deterministic v1 practice notebook, and generated pipeline syntax/default dependencies | CPU |
| `artifact_01_release_composition.py --scope source-tree` | manifest-owned release identities, precision profiles, ABI, platforms, artifact names, workflow globs, optional extras, and generated matrix | CPU |
| `gafime-gpu-sys::tests::metal::metal_continuous_metrics_match_cpu_on_high_dynamic_and_nonfinite_inputs_when_available` | current Rust host ABI 1.0 execution against both older same-ABI Metal payloads, including CPU-oracle parity and immutable-protocol capability negotiation, without labeling the legacy arithmetic as a canonical precision profile | Apple GPU plus older payload |
| `v1_architecture_gate.py` | package layout, forbidden legacy imports, native report view, CPU/GPU payload structure | CPU/GPU |
| `installed_wheel_smoke.py` | clean installed-package import, PyO3 symbols, typed Arrow ingest, all three Core profiles, adversarial fp64 preservation, significance identity, and eager/compiled value parity | installed Core wheel |
| `installed_payload_smoke.py` | payload separation, RT exclusion, exact additive precision ABI exports, optional hash-bound CUDA/ROCm device-code evidence, exact capability masks, Metal fp32-only behavior, and optional physical execution of every supported profile | installed Core/payload pair; device only for `--execute-profiles` |

### decision_path

| script | measures | needs |
|---|---|---|
| `dp_02_openml_tour_logged.py` | baseline vs assisted lift across datasets | CPU |
| `dp_03_method_effect_gated_soft.py` | hard/gated path strategy lift comparisons | CPU |
| `dp_05_dataset_structure_map.py` | where decision-path lift appears by dataset structure | CPU |
| `dp_06_depth_rounds_sweep.py` | depth/rounds lift vs cost | CPU |
| `dp_07_boosting_residual_reduction.py` | boosting residual reduction and path growth | CPU |
| `dp_08_leakage_safety.py` | train-mined vs leaked feature generation gap | CPU |

### compile

| script | validates | needs |
|---|---|---|
| `compile_01_plan_correctness.py` | native compile artifact and plan shape | CPU |
| `compile_02_compiled_vs_eager.py` | one-shot, resident-cache, and explicit-compiled parity across first/repeat/target-update runs, including non-finite inputs, large seeds, warnings, significance, and final decisions; timing is context only | CPU/GPU |

### graph

| script | validates | needs |
|---|---|---|
| `graph_01_replay_parity.py` | graph replay equals plain launch within approved tolerance | GPU |
| `graph_02_launch_shaping_timing.py` | graph vs plain launch latency | GPU |

### backends

| script | validates | needs |
|---|---|---|
| `backend_01_availability_smoke.py` | public API backend resolution and explicit errors | CPU/GPU |
| `backend_02_cross_backend_parity.py` | core vs CUDA vs ROCm numerical parity | GPU |
| `backend_03_e2e_smoke_per_backend.py` | per-backend end-to-end smoke through top-level API | CPU/GPU |

### performance

| script | measures | needs |
|---|---|---|
| `perf_01_residency_session_benefit.py` | resident compile/session reuse vs fresh analyze | CPU/GPU |
| `perf_02_metric_cache_benefit.py` | metric-cache hit rate and counters | GPU |
| `perf_04_cpu_native_kernels.py` | CPU SIMD dispatch, column layout, and scratch-reuse guardrails | CPU |
| `gpu_static_kernel_report.py` | CUDA SASS and HIP code-object size, register, shared/LDS, spill, exact fp32/mixed/fp64 device specialization, hash-bound exact-wheel evidence, and top-k topology checks | CUDA/HIP toolchains, no GPU |
| `perf_06_gpu_mi_specializations.py` | resident MI throughput by candidate count, candidate-sample pairs, and bins | CUDA/HIP GPU |
| `perf_07_rocm_mi_wave_ab.py` | provenance-checked, numerically guarded interleaved HIP high-bin A/B with control normalization and JSON output | HIP GPU and two payload builds |
| `perf_08_v047_distribution_ab.py` | isolated v0.4.7 or `v0.5.0-legacy` Core/CUDA/ROCm/Metal distributions vs current one-shot, eager-cache, and compiled paths; report order, tuple/family identity, candidate-id stability, warnings, deterministic decisions, optional stochastic snapshots, numeric/performance thresholds, and provenance. Cross-distribution stochastic values are recorded but not value-gated because legacy candidate-wise permutation streams and current family-wise maxT are different statistical methods; current one-shot/resident/compiled stochastic parity remains strict. | CPU/GPU, scikit-learn/OpenML preparation |
| `perf_09_interaction_diagnostics_overhead.py` | public safe-path one-shot and resident timing distributions for base/candidate diagnostic A/B; also validates candidate count, availability, and zero false-positive diagnostics | CPU/GPU and separate base/candidate installs |
| `perf_10_cpu_covariance_finite_pass.py` | public resident Core Pearson-only and Pearson+R2 timing distributions for the finite-input SIMD covariance A/B | CPU with NumPy input |
| `perf_11_cpu_mi_histogram.py` | public resident Core fixed-bin MI timing distributions, kept separate from the ignored internal histogram/helper microbenchmark | CPU with NumPy input |
| `perf_12_precision_profiles.py` | historical precision-profile measurements only; its fixed-order, pre-probed, three-repeat output is explicitly provisional and invalid for comparison | CPU/GPU; do not use for release performance claims |
| `core_precision_native_benchmark.rs` + `run_core_precision_native_benchmark.py` | supplemental intentionally single-core direct leaf-kernel SIMD/code-generation diagnostic; never Core product throughput | exact Core rlib and wheel |
| `core_precision_production_benchmark.rs` + `run_core_precision_production_benchmark.py` | real planner/protocol -> resident matrix -> `PrecisionComputeBackend` -> ranked-result Core production timing, with default-worker primary evidence and bounded 1/2/4 scaling diagnostics | exact Core/orchestrator/types/Rayon rlibs and wheel |
| `cold_lifecycle.py` | fresh-process canonical ABI phase timing for import, discovery, dynamic load, runtime initialization, route/capability query, allocation, upload, planning, execution, typed result access, cleanup, and an honestly combined exit residual | exact installed payload/wheel and physical hardware |
| `perf_13_precision_profiles.py` | public one-shot/resident/compiled/graph precision measurements plus a diagnostic fresh-worker public cold envelope across all six profile orders, both source-dtype policies, multiple workloads, raw distributions, bootstrap intervals, order sensitivity, and randomized A/B plus B/A provenance; canonical 30-sample cold lifecycle evidence remains owned by `cold_lifecycle.py` | installed Core/payload wheels and physical hardware |

### Precision-profile performance evidence

`perf_13_precision_profiles.py` is the only precision-profile public benchmark
accepted for new release comparisons, while `cold_lifecycle.py` is the
required separate canonical-ABI cold layer. Its comparative mode requires four
hash-bound artifacts: A/B and reversed B/A, each with at least 30 fresh-process
samples per profile, a clean exact product commit, and payload bytes verified
against the exact wheel member. The perf13 driver imports no
GAFIME or NumPy code. Its public cold-envelope samples run one profile per
fresh worker; public trials run in a
fresh worker for each backend, workload, input policy, profile-order block,
variant, and A/B block. Supplying all three profiles exercises all six possible
orders without converting the requested order through a set.

The cold comparison revalidates the raw count for every canonical phase in
both variants; the top-level repetition declaration alone is not evidence.
Loader-constructor registration and the process-exit residual use their
explicit combined timing buckets. Metal runtime/context initialization is the
only current diagnostic-only phase because Metal exposes no separate public C
runtime operation; CUDA and ROCm must provide the complete measured phase.

The hosted Metal Beast workflow requires an explicit full
`expected_candidate_sha`, verifies that the checkout and live PR #70 head match
it before building, and checks the live head again around final evidence. Its
canonical cold lane runs this same current tracked harness against the exact
wheel-extracted baseline and candidate dylibs in four isolated A/B plus B/A
cells. It uploads all raw samples and the hash-bound comparison manifest and
fails unless `valid_for_canonical_cold_lifecycle_claims` is true. The typed
historical ABI surface is accepted for the baseline only; the candidate must
resolve the generic numeric-route surface.

The public cold envelope never invents sub-times for boundaries that the
top-level compile/analyze API combines. Phase-by-phase cold claims come only
from `cold_lifecycle.py`, which times the canonical generic ABI directly and
labels loader-constructor registration and process-exit residuals as combined
where the platform exposes no safe narrower boundary. The native arithmetic
helpers remain the third, device-event or Rust-kernel layer.

Every real run must pass `--native-evidence PATH`. This is a machine-readable
manifest, not a source of invented timings. An E2E-only run may explicitly
declare that native evidence was not collected:

```json
{
  "schema": "gafime.precision-profile-native-evidence.v1",
  "status": "not_collected",
  "arithmetic_claims_valid": false,
  "artifacts": []
}
```

For arithmetic or kernel claims, use `status: "validated"`, set
`arithmetic_claims_valid` to `true`, and list each independently collected
native decomposition/device-event or Core production-executor artifact with its
variant, backend, kind, path, and SHA-256. The harness verifies every listed
file and records only those identities; it never creates native timings from
public wall-clock measurements. Each backend artifact must use its
backend-specific schema, carry a full source commit and real file identities,
and cover every supported profile and required decomposition operation. CUDA
and ROCm artifacts must also declare and record all six profile orders in a
fixed, predeclared minimum of 30 complete cycles. The 30-cycle default is not a
staged target: release evidence may not stop early when an interval happens to
look clean or add cycles conditionally after seeing an inconclusive result.
Artifacts serialize the exact ordered sequence of six
permutations in `profile_order_cycles`, cross-check it against every record's
absolute order index, and use at least two distinct cycle sequences with no
adjacent exact reuse. Device-timed records must identify their synchronized
event clock and timing boundary. A SHA-256 over arbitrary JSON is rejected even
when the manifest hash matches. A validated claim must provide a
schema-validated artifact for
every requested variant/backend pair (`core_production_executor` or
`native_decomposition` for Core; `cuda_events`/`rocm_events`/`metal_events`,
`device_events`, or `native_decomposition` for the corresponding GPU backend).

CUDA and ROCm timing artifacts must authenticate a discarded
`calibration_prepass`: it is performed in the canonical `fp32,mixed,fp64`
order, uses the shared fixed-loop cache, reports positive discarded
record/sample counts and required cache-key coverage, and is excluded from
`profile_order_cycles`. Their clock boundary must say exactly that
measurement-before is captured after this discarded pass and before the
randomized cycles, while measurement-after follows cycle collection and
record verification. CUDA and ROCm A/B native cells are comparable only when
their `loop_count_per_sample` values are identical; mismatched counts are
classified as incomparable and fail the native claim gate. Production Core
cells preserve each fresh child's fixed calibrated count and compare normalized
one-call samples; both raw calibrated-region durations and counts remain in
the artifact, and no sample is rescaled after collection.

The CUDA helper is benchmark-only and is never part of the payload or CTest:

```bash
cmake -S src/cuda -B build/cuda-native-benchmark \
  -DGAFIME_CUDA_BUILD_BENCHMARKS=ON \
  -DGAFIME_CUDA_BENCHMARK_PRODUCT_ROOT=/clean/product-worktree
cmake --build build/cuda-native-benchmark \
  --target gafime_cuda_precision_native_timing \
           gafime_cuda_precision_native_timing_canonical \
           gafime_cuda_precision_native_timing_host \
           gafime_abi_1_1_c_consumer_cuda
python tests/release_measure/canonical_abi_lifecycle_evidence.py \
  --backend cuda \
  --consumer build/cuda-native-benchmark/abi_consumers/gafime_abi_1_1_c_consumer_cuda \
  --payload "$GAFIME_CUDA_V1_LIB" \
  --wheel /artifacts/gafime-cuda.whl \
  --source-root "$PWD" \
  --output /artifacts/cuda-canonical.json
build/cuda-native-benchmark/gafime_cuda_precision_native_timing \
  --workload release --rows 4096 --features 8 --arity 1 --mi-bins 32 \
  --profiles fp32,mixed,fp64 \
  --payload "$GAFIME_CUDA_V1_LIB" --wheel /artifacts/gafime-cuda.whl \
  --source-root "$PWD" --harness-source-root "$PWD" \
  --canonical-evidence /artifacts/cuda-canonical.json \
  --json /artifacts/cuda-native.json
```

Configure those targets once per A/B variant from the same final harness
checkout, changing only `GAFIME_CUDA_BENCHMARK_PRODUCT_ROOT`. The direct target
links `precision_kernels.cu` and its matching header from that clean product
tree, embeds their commit and SHA-256 identities, and refuses to run when those
compile-time identities do not match `--source-root`. Baseline and candidate
direct binaries are therefore expected to differ. Canonical and host control
binaries must instead be byte-identical across variants when compiler and
flags match; all lanes retain the same tracked harness source/blob identity.

The helper auto-detects the exact payload surface and emits the canonical
`abi_surface` value `precision-typed-v1.1` for the historical pre-freeze PR-70
typed baseline payload or
`numeric-route-v2` for the current generic payload. Its
`canonical_payload_api` records are host steady-clock measurements bracketed
by device synchronization; they include ABI validation, payload-private
launches, device completion, and caller-owned result visibility, so they are
payload-boundary arithmetic evidence rather than pure-kernel timings. The
existing `supplemental_internal_kernel` records remain a separate lane, and
host-only controls use the predeclared `supplemental_host_phase` lane. The
helper repeats the complete six-order CUDA set for 30 fixed cycles by default;
`order_repetitions` and raw per-order records are retained for contamination
analysis. Each cycle starts from the canonical set and performs a deterministic
seeded reshuffle; the emitted `order_schedule` marker and `order_index` preserve
that cycle/slot provenance, and an exact shuffle collision is rotated rather
than reusing the preceding temporal schedule. The payload lane emits separate
actual `payload_execute` records for
Pearson, Spearman, mutual information, and R2; neither lane may be pooled
across ABI surfaces.
When one common helper runs both variants, pass the product checkout with
`--source-root` and the clean common helper checkout with
`--harness-source-root`; the JSON keeps product commit/tree identity separate
from harness commit/tree/blob identity.

The CUDA direct helper is the only native timing target compiled by NVCC and
linked with `precision_kernels.cu`.  The canonical and host targets include the
same tracked harness through `cuda_precision_native_timing_host.cpp` and are
ordinary C++ executables.  Their section/symbol gate must therefore find no
`.nv_fatbin`, `__cudaRegister*`, cubin, or device-module registration.  The
canonical target loads the exact runtime payload and resolves its ABI inside
that process. Direct and host targets do not load it: they report unresolved
in-process `canonical_payload_resolution` and `payload_not_loaded=true`, then
bind the separately captured canonical lifecycle through
`canonical_payload_lifecycle` with `binding=external_canonical_evidence`,
`status=validated`, and an authenticated `path`/`sha256` pair. This external
binding is evidence provenance, not payload loading in the direct or host
process.

It preserves the caller's profile list, executes every permutation (all six
orders for three profiles), and records ingest conversion, planning,
allocation, H2D, supplemental target-stat preparation, feature-stat
preparation, candidate materialization, each metric, actual top-k selection,
selected-row gather, D2H, and report construction with at least ten warmups and
thirty raw repetitions. The two stat-preparation records use synchronized CUDA
events and stay in the supplemental/direct lane; payload-private preparation
cannot be separated from `payload_execute` by this harness. Direct
internal-kernel timings
are explicitly supplemental; they require separate canonical stable-ABI
payload lifecycle evidence, with matching payload/wheel identities, before they
can support an arithmetic claim. The lifecycle producer rejects dirty source
trees and payload bytes that do not exactly match the wheel member. It accepts
only the structured success marker emitted after the standalone public-header C
consumer has exercised route enumeration, typed allocation/upload/target
replacement, execution, both memory forecasts, significance, diagnostics, and
free for every advertised route; symbol resolution alone is not evidence. The
Supplemental Core leaf-kernel evidence comes from the standalone tracked source
`core_precision_native_benchmark.rs`. Compile and run its benchmark `main`
only through `run_core_precision_native_benchmark.py`, which supplies one
common harness source directly to `rustc` and links it with an exact
`--extern gafime_cpu=<product rlib>` argument. Cargo validation includes the
same source as a module only to execute its methodology adversarial tests; it
never invokes benchmark `main` or produces performance evidence. A baseline's
product-local benchmark source therefore cannot affect comparative evidence.
For example:

```text
python tests/release_measure/run_core_precision_native_benchmark.py \
  --product-source-root /clean/baseline \
  --harness-source-root /clean/current-harness \
  --product-rlib /clean/baseline-target/release/deps/libgafime_cpu-....rlib \
  --wheel /artifacts/baseline/gafime.whl \
  --binary /evidence/bin/core-baseline-common-f64 \
  --output /evidence/core-baseline-common-f64.json \
  --input-policy common-f64 --toolchain 1.97.1
```

Run the same command with `--input-policy native` for the native-source lane.
The common-f64 lane derives fp32/mixed vectors from one f64 source; the native
lane constructs f32 sources for fp32/mixed and an independent f64 source for
fp64. Both exclude input construction from their native arithmetic timers.

The helper writes `gafime.core-leaf-kernel-diagnostic.v1` and emits every raw
duration, median, MAD, p05, p95, bootstrap interval, exact input hash, and
source/compiler/runtime-command/affinity provenance. Its recorded seed
randomizes each of twenty complete balanced cycles. Every cycle contains the
full cross product of all six profile orders and all four metric rotations, so
the run has 480 blocks and each profile/metric cell has 480 observations, 160
at each profile position. Metric ordinal is therefore crossed with, rather
than inferred from, profile position.

Order inference uses 200,000 resamples of complete 24-block balanced cycles;
all four rotations from a sampled cycle stay together, preserving macro
temporal and thermal correlation. It reports all three signed position-pair
contrasts in each of the twelve profile/metric cells. Two-sided Bonferroni
intervals cover all 36 inspected contrasts with at least 95 percent familywise
confidence. A cell is contaminated only when a corrected interval lies wholly
beyond plus or minus one percent. The artifact is clean only when every
corrected interval lies wholly inside that band. Any overlap with a boundary
is inconclusive, exits nonzero, and requires a rerun or investigation; absence
of evidence is never reported as a pass.

Each sample also receives at least ten untimed same-cell preconditions and at
least 100 ms of same-cell stabilization to normalize code, input-cache,
allocator, CPU-frequency, and thermal state. Calibration targets 200 ms once
per cell and then fixes the loop count; it never rescales a measured sample.
Every raw measured region must still reach 100 ms or the artifact exits
nonzero. A portable, reliable per-region involuntary-context-switch counter is
not available across supported platforms, so scheduler effects are retained
instead of selectively deleting samples and are handled by the long regions
and conservative whole-cycle intervals.

This is deliberately **supplemental single-core leaf-kernel diagnostic
evidence**, not GAFIME Core throughput evidence. It calls metric kernels
directly and does not construct a planner/protocol, resident
`CpuPrecisionMatrix`, `PrecisionComputeBackend`, candidate-level Rayon work,
or a ranked typed result. It may intentionally use one CPU to inspect SIMD and
code generation; it cannot satisfy a Core product-throughput or release
comparison claim.

Core product-throughput evidence instead comes from the tracked
`core_precision_production_benchmark.rs` child and
`run_core_precision_production_benchmark.py` runner. The child links exact
Core, orchestrator, types, and Rayon rlibs and measures the real
`planner/protocol -> CpuPrecisionMatrix -> PrecisionComputeBackend -> ranked
typed result` surface for unary candidate plans in fp32, mixed, and fp64 across Pearson, Spearman, MI,
and R2. It records latency, medium, and kernel-dominant workloads under both
common-f64 and native-source policies. Each cell runs in a fresh child process;
the primary result is explicitly labeled
`primary_default_worker_production_result`, while attainable 1/2/4 worker
cells are separately labeled `thread_scaling_diagnostic`. The runner records
the complete allowed affinity mask, requested/effective Rayon workers,
pool-start construction evidence (not candidate-work participation), logical
and observable physical CPU counts, and before/after governor/clock/power
state. If the allowed CPU set is smaller than 2 or 4, that scaling label is
skipped with an explicit bound reason rather than oversubscribing the CPU set.
On Linux, every dedicated Rayon worker's OS TID is sampled through `/proc`
before and after the real production measurement. Stable evidence requires a
positive CPU-tick delta for every effective worker. Other platforms record the
observation as unavailable and cannot self-promote that cell to stable
evidence; the cfg(test) production-executor topology test remains the stronger
semantic proof that candidate work itself reached multiple workers.

For example, an evidence producer supplies exact clean-product rlibs and a
wheel (the full default matrix is release sampling, so do not run it casually):

```text
python tests/release_measure/run_core_precision_production_benchmark.py \
  --product-source-root /clean/candidate \
  --harness-source-root /clean/current-harness \
  --product-rlib /clean/candidate-target/release/deps/libgafime_cpu-....rlib \
  --orchestrator-rlib /clean/candidate-target/release/deps/libgafime_orchestrator-....rlib \
  --types-rlib /clean/candidate-target/release/deps/libgafime_types-....rlib \
  --rayon-rlib /clean/candidate-target/release/deps/librayon-....rlib \
  --dependency-dir /clean/candidate-target/release/deps \
  --wheel /artifacts/candidate/gafime.whl \
  --binary /evidence/bin/core-production-candidate \
  --output /evidence/core-production-candidate-ab0.json \
  --variant candidate --ab-block 0 \
  --variant-sequence baseline,candidate
```

Every A/B block uses a persisted seeded schedule with balanced assignments of
all six profile orders. Baseline and candidate use the exact same schedule
inside a block, while the reversed block uses a different seed, schedule hash,
and order. The runner records its PID, requires every fresh child PID to be
distinct, removes inherited `RAYON_NUM_THREADS`, and preserves a canonical
nonempty view of PATH and relevant thread/runtime variables.

One artifact is raw integrity and production-sampling evidence only; it always
sets both `performance_claim_ready=false` and
`comparative_performance_claim_ready=false`, while
`raw_measurement_claim_ready` reports only whether its own cells met the raw
contract. A comparative Core claim is
made only by a later perf13 aggregation that authenticates distinct baseline
and candidate products in both `baseline,candidate` and `candidate,baseline`
blocks. The aggregate retains every raw child duration and per-child
provenance. Perf13 reopens each raw child, rechecks its SHA-256 and byte length,
and compares the authenticated JSON against the aggregate record with only
derived distribution fields excluded. It independently re-derives the repeat
count, positive loop count, 100 ms raw floor, `raw/loop` normalization, target
and observed minimum. It never lets the leaf diagnostic satisfy the production
claim.
For PR #70, the before-product is the recorded pre-repair precision head
`d52199f44aa80ab8ef50c18db95dd1630961cdaf`. The PR base on `main` does not
contain the precision executor API consumed by this common harness and is not
substituted as a fictitious A/B baseline. The workflow separately verifies
that the PR still targets an unchanged `main` base while binding performance
to the exact before-fix precision commit.

The runner and helper jointly require clean product and harness trees and bind
the report to both full commits and Git tree IDs, the tracked Rust harness
source blob and SHA-256, the separately authenticated Python runner blob and
SHA-256, the exact
compiler argument vector, normalized Rust toolchain/edition/codegen flags,
linked rlib, compiled executable, Core wheel, and
Python executable. The source, runner and rlib identities are embedded at
compile time; a fixed-width SHA-256 of the exact compiler argument vector is
embedded instead of its variable-length path strings so evidence paths cannot
change hot-function alignment. All identities are checked again by the
executable. Observable CPU frequency
policy fields, governors and safe platform power profiles are captured before
and after; unavailable power state is explicitly reported as unobservable.
Baseline and candidate runs must use the same harness commit, Rust source blob,
and Python runner blob even though their product commits, rlibs, wheels, and
benchmark binaries differ. The 100 ms hard raw-region floor and 200 ms
calibration target are reported as `target_region_ns` and
`calibration_target_region_ns`; they are independently validated from perf13's
public sample-region gate. Input generation, planner/protocol construction,
and resident-matrix construction happen before the timed region and are
reported separately. The timed region includes the production executor's
candidate interaction/scoring work plus typed ranked-result allocation and
materialization; it does not claim Python public-report construction.
After timing, the child captures an untimed complete ordered snapshot of combo
indices, ranks, families, candidate IDs, row flags, result-table flags, ordered
metric IDs, and every visible metric value. A/B and B/A validation requires
exact structural metadata, bit-exact fp32 and mutual-information values, and
exact value classification. Other finite mixed values use an absolute-only
`1e-12` tolerance; other finite fp64 values use an absolute-only `2e-12`
tolerance. The timed black-box digests authenticate the dtype, dimensions,
result flags, metric ordering, complete structural snapshot, and every visible
metric bit; both validators additionally require the emitted text and
classification to be exact derivations of those bits. Thread
scaling tables derive `speedup(N) = T1/TN` and efficiency `speedup(N)/N` for
each variant; these remain diagnostics separate from the primary default-worker
product result. An interval overlapping the one-percent regression margin is
not stable-release-ready. A lower 95 percent bound above one percent triggers
investigation, a lower bound above three percent is a hard blocker, and an upper
bound at or below one percent is clean even when the interval crosses zero.
The stable workflow runs only on the pinned
`self-hosted,linux,x64,gafime-core-stable` runner label; ordinary PR
comparisons remain informational on GitHub-hosted Ubuntu. The informational
workflow's declared reduced workload/input-policy matrix is validated exactly
as raw diagnostic evidence, while all public release/comparative-claim booleans
remain false; stable mode requires the complete canonical matrix.

Metal event evidence is produced by the test-only
`gafime_metal_precision_native_timing` CMake target. It records allocation,
unified-memory/host upload, descriptor planning, Pearson, R2, MI, cached-rank
Spearman, top-k/gather, result readback, and report-construction samples. Device
records use `MTLCommandBuffer.GPUStartTime`/`GPUEndTime` only after `commit` and
`waitUntilCompleted`; synchronized host wall time is retained separately. A
validated `metal_events` artifact requires 10 or more warmups, at least 30 raw
samples per record, complete Metal GPU timestamps, the genuine fp32 route, and
SHA-256 identities for the benchmark source/binary, shader/metallib, payload,
wheel, and exact source commit. It also records a verified-clean source tree,
the selected `common-f64` or native-fp32 source policy, and exact source plus
execution matrix/target identities. Hosted comparative evidence runs both
policies through both A/B and reversed B/A blocks, producing eight fresh native
helper artifacts rather than letting input conversion contaminate arithmetic
claims. The direct
metallib event records are explicitly supplemental: the same artifact must also contain
`canonical_payload_records` from the exact wheel-extracted Metal dylib, loaded
with `dlopen` and exercised through ABI 1.1 route enumeration, matrix
allocation/upload, execute, and free. It resolves the complete canonical ABI
1.1 operation set, including target update, forecasts, significance, and
diagnostics, before recording lifecycle evidence. The canonical lane uses
synchronous host timing because the stable ABI does not expose the payload's private command
buffer; its lifecycle and symbol set are hash-bound in
`canonical_payload_lifecycle`. Candidate interaction materialization is fused
into each shipped Metal metric kernel and is labeled as fused rather than given
an invented standalone duration.

`.github/workflows/metal_beast_benchmark.yml` builds this helper outside the
source checkout, validates its hash-bound JSON, constructs the native-evidence
manifest, and then runs this perf13 public harness from a fresh installed-wheel
environment. Smoke/beast/absurd are bounded hosted presets; subset presets keep
their claim-gate limitations in the raw JSON and are never described as the
complete release workload matrix.

The complete release matrix keeps every workload containing Spearman at or
below 4,096 rows, the current Metal target-rank-cache boundary. Above that
boundary the current pairwise rank construction is not a practical hosted
release-timing workload. The 65,536-row `large-kernel` case still measures
Pearson, mutual information, and R2; Spearman remains covered by the latency,
mixed-overhead, metric-specific, all-metrics, and arity 3--5 cases. This is an
explicit benchmark-capacity bound, not a claim of a new ranking algorithm or
improved large-row Spearman support.

The canonical run uses five complete six-order cycles, 10 untimed warmups, at
least 30 recorded repetitions, automatic loop scaling to a 100 ms sampled
region, and emits every raw duration plus median, MAD, p05, p95, and a bootstrap
median confidence interval. The CLI defaults to five order repetitions and
rejects smaller release-claim schedules:

Run it from a clean source checkout (keep build products and the output JSON
outside that checkout); the source commit and wheel must describe the same
package contents.

```bash
PY=.venv-release/bin/python
$PY tests/release_measure/perf_13_precision_profiles.py \
  --source-root current="$PWD" \
  --wheel current=/artifacts/gafime-core.whl \
  --wheel current=/artifacts/gafime-cuda.whl \
  --wheel current=/artifacts/gafime-rocm.whl \
  --native-evidence /artifacts/native-evidence.json \
  --backend core --backend cuda --backend rocm \
  --profile fp32,mixed,fp64 \
  --order-repetitions 5 \
  --workload release \
  --input-policy common-f64,native \
  --output precision-profile-perf-v2.json
```

The public matrix includes safe continuous-family arity-3, arity-4, and
arity-5 workloads. Generated feature families are not claimed by perf13: the
current public benchmark configuration deliberately sets
`max_generated_features=0`, so generated-family performance must not be
inferred from these measurements.

For an exact-head comparison, use independent installs and bind each result to
its source tree, wheel, and native manifest. The A/B environments execute the
same frozen perf13 driver/worker script; the baseline source tree does not need
to contain perf13 (8df predates it):

```bash
$PY tests/release_measure/perf_13_precision_profiles.py \
  --variant baseline=/opt/gafime-baseline/bin/python \
  --variant candidate=/opt/gafime-candidate/bin/python \
  --source-root baseline=/src/gafime-baseline \
  --source-root candidate=/src/gafime-candidate \
  --wheel baseline=/artifacts/gafime-core-baseline.whl \
  --wheel baseline=/artifacts/gafime-cuda-baseline.whl \
  --wheel candidate=/artifacts/gafime-core-candidate.whl \
  --wheel candidate=/artifacts/gafime-cuda-candidate.whl \
  --native-evidence baseline=/artifacts/native-evidence-baseline.json \
  --native-evidence candidate=/artifacts/native-evidence-candidate.json \
  --backend cuda --profile fp32,mixed,fp64 \
  --order-repetitions 5 \
  --workload release --input-policy common-f64,native \
  --output precision-profile-cuda-ab.json
```

The JSON calls its rate “candidate-sample pairs per second for the configured
metric set”; it never labels a four-metric public call as generic GEval/s. Only
the public analyze/replay call is inside the timed region. Candidate-count,
finite-value, and graph-replay validation runs after the clock stops. Compiled
timing includes artifact replay only; compilation/planning is outside that
public timing. Resident timing includes the warmed public resident-cache hit,
including input coercion/ownership checks, digest hashing, cache lookup,
execution, and report materialization. It is not a pure resident-device timer.
One-shot uses a zero-capacity analysis cache. Graph timing is replay-only and
is unsupported by contract on Core/Metal. The JSON names public and native
measurement categories separately and lists phases that cannot be observed
independently.

Two-variant runs use the same workload, surface, input-policy, profile-order,
and A/B block keys for baseline and candidate (the method is suitable for an
8df baseline versus the current head). Every matched cell reports a bootstrap
confidence interval for the candidate-minus-baseline median delta. The point
estimate is descriptive: the confidence bounds decide the gate. A lower bound
above one percent triggers investigation, a lower bound above three percent is
a hard blocker, an upper bound at or below one percent is clean even if the
interval crosses zero, and an interval overlapping one percent is inconclusive.
Because
each cell is measured in separate fresh workers, baseline and candidate raw
durations are resampled independently; the artifact never claims paired
observations. The perf13 cold summaries include the overall clean interval and
every phase with an actually observed duration, but are order-contamination
diagnostics only. The canonical 30-sample lifecycle distributions must be
produced by `cold_lifecycle.py`. Native comparisons retain workload,
input identity/policy, order index/order, clock, synchronization boundary, and
all repeated records instead of overwriting same-operation rows. Their key also
retains A/B block, variant sequence, canonical-payload versus direct-kernel
category, and comparability declaration so wrapper and kernel lanes cannot be
collapsed into one number. GPU native artifacts also retain the exact runtime
argument vector and a validated process-affinity identity. Comparative checks
normalize only authenticated per-variant paths and schedule values; workload,
timing, common-harness, and all other arguments must remain exactly equal.
CUDA and ROCm additionally require at least 10 iterations and 100 ms of
same-cell untimed preconditioning, bounded device-event batches, and one fixed
cached loop count per semantic cell across all six-order cycles. Per-sample
loop rescaling is invalid evidence.

Every CUDA/ROCm native timing artifact and every record must declare the
producer's 5,000 us sampled-region target. The record target must equal the
artifact target, `sample_region_target_met` must be true, the reported minimum
must match the raw minimum, and every raw region must reach the declared target.
Every normalized `samples_us` value must also match `raw_samples_us` divided by
the record's one fixed loop count within tight floating-point tolerance.
The 5 ms floor is the current synchronized CUDA/HIP event-region stability
contract; it is distinct from both the 100 ms untimed same-cell precondition
and perf13's 100 ms public wall-clock sampled-region floor. Missing metadata or
a nominal 1 us region, inconsistent normalization, a missing cycle schedule, or
an exactly reused schedule across all cycles fails closed.

Metal native timing enforces the same 5 ms measured-region floor. It calibrates
the one fixed loop count for each cell against a 20 ms target, providing
fourfold headroom for post-calibration host or GPU clock ramp while retaining
every measured region's fail-closed 5 ms check. It never rescales or filters an
individual measured sample.

GPU-native order inference keeps each 30-timing record as one clustered order
assignment and resamples complete six-order cycles; it never counts the 30 raw
timings as 30 independent order assignments. The same sampled cycle vector is
applied to every semantic cell in that schedule stratum, preserving shared
thermal, clock, and temporal movement. A 10,000-resample joint
maximum-standardized bootstrap
constructs 95 percent familywise intervals across every profile, operation,
metric, timing lane, and all three position-pair contrasts. A native artifact
whose complete-cycle sample multisets are exactly identical at every position
uses the mathematically equivalent zero-effect bound and records that analytic
degenerate path in `bootstrap_execution` instead of iterating identical draws.
A native order claim is clean only when every simultaneous upper absolute bound
is at most one percent. It is contaminated when any simultaneous lower absolute
bound exceeds one percent. Every boundary-overlapping interval is inconclusive
and cannot support that performance claim. Evidence integrity and claim
readiness are separate: valid, complete raw evidence remains auditable when its
order result is inconclusive or contaminated, while normalization, schedule,
coverage, provenance, or schema defects still invalidate the artifact. The
same separation applies to Core's balanced order analysis: a structurally
complete raw Core artifact remains evidence-integrity-valid while its
inconclusive or contaminated statistics are recorded as claim failures and
block performance claims. The
`canonical_payload_api`, `supplemental_internal_kernel`, and
`supplemental_host_phase` families are assessed separately; a non-clean
supplemental family blocks only that family and a combined all-phase claim, not
an independently clean canonical family. Such a clean family may be reported
only as a narrowly labelled, lane-scoped diagnostic comparison; it does not set
release-wide `performance_claim_ready`, `arithmetic_claims_valid`, or backend
readiness true while any required family remains non-clean. Missing raw samples, variable loop
counts, fewer than 30 complete predeclared cycles, or incomplete six-order
blocks are structurally insufficient evidence, never a pass. Cycle count is
fixed before collection; optional stopping is forbidden.

The fresh-worker public order control and the already-warmed interleaved
control use the same three-state equivalence rule. Fresh-worker public cycles
are stratified by A/B block; interleaved samples retain each randomized
six-order block and must contain the backend's complete profile set. Their
point spreads remain diagnostics only and cannot replace the simultaneous
intervals.

Each cold worker records a clean interval from worker entry through explicit
artifact cleanup; report validation also occurs after the interval stops. That
interval excludes provenance capture, JSON serialization, process startup, and
exit residual; the output reports raw intervals, median,
MAD, p05, p95, and bootstrap median confidence intervals. ABI-combined fields
such as dynamic loading, allocation/upload/planning, or GPU
execution/report-materialization are marked combined rather than assigned
invented phase timings.

The harness rejects release claims when the declared source tree is dirty, a
loaded `gafime` module comes from the source checkout instead of the installed
wheel, a wheel's embedded distribution metadata/version does not match the
runtime, the canonical benchmark script is not hash-bound and identical for
both variants, or the benchmark script/native binaries are not hash-bound. It
also compares the interpreter's SHA-256 and size rather than requiring two
isolated environments to share one path. Variant-bound library, module,
`PYTHONPATH`, and virtual-environment paths may differ only when their presence
matches; wheel/runtime hashes authenticate the bytes they load. Threading,
device visibility, runtime search paths, and other semantic environment values
must match exactly. NumPy and Polars versions plus installed `RECORD` hashes,
stable device/driver identity, CPU governor, toolchain, and process affinity
must match. Raw before/after device clock and power readings
remain attached for drift and order-sensitivity analysis, but instantaneous
dynamic values are not required to be byte-identical across fresh processes.
macOS records when an affinity mask is unavailable. Public cells carry SHA-256
identities for the generated matrix, target, and feature names; baseline and
candidate mismatches are rejected before comparison. The
canonical gate also requires the full public surface/workload/input matrix,
all six orders where three profiles are supported, at least 10 warmups and 30
recorded repetitions, every sampled region to meet its auto-scaling target,
enabled interleaved control, alternating randomized A/B and B/A blocks, and no
unresolved one-percent order/regression threshold.

Native A/B evidence has a separate schedule gate. Each native cell is collected
in a fresh helper process for both `baseline,candidate` and
`candidate,baseline` blocks. Every artifact or its manifest schedule entry must
record `ab_block`, `variant`, `variant_sequence`, and
`process_isolation=fresh_helper_process_per_variant_trial`. The gate binds the
exact artifact, benchmark binary, wheel, payload where applicable, common
harness source/blob, product commit/tree, normalized environment, structured
workload, input policy, and dataset identity. A missing reversed block or an
identity change between blocks invalidates the comparative claim. Native
arithmetic comparisons cover only input policies explicitly represented by
that schedule; the public matrix remains responsible for both `common-f64` and
`native` source policies.

Current CUDA and ROCm native evidence is additionally lane-isolated. Every
artifact declares exactly one of `canonical_payload_api`,
`supplemental_internal_kernel`, or `supplemental_host_phase`, and every record
must carry the same lane. The canonical lane is ABI/payload-resolution-bound;
it must resolve and authenticate the exact canonical route in that helper
process. The direct and host lanes must positively record
`payload_not_loaded=true`, their non-payload execution marker, and a
path/SHA-256-bound external canonical lifecycle; live payload resolution in a
supplemental process is forbidden. ROCm additionally binds `compiled_lane` to
the declared evidence lane and authenticates the direct-kernel product root,
commit, kernels/header hashes, direct-source hash, and lane-specific compiled
marker. Backend readiness requires the manifest
union for each variant to cover all three lanes, both input policies, and both
AB/BA blocks, with unique artifact path, SHA-256, and runner/child-process
attestation. Operation coverage and order-family readiness are accumulated
within a lane; they are never pooled across lanes. Native A/B cells are
comparable only when lane, input policy, workload/input identity, and immutable
loop-plan identities all match. The plan's embedded `plan_sha256` is the
semantic digest of canonical unsigned contents; the separate `file_sha256` is
the raw serialized-file digest. The runner and perf13 rehash both, reopen each
plan-bound calibration file, compare its binding metadata, require trusted
Git/source/harness provenance, and prove every loop count equals
`max(baseline,candidate) * headroom_factor` under the plan cap. Perf13 resolves
the plan and calibration bindings only inside the explicit manifest evidence
root; the runner's `artifacts/../calibration` sibling path is accepted only
while it remains inside that root, and absolute or traversal escapes are
rejected even when their hashes match. The independent verifier also requires
plan version 1, exactly two sources and the exact baseline/candidate variants,
distinct full product commits, and equality between root and binding commit
sets. When a manifest schedule is present, its `input_policy` and all other
schedule fields must exactly match the hash-verified payload; the payload is
authoritative. Native A/B readiness independently rejects identical baseline
and candidate product identities even when no public result matrix is present.

`valid_for_e2e_performance_claims` covers only the public wall-clock evidence.
`valid_for_native_arithmetic_claims` (also emitted as
`valid_for_arithmetic_claims` and `valid_for_kernel_claims`) additionally
requires a valid native manifest. The top-level comparative claim additionally
requires the independent native A/B/B/A schedule gate;
`valid_for_performance_claims` requires that native gate plus the two-variant
comparative A/B/B/A gate. A local exploratory run may omit `--wheel`, but its
claim fields remain false; do not copy its summary into a PR performance table.
Repeat `--wheel NAME=PATH` for every Core and payload wheel used by that
variant.

The Python validator derives native raw-duration median, MAD, p05, p95, and
bootstrap-median intervals from the helper records. The current native helper
artifacts report fixed 10-warmup/30-repetition captures; their automatic
sample-region scaling is recorded when present, but native auto-scaling is not
manufactured by perf13 (`auto_scaling.status` is
`not_observed_in_native_artifact` when the helper did not emit it). The public
worker independently enforces its 100 ms sampled-region target.

`_measure_common.py` contains shared loaders, telemetry helpers, candidate
materialization helpers, and model baselines. `run_cpu_suite.sh` and
`run_gpu_suite.sh` run focused subsets, continue long enough to report every
failed script, and return nonzero if any script failed. Metal excludes the
unsupported graph and CUDA RT measurements.

## Guardrails

- Feature generation is validated from the public API.
- CPU and GPU backends must not silently fall back to another backend.
- Numerical output must be bit-equal where policy says bit parity is possible,
  otherwise the approved tolerance must be documented and tested.
- Mutual-information parity is split by estimator: default adaptive CPU MI is
  not compared to GPU fixed-bin MI; GPU MI parity is tested only through the
  explicit `mi_approximate=True` fixed-bin path. In both cases `mi_bins` is an
  adaptive maximum; fixed-bin parity uses the shared sample-size-selected
  template rather than forcing the configured ceiling.
- Performance artifacts are useful only when generated by these scripts or the
  architecture gate, not from hand-written numbers.
