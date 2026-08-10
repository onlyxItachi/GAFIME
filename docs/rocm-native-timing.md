# ROCm native precision timing

`gafime_rocm_native_timing` is the direct, benchmark-only helper for release
evidence. ROCm evidence is compile-time lane-isolated into three byte-distinct
helpers: `gafime_rocm_native_timing` (direct product-source kernels),
`gafime_rocm_native_timing_canonical` (canonical payload API), and
`gafime_rocm_native_timing_host` (host-only control). Only the canonical helper
loads the exact shared object named by `--payload` and auto-detects one of two
ABI 1.1 surfaces, including the historical pre-freeze typed baseline:

- `numeric-route-v2` resolves `gafime_gpu_numeric_routes_v2`, generic
  route/view/protocol/result calls, and `gafime_gpu_matrix_free_v2`;
- `precision-typed-v1.1` is selected only when the generic route symbol is
  absent. It uses private, size-checked local copies of the historical
  pre-freeze typed capability,
  matrix descriptor, launch wrapper, and f64 result layouts, plus the typed
  f32/f64 upload, target-update, execute, forecast, and legacy void free
  symbols.

The canonical and host helpers do not link the product library or embed product
HIP translation units. The direct helper links the clean product
`src/rocm/kernels.hip` translation unit plus the common direct wrapper; it does
not link the product shared library or call the canonical ABI. A typed baseline
route is synthesized only from the fixed profile contract after validating the
payload capability masks; the canonical JSON marks `route_synthesized: true`
and never presents it as an exported generic route.
Generic `numeric-route-v2` resolution is accepted only after all ten canonical
symbols (routes, allocation, upload, target update, execute, both admission
and permutation helpers, diagnostics, and free) resolve. Typed
`precision-typed-v1.1` resolution requires its eleven-symbol mandatory
inventory, including diagnostics; the three typed permutation symbols are
reported as optional present/missing metadata and are not inferred when absent.
The corresponding `optional_permutation_status` is `not_exported_typed_optional`
for the exact 8df ROCm payload; a partially exported typed family is rejected.
The helper records optional present/missing metadata for every optional symbol.
The direct lane labels its selected-row boundary `representative_direct_transfer`.

The helper runs fp32, mixed, and fp64 routes in all six deterministic profile
orders for a fixed, predeclared minimum of 30 complete cycles and writes the complete
`profile_order` plus global `order_index` into every record. Each cycle starts
from the canonical six-order set and consumes the deterministic `order_seed`
stream once; an exact shuffle collision is rotated so adjacent cycles never
reuse one temporal sequence. Collection may not stop early after a favorable
interval or add cycles conditionally after an inconclusive result. Before the
recorded cycles, one canonical fp32/mixed/fp64 pass initializes each lane's
runtime/helper state and populates the shared fixed-loop calibration cache; its records are discarded
and the JSON reports that prepass separately. Validation authenticates its
canonical order, positive record/sample counts, required cache-key coverage,
shared-cache marker, and exclusion from the recorded cycles. The
`clock_and_power_capture_point` marker is exact: measurement-before follows
the discarded prepass and precedes randomized cycles, while measurement-after
follows cycle collection and record verification. The JSON `order_schedule` marker
makes this schedule contract explicit. Each recorded order
exercises ingest conversion, candidate descriptor materialization, host
planning, allocation, upload, target update, the state-aware execution-memory
forecast, Pearson/R2/MI/Spearman, top-k plus selected-row gather, result
visibility/readback, and host report construction. The direct helper also
times a common-harness target-rank preparation and materialization helper; its
metric/ranking/gather operations are launched from the exact clean product
source. Direct Spearman uses the product static rank kernel for every arity;
the candidate-only cached-unary optimization remains a canonical payload path,
not a direct-lane substitute. The fp32 route stores/reduces/returns float;
mixed stores float input but reduces/returns double; fp64 uses double
throughout. Metric/ranking samples use synchronized HIP events
(`hipDeviceSynchronize`, event start/stop, and `hipEventSynchronize`); host-only
and forecast boundaries use `steady_clock`. The JSON artifact records raw samples, route/workload
configuration, source commit, payload/helper hashes, HIP runtime/driver/device
details, environment, CPU affinity, clock metadata, `abi_surface`, and
`route_source`. The exact runtime argument vector is recorded as
`command_line`; its executable must resolve to the same file authenticated by
`provenance.benchmark_binary`.

The top-level `wrapper_comparability` map is mandatory evidence metadata:
symbol/capability/planning phases are `not_comparable` across the two wrapper
surfaces; allocation, upload, target update, forecast, execute, ranking, and
cleanup are `semantic_only`; and `d2h_transfer` is
`host_only_d2h_unobservable`. The synchronous execute call owns vendor D2H and
device synchronization, so the bundled payload-boundary D2H record must not be
interpreted as a standalone vendor transfer timer. The direct lane stops at
product selected-row gather; the host lane's
`d2h_transfer` is a helper-owned control and is not a claim about payload
 internal copy time. Product source provenance and common helper/harness provenance
 are emitted separately. The direct binary binds the product root,
commit, kernel/header hashes, and common direct-source hash; canonical and host
binary identities contain no product provenance or product device code.

Run the helper once per variant in each schedule block. Current evidence must
contain both `baseline,candidate` and reversed `candidate,baseline` blocks, each
from a fresh helper process. `--variant`, `--ab-block`, and
`--variant-sequence` bind those fields into the artifact. Run `--input-policy
common-f64` and `--input-policy native` as separate cells: the former converts
one shared f64 source to the selected storage dtype, while the latter starts
fp32/mixed from f32 and fp64 from f64. Canonical wrapper timings, direct
product-source operations, and host controls remain different measurement
categories; payload-private per-kernel and internal D2H phases are explicitly
unobservable.

Pin every helper process to the same CPU-affinity mask. The artifact records
that mask, and comparative validation rejects missing or different affinity
values across A/B and reversed B/A cells.

Every calibration and recorded JSON artifact carries the lane contract fields
`execution_mode`, `payload_loaded`, `payload_not_loaded`, and
`payload_execution_mode`. Canonical artifacts emit
`canonical_payload`, `true`, `false`, and `canonical_payload`; direct and host
artifacts emit their lane name, `false`, `true`, and `payload_not_loaded`.
The direct and host helpers also require `--canonical-evidence PATH` before
any HIP setup, payload discovery, or allocation. They authenticate the
absolute regular-file path and SHA-256 without parsing it, keep the payload
unloaded, and emit `canonical_payload_lifecycle` with
`binding: external_canonical_evidence`; perf13 independently reopens and
validates that exact canonical lifecycle file. The canonical helper emits its
truthful live `canonical_helper` lifecycle binding.

Every native helper timing record receives at least 10 same-cell untimed
precondition iterations and at least 100 ms of untimed preconditioning before
30 measured samples (smaller values are rejected). Device records use
synchronized HIP events and bounded, adaptively sized precondition batches;
host records use the steady clock, with HIP synchronization around
device-owning wrapper preconditioning. The helper then calibrates one bounded
inner-loop count per semantic profile/operation/metric cell to a 2x guard band
above the 5 ms sampled-region floor. That fixed count is cached and reused for
the cell across every profile-order position; it is never rescaled between
measured samples. The artifact keeps both `raw_samples_us` (the complete
calibrated region) and `samples_us` (the normalized per-call value), and both
loop-count fields must prove that fixed calibration. Each record also emits
the precondition duration/count/batch/clock, median, MAD, p05, p95, and a
2,000-resample bootstrap median 95% interval. Bootstrap resampling uses the
fixed seed `20260809`, mixed with the record identity, so the statistical
summary is reproducible without changing the synchronized HIP/CUDA/Metal
clock boundary. The public perf13 harness retains its separate 100 ms sample
region gate.

Native A/B matching includes `loop_count_per_sample` in the comparison key.
Baseline and candidate cells with different fixed counts are explicitly
incomparable and fail the native comparative-claim gate; their normalized
durations are never paired.

The loop plan is immutable and has two distinct identities: the embedded
`plan_sha256` authenticates canonical unsigned JSON contents, while the
external `file_sha256` authenticates the exact serialized bytes. Calibration
bindings are plan-relative siblings and carry their own file hashes plus
source/product/harness roots, clean-tree states, tracked helper blobs, and
trusted absolute Git identity. The runner and perf13 reopen and rehash both
calibration files, compare all bound identity fields, and verify each planned
count is exactly the fixed headroom factor applied to the larger baseline or
candidate calibration count under the cap. Manifest schedule fields cannot
override payload fields: when both are present they must be identical.

These helpers do not claim a component-level cold-start decomposition. The
payload is `dlopen`ed and symbols are resolved once before the repeated
release measurements, so dynamic loading, first route/capability discovery,
allocation/upload, first execution, result validation, and cleanup are not
separately attributable by the steady-state records. A cold probe may report
one explicitly combined fresh-process interval, but it must not assign that
interval to individual runtime/module-registration components. Use the
separate `tests/release_measure/cold_lifecycle.py` driver for that evidence:
it launches one fresh worker per sample and labels ABI-unobservable phases as
combined or not observable. The native GPU helpers intentionally have no
`--cold-once` branch, so their release-mode sample contract cannot be weakened
by a one-sample startup path.

Example (gfx1150):

```sh
cmake -S src/rocm -B /tmp/gafime-rocm-timing \
  -DGAFIME_ROCM_BUILD_TESTS=ON -DGAFIME_ROCM_BUILD_BENCHMARKS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1150 \
  -DGAFIME_ROCM_BENCHMARK_PRODUCT_ROOT=/path/to/clean-product-source
cmake --build /tmp/gafime-rocm-timing --target \
  gafime_rocm_v1 gafime_rocm_native_timing \
  gafime_rocm_native_timing_canonical gafime_rocm_native_timing_host -j2
/tmp/gafime-rocm-timing/gafime_rocm_native_timing \
  --payload /tmp/gafime-rocm-timing/libgafime_rocm_v1.so \
  --source-root /path/to/product-source \
  --harness-source-root /path/to/clean-common-harness-source \
  --input-policy common-f64 \
  --evidence-lane supplemental_internal_kernel --artifact-kind rocm_events \
  --variant baseline --ab-block 0 \
  --variant-sequence baseline,candidate \
  --canonical-evidence /path/to/rocm-canonical-lifecycle.json \
  --json /tmp/rocm-native-timing.json \
  --workload release-small \
  --rows 4096 --features 8 --candidates 8 --arity 1 --mi-bins 64 \
  --top-k 2 --warmups 10 --repeats 30
```

Workload label, rows, features, candidates, arity, MI bins, top-k, warmups,
repeats, seeds, device, and an optional wheel path are configurable. The benchmark itself does
not require a wheel, but strict `perf_13_precision_profiles.py` provenance
validation requires a `wheel` identity; pass `--wheel /path/to/wheel.whl`
when producing a release-manifest artifact.
