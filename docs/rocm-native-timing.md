# ROCm native precision timing

`gafime_rocm_native_timing` is a benchmark-only helper for release evidence. It
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

The helper does not link the product library, call private kernels, or change
the public ABI. A typed baseline route is synthesized only from the fixed
profile contract after validating the payload capability masks; the JSON marks
`route_synthesized: true` and never presents it as an exported generic route.
Generic `numeric-route-v2` resolution is accepted only after all ten canonical
symbols (routes, allocation, upload, target update, execute, both admission
and permutation helpers, diagnostics, and free) resolve. Typed
`precision-typed-v1.1` resolution requires its eleven-symbol mandatory
inventory, including diagnostics; the three typed permutation symbols are
reported as optional present/missing metadata and are not inferred when absent.
The corresponding `optional_permutation_status` is `not_exported_typed_optional`
for the exact 8df ROCm payload; a partially exported typed family is rejected.

The helper runs fp32, mixed, and fp64 routes in all six deterministic profile
orders for at least five complete cycles and writes the complete
`profile_order` plus global `order_index` into every record. Each order
exercises ingest conversion, candidate descriptor
materialization, host planning, allocation, upload, target update, the
state-aware execution-memory forecast, Pearson/R2/MI/Spearman, the
route-typed Spearman target-rank build after target replacement, the subsequent
cached unary Spearman route, top-k plus selected-row gather, result
visibility/readback, and host report construction. The fp32 cache stores float
ranks and reduces/returns float; mixed stores float input but caches double
ranks and reduces/returns double; fp64 uses double throughout. Metric/ranking
samples use synchronized HIP events (`hipDeviceSynchronize`, event start/stop,
and `hipEventSynchronize`); host-only and forecast boundaries use
`steady_clock`. The JSON artifact records raw samples, route/workload
configuration, source commit, payload/helper hashes, HIP runtime/driver/device
details, environment, CPU affinity, clock metadata, `abi_surface`, and
`route_source`.

The top-level `wrapper_comparability` map is mandatory evidence metadata:
symbol/capability/planning phases are `not_comparable` across the two wrapper
surfaces; allocation, upload, target update, forecast, execute, ranking, and
cleanup are `semantic_only`; and `d2h_transfer` is
`host_only_d2h_unobservable`. The synchronous execute call owns vendor D2H and
device synchronization, so the bundled payload-boundary D2H record must not be
interpreted as a standalone vendor transfer timer. A second
`representative_direct_transfer` record copies the full result-buffer byte
count with HIP D2H on helper-owned buffers; it is a transfer reference, not a
claim about payload-internal copy time. Product source provenance and the
common helper/harness provenance are emitted separately, allowing the same
clean helper source hash to bind baseline and candidate runs.

Run the helper once per variant in each schedule block. Current evidence must
contain both `baseline,candidate` and reversed `candidate,baseline` blocks, each
from a fresh helper process. `--variant`, `--ab-block`, and
`--variant-sequence` bind those fields into the artifact. Run `--input-policy
common-f64` and `--input-policy native` as separate cells: the former converts
one shared f64 source to the selected storage dtype, while the latter starts
fp32/mixed from f32 and fp64 from f64. Canonical wrapper timings and the
representative direct-transfer reference remain different measurement
categories; payload-private per-kernel and internal D2H phases are explicitly
unobservable.

Every native helper timing record uses at least 10 untimed warmups and 30
measured samples (smaller values are rejected). Before those samples are
recorded, the helper doubles an inner loop until a sampled region reaches a
5 ms floor, with a 2x calibration guard band and a bounded loop count. The
artifact keeps both `raw_samples_us` (the complete calibrated region) and
`samples_us` (the normalized per-call value); `loop_count_per_sample` makes
the conversion explicit, while `loop_counts_per_sample` records any adaptive
per-sample increases. Each record also emits median, MAD, p05, p95, and a
2,000-resample bootstrap median 95% interval. Bootstrap resampling uses the
fixed seed `20260809`, mixed with the record identity, so the statistical
summary is reproducible without changing the synchronized HIP/CUDA/Metal
clock boundary. The public perf13 harness retains its separate 100 ms sample
region gate.

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
  -DGAFIME_ROCM_BUILD_TESTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1150
cmake --build /tmp/gafime-rocm-timing --target gafime_rocm_native_timing -j2
/tmp/gafime-rocm-timing/gafime_rocm_native_timing \
  --payload /tmp/gafime-rocm-timing/libgafime_rocm_v1.so \
  --source-root /path/to/product-source \
  --harness-source-root /path/to/clean-common-harness-source \
  --input-policy common-f64 \
  --variant baseline --ab-block 0 \
  --variant-sequence baseline,candidate \
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
