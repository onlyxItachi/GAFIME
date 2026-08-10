# Cold lifecycle benchmark

`tests/release_measure/cold_lifecycle.py` is the isolated lifecycle layer for
precision-profile performance evidence. It is deliberately separate from the
public end-to-end and native arithmetic layers.

## Contract

The driver starts one fresh subprocess for every profile sample. With the
default three profiles and 30 repetitions it therefore launches 90 workers.
The profile order is randomized, and all six permutations are represented when
all three profiles are selected. The output keeps every raw duration and binds
each worker to the exact payload and benchmark-script SHA-256. `--wheel` and
`--source-root` are mandatory: the source tree must be clean at a full commit,
and the payload bytes must match the backend member in that exact wheel.

Each worker imports the installed `gafime` package, performs installed-payload
discovery, loads the requested payload with `dlopen`/`LoadLibrary`, and selects
the ABI surface from the exact exported symbols. Candidate/current payloads use
`gafime_gpu_numeric_routes_v2` plus the complete ten-symbol generic operation
table; resolution fails before allocation if any member is absent. The exact
historical pre-freeze ABI 1.1 baseline does not export
that route symbol, so the same script automatically falls back to
`gafime_gpu_precision_capabilities`, `gafime_gpu_matrix_alloc_v2`, the
dtype-suffixed upload/update/execute functions, the typed execution-memory
function, and the legacy void `gafime_gpu_matrix_free` entry point. The typed
path synthesizes only the route domains already fixed by the profile contract;
it does not pretend that a route record was exported.

Both surfaces allocate and upload the canonical typed 4x2 matrix, replace its
target, build one Pearson candidate, forecast execution memory, execute, read
one result row, and free the matrix. The report records `abi_surface`,
`route_source`, the capability masks, and `route_synthesized` so a baseline
typed sample cannot be mistaken for a generic-route sample. The worker does not
use the Rust loader.

The report contains median, MAD, p05, p95, bootstrap 95% confidence intervals,
and raw samples for every observed phase. It also records status counts and
the parent-side subprocess duration. The benchmark does not claim a phase that
cannot be separated honestly:

- CUDA and HIP runtime/context initialization is the first explicit
  `cudaFree(0)` or `hipFree(0)` after the payload is loaded; Metal has no
  equivalent separate C runtime boundary and reports that phase as
  `not_observable`;
- host code-object/fatbinary registration runs in loader constructors, so it is
  reported as `observed_combined` with the exact payload's
  `dlopen`/`LoadLibrary` duration instead of assigning invented sub-times;
- target replacement and the state-aware execution-memory forecast are timed
  as their own canonical calls;
- caller-side candidate/protocol/result-buffer planning is timed separately;
- result materialization times the first typed host read of the caller-owned
  structural and metric result buffers after execute returns. Vendor D2H and
  synchronization happen inside the backend execute call and are not separately
  observable at this ABI boundary, so this phase is marked
  `host_only_d2h_unobservable`;
- process exit cannot be isolated from interpreter startup and JSON/provenance
  work; the parent-minus-worker residual is retained as `observed_combined`
  and is never labeled pure teardown.

The JSON contains a `phase_comparability` map and each phase carries the same
classification. `symbol_resolution`, `first_capability_query`, and `planning`
are `not_comparable` because the two ABI surfaces have different symbols,
capability operations, and wrapper layouts. Allocation, upload, target update,
execution-memory forecast, execution, and cleanup are `semantic_only`: they
measure corresponding backend operations but retain ABI-specific validation
and wrapper overhead. Loader registration and process exit are
`combined_not_separately_observable`. Only like-for-like phase deltas should be
used as performance gates; the typed fallback is compatibility evidence, not a
claim that the old and new wrapper costs are identical.

## Example

```sh
python tests/release_measure/cold_lifecycle.py \
  --backend cuda \
  --payload /absolute/path/to/libgafime_cuda.so \
  --profile fp32,mixed,fp64 \
  --source-root /absolute/path/to/source \
  --wheel /absolute/path/to/gafime_cuda.whl \
  --repetitions 30 \
  --output cold-cuda.json
```

The CUDA and ROCm commands should be run on their corresponding physical
devices with the exact candidate payload. Metal uses `--profile fp32` only.
The report is evidence for the cold lifecycle; it must not be substituted for
the public or device-event benchmark layers.

For comparative evidence, run two blocks with reversed variant order. Each
artifact declares `--variant`, `--ab-block`, and `--variant-sequence`, and each
block contains at least 30 fresh-process samples per profile. Put the four
artifact paths and SHA-256 values in a
`gafime.cold-lifecycle-comparison-manifest.v1` manifest, then validate and
classify them with the same tracked script:

```sh
python tests/release_measure/cold_lifecycle.py \
  --compare-manifest cold-comparison-manifest.json \
  --bootstrap-resamples 2000 \
  --output cold-comparison.json
```

The comparison requires one common benchmark blob and physical device, stable
payload/wheel bytes per variant, distinct clean product commits, complete A/B
plus reversed B/A scheduling, and exact wheel-member binding. Repeatable
regressions above one percent and order sensitivity above one percent block a
performance claim; repeatable regressions above three percent are
release-ineligible without explicit maintainer approval.

## Hosted Metal gate

The manually dispatched `Metal Beast Benchmark` requires the full current PR
#70 head in `expected_candidate_sha`. It checks out that exact commit and reads
the live PR head before any build. A second live-head check runs after the
baseline/candidate builds, and a final check runs after evidence collection, so
a moving PR cannot leave an artifact labeled as current-head evidence.

The workflow builds and installs the exact baseline and candidate wheels,
extracts each wheel's Metal dylib, and invokes this one tracked current harness
with the corresponding isolated interpreter. It collects block 0 in
`baseline,candidate` order and block 1 in reversed `candidate,baseline` order,
with 30 fresh processes for Metal fp32 in every cell. The four raw artifacts,
their SHA-256-bound comparison manifest, the validated comparison, and the
pre/post live-head record are uploaded together. The hosted job fails unless
`valid_for_canonical_cold_lifecycle_claims` is true. The baseline is explicitly
required to report the historical `precision-typed-v1.1` surface, while the
candidate is required to report `numeric-route-v2`.

This canonical fp32 lifecycle uses one fixed typed ABI input route. The
separate native Metal helper runs both `common-f64` and native-fp32 source
policies in its own eight-artifact A/B+B/A matrix; those conversion-policy
records are not fabricated inside the canonical ABI lifecycle layer.
