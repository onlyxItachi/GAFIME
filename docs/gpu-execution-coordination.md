# Ordinary CUDA/HIP execution coordination

## Boundary

Each loaded ordinary CUDA or HIP payload owns one nonrecursive host mutex.
Every public ordinary ABI 1.0 and generic numeric-route ABI 1.1 entry acquires
that same mutex before validation, device selection, or vendor-runtime work.
This includes capability/device probes, route enumeration, allocation, upload,
target replacement, execution, memory forecasts, significance, diagnostics,
and both free interfaces. Internal adapters do not acquire the mutex again.

The gate covers the entire native call, including graph preparation/capture,
launch, synchronization, failure cleanup, and restoration of the caller's
current device. The guard is the first local in each export so all nested
device guards are destroyed before the mutex unlocks. The mutex is deliberately
payload-wide: calls using different device IDs also serialize. This is a
conservative ordinary-payload correctness boundary, not a throughput claim.

This prevents independent foreign callers using both ABI generations of the
**same loaded payload** from entering capture and incompatible runtime work
concurrently. It does not depend on the Python GIL and does not release or
otherwise change that GIL. Capture mode, graph/cache identity, numeric kernels,
precision profiles, ABI layouts/symbols and numerical compiler policy are
unchanged. Mixed precision uses this boundary too; it is not inherently free
of device-wide synchronization.

## Limits and ownership

- A separately loaded payload instance, CUDA versus HIP payload, or another
  framework/direct vendor-runtime caller does not share this mutex. There is
  no process-wide GPU lock or general cross-framework capture interoperability
  guarantee. The host must externally coordinate those domains when necessary.
- This is call serialization, not a whole-operation transaction or device-memory
  reservation. Resident matrices can coexist between calls. Allocation peaks,
  forecasts, memory admission, fairness and useful parallel throughput are not
  guaranteed by this gate.
- Matrix handles and host buffers still require valid caller-owned lifetimes.
  A caller must not free a handle while another caller might use or wait to use
  it. Serialization alone cannot turn a stale raw pointer into a valid handle.
  Multi-call update/execute sequences also require caller coordination if they
  share mutable state.
- There are no callbacks into Python while this gate is held. The gate is
  released by the native return; no lock is carried into interpreter reentry.
- Metal and experimental local CUDA RT/OptiX entry points are outside this
  change. Standard distribution payloads remain RT-disabled. The ordinary gate
  does not promise coordination with experimental RT calls.

The guard catches host mutex-acquisition exceptions. An integer-returning
entry fails closed with `GAFIME_STATUS_DEVICE_ERROR`, before touching the
runtime or output buffers. Legacy `void` free cannot report a status: if
acquisition fails it leaves the allocation untouched rather than destroying it
unguarded. This exceptional case may leak unless the caller can retry. The
status-returning free can be retried with the still-live handle. Existing
vendor cleanup/device-restoration error behavior is not changed; this is not a
general cleanup-error redesign.

## Evidence and regression layers

The motivating operator report is an RC2 installed CUDA graph-versus-foreign
ABI 1.0 eager collision: three failures in thirteen reported runs, with clean
graph-free and eager-only controls. Retained crash artifacts examined during
the audit showed two `libcuda`/`cudaDeviceSynchronize` failures; the complete
three-failure aggregate and exact scheduling are reported evidence, not a new
independent reproduction. No baseline GPU crash run is needed or authorized by
this change. A source-level interleaving risk and that hardware report justify
the coordination boundary, but source tests alone do not prove a physical fix.

Host-only checks:

- `python3 tests/gpu/test_execution_gate_source.py` checks all ordinary exports,
  first-statement acquisition/fail-closed exits, absence of nested export calls,
  and source-distribution inclusion.
- The `gafime_payload_execution_gate_*` CTest target tests cross-thread
  exclusion, contention, exception unwinding, restoration-before-unlock, and
  injected acquisition failure including the legacy free limitation. It does
  not import, load, probe or execute a GPU runtime. Its timeout is 15 seconds.

Physical qualification is separate and opt-in. Build the normal fixed candidate
and install its matching Core/payload outside the source tree under the normal
preflight policy. `GAFIME_ABI_CONSUMER_BUILD_CONCURRENCY_SHIMS=ON` additionally
builds two test-only shared libraries from the existing frozen ABI 1.0 and
canonical generic typed-view ABI 1.1 C consumers. These are foreign `ctypes.CDLL`
callers (the calls release the GIL), not new product exports. The generic
consumer exercises all three numeric routes and update/forecast/significance/
diagnostic/free paths. The shims and collision runner are not default CTest
workloads or distribution payload members.

`tests/gpu/execution_coordination_regression.py --help` is host-only. An actual
run requires explicit paths for the installed interpreter, candidate payload,
both shims, expected payload SHA256, backend, precision, case, a new output
directory and `--acknowledge-fixed-candidate`. The operator must first confirm
build provenance, device availability/memory and that desktop/display or other
GPU users will not be disrupted. The known unfixed RC2 CUDA hash is rejected;
the script does not establish arbitrary candidate build provenance by itself.
Do not run the retained unfixed reproducer or delete existing crash records.

Each invocation selects exactly one case (`graph-abi10`, `graph-abi11`,
`graph-both`, `eager-both`, or `foreign-only`) in an isolated installed-package
subprocess. Runtime, worker and foreign-call counts are bounded. The controller
records source HEAD/status/file hashes, payload/shim hashes, child Core hash,
raw stdout/stderr, timeout and return status in a new directory. A signal,
timeout, unavailable status, nonzero consumer result, fallback, missing progress
or parity difference fails the run; it is not silently skipped. Process exit
also exercises teardown. Existing crash artifacts are never removed.

Graph cases repeatedly create, analyze and close independent matrices. Every
deterministic public report field is compared against its serial reference,
including row order/identity, all metrics, diagnostics, configuration,
significance, decision, warnings and backend precision facts. Only the live
`backend.memory_free_mb` snapshot is omitted. Floating values compare their
binary64 representation, including signed zero; this is stronger than Python
numeric equality. Volatile telemetry/Arrow metadata bytes are not used as a
parity proxy. The runner requires actual graph replay when requested.

Qualification should cover each affected backend and `fp32`, `mixed`, `fp64`,
the ABI-specific and combined graph cases and graph-free controls, with bounded
repetitions chosen at preflight. Add multi-device caller-device restoration
checks where a second device exists; report them as untested otherwise. Normal
ABI/error-path and serial numeric gates remain required. Until those physical
and packaging gates run against identified candidate artifacts, this is an
implemented and host-checkable fix candidate, not a qualified release or a
guarantee that all driver failures are eliminated.
