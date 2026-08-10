# GPU numeric-route ABI evolution

This document defines the versioning and negotiation rules for the native GPU
boundary. It does not change the Python precision API, backend support matrix,
numeric contracts, packaging topology, or runtime-ownership policy described in
[`precision-contract.md`](precision-contract.md) and [`contract.md`](contract.md).

## ABI generations

ABI 1.0 is frozen. Its exported symbols, structures, offsets, status behavior,
and accepted `float*` inputs remain byte-compatible. An ABI 1.0 pointer is never
reinterpreted as a `double*`. The implementation may translate a 1.0 call into
shared modern internals, then adapt the result back to the frozen float result
layout. That compatibility adapter is not a fourth public precision profile and
does not require a separate complete device engine.

ABI 1.1 is the canonical numeric-route ABI. It has one generic operation per
lifecycle action, caller-owned typed buffer views, and one authoritative route
record describing all four numeric domains. Dtype masks may be exposed as quick
summaries, but a mask cannot establish support for a dtype combination; only an
enumerated route can do so.

ABI 1.2 is reserved for additive evolution. A future integer engine may add
dtype values, numeric routes, overflow policies, and internal specializations.
It must use the generic 1.1 upload, execute, and result operations rather than
creating dtype-suffixed symbol families. This repository does not currently
implement or advertise an integer route.

### Frozen ABI 1.0 arithmetic adapter

Source inspection, frozen fixtures, and generated-device-code comparison show
that no single ABI 1.1 profile is the complete ABI 1.0 arithmetic contract.
Legacy GPU input, resident storage, pointwise values, and visible results are
f32. Pearson, R2, and MI use f32 device reduction/finalization, while Spearman
rank-derived covariance and normalization use f64 before the visible result is
rounded to the frozen f32 table. Legacy host column means also accumulate in
f64 and then round to f32; ABI 1.1 `fp32` intentionally keeps that reduction in
f32. The historical MI path additionally preserves its established non-finite
bin policy.

The payload therefore does not advertise ABI 1.0 as a fourth route. For
contracted arities 1 through 5, ABI 1.0 Pearson/R2/MI call the shared fp32 route
primitives, with only the compatibility mean and MI policy applied; Spearman
uses the shared f32 pointwise primitive with its narrow f64-reduction/f32-result
adapter. The previously frozen ABI also accepted higher arities, so one dynamic
compatibility primitive per differing metric path remains for those calls.
There is no complete legacy device-kernel tree. ABI 1.0 C fixtures cover the
ordinary lifecycle and an arity-6 lifecycle against the new payload.

## Version and structure negotiation

`abi_version` stores the incompatible major in its high 16 bits and the additive
minor in its low 16 bits. Every extensible ABI 1.1 structure starts with
`abi_version` followed by `struct_size`.

Compatibility is checked against the stable numeric-route floor
`GAFIME_NUMERIC_ROUTE_ABI_MIN_MINOR`, not against the payload's newest minor.
Consequently, increasing a payload implementation to ABI 1.2 does not by itself
exclude an ABI 1.1 caller that supplies the complete 1.1 prefix.

- A major-version mismatch fails with `GAFIME_STATUS_ABI_MISMATCH`.
- A newer minor is accepted when the complete known stable prefix is present.
- A shorter-than-required stable prefix fails closed.
- A consumer ignores fields after the smaller of its known structure size and
  the producer record's `struct_size`.
- Current producers zero reserved fields. A current record with a nonzero known
  reserved field is invalid.
- Bits `0x00000001` through `0x00008000` are required-semantics flags. An
  unknown required bit fails closed. Bits in `0xffff0000` are ignorable hints;
  consumers may ignore unknown values there.
- Unknown dtype, profile, route, and overflow-policy values are never executed
  accidentally.

The route enumeration call first supports a count query with `routes_out ==
NULL` and `route_capacity == 0`. The consumer then supplies caller-owned storage
and an explicit record stride. The payload writes no more than that stride per
record. The producer's `struct_size` may be larger than the supplied stride:
that value describes the full producer record, while the caller-provided
stride bounds the bytes the producer may write. The consumer advances by the
supplied stride, interprets only the copied prefix, and uses each record's
`struct_size` to identify its available prefix. No backend-owned route pointer
or lifetime is exposed.

Enumeration records may grow beyond the ABI 1.1 route size. A route embedded by
value in an ABI 1.1 matrix or launch structure is deliberately the fixed ABI 1.1
selection prefix: an older consumer copies only that known prefix and reports
its known inner `struct_size`. It never copies a larger record over the fields
that follow the embedded prefix. A future operation that needs semantics beyond
the route ID and current numeric-domain fields must carry them in an additive
outer-structure tail; it must not enlarge an embedded record in place. Generic
operations that receive a standalone route pointer may accept a larger record
and ignore its unknown tail.

An ABI 1.1 consumer validates an enumerated set in this order:

1. validate the major, stable-prefix length, and flag policy for every record;
2. reject duplicate route IDs, including IDs whose future semantics are
   otherwise unknown to the consumer;
3. skip unknown additive routes and dtypes;
4. validate the complete tuple for every recognized route and reject
   contradictory declarations;
5. select one exact recognized route and preserve it for allocation, upload,
   execution, significance, diagnostics, and free.

An ABI 1.1 consumer inspecting an ABI 1.2 payload can therefore retain the
known float routes while skipping larger unknown records. Conversely, an ABI
1.1 payload rejects a future route or dtype when a caller tries to request it.

## Current route set

| Route | Storage | Pointwise | Reduction | Result and ranking |
| --- | --- | --- | --- | --- |
| `fp32` | f32 | f32 | f32 | f32 |
| `mixed` | f32 | f32 | f64 | f64 |
| `fp64` | f64 | f64 | f64 | f64 |

Core, CUDA, and ROCm advertise all three records. Metal advertises only `fp32`.
An explicit unsupported Metal route fails before allocation. Automatic backend
selection may select Core for `mixed` or `fp64` on Apple systems.

The result dtype is both the comparison dtype for ranking and the visible dtype
for metric and significance output. A payload must not rank with a hidden wider
value and expose a narrower value.

## Typed buffers and generic operations

`GafimeConstBufferView` and `GafimeMutableBufferView` carry the dtype, ownership
flags, pointer, element count or capacity, byte length, and byte stride. The
current operations accept caller-owned contiguous host memory for the duration
of a synchronous call. Before reading or writing, a payload validates:

- the requested route and expected domain dtype;
- pointer nullability and natural alignment;
- element count or capacity;
- exact byte length and stride;
- multiplication overflow;
- matrix storage dtype and result dtype;
- current host ownership flags.

Structural output arrays remain explicitly typed as integers. Metric,
significance, and other numeric values cross the boundary through typed buffer
views.

The canonical dynamic symbol set is:

- `gafime_gpu_numeric_routes_v2`
- `gafime_gpu_matrix_alloc_v2`
- `gafime_gpu_matrix_upload_v2`
- `gafime_gpu_matrix_update_target_v2`
- `gafime_gpu_execute_v2`
- `gafime_gpu_execution_memory_peak_v2`
- `gafime_gpu_permutation_memory_peak_v2`
- `gafime_gpu_permutation_pvalues_v2`
- `gafime_gpu_interaction_diagnostics_v2`
- `gafime_gpu_matrix_free_v2`

These ten symbols are one normative ABI 1.1 operation table. A payload that
advertises `gafime_gpu_numeric_routes_v2` must export all ten; a partial table
is rejected before allocation. Dynamic loaders may represent individual
lookups as optional values while probing a library, but that representation
does not promise an optional generic operation or a fallback. The unsuffixed
ABI 1.0 capability symbols retain their separate legacy optional semantics.

Dtype-specific helpers may exist as internal functions or compile-time header
wrappers, but they are not independent exported implementation owners. Generic
ABI dispatch selects a route once; backend hot loops remain statically
specialized.

## Enum-ID allocation

Zero is invalid for dtype, profile, route, and overflow-policy enums. Existing
nonzero IDs are permanent and are never renumbered, reused, or given new
semantics. A future version allocates a new ID by adding the next explicit value
to the shared ABI header and an associated route record. Support exists only
when the payload enumerates the complete route; the presence of an enum value or
summary-mask bit alone is not support. Reserved or proposed integer IDs must not
be advertised before their arithmetic and overflow contracts are implemented.

An illustrative future route could combine integer storage, widened integer
pointwise arithmetic, an i64-or-wider exact reduction, and f64 ranking/results.
That example demonstrates extensibility only: it commits neither concrete dtype
IDs nor integer arithmetic semantics.

## Independent compatibility evidence

`tests/gpu/abi_consumers` provides consumers that do not rely on private loader
or backend types:

- `abi_1_0_c_consumer.c` declares only the frozen 1.0 prefix, dynamically loads
  a current payload, and performs allocate/upload/execute/free.
- `abi_1_0_dynamic_arity6_consumer.c` proves that the frozen accepted
  higher-arity lifecycle remains executable without retaining the old complete
  kernel tree.
- `abi_1_1_c_consumer.c` dynamically resolves only canonical generic symbols,
  enumerates routes into caller storage, exercises typed lifecycles, and checks
  fail-closed malformed inputs.
- `abi_1_1_rust_consumer.rs` independently declares the published C layouts,
  negotiates routes, and executes each supported route without using
  `gafime-gpu-sys`.
- `abi_1_2_route_fixture.c` emits larger records containing the three known
  float routes plus an unknown future route. `abi_route_compatibility_test.c`
  proves that known routes survive, the unknown route is skipped, ignorable
  flags are tolerated, and major mismatch, short prefixes, required unknown
  flags, nonzero reserved fields, duplicates, and contradictions are rejected.

Both C and Rust consumers assert the sizes, alignments, and key offsets of the
published stable layouts. Backend CTest builds run these consumers against the
actual payload produced in that build.
