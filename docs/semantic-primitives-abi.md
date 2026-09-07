# Optional tabular semantic primitive ABI

This document records the narrow native boundary used by the additive
`gafime.semantic` accelerator lowering. It is an implementation and
negotiation contract, not a claim that a GPU payload is installed, a device is
available, or a physical workload has run.

## Separate version domain

The source authority is
[`src/common/gafime_semantic_primitives_abi.hpp`](../src/common/gafime_semantic_primitives_abi.hpp).
Its optional semantic-primitive table has its **own** `1.2` major/minor version
domain. That number does not revise or replace either the frozen ABI 1.0
contract or the standard ten-symbol numeric-route ABI 1.1 table described in
[ABI evolution](abi-evolution.md). Those existing tables and their payload
policy remain unchanged.

The semantic table is additive and all-or-nothing: a payload that advertises
it must export the complete eleven-entry table, with a compatible major version
and stable-prefix `struct_size`. A missing, old, partial, or otherwise
incompatible table is rejected for this semantic route; it is not repaired by
silently executing Core. The `forecast` entry makes native scratch and retained
storage explicit for resource admission. Minor 2 requires total operand and
mean counts for immutable batch-wide descriptors; the earlier reusable
per-node span cannot bound that storage. An older prefix is therefore not
enough to admit this lowering's materialization peak.

The eleven entries cover capability discovery, resident-bank allocation/upload/
materialization/retention/download/free, pairwise Pearson, ordered graph energy,
sparse gather, and storage forecasting. They transfer typed physical column
slots, shapes, selected numeric profile, and bounded device resource requests.
They do **not** transfer a `FeatureId`, candidate identity, evidence name,
labels as a semantic target, graph provenance, frame/context identity,
selection policy, accepted-program authority, or any Python object. Rust
continues to own those semantics and resolves them to checked native operands.

Native operations are synchronous. Every bank keeps its producing device and
payload alive through arithmetic, transfer and teardown. `bank_free_v1` reports
device-selection/free errors and invalidates the handle only after successful
cleanup. Normally a failed `bank_retain_v1` leaves its output null. If its
error-path cleanup also fails, it returns that cleanup error with a non-null
**owned, free-only** output handle; a direct ABI caller must retry cleanup, not
use its incomplete values. The Rust wrapper adopts any such error output before
propagating the failure so RAII attempts cleanup rather than losing ownership.
Final destruction is best-effort after device/context loss: this is not a
guarantee that an unavailable runtime can reclaim device resources. No failed
arithmetic or teardown is reported as successful physical validation.

A bank represents one immutable content epoch. Source upload is single-use;
materialization and gather may initialize fresh slots but cannot overwrite
initialized values. Gathered slots can be dependencies of later operations.
This prevents a changed source or failed overwrite from leaving apparently
valid derived values. Rebinding a frame uses a new bank, while retention creates
an independently owned bank for the selected columns.

Materialization uploads operand/centering descriptors once per batch and keeps
them immutable until queued kernels finish. In particular, host writes into
managed memory must not race a preceding HIP kernel's descriptor reads. This
invariant is independent of stream ordering between kernels; no per-node host
scratch rewrite is allowed while a device may still consume it.

## Current public negotiation boundary

`TabularSession.capabilities` is the public, operation-specific record. For an
explicit CUDA or ROCm request, it reports the intersection of the loaded
payload's table/capability bits and the Rust lowering actually present in the
installed extension. Backend names alone do not establish support. A selected
payload must cover the requested profile as well as the operation and context;
otherwise the request fails explicitly without Core substitution.

| Request | Public semantic vocabulary | Evidence boundary | Status source |
|---|---|---|---|
| `core` | Source, absolute difference, softsign, ordered frozen centered product; fp32/mixed/fp64 | Pearson, Spearman, fixed corrected NMI, graph energy; reference, paired, labels, graph | Static Core policy |
| `auto` | The complete Core vocabulary above | The complete Core vocabulary above | Deliberately selects Core; a partial accelerator vocabulary is insufficient |
| explicit `cuda` / `rocm` | Only operations advertised by the complete optional table and lowered by Rust, for the selected profile | Current lowering can negotiate Pearson reference/paired measurements, sparse partial labels, and graph energy only when their relevant bits are present | Runtime payload plus Rust-lowering intersection |
| `metal` | None for this product | None for this product | Explicit semantic unsupported error |

The current accelerator semantic path intentionally does not advertise
Spearman or fixed corrected NMI. Such a request must fail as unsupported; it
must never become a hidden Core evaluation. `capabilities` is therefore more
specific than legacy backend availability. For a negotiated GPU session,
`diagnostics` reports only backend identity, retained bytes, and
`native_work_counters_available=False`; it does not invent kernel counters,
timing, occupancy, or cache-performance facts.

## Physical-validation status

Development validation on 2026-09-07 executed the installed extension with each
explicit payload sequentially: **29 physical semantic tests passed on CUDA and
29 on ROCm**, with no skipped cell in either selected-backend run. The devices
were an RTX 4060 Laptop GPU (`sm_89`, CUDA 13.3) and AMD Radeon Graphics
(`gfx1150`, system HIP runtime `70253211`, LLVM 21.1.8). The tests cover all
three precision profiles, row counts 2/33/65/257/1025/8192, heterogeneous
evidence, partial labels, retained reuse, third-round composition, unlabeled
inference, descriptor lifetime, and arithmetic definedness/overflow. Existing
native ABI fixtures also passed: 11 CUDA and 10 ROCm tests, including frozen
ABI 1.0 and generic-route consumers.

These are correctness results for the enumerated local development payloads,
not a release artifact qualification or a comparative throughput claim. Exact
source/binary hashes and logs are bound in the child PR's validation record.
Loader discovery, compilation, capability reporting, and host-only skips are
not physical execution evidence. Metal remains an explicitly unsupported
semantic route, not a pending implementation claim.

The frozen Core checkpoint has a deliberately bounded counter diagnostic, not a
throughput result: `semantic_core_sanity` processed 8,192 rows and 78 candidates
for fp32, mixed, and fp64 with workers 1/4/24 and five samples per mode. Every
cold sample recorded 180 materialized nodes and zero retained hits; the
accepted-resident mode recorded 91 materialized nodes and 78 hits; the paired
view mode recorded 90 paired-view nodes plus one required reference node. The
retained-byte snapshots were 2,555,904 for fp32/mixed and 5,111,808 for fp64,
with value-equality assertions passing. The raw
`semantic-core-sanity-8219.csv` SHA-256 is
`8f273736ee5f86ad0d3147b38b425fee5559981578fc43a896c4bdb18c2cb2c2`.
Its timing columns are non-isolated diagnostics only; this Core observation
neither measures accelerator execution nor settles accelerator performance.

This record does not expose the internal descriptor structs as public Python
objects and does not extend the public API reference with native slot layouts.
Users work through opaque `gafime.semantic` handles and the public capability
record; maintainers use the header named above for ABI detail.
