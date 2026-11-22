# Native boundary snapshots and developer smoke selection

This design and implementation note describes bounded native-boundary
hardening. It does not change public ABI layouts, numeric profiles, candidate
planning, Python APIs, package topology, or the GIL contract. The source of
truth for ABI compatibility remains [ABI evolution](abi-evolution.md).

## Metal result finalization

Metal finalizes GPU-produced metrics into caller-owned host result spans.
The writer uses a private snapshot of the result pointers and strides and the
launch header, so writes cannot change the bounds used by later writes.
If an actual output-write span intersects the chunk or combination descriptors
that finalization will read, those descriptors are copied before scattering.
The ordinary disjoint Rust-owned path needs no additional descriptor copy.
Only the completed row count is committed back; the result header is not
wholesale copied over caller storage.

This does not promise meaningful simultaneous values for output arrays that
overlap each other, nor make inaccessible pointers valid. Callers still own
the synchronous buffer lifetimes. No new blanket non-overlap requirement is
imposed on the frozen C ABI. Arrow release callbacks and resident ownership are
unchanged. Host-only writer tests do not establish execution on Apple hardware.

## CUDA chunk-derived state

The nonzero immutable descriptor generation identifies reusable combo/metric
device storage. Host covariance-conditioning state additionally depends on
the current validated chunk partition. CUDA keeps an exact copy of the chunk
arity, descriptor offset and candidate-count interpretation; changed count or
partition refreshes that derived state without re-uploading unchanged device
descriptors. Family-only changes continue to use the existing graph identity.
Upload and target update invalidate this host state along with descriptors.

All scoring metrics and graph comparison/storage check the mode-vector length
before indexed access. The same helpers serve ordinary and permutation graph
bookkeeping, both ABI generations, and all three CUDA precision profiles.
Changing combo/metric contents without advancing the generation remains a
caller-contract violation; this change does not make such mutation supported.
No kernel equation, ordering rule, or graph/lifecycle ownership changes.

## Opt-in developer GPU gate

`v1_architecture_gate.py --include-gpu` requires explicit smoke executables
built from the trusted selected source. It accepts regular executable files
owned by the invoking user or root, with no group/other write permission, and
checks canonical parent ownership and permissions. A private directory below
root-owned sticky `/tmp` is allowed; an executable directly in a shared writable
directory is not. Canonical symlink targets are checked and executed.

The boundary is replacement by another local UID. The developer/root, compiler,
source selection and explicitly selected payloads remain trusted. Filesystem
ownership alone is not source provenance. The POSIX-only selection check fails
closed elsewhere; ordinary non-GPU and `--skip-cargo` behavior is unchanged.
All three CUDA, local RT-CUDA and ROCm smoke routes retain their original order
and required execution checks. This does not put RT into normal distributions.

## Focused evidence and remaining release work

- `tests/gpu/metal_result_metadata_test.mm` exercises the actual host writer,
  valid disjoint/padded/ranked results and alias controls, without device calls.
- `tests/gpu/cuda_descriptor_state_test.cu` is compiled by NVCC and exercises
  actual host cache/graph helpers without allocating or executing on a GPU.
  The Linux CUDA CTest build registers it; this is not Windows CUDA evidence.
- `tests/python/test_gpu_gate_smoke_paths.py` checks every developer smoke
  selector without running any smoke, Cargo or GPU work. Ownership cases are
  emulated; they are not a cross-UID deployment test.

Host sanitizers and these focused checks supplement, not replace, the configured
CI, installed-payload numerical/ABI/lifecycle checks and exact-source security
qualification. Historical sealed scans remain tied to their original SHA;
these local source changes are not already qualified by those scans.
