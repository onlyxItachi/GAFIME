# GAFIME Repository Threat Model

## Overview

GAFIME is an in-process native feature-interaction engine. Its supported public
surface begins in Python, crosses PyO3 into Rust, and may execute on the Rust
Core backend or through a stable C ABI implemented by CUDA, ROCm/HIP, and Metal
payloads. The repository also owns wheels, source distributions, native payload
staging, frozen release bundles, checksums, provenance, and publication tooling.

The assets that matter most are:

- host-process memory and the integrity of Rust-owned and caller-owned buffers;
- CUDA, ROCm/HIP, and Metal buffers, streams, queues, command encoders, and
  device-local cached resources;
- result integrity, deterministic ranking, and precision/profile identity;
- native payload identity and the integrity of resident library and analysis
  caches;
- ABI compatibility and the safe interpretation of buffers across languages;
- wheel, sdist, frozen-bundle, checksum, and provenance integrity; and
- the isolation of experimental local RT/OptiX code from standard artifacts and
  runtime paths.

GAFIME is a local library, not a network service, authorization system,
multi-tenant compute service, or sandbox for untrusted native code. Web
authentication, CSRF, SSRF, and tenant-isolation threats are therefore not
primary repository concerns unless future product surfaces introduce them.

## Threat Model, Trust Boundaries, and Assumptions

### Actors and controllable inputs

An application may pass attacker-influenced numeric data, Arrow metadata,
feature names, dimensions, dtypes, shapes, strides, counts, enum values,
precision, backend and device selections, metric sets, budgets, and workload
sizes through supported Python APIs. A native embedding may additionally supply
ABI-visible versions, structure sizes, flags, routes, capacities, pointers, and
device identifiers. Package archives and workflow inputs are externally
supplied at the build and release boundary.

The process owner controls environment variables, explicit native-library and
module paths, the installed packages, drivers, compilers, and vendor runtimes.
Those are trusted operator choices. If a surrounding application lets an
untrusted user influence them, that application has promoted them into direct
native-code-loading boundaries and must constrain them before starting GAFIME.

Repository maintainers and CI workflows control source changes, build
credentials, artifact freezing, and publication. A compromise in that domain
can affect every downstream installer and is more severe than an ordinary local
availability failure.

### Trust boundaries

1. **Python and Arrow to PyO3/Rust.** Python objects, arrays, and Arrow C stream
   capsules become typed Rust data. Shape, dtype, length, and ownership must be
   established before constructing slices or transferring callbacks
   (`crates/gafime-py/src/common.rs`, `crates/gafime-py/src/py_api.rs`).
2. **Safe Rust to unsafe Rust and SIMD.** Safe APIs enter scoped raw-pointer,
   row-compaction, and ISA-specific code. Runtime feature detection and exact
   slice bounds must uphold the preconditions of intrinsics and pointer access
   (`crates/gafime-cpu/src/lib.rs`, `rank.rs`, and `simd/`).
3. **Rust to the native C ABI.** Rust-owned plans and buffers become ABI 1.0 or
   ABI 1.1 descriptors consumed by C++, CUDA, HIP, and Objective-C++. Versions,
   stable prefixes, routes, dtypes, counts, strides, alignment, capacities,
   reserved fields, and checked byte sizes must be validated before native
   dereference (`crates/gafime-gpu-sys/src`, `src/common/gpu_abi_impl.hpp`).
4. **Host to device.** Host descriptors and buffers cross into device memory,
   streams, queues, graphs, and encoders. The selected device, payload,
   precision profile, content generation, and asynchronous lifetime must remain
   consistent through completion and teardown (`src/cuda`, `src/rocm`,
   `src/metal`).
5. **Filesystem and environment to executable payload.** Installed-package
   discovery or an explicit `GAFIME_*_V1_LIB`/module/metallib choice selects
   executable native or Python code. Automatic discovery must reject ambiguity
   and package escape; explicit paths remain a trusted operator action
   (`python/gafime/_payloads.py`, `crates/gafime-gpu-sys/src/loader.rs`).
6. **Inputs to resident state.** Analysis, library, graph, descriptor, and
   device caches reuse native resources. Cache identity must include every
   backend, profile, device, payload, ABI, content, and lifetime dimension that
   changes interpretation (`python/gafime/v1_adapter.py` and backend launchers).
7. **Repository to release artifact.** Source and CI output become wheels,
   sdists, evidence records, and a frozen bundle. Exact manifests, checksums,
   provenance, package composition, and no-rebuild publication rules preserve
   the reviewed identity (`.github/scripts/release_bundle.py`,
   `.github/release-artifacts.json`, `.github/workflows/`).

### Required invariants

- Safe Python and Rust entry points validate all representable layout and
  capability metadata before native access.
- Element, byte-count, allocation, indexing, grid, and offset arithmetic fails
  closed on overflow or inconsistent lengths.
- ABI 1.0 stays byte-compatible; ABI 1.1 validates its stable prefix and typed
  routes; additive future records cannot overwrite known layouts or activate
  unknown required semantics.
- Unsafe Rust and ISA-specific code remain bounded by safe wrappers with
  documented ownership, alignment, feature, and lifetime assumptions.
- Device pointers, streams or queues, cached resources, and matrix handles stay
  bound to the selected backend, device, precision profile, and generation.
- Caller-owned memory remains live until synchronous return or explicit native
  completion; update, eviction, teardown, and error paths do not race in-flight
  work or free resources twice.
- Unsupported explicit backends, routes, profiles, and capabilities fail closed
  rather than silently selecting another interpretation.
- Dynamic payload auto-discovery rejects ambiguous or mismatched identities;
  standard package paths cannot escape their package root.
- Release files remain byte-identical to their reviewed frozen-bundle records,
  and standard artifacts exclude the local RT/OptiX experiment.

### Assumptions and accepted limitations

- A direct C ABI caller supplies actually allocated, aligned, accessible, and
  live buffers of the declared extent. An address alone cannot prove this;
  supported safe APIs must not expose that responsibility to untrusted data.
- Explicitly selected native libraries, Python modules, drivers, vendor
  runtimes, and compilers are trusted. GAFIME does not sandbox malicious native
  code or a malicious process owner.
- Installed payload layout/version checks and frozen SHA-256 records establish
  repository workflow identity, not general code signing or transparency.
- Some selected libraries and backend state are deliberately retained for a
  thread or process lifetime, provided identity and synchronization invariants
  hold.
- Expected out-of-memory conditions, unsupported hardware, documented numeric
  variance, and performance regressions are not security findings without a
  concrete integrity, confidentiality, memory-safety, or code-execution impact.

## Attack Surface, Mitigations, and Attacker Stories

### Python, NumPy, and Arrow inputs

Relevant failures include unchecked dimension products, dtype confusion,
malformed Arrow capsules or callbacks, premature release of externally owned
memory, and safe-API construction of invalid native layouts. Existing controls
include exact dtype and dimensionality checks, checked multiplication, owned or
borrowed-lifetime discipline, and fail-closed configuration validation.

A realistic high-impact story is an attacker-controlled array shape or Arrow
descriptor reaching a slice construction or transpose with a smaller backing
allocation, causing host out-of-bounds access. A mere Python exception or
rejection of an unsupported dtype is not a vulnerability.

### Rust ownership, raw pointers, and SIMD

The critical question is whether every safe Rust entry point proves the
preconditions of raw slice construction, unsafe row compaction, native handles,
and architecture intrinsics. Runtime ISA detection, vector-prefix bounds,
checked row layouts, and narrow unsafe functions reduce exposure. Candidate
parallelism must preserve worker-local scratch ownership so no concurrent
aliasing is introduced.

A reachable use-after-free, invalid `Send`/`Sync`, out-of-bounds compaction, or
safe wrapper that accepts dangling raw fields can be high severity. Unsafe code
that is unreachable without an already-invalid direct native caller may instead
be an accepted ABI limitation.

### ABI 1.0, ABI 1.1, and future records

ABI-visible versions, record sizes, flags, routes, dtypes, pointers, counts,
strides, and output capacities are attacker-influenced for an embedding that
parses less-trusted input. ABI 1.1 central validators check stable prefixes,
required flags, route identity, alignment, exact byte lengths, result capacity,
reserved fields, and arithmetic overflow. The ABI 1.0 compatibility paths need
equivalent independent validation. Synthetic larger records must be safely
ignored unless a required unknown semantic is present.

The important attacker story is metadata that passes validation but induces a
native out-of-bounds access or incompatible typed route. Returning a stable
validation error for an unknown route is intended behavior.

### CUDA, ROCm/HIP, and Metal execution

Device selection, matrix handles, host/device pointer consistency, stream and
queue ownership, graph/descriptor cache invalidation, and completion-before-
release are the primary concerns. Backends validate protocol/result layouts,
bind resources to a device/profile/generation, synchronize borrowed data before
return, and fail closed on unsupported routes. Metal intentionally exposes only
fp32.

A realistic high finding would require malformed supported input to cause a
device out-of-bounds access, cross-device resource confusion, or asynchronous
use after host memory was released. A missing driver, unsupported precision, or
vendor-runtime failure without a GAFIME misuse path is not reportable.

### Payload discovery and resident caches

Automatic discovery searches known distributions, requires an exact package
version and recognized payload layout, resolves package-contained paths, and
rejects ambiguous candidates. Explicit environment paths and boundary modules
deliberately load operator-selected executable code. Resident caches must not
reuse analysis or device state across incompatible payload, backend, profile,
device, content, or generation identities.

An unintended library selected despite automatic-discovery controls, a package
path escape, or an under-keyed cache that exposes stale memory could be
reportable. Deliberately setting an explicit library environment variable to an
untrusted DSO is outside the sandbox model.

### Packages, archives, CI, and publication

Release tooling handles untrusted archive metadata and privileged artifact
identities. Relevant classes include archive path traversal, duplicate archive
entries, decompression or allocation exhaustion, shell injection through
workflow inputs, artifact substitution, and publication of bytes not bound to
the reviewed source. Existing controls reject symlinks and unexpected files,
derive the exact package set from the manifest, recompute SHA-256, bind source
and run identity, and publish the frozen bundle without rebuilding.

Compromise that substitutes a release artifact for downstream users can be
critical. A developer-only command that consumes only trusted maintainer input
is lower risk unless CI or a release boundary makes that input externally
controllable.

### Experimental local RT/OptiX path

RT/OptiX is off by default, feature-gated, locally loaded, and excluded from
standard payload staging and distributions. Its raw handles, per-device native
state, callbacks, and teardown still require memory-safety and synchronization
review, but its isolation narrows reachability. Accidental inclusion or
activation in a standard artifact is reportable; ordinary failure of an
explicit local experiment is not automatically a security issue.

## Severity Calibration

### Critical

- Practical arbitrary code execution through a supported Python/Rust input or
  default payload-discovery path.
- Release-bundle or publication-integrity bypass that substitutes malicious
  artifacts for downstream users.

Critical severity requires realistic reachability without already trusting the
attacker as the process owner, payload author, driver, or compiler provider.

### High

- Reachable host or device memory corruption, meaningful arbitrary read/write,
  use-after-free, or cross-resource sensitive-data disclosure.
- Malformed supported ABI metadata that bypasses validation and reaches an
  out-of-bounds native operation.
- Automatic loading of an unintended executable payload or exploitable
  asynchronous device/resource lifetime confusion.

### Medium

- Constrained unauthorized file access, integrity loss, cache-identity
  confusion, or information disclosure with a realistic supported entry point.
- A CI or archive-handling weakness that affects pre-release evidence or a
  bounded artifact but does not enable general release substitution.
- A resource-lifetime or validation defect with demonstrable security impact
  under non-default but supported embedding conditions.

### Low

- Defense-in-depth gaps whose exploitation requires a trusted operator to
  supply a malicious explicit payload or otherwise violate the documented
  process trust model.
- Bounded resource exhaustion or diagnostic information exposure without
  sensitive contents, cross-boundary impact, or release-integrity consequences.
- Hardening opportunities where current validation and reachability evidence
  do not establish a concrete security outcome.

Exact candidate binding: the canonical Codex Security scan record binds this
version-controlled model to the reviewed source SHA. The human-readable model
does not embed a self-referential repository hash.
