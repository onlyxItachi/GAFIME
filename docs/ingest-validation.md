# Dataload native acquisition, validation and execution

`gafime.dataload` is a Polars loader and an adapter to the existing native
analysis routes. It must preserve the complete `EngineConfig`, including the
integer planning seed and binding candidate caps. It does not define a separate
numeric policy or runtime API.

## Preserve source values before checked ingest

The loader selects columns without narrowing their dtype or forcing a rechunk.
Casting a Float64 column to Float32 in Polars can turn finite `1e39` or `1e100`
into infinity before the configured ingest path can reject it. That loses the
distinction between disallowed finite overflow and allowed source NaN/Inf.
The same rule applies to both features and the target.

Numeric Polars frames now use private Arrow acquisition independently of the
execution configuration. Rust imports the Arrow C Stream incrementally, checks
schema, nulls, dimensions, counts and numeric range, and writes profile-typed
owned input. Float32/Float64, primitive integer and Boolean columns can be
converted there; fp32/mixed finite overflow is checked **before** narrowing.
Features and targets may have different batch boundaries, but their total row
counts and ordering must align. No Python row/scalar collection or full-buffer
Python byte serialization is used on this numeric path.

The resulting input feeds the existing full-config continuous, time-series or
decision-path state construction. Defaults, significance, metric options,
candidate budgets and backend selection do not become Arrow-specific policy.
Explicit unsupported backend/precision requests still fail closed. Rust owns
validation and execution; schema routing in Python inspects metadata only.

Compatible scalar schemas outside this native primitive set, such as numeric
strings, use a bounded iterator consumed by Rust: Python compatibility conversion
holds one row at a time, not a list of the entire dataset. This is a slower
compatibility path, not a newly optimized numeric producer. Native resident
storage remains Float32 for `fp32`/`mixed` and Float64 for `fp64`, with no fp32
intermediate for fp64.

Integer columns retain the loader's existing scalar conversion order (integer
to f64, then selected storage). At extreme 64-bit integer rounding midpoints,
direct NumPy integer-to-f32 conversion can differ because it skips that
intermediate. This pre-existing cross-input distinction is not silently
redefined by transport hardening; its regression oracle is the scalar loader
semantics, not a claim of universal integer/NumPy bit equivalence.

The older low-level `analyze_continuous_arrow` convenience function remains a
strict, matching-float, single-batch CPU boundary. Its restrictions no longer
control acquisition in the installed public loader. Custom older boundary
modules retain their legacy compatibility path; they do not establish the
optimized current-wheel acceptance gate.

Input already narrowed by its producer cannot be reconstructed. An infinity
already stored in a Float32 file remains a source infinity; this fix prevents
the loader itself from creating that ambiguity.

## Seed identity

Native acquisition forwards the same integer seed as the direct configured path.
It preserves the existing entropy/reseeding policy: public `random_seed=None`
uses a fresh effective execution stream rather than a deterministic cache seed.
Rust's existing
`parse_python_integer_seed` retains all little-endian seed words for planning
and derives the established bounded significance seed. Seeds are not truncated
to u64. Default seed 7 and explicit 123 keep their established identities.

The low-level raw Arrow function retains its optional keyword seed; omitting it
keeps that convenience function's existing default seed. Non-default budgets
which that convenience function cannot express are preserved by the public
loader's full-config planner. Thresholds, count ranges, MI bins and family
conflicts pass through the ordinary Rust config validation, not a parallel
Python validation policy.

## Ownership, cache identity and copies

Arrow buffer import is not resident zero-copy. Imported owners remain alive
until native validation/conversion completes and are released on both success
and failure. Each stream capsule is consumed once. Hashes and execution refer
to the same selected, Rust-owned snapshot; mutable foreign buffers are not kept
as resident storage. Callers must still prevent mutation during acquisition;
this does not provide an atomic snapshot against concurrent external writers.

When Polars provides its frame height, Rust uses it for one checked capacity
reservation and still verifies every batch and the final row count. This avoids
chunk-dependent geometric spare capacity; unknown-length streams retain amortized
growth, not repeated exact-size reallocations. The loader releases its original
frame and both projected views after acquisition, before resident construction.
The prepared input contains no foreign frame or iterator. These releases do not
promise that the source allocator immediately returns freed pages to the OS.

The acquisition buffer is row-major. The existing Core constructor copies it
into column-major compute storage; both may coexist during construction. GPU
upload, adaptive-family originals/expansion and host significance storage can
require further copies. File parsing and resident cache retention also affect
peak memory. These are justified stages to measure, not a one-copy claim.

Content digests use the established precision/count/value-byte identity. The
normal thread-local resident cache reuses features and updates changed targets
through the existing lifecycle operation. Unchanged narrow convenience-compatible
calls keep their former uncached/reporting behavior. No second loader cache,
new borrowed-memory API or GIL behavior is introduced.

## Report diagnostics

Public file analysis includes the same continuous candidate-cap warnings as
configured analysis, including unary/per-arity limits and an arity exceeding
the feature count. These warnings coexist with native aggregate interaction
overflow warnings; they do not change candidate IDs, order, metric bits, or
the no-significance decision boolean. Convenience-compatible calls retain their existing
Arrow-specific `Decision.message`, so complete report text equality is not
claimed. Full-config calls use the normal configured decision/report construction.

## Focused evidence and remaining gates

`tests/python/test_v1_public_truthfulness.py` checks seed forwarding, per-call
entropy, and conservative schema admission without a native build.
`tests/python/test_v1_ingest_contracts.py` checks top-level dataload against
direct analysis with binding per-arity and feature caps, seeds 7/123/None and
integers wider than u64, finite `1e39`/`1e100` feature and target failures for
fp32/mixed, exact f32 limits and existing NaN/Inf behavior with both Float32 and
Float64 source schemas, strict raw Arrow dtype rejection, and fp64 preservation.
Both shortcut-compatible and incompatible configurations are covered.

`tests/python/test_v1_native_ingestion.py` additionally forces the numeric route
to reject any Python row materialization or byte transport, compares complete
candidate/metric/significance content with independent NumPy acquisition, and
checks profile-typed digests, chunks, bounded compatibility and cache invalidation.
The cost/acceptance plan is in [RC3 ingestion hardening](rc3-ingestion-hardening.md).

Run native cases with `GAFIME_TEST_INSTALLED_PACKAGE=1` from outside the checkout
against a freshly built installed wheel; missing native dependencies then fail
instead of skipping. Focused tests do not replace the installed-package feature
generation gate, precision gates, physical GPU validation, performance evidence,
or release qualification. GIL, thread-local cache, Rayon and shutdown behavior
are outside this change.

The `test_integrated_dataload_*` cases specifically require the combined ingest
and Rust config-validation fixes. They check uniform ValueError rejection for
invalid omitted fields and count ranges, finite non-negative thresholds without
an invented upper-one limit, and threshold representability in the result lane.
They are installed-package tests, not evidence for the ingest-only parent wheel.
