# Dataload numeric validation and seed parity

`gafime.dataload` is a Polars loader and an adapter to the existing native
analysis routes. It must preserve the complete `EngineConfig`, including the
integer planning seed and binding candidate caps. It does not define a separate
numeric policy or runtime API.

## Preserve source values before checked ingest

The loader selects columns and rechunks them without narrowing their dtype.
Casting a Float64 column to Float32 in Polars can turn finite `1e39` or `1e100`
into infinity before the configured ingest path can reject it. That loses the
distinction between disallowed finite overflow and allowed source NaN/Inf.
The same rule applies to both features and the target.

The raw Arrow CPU shortcut remains a strict typed boundary: Float32 for
`fp32`/`mixed`, Float64 for `fp64`. It rejects mismatched dtypes; it does not
perform implicit numeric conversion. The adapter uses it only when the complete
configuration is supported and both frame schemas already match the selected
resident dtype. Schema inspection is metadata-only. Unknown or mixed schema
types take the existing configured ingest path, which performs its established
checked conversion before native execution. No new Python numeric validation or
data-plane loop is introduced by this change.

Consequently, a Float64 CSV/Parquet source requesting `fp32`/`mixed` uses the
configured route, not the raw Arrow shortcut. This may materialize rows and has
different ingest cost from an already-Float32 producer. It is not a zero-copy
compute path or a new performance claim. Native owned storage remains Float32
for `fp32`/`mixed` and Float64 for `fp64`; no fp32 intermediate is introduced for
the fp64 route. Existing source NaN/Inf handling, exact `f32::MAX` admission and
raw Arrow dtype rejection remain unchanged.

Input already narrowed by its producer cannot be reconstructed. An infinity
already stored in a Float32 file remains a source infinity; this fix prevents
the loader itself from creating that ambiguity.

## Seed identity

The shortcut forwards the same integer seed as the configured path. It uses
the existing config-payload entropy policy: public `random_seed=None` resolves
to a fresh integer exactly once per analysis. Rust's existing
`parse_python_integer_seed` retains all little-endian seed words for planning
and derives the established bounded significance seed. Seeds are not truncated
to u64. Default seed 7 and explicit 123 keep their established identities.

The low-level raw Arrow function adds an optional keyword seed; omitting it
keeps that convenience function's existing default seed. Non-default budgets
which the shortcut cannot express continue through the configured route rather
than being silently discarded.

Shortcut eligibility also requires default values for omitted threshold fields,
`significance_top_n`, and `mi_bins`, even when no significance or MI work is
requested. Non-default or invalid values must reach the configured Rust parser's
universal checks. The two shortcut count arguments must already be positive
integers representable as u32/u64; other requests use the configured path rather
than allowing PyO3's convenience signature to choose a different exception.
These conditions admit the shortcut; they do not replace Rust validation or
raise their own validation errors.

## Focused evidence and remaining gates

`tests/python/test_v1_public_truthfulness.py` checks seed forwarding, per-call
entropy, and conservative schema admission without a native build.
`tests/python/test_v1_ingest_contracts.py` checks top-level dataload against
direct analysis with binding per-arity and feature caps, seeds 7/123/None and
integers wider than u64, finite `1e39`/`1e100` feature and target failures for
fp32/mixed, exact f32 limits and existing NaN/Inf behavior with both Float32 and
Float64 source schemas, strict raw Arrow dtype rejection, and fp64 preservation.
Both shortcut-compatible and incompatible configurations are covered.

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
