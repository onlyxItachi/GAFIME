# CPU Fused Continuous Accumulation

The CPU continuous kernel uses compile-time arity specializations for Pearson
and R2 interaction scoring at arities 2 through 5. Each pass recreates the
centered interaction in the same ordered `f32` multiplication sequence as the
former vector materialization, then accumulates finite Pearson pairs in `f64`.
The second pass computes centered covariance terms without retaining a
row-sized interaction vector.

Arity 1 remains on the existing slice-based SIMD covariance path. Arity values
outside 2 through 5 and metric requests that need a signal slice retain the
materialized path. In particular, Spearman behavior is unchanged.

Fixed-bin MI was deliberately not fused in this change. Its parity-tested
implementation filters compact finite pairs before invoking the existing SIMD
bin arithmetic and histogram scatter. Removing its retained interaction
vector safely would require a dedicated fused min/max and histogram design
that preserves those exact counts and non-finite semantics across scalar and
SIMD dispatch. Adaptive MI also needs signal-shaped data for sorting. No
throughput claim is made here because this change has no benchmark or profiler
capture.

Focused CPU tests cover arities 1 through 5, scalar and materialized Pearson/R2
references, NaN/Infinity filtering, constant columns, mixed Spearman/fixed-MI
requests, and ranked arity-2 output. The fused scalar reduction is bit-equal to
the scalar reference for arities 2 through 5; comparison with the pre-existing
dispatched covariance path uses its established `5e-5` Pearson and `1e-4` R2
test bounds.
