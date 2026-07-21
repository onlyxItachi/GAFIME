# CPU Continuous Interaction Fusion Evaluation

The arity-specialized fused Pearson/R2 candidate was evaluated and is not
enabled in the production CPU scorer. Arity 1 remains on the existing
slice-based SIMD covariance path. Every higher-arity request materializes one
reusable interaction vector and runs all requested metrics through the existing
slice kernels. This includes the default Pearson, Spearman, mutual information,
and R2 set, so it does not combine fused passes with materialization.

Fixed-bin MI was not fused. Its parity-tested implementation filters compact
finite pairs before invoking the existing SIMD bin arithmetic and histogram
scatter. Removing its retained interaction vector safely still requires a
dedicated fused min/max and histogram design that preserves exact counts and
non-finite semantics across scalar and SIMD dispatch. Adaptive MI also needs
signal-shaped data for sorting.

## Release Measurement

Temporary untracked harnesses compared reusable fused scoring with the former
reusable materialized-vector route in an AVX-512 release build. Each required
shape used 15 timed samples and about 4.2 million candidate-row visits per
sample; the table reports the median ratio from three independent runs. Ratio
is materialized time divided by fused time, so values above 1 favor fusion.
The arity-2 values use the exact dynamic metric-loop route; arities 3 through 5
use the direct candidate/reference comparison.

| Rows | Arity 2 | Arity 3 | Arity 4 | Arity 5 |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.972x | 0.886x | 0.900x | 0.793x |
| 16,384 | 0.946x | 0.860x | 0.875x | 0.769x |
| 262,144 | 1.008x | 0.895x | 0.904x | 0.792x |

The isolated arity-2 gain at 262,144 rows did not define a safe gate: an
additional 1,048,576-row, 15-sample run produced `0.953x`. Pearson-only and
R2-only arity-2 requests were also consistently slower at the required sizes:

| Rows | Pearson only | R2 only | Pearson + R2 |
| ---: | ---: | ---: | ---: |
| 1,024 | 0.540x | 0.556x | 0.972x |
| 16,384 | 0.522x | 0.521x | 0.946x |
| 262,144 | 0.558x | 0.560x | 1.008x |

The benchmark dataset observed zero absolute Pearson/R2 drift and equal output
bits against the previous SIMD materialized implementation at every required
shape. That is local measurement evidence, not a public bit-parity guarantee.
Focused tests continue to enforce the established scalar-reference tolerances,
exact NaN/Infinity and constant-column behavior, and materialized output/ranking
identity for arities 1 through 5.

No Criterion benchmark or profiler report was generated, so PerfDigest was not
applicable. The production decision is based on measured runtime ratios, not an
allocation-theory claim.
