# GAFIME Design Rationale (Internal)

## Scope Boundaries

GAFIME targets small to medium tabular datasets where feature–target signal
must be assessed quickly and safely. The system is not intended to train
predictive models; it diagnoses whether signal exists and where it may reside.

## Why Binary Search Is Excluded From Core

Binary search or hierarchical chunking introduces coupling between search
strategy and feature scoring. In early internal development we want stable,
transparent diagnostics where metrics are computed directly on features or
low-order interactions. Chunking may be explored later as a diagnostic add-on,
but it is excluded from the core pipeline to avoid hidden assumptions and
premature optimization.

## Why GPU Acceleration Is Used

Interaction scoring and metric computation are embarrassingly parallel when
structured as batched tensor operations. GPU acceleration allows GAFIME to
scale to thousands of combinations while keeping a predictable runtime profile.
CPU fallback remains the default for small workloads or when GPU resources are
not available.

## Why Multiple Relationship Metrics Are Required

No single metric captures all relationships:

- Pearson correlation detects linear dependencies.
- Mutual information (binning-based) detects non-linear associations.

GAFIME uses multiple metrics to reduce false confidence in a single view and to
support explicit disagreement when relationships are weak or inconsistent.
