# GAFIME Design Notes

## Binary Search Exclusion

Binary search or hierarchical chunking is intentionally excluded from the core
pipeline to avoid adding opaque heuristic control flow at this stage. The goal
is a predictable, debuggable, and linear pipeline; optional diagnostics can be
added later.

## GPU Acceleration

GPU acceleration is used to speed up embarrassingly parallel metric computation
and batched interaction scoring. This aligns with the objective of rapid
diagnosis on moderate-sized datasets while retaining deterministic logic.

## Multiple Relationship Metrics

Both linear and non-linear metrics are required to avoid over-reliance on a
single signal type. Pearson correlation captures linear relationships, while
mutual information (binning-based) surfaces non-linear dependencies.
