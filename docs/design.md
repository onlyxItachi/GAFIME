# GAFIME Internal Design Notes

## Excluding binary search from the core pipeline
Binary search and hierarchical chunking are intentionally excluded from the initial core. They can bias diagnostics, complicate deterministic evaluation, and introduce additional heuristics that obscure failure modes. This keeps the initial system fully transparent and easier to validate.

## Why GPU acceleration
The core metrics (correlation and mutual information) are embarrassingly parallel over features and feature pairs. GPU acceleration provides a high-throughput path for large numbers of comparisons while keeping the logic simple and testable.

## Why multiple relationship metrics
No single metric is sufficient to capture the breadth of relationships in tabular data. Pearson correlation detects linear relationships while binned mutual information captures non-linear dependencies. Reporting both is required for a reliable diagnostic signal.
