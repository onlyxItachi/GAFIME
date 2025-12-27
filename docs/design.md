# GAFIME internal design notes

## Core intent
GAFIME is an internal engine for diagnosing feature–target relationships in
small to medium tabular datasets. The design emphasizes transparent metrics,
controlled computation, and clear negative results.

## Why binary search is excluded from the core
Binary search and hierarchical chunking are intentionally excluded from the
core pipeline to avoid adding speculative heuristics during early research.
The baseline system uses direct, bounded enumeration with explicit limits so
results are predictable and debuggable. Optional experiments may reintroduce
hierarchical search later.

## Why GPU acceleration is used
Some metric computations and interaction scoring are embarrassingly parallel.
GPU acceleration reduces wall-clock time when evaluating many feature
combinations, provided that memory usage is strictly controlled.

## Why multiple relationship metrics are required
A single linear metric is insufficient for diagnosing complex dependencies.
GAFIME requires both Pearson correlation and binning-based mutual information
so that linear and non-linear relationships can be detected with consistent,
interpretable outputs.
