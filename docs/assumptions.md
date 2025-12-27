# Modeling assumptions

- Input data is tabular and fits in host memory.
- Features are numeric or can be pre-encoded as numeric values.
- Targets are numeric for correlation and mutual information scoring.
- Relationships are assessed without training predictive models.

## When NOT to use GAFIME
- Very high-dimensional feature spaces without tight combination limits.
- Streaming or real-time inference workloads.
- Problems requiring causal inference or counterfactual analysis.
- Tasks needing model training, ensembling, or AutoML capabilities.
