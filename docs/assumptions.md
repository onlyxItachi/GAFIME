# Modeling Assumptions

- Data is tabular with numeric features.
- Target is numeric or can be mapped to numeric values.
- Feature scales are reasonable for correlation and binning-based metrics.
- Dependencies are expected to be low-order (k ≤ 3).

## Not Suitable For

- Extremely high-dimensional sparse data.
- Domains requiring causal attribution.
- Tasks where deep learned representations are required.
- Data with heavy missingness or complex mixed types without preprocessing.
