# Modeling Assumptions

- Inputs are tabular and fit in host memory.
- Features are numeric and can be converted to floating point.
- Targets are numeric and continuous or ordinal for diagnostic purposes.
- Relationships are summarized via low-order statistics (correlation and mutual information).

## Where GAFIME should NOT be used
- High-dimensional sparse text or image data.
- Deep feature representation learning.
- Causal inference or policy evaluation.
- Streaming datasets without a fixed batch.
