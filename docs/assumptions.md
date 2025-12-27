# Modeling Assumptions (Internal)

- Signal, if present, is captured by unary or low-order (k ≤ 3) interactions.
- Features and target are provided as numeric arrays after basic preprocessing.
- Relationship metrics are approximate and intended for ranking, not estimation.

## When Not To Use GAFIME

- Very large datasets where exhaustive interaction scoring is impractical.
- High-dimensional sparse data where interactions are predominantly high-order.
- Domains requiring causal inference or counterfactual reasoning.
- Streaming or real-time systems without controlled batch windows.
