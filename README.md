# GAFIME (GPU-Accelerated Feature Interaction Mining Engine)

**Status:** Private development repository

GAFIME is an internal analytical engine for diagnosing feature–target relationships
in small to medium-sized tabular datasets. It focuses on explainable metrics,
controlled compute cost, and clear failure modes.

## Internal Purpose

- Evaluate whether a dataset contains learnable, feature-based signal
- Quantify linear and non-linear dependencies using multiple metrics
- Identify low-order feature interactions (k ≤ 3) when warranted
- Provide a structured, deterministic diagnostic report

## Intended Usage Scenarios

- Early feasibility checks before investing in model building
- Feature interaction diagnostics for tabular datasets
- Research-driven comparison of linear vs. non-linear relationships

## Non-Goals

- AutoML or model training
- Deep learning
- Causal inference
- High-order exhaustive interaction search

## Minimal Usage Example (planned API)

```python
from gafime.engine import GAFIMEEngine

engine = GAFIMEEngine(
    max_comb_size=2,
    max_combinations_per_k=500,
    top_features_for_higher_k=20,
    max_generated_features=100,
    keep_in_vram=True,
)

report = engine.analyze(features, target, feature_names)
print(report["summary"]["diagnosis"])
```

## Explicit Limitations

- Assumes low-order interactions carry signal
- Not designed for large-scale datasets or streaming workloads
- Uses binning-based mutual information (approximate)
- GPU acceleration is limited to embarrassingly parallel scoring

See `docs/` for internal design rationale and safety policies.
