# GAFIME (internal)

GPU-Accelerated Feature Interaction Mining Engine (GAFIME) is an internal analytical engine for diagnosing feature-target relationships in tabular datasets. The focus is early signal detection, explainability, and compute safety.

## Purpose (internal)
- Measure linear and non-linear relationships between features and a target.
- Identify low-order feature interactions (k <= 3) when configured.
- Provide early negative diagnoses when no learnable signal exists.

## Intended usage
- Small to medium-sized tabular datasets.
- Early-stage data diagnostics before model training.
- Controlled, repeatable experiments for feature interaction discovery.

## Non-goals
- AutoML or model training orchestration.
- Deep learning systems.
- Causal inference.
- User interfaces, notebooks, or CLI tooling.

## Minimal usage example
```python
import numpy as np
from gafime.engine import GAFIMEEngine

X = np.random.randn(100, 5)
y = X[:, 0] * 0.5 + np.random.randn(100) * 0.1

engine = GAFIMEEngine(max_comb_size=2)
report = engine.analyze(X, y)
print(report["summary"]["status"])
```

## Limitations
- Relationships are based on Pearson correlation and binned mutual information.
- Interaction discovery is capped and controlled by user parameters.
- GPU acceleration is best-effort and falls back to CPU when unavailable.
- This repository is internal and not designed for external distribution.
