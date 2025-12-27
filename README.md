# GAFIME (internal)

GAFIME (GPU-Accelerated Feature Interaction Mining Engine) is an internal analytical
engine for diagnosing feature–target relationships in small to medium tabular datasets.
It focuses on explainability, robustness, and compute safety, rather than model training.

## Intended usage
- Rapidly assess whether a dataset contains learnable feature-based signal.
- Examine unary and low-order interaction effects (k ≤ 3) without deep learning.
- Produce a structured JSON report for downstream analysis.

## Non-goals
- AutoML, hyperparameter search, or model training frameworks.
- Deep learning, representation learning, or causal inference.
- UI/CLI tools or notebooks.

## Minimal usage example
```python
import numpy as np
from gafime.engine import GAFIMEEngine

X = np.random.randn(100, 4)
y = X[:, 0] * 0.7 + np.random.randn(100) * 0.1

engine = GAFIMEEngine()
report = engine.analyze(X, y)
print(report["summary"])
```

## Limitations
- Designed for small to medium tabular data; very wide datasets may require
  aggressive combination limits.
- GPU acceleration is optional; CPU-only execution is supported.
- Metrics are correlation and binning-based mutual information only.
- No model training or predictive guarantees are provided.
