# GAFIME (GPU-Accelerated Feature Interaction Mining Engine)

Internal research and engineering repository. Private development only.

## Purpose

GAFIME is a GPU-accelerated analytical engine for probing feature–target
relationships in small to medium tabular datasets. It prioritizes explainability,
robustness, and compute safety, and explicitly reports when no learnable signal
is detected.

## Intended Usage Scenarios

- Early dataset diagnostics prior to committing to modeling work.
- Identifying potential linear and non-linear dependencies.
- Discovering low-order feature interactions (k ≤ 3).

## Non-Goals

- Not an AutoML system.
- Not a model training framework.
- Not a deep learning system.
- Not a causal inference engine.

## Minimal Usage Example

```python
import numpy as np
from gafime.engine import GAFIMEEngine

x = np.random.randn(100, 4)
y = x[:, 0] * 0.5 + np.random.randn(100) * 0.1

engine = GAFIMEEngine()
report = engine.run(x, y)
print(report)
```

## Limitations

- Limited to tabular data with numeric features.
- Focused on low-order interactions (k ≤ max_comb_size).
- Not intended for high-dimensional sparse data or deep feature extraction.
