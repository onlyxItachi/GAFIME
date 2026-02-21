# GAFIME: GPU-Accelerated Feature Interaction Mining Engine 🚀

![PyPI version](https://img.shields.io/pypi/v/gafime)
![Python Versions](https://img.shields.io/pypi/pyversions/gafime)
![License](https://img.shields.io/github/license/onlyxItachi/GAFIME)

GAFIME is a high-performance computing engine engineered to eliminate the biggest bottleneck in modern machine learning workflows: **Feature Interaction Discovery.**

While most data science tools prioritize ease-of-use over execution efficiency, GAFIME treats feature engineering as a low-level systems problem. By combining C++ optimization, Rust memory-safety pipelines, and cross-platform native bindings (CUDA/Metal), GAFIME bridges the gap between high-level data science and the raw power of modern hardware architectures.

## 📦 Installation

GAFIME ships natively compiled wheel binaries for Windows, macOS (Apple Silicon), and Linux heavily optimized for performance out-of-the-box.

**Basic Install (Engine Only):**

```bash
pip install gafime
```

**Data Science Install (Includes Scikit-Learn Wrapper):**

```bash
pip install gafime[sklearn]
```

## ⚡ Quickstart: The Interactive Tutorial

The fastest way to understand GAFIME's speed is to try our built-in interactive tutorial generator. Running this command will generate a pre-configured `gafime_tutorial.ipynb` Jupyter Notebook in your current directory with dummy feature data to instantly evaluate against:

```bash
gafime --init
```

## 🧩 Scikit-Learn Pipeline Integration

You don't need to rewrite your data pipelines to use GAFIME. By importing the `GafimeSelector`, you can inject GPU-accelerated feature discovery natively into `sklearn.pipeline.Pipeline` or `GridSearchCV`:

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from gafime.sklearn import GafimeSelector

# Define dummy data
X_train = np.random.randn(1000, 50).astype(np.float32)
y_train = np.random.randint(0, 2, size=1000).astype(np.float32)

# Create a pipeline that automatically discovers the Top 5 best Feature Interactions
# Evaluated instantly against the GPU logic and appends them to your training dataset
pipe = Pipeline([
    ('interaction_miner', GafimeSelector(k=5, backend='auto', operator='multiply')),
    ('classifier', LogisticRegression())
])

pipe.fit(X_train, y_train)
```

## 🌌 Why GAFIME? The Performance Ceiling

In the current data science landscape, mining interaction data (like checking `Feature X * Feature Y` against the target) is painfully slow on CPUs or inefficiently memory-managed on GPUs. GAFIME achieves:

1. **Hardware-Bound Execution**: GAFIME targets physical memory bandwidth limits, minimizing the overhead of standard GPU python workflows. You hit the system's ceiling.
2. **Zero-Overhead Scaling**: Utilizing Rust's FFI capabilities on top of optimized CUDA C++, GAFIME bypasses the Python Global Interpreter Lock (GIL) ensuring every clock cycle executes pure feature logic.
3. **Cross-Platform Scalability**: Whether you're on a MacBook executing `Metal` fallback logic via Rust, or an RTX workstation targeting `CUDA` registers, GAFIME auto-discovers and optimizes for your hardware at runtime.

### Caching and Branch-less Operations

GAFIME's specialized memory management layout ensures tabular feature data is cached and aligned, dramatically minimizing GPU cycle stalls regardless of noisy dataset inputs.

## 🛠️ Technology Stack

- **Core Engine**: C++ / CUDA (Performance-critical computation paths) and **Metal** (Apple Silicon native acceleration)
- **Safety Pipeline & Schedulers**: Rust (Memory safe FFI interface scheduling)
- **Data Science Interfacing**: Python (Polars / Numpy bindings seamlessly communicating across boundaries)

## ✅ For being honest

-> Current state of the project is its **first v0.1 open beta release**.

-> The project is developed with the help of current frontier SOTA models such as Gemini 3.1 Pro (high reasoning effort) and Claude Opus 4.6 (high). The state of project is clearly working on my personal computer! But I am not guaranteeing that at this stage "you could run it on your device as well!"

## 🤝 If you want

You could collaborate with me via using email to communicate 🥰

Email: <hamzausta2222@gmail.com>

---

### Contributing and Advanced Usage

Looking to expand the engine metrics or compile natively yourself?
Please see our detailed references:

- [USAGE.md](/USAGE.md) - Advanced `EngineConfig` features and API logic.
- [CONTRIBUTING.md](/CONTRIBUTING.md) - Local compilation instructions for OS developers.

*GAFIME was conceptualized and engineered for extreme high-frequency feature permutations in complex categorical environments like Banking models.*
