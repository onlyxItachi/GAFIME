from __future__ import annotations

import json
from pathlib import Path


def generate_tutorial(output_path: str = "gafime_tutorial.ipynb") -> str:
    cells = [
        _md("# GAFIME v0.4.6 Native API Reference"),
        _md("## 0. Environment"),
        _code("import gafime\nprint(gafime.__version__)\nprint(gafime.__all__)"),
        _md("## 1. Native Engine"),
        _code(
            "from gafime import GafimeEngine, EngineConfig, ComputeBudget\n"
            "X = [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [3.0, 8.0]]\n"
            "y = [0.0, 1.0, 2.0, 3.0]\n"
            "cfg = EngineConfig(metric_names=('pearson', 'r2'), backend='auto', permutation_tests=0, num_repeats=1)\n"
            "report = GafimeEngine(cfg).analyze(X, y, feature_names=['a', 'b'])\n"
            "print(report.backend)\n"
            "print(report.interactions[:3])"
        ),
        _md("## 2. ComputeBudget"),
        _code(
            "budget = ComputeBudget(max_comb_size=3, max_combinations_per_k=1000)\n"
            "print(budget)"
        ),
        _md("## 3. Discrete Functions"),
        _code(
            "cfg = EngineConfig(\n"
            "    backend='auto',\n"
            "    metric_names=('pearson', 'mutual_info'),\n"
            "    enable_discrete_functions=True,\n"
            "    discrete_mode='soft',\n"
            "    permutation_tests=0,\n"
            "    num_repeats=1,\n"
            ")\n"
            "report = GafimeEngine(cfg).analyze(X, y, feature_names=['a', 'b'])\n"
            "print([r.family for r in report.interactions])"
        ),
        _md("## 4. Native-Only Backend Policy"),
        _code(
            "from gafime.backends import resolve_backend\n"
            "from gafime.native_data import coerce_inputs\n"
            "Xn, yn, _ = coerce_inputs(X, y)\n"
            "for backend in ['auto', 'cuda', 'core']:\n"
            "    try:\n"
            "        b, warnings = resolve_backend(EngineConfig(backend=backend), Xn, yn)\n"
            "        print(backend, '->', b.info())\n"
            "    except Exception as exc:\n"
            "        print(backend, '->', exc)"
        ),
        _md("## 5. Streamer"),
        _code("from gafime import GafimeStreamer\nprint(GafimeStreamer)"),
        _md("## 6. sklearn-Style Selector"),
        _code(
            "from gafime.sklearn import GafimeSelector\n"
            "selector = GafimeSelector(k=1, backend='auto', metric='pearson')\n"
            "try:\n"
            "    print(selector.fit_transform(X, y))\n"
            "except Exception as exc:\n"
            "    print(exc)"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "gafime_reference": {
                "purpose": "Native-only GAFIME v0.4.6 API reference",
                "sections": len(cells),
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = Path(output_path)
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return str(path)


def _md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source + "\n"]}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }
