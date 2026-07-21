from __future__ import annotations

import json
from pathlib import Path


def generate_tutorial(output_path: str = "gafime_tutorial.ipynb") -> str:
    """Generate a starter notebook using the current public Python API."""

    cells = [
        _md("# GAFIME Starter Tutorial"),
        _md("## 0. Environment"),
        _code("import gafime\nprint(gafime.__version__)\nprint(gafime.__all__)"),
        _md("## 1. Eager Analysis"),
        _code(
            "from gafime import ComputeBudget, EngineConfig, GafimeEngine\n"
            "X = [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [3.0, 8.0]]\n"
            "y = [0.0, 1.0, 2.0, 3.0]\n"
            "names = ['a', 'b']\n"
            "config = EngineConfig(\n"
            "    metric_names=('pearson', 'r2'),\n"
            "    backend='auto',\n"
            "    permutation_tests=0,\n"
            "    num_repeats=1,\n"
            "    budget=ComputeBudget(max_comb_size=2),\n"
            ")\n"
            "report = GafimeEngine(config).analyze(X, y, names)\n"
            "print(report.backend)\n"
            "print(list(report.interactions))"
        ),
        _md("## 2. Compiled Analysis"),
        _code(
            "from gafime import CompileFlags, compile\n"
            "artifact = compile(\n"
            "    X, y, names, config=config, flags=CompileFlags(export=True)\n"
            ")\n"
            "try:\n"
            "    compiled_report = artifact.analyze()\n"
            "    print(compiled_report.interactions.top_k(2, 'pearson'))\n"
            "finally:\n"
            "    artifact.close()"
        ),
        _md("## 3. Optional Candidate Families"),
        _code(
            "decision_config = EngineConfig(\n"
            "    enable_decision_path_functions=True, permutation_tests=0\n"
            ")\n"
            "time_series_config = EngineConfig(\n"
            "    enable_time_series_functions=True, time_series_lags=(1, 2)\n"
            ")\n"
            "print(decision_config)\n"
            "print(time_series_config)"
        ),
        _md("## 4. File Streaming"),
        _code("from gafime import GafimeStreamer\nprint(GafimeStreamer)"),
        _md("## 5. sklearn-Style Selection"),
        _code(
            "from gafime import GafimeSelector\n"
            "selector = GafimeSelector(k=1, metric='pearson')\n"
            "print(selector.fit_transform(X, y))"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "gafime_reference": {
                "purpose": "Starter tutorial for the public GAFIME API",
                "release_scope": "Current v1 Python API",
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


__all__ = ["generate_tutorial"]
