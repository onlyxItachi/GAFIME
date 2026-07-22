from __future__ import annotations

import json
from pathlib import Path


def generate_tutorial(output_path: str = "gafime_tutorial.ipynb") -> str:
    """Generate a runnable practice notebook for the current v1 public API."""

    cells = [
        _md(
            "# GAFIME v1 Practice Notebook\n\n"
            "This notebook uses bounded deterministic data and the public API. "
            "It probes capabilities before running continuous, compiled, "
            "time-series, decision-path, and selector examples."
        ),
        _md("## 1. Version and Capability Probe"),
        _code(
            "import gafime\n"
            "from gafime import backend_capabilities\n\n"
            "print('GAFIME', gafime.__version__)\n"
            "caps = backend_capabilities('auto', probe=True)\n"
            "print('configured:', caps.configured_backend)\n"
            "print('selected:', caps.selected_backend)\n"
            "print('status:', caps.selection_status)\n"
            "print('device:', caps.device.value)"
        ),
        _md("## 2. Deterministic Practice Data"),
        _code(
            "X = [\n"
            "    [float(i), float((i * 7) % 11), float((i % 5) - 2)]\n"
            "    for i in range(64)\n"
            "]\n"
            "y = [0.4 * row[0] * row[1] - 0.2 * row[2] for row in X]\n"
            "feature_names = ['trend', 'cycle', 'offset']\n"
            "len(X), len(X[0])"
        ),
        _md("## 3. Reproducible Core Analysis"),
        _code(
            "from gafime import ComputeBudget, EngineConfig, GafimeEngine\n"
            "config = EngineConfig(\n"
            "    metric_names=('pearson', 'r2'),\n"
            "    backend='core',\n"
            "    permutation_tests=0,\n"
            "    num_repeats=1,\n"
            "    budget=ComputeBudget(\n"
            "        max_comb_size=2, max_combinations_per_k=64\n"
            "    ),\n"
            ")\n"
            "report = GafimeEngine(config).analyze(X, y, feature_names)\n"
            "print(report.backend.selected_backend, report.backend.execution_placement)\n"
            "list(report.interactions.top_k(5, metric_name='pearson'))"
        ),
        _md("## 4. Auto Backend Selection"),
        _code(
            "from dataclasses import replace\n\n"
            "auto_config = replace(config, backend='auto')\n"
            "auto_report = GafimeEngine(auto_config).analyze(X, y, feature_names)\n"
            "print('selected:', auto_report.backend.selected_backend)\n"
            "print('warnings:', auto_report.warnings)"
        ),
        _md("## 5. Explicit Compiled Artifact"),
        _code(
            "from gafime import CompileFlags, compile\n"
            "artifact = compile(\n"
            "    X, y, feature_names, config=config,\n"
            "    flags=CompileFlags(plan=True, graph=False, export=False),\n"
            ")\n"
            "try:\n"
            "    compiled_report = artifact.analyze()\n"
            "    print('compiled backend:', artifact.backend)\n"
            "    print(compiled_report.interactions.top_k(2, 'pearson'))\n"
            "finally:\n"
            "    artifact.close()"
        ),
        _md("## 6. Time-Series Generated Family"),
        _code(
            "time_series_config = EngineConfig(\n"
            "    backend='core',\n"
            "    metric_names=('pearson', 'r2'),\n"
            "    enable_time_series_functions=True,\n"
            "    time_series_lags=(1, 2),\n"
            "    time_series_windows=(4,),\n"
            "    permutation_tests=0,\n"
            "    num_repeats=1,\n"
            "    budget=ComputeBudget(\n"
            "        max_comb_size=1,\n"
            "        max_combinations_per_k=128,\n"
            "        max_time_series_candidates=32,\n"
            "        top_k_features_for_time_series=3,\n"
            "    ),\n"
            ")\n"
            "time_report = GafimeEngine(time_series_config).analyze(\n"
            "    X, y, feature_names\n"
            ")\n"
            "print(time_report.warnings)\n"
            "list(time_report.interactions.top_k(5, metric_name='pearson'))"
        ),
        _md(
            "Time-series lags and windows use the supplied row order. Sort input "
            "first and partition entity groups outside GAFIME so windows do not "
            "cross group boundaries."
        ),
        _md("## 7. Decision-Path Generated Family"),
        _code(
            "decision_config = EngineConfig(\n"
            "    backend='core',\n"
            "    metric_names=('pearson', 'r2'),\n"
            "    enable_decision_path_functions=True,\n"
            "    decision_path_max_depth=2,\n"
            "    decision_path_max_paths=8,\n"
            "    decision_path_min_leaf=4,\n"
            "    permutation_tests=0,\n"
            "    num_repeats=1,\n"
            "    budget=ComputeBudget(max_comb_size=1),\n"
            ")\n"
            "decision_report = GafimeEngine(decision_config).analyze(\n"
            "    X, y, feature_names\n"
            ")\n"
            "print(decision_report.warnings)\n"
            "list(decision_report.interactions.top_k(5, metric_name='pearson'))"
        ),
        _md(
            "Decision-path bootstrap stability is supported. Permutation "
            "significance is unavailable because every permuted target would "
            "require path rediscovery, so this family must use "
            "`permutation_tests=0`."
        ),
        _md("## 8. Family Capability Disclosure"),
        _code(
            "from gafime import available_families\n\n"
            "for family in available_families():\n"
            "    significance = family.significance_support\n"
            "    print(\n"
            "        family.name,\n"
            "        'generation=', family.generation_placement,\n"
            "        'scoring=', family.scoring_backends,\n"
            "        'permutation=', significance.permutation,\n"
            "        'stability=', significance.stability,\n"
            "    )"
        ),
        _md("## 9. sklearn-Style Pair Selection"),
        _code(
            "from gafime import GafimeSelector\n"
            "selector = GafimeSelector(k=1, metric='pearson')\n"
            "augmented = selector.fit_transform(X, y)\n"
            "print('selected pairs:', selector.top_interactions_)\n"
            "print('shape:', len(augmented), 'x', len(augmented[0]))"
        ),
        _md(
            "For model evaluation, place `GafimeSelector` inside a scikit-learn "
            "Pipeline so discovery is refit on every training fold. Install the "
            "optional integration with `pip install \"gafime[sklearn]\"`."
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
                "purpose": "Runnable practice notebook for the public GAFIME API",
                "release_scope": "GAFIME v1 public API",
                "generator": "python/gafime/tutorial.py",
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
