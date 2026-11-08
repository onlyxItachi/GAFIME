"""compile_01 | Compiled native candidate and resident-lifecycle correctness.

Checks result identities and configured metrics, eager/compiled parity, target
replacement, and close behavior. The Rust-owned plan is not projected to Python.

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 compile_01_plan_correctness.py
"""
import gafime
import os
from itertools import combinations

from gafime import EngineConfig

import _measure_common as mc
from compile_02_compiled_vs_eager import assert_report_parity, result_map


def main():
    backend = os.environ.get("GAFIME_BACKEND", "core")
    X, y, names, _meta, _ = mc.load_synthetic_and(n=800, f=8)
    config = EngineConfig(backend=backend)
    compiled = gafime.compile(
        X.tolist(),
        y.tolist(),
        names,
        config=config,
    )
    try:
        assert compiled.feature_names == names
        backend_info = compiled.backend
        assert backend_info is not None and backend_info.name, (
            "compiled artifact did not expose resolved backend information"
        )
        print(f"backend: {backend_info.name}")
        report = compiled.analyze()
        assert report.backend is not None and report.backend.name == backend_info.name, (
            f"analyze backend {report.backend!r} != compiled backend {backend_info.name!r}"
        )
        results = result_map(report)
        expected_combos = {
            combo
            for arity in range(1, config.budget.max_comb_size + 1)
            for combo in combinations(range(len(names)), arity)
        }
        actual_combos = [tuple(sorted(result["combo"])) for result in results.values()]
        assert len(actual_combos) == len(set(actual_combos)), "duplicate combinations"
        assert set(actual_combos) == expected_combos, "compiled candidate family differs"
        for result in results.values():
            combo = result["combo"]
            assert 1 <= len(combo) <= config.budget.max_comb_size
            assert all(0 <= index < len(names) for index in combo)
            assert set(result["metrics"]) == set(config.metric_names)
        eager = gafime.GafimeEngine(config).analyze(X.tolist(), y.tolist(), names)
        assert_report_parity(eager, report)
        updated_target = y[::-1].tolist()
        assert compiled.update_target(updated_target) is compiled
        updated = compiled.analyze()
        eager_updated = gafime.GafimeEngine(config).analyze(
            X.tolist(), updated_target, names
        )
        assert_report_parity(eager_updated, updated)
        print(f"compiled.analyze() interactions: {len(results)}")
    finally:
        compiled.close()
    assert result_map(report) == results, "report must remain readable after close"
    try:
        compiled.analyze()
    except RuntimeError as exc:
        assert "closed" in str(exc)
    else:
        raise AssertionError("closed artifact accepted analyze")
    print("COMPILED CORRECTNESS: PASS")


if __name__ == "__main__":
    main()
