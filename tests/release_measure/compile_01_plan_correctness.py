"""compile_01 | gafime.compile (PLAN track): a compiled artifact with plan=True
produces a scenario plan ahead of execution. Verifies the native plan shape and
that analyze() off the compiled artifact returns results. CPU-safe.

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 compile_01_plan_correctness.py
"""
import gafime
import os

from gafime import CompileFlags, EngineConfig

import _measure_common as mc


def main():
    backend = os.environ.get("GAFIME_BACKEND", "core")
    X, y, names, _meta, _ = mc.load_synthetic_and(n=800, f=8)
    config = EngineConfig(backend=backend)
    compiled = gafime.compile(
        X.tolist(),
        y.tolist(),
        names,
        config=config,
        flags=CompileFlags(plan=True),
    )
    try:
        plan = compiled.scenario_plan
        assert plan is not None, "CompileFlags(plan=True) did not expose a scenario plan"
        assert int(plan.rows) == len(X), (
            f"plan row count {plan.rows} != input row count {len(X)}"
        )
        assert int(plan.cols) == len(names), (
            f"plan column count {plan.cols} != feature count {len(names)}"
        )
        assert int(plan.max_arity) == config.budget.max_comb_size, (
            f"plan max_arity {plan.max_arity} != requested {config.budget.max_comb_size}"
        )
        metric_ids = tuple(int(value) for value in plan.metric_ids)
        assert len(metric_ids) == len(config.metric_names), (
            f"plan metric count {len(metric_ids)} "
            f"!= requested {len(config.metric_names)}"
        )
        assert len(set(metric_ids)) == len(metric_ids), (
            f"plan contains duplicate metric ids: {metric_ids}"
        )
        backend_info = compiled.backend
        assert backend_info is not None and backend_info.name, (
            "compiled plan did not expose resolved backend information"
        )

        print(
            f"scenario_plan: rows={plan.rows} cols={plan.cols} "
            f"max_arity={plan.max_arity} metrics={len(metric_ids)}"
        )
        print(f"backend: {backend_info.name}")
        report = compiled.analyze()
        assert report.backend is not None and report.backend.name == backend_info.name, (
            f"analyze backend {report.backend!r} != compiled backend {backend_info.name!r}"
        )
        n_inter = len(list(getattr(report, "interactions", []) or []))
        assert n_inter > 0, "compiled analyze produced no interactions"
        print(f"compiled.analyze() interactions: {n_inter}")
        print("PLAN CORRECTNESS: PASS")
    finally:
        compiled.close()


if __name__ == "__main__":
    main()
