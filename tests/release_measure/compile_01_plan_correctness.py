"""compile_01 | gafime.compile (PLAN track): a compiled artifact with plan=True
produces a scenario plan ahead of execution. Verifies the plan exists, carries
chunk/scenario structure, surfaces warnings, and that analyze() off the compiled
artifact returns results. CPU-safe.

  PYTHONPATH=/home/hamza-usta/GAFIME-integration \
  /home/hamza-usta/.venvs/gafime-dl-py314/bin/python compile_01_plan_correctness.py
"""
import gafime
import os

from gafime import CompileFlags, EngineConfig

import _measure_common as mc


def main():
    backend = os.environ.get("GAFIME_BACKEND", "core")
    X, y, names, meta, _ = mc.load_synthetic_and(n=800, f=8)
    compiled = gafime.compile(X.tolist(), y.tolist(), names,
                              config=EngineConfig(backend=backend),
                              flags=CompileFlags(plan=True))
    try:
        plan = compiled.scenario_plan
        print(f"scenario_plan present: {plan is not None}")
        print(f"backend: {compiled.backend.name}")
        print(f"plan warnings: {list(getattr(plan, 'warnings', []) or [])[:3]}")
        # introspect whatever structure the plan exposes (chunk ranges / scenarios / combos)
        for attr in ("scenarios", "batches", "chunk_ranges", "combos", "total_combinations"):
            if hasattr(plan, attr):
                v = getattr(plan, attr)
                try:
                    print(f"  plan.{attr}: len={len(v)}")
                except TypeError:
                    print(f"  plan.{attr}: {v}")
        report = compiled.analyze()
        n_inter = len(list(getattr(report, "interactions", []) or []))
        print(f"compiled.analyze() interactions: {n_inter}")
        print(f"PLAN CORRECTNESS: plan present={plan is not None}, analyze produced={n_inter>0}")
    finally:
        compiled.close()


if __name__ == "__main__":
    main()
