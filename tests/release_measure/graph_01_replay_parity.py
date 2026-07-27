"""graph_01 | CUDA/HIP GRAPH parity: capturing+replaying the launch sequence must
not change results. Compares interaction metrics with graph=True vs graph=False
on the same data (within fp atomic-ordering tolerance). GPU-gated: requires a
CUDA or ROCm backend; skips on core.

  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 graph_01_replay_parity.py   # CUDA
"""

import math
import os

import gafime
from gafime import CompileFlags, EngineConfig

import _measure_common as mc

TOL = 1e-4  # fp32 atomic-ordering noise budget (per graph-track doc ~1e-6..1e-4)


def metrics_map(report):
    out = {}
    for ir in getattr(report, "interactions", []) or []:
        combo = tuple(ir.combo)
        if combo in out:
            raise AssertionError(f"duplicate graph result combo: {combo}")
        out[combo] = {key: float(value) for key, value in ir.metrics.items()}
    return out


def diagnostics_map(report):
    if report.backend is None or not report.backend.interaction_diagnostics_available:
        raise AssertionError("current graph payload omitted interaction diagnostics")
    out = {}
    for interaction in getattr(report, "interactions", []) or []:
        combo = tuple(interaction.combo)
        out[combo] = (
            interaction.interaction_overflow_rows,
            interaction.interaction_overflow_ratio,
            interaction.source_nonfinite,
            interaction.precision_diagnostics_available,
        )
    return out


def run(backend):
    X, y, names, _meta, _ = mc.load_synthetic_and(n=4000, f=12)
    Xl, yl = X.tolist(), y.tolist()
    cfg = dict(
        backend=backend,
        metric_names=("pearson", "r2"),
        num_repeats=1,
        permutation_tests=20,
    )  # repeated same-shape covariance launches exercise graph replay
    base = gafime.compile(
        Xl,
        yl,
        names,
        config=EngineConfig(**cfg),
        flags=CompileFlags(plan=True, graph=False),
    )
    try:
        graph = gafime.compile(
            Xl,
            yl,
            names,
            config=EngineConfig(**cfg),
            flags=CompileFlags(plan=True, graph=True),
        )
        try:
            rb, rg = base.analyze(), graph.analyze()
        finally:
            graph.close()
    finally:
        base.close()

    vb, vg = metrics_map(rb), metrics_map(rg)
    db, dg = diagnostics_map(rb), diagnostics_map(rg)
    if vb.keys() != vg.keys():
        raise AssertionError(
            f"[{backend}] graph candidate identities differ: "
            f"plain_only={len(vb.keys() - vg.keys())} "
            f"graph_only={len(vg.keys() - vb.keys())}"
        )
    if db != dg:
        raise AssertionError(
            f"[{backend}] graph diagnostics differ from plain execution"
        )
    if any(value != (0, 0.0, False, True) for value in db.values()):
        raise AssertionError(
            f"[{backend}] safe graph workload reported invalid diagnostics"
        )
    maxd = 0.0
    for combo in vb:
        if vb[combo].keys() != vg[combo].keys():
            raise AssertionError(f"[{backend}] metric identities differ for {combo}")
        for metric, plain_value in vb[combo].items():
            graph_value = vg[combo][metric]
            if not math.isfinite(plain_value) or not math.isfinite(graph_value):
                raise AssertionError(
                    f"[{backend}] non-finite {metric} value for {combo}: "
                    f"plain={plain_value} graph={graph_value}"
                )
            maxd = max(maxd, abs(plain_value - graph_value))
    if maxd > TOL:
        raise AssertionError(
            f"[{backend}] graph-vs-plain max delta {maxd:.6g} exceeds {TOL}"
        )
    print(
        f"[{backend}] results={len(vb)} max|delta| graph-vs-plain={maxd:.2e} "
        f"-> PASS (tol {TOL})"
    )


def main():
    backend = os.environ.get("GAFIME_GRAPH_BACKEND", "cuda")
    payload_env = {
        "cuda": "GAFIME_CUDA_V1_LIB",
        "rocm": "GAFIME_ROCM_V1_LIB",
    }.get(backend)
    try:
        run(backend)
    except Exception as exc:
        if payload_env is not None and os.environ.get(payload_env):
            raise
        print(
            f"[{backend}] skipped/error (needs that GPU): {type(exc).__name__}: {str(exc)[:80]}"
        )


if __name__ == "__main__":
    main()
