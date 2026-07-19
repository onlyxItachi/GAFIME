"""perf_05 | CUDA RT first-hit partition scale proof.

Runs the CUDA RT decision-path benchmark in the tree-leaf-like first-hit mode
when a benchmark binary is explicitly provided. This is release evidence for
RT-core saturation work, not a generic unit test.

Required:
  GAFIME_CUDA_RT_SCALE_BENCH=/tmp/cuda_rt_membership_scale_bench_new
  GAFIME_CUDA_V1_LIB=/tmp/libgafime_cuda_v1.so

Optional:
  GAFIME_CUDA_RT_FIRSTHIT_CASE=262144x8192
  GAFIME_CUDA_RT_FIRSTHIT_REPEATS=6
  GAFIME_CUDA_RT_FIRSTHIT_MIN_GEVALS=1000
  GAFIME_CUDA_RT_FIRSTHIT_MAX_ABS=1e-4

Run:
  PYTHONPATH=/home/hamza-usta/GAFIME/python:/home/hamza-usta/GAFIME/tests/release_measure \
  python3 perf_05_cuda_rt_firsthit_scale.py
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import _measure_common as mc


RT_SCORE = re.compile(
    r"^gpu_rt_score\s+(?P<ms>\d+(?:\.\d+)?) ms\s+"
    r"(?P<gevals>\d+(?:\.\d+)?) G eval/s",
    re.MULTILINE,
)
PARITY = re.compile(r"rt_max_abs=(?P<diff>[0-9.eE+-]+)")
TIMING = re.compile(
    r"^gpu_rt_score_timing\s+first_ms=(?P<first>[0-9.eE+-]+)\s+"
    r"warm_p50_ms=(?P<warm_p50>[0-9.eE+-]+)\s+"
    r"warm_best_ms=(?P<warm_best>[0-9.eE+-]+)\s+"
    r"warm_samples=(?P<warm_samples>\d+)",
    re.MULTILINE,
)
FIRSTHIT_WORK = re.compile(
    r"^firsthit work\s+groups=(?P<groups>\d+)\s+"
    r"paths_per_group=(?P<paths_per_group>\d+)\s+"
    r"rays=(?P<rays>\d+)\s+ray_rate=(?P<ray_rate>[0-9.eE+-]+) G ray/s\s+"
    r"hits=(?P<hits>\d+)\s+hit_rate=(?P<hit_rate>[0-9.eE+-]+)",
    re.MULTILINE,
)


def main() -> int:
    bench = os.environ.get("GAFIME_CUDA_RT_SCALE_BENCH")
    cuda_lib = os.environ.get("GAFIME_CUDA_V1_LIB")
    if not bench or not cuda_lib:
        print("skipped: set GAFIME_CUDA_RT_SCALE_BENCH and GAFIME_CUDA_V1_LIB")
        return 0

    bench_path = Path(bench)
    cuda_lib_path = Path(cuda_lib)
    if not bench_path.exists() or not cuda_lib_path.exists():
        print(f"skipped: missing benchmark or CUDA payload: {bench_path} {cuda_lib_path}")
        return 0

    case = os.environ.get("GAFIME_CUDA_RT_FIRSTHIT_CASE", "262144x8192")
    repeats = int(os.environ.get("GAFIME_CUDA_RT_FIRSTHIT_REPEATS", "6"))
    if repeats < 2:
        raise ValueError("GAFIME_CUDA_RT_FIRSTHIT_REPEATS must be at least 2")
    min_gevals = float(os.environ.get("GAFIME_CUDA_RT_FIRSTHIT_MIN_GEVALS", "1000"))
    max_abs = float(os.environ.get("GAFIME_CUDA_RT_FIRSTHIT_MAX_ABS", "1e-4"))
    cmd = [
        str(bench_path),
        "--score-only",
        "--partitioned-grid",
        "--overlap-axis-pairs=8",
        "--firsthit-score",
        "--rt-only",
        f"--repeats={repeats}",
        case,
    ]
    env = os.environ.copy()
    lib_parent = str(cuda_lib_path.parent)
    env["LD_LIBRARY_PATH"] = (
        lib_parent if not env.get("LD_LIBRARY_PATH") else f"{lib_parent}:{env['LD_LIBRARY_PATH']}"
    )
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    score = RT_SCORE.search(proc.stdout)
    parity = PARITY.search(proc.stdout)
    timing = TIMING.search(proc.stdout)
    firsthit_work = FIRSTHIT_WORK.search(proc.stdout)
    if score is None or parity is None or timing is None or firsthit_work is None:
        raise AssertionError(
            "benchmark output did not include score, cold/warm timing, first-hit work, and parity"
        )

    rt_ms = float(score.group("ms"))
    rt_gevals = float(score.group("gevals"))
    rt_max_abs = float(parity.group("diff"))
    first_ms = float(timing.group("first"))
    warm_p50_ms = float(timing.group("warm_p50"))
    warm_best_ms = float(timing.group("warm_best"))
    warm_samples = int(timing.group("warm_samples"))
    groups = int(firsthit_work.group("groups"))
    paths_per_group = int(firsthit_work.group("paths_per_group"))
    rays = int(firsthit_work.group("rays"))
    ray_rate = float(firsthit_work.group("ray_rate"))
    hits = int(firsthit_work.group("hits"))
    hit_rate = float(firsthit_work.group("hit_rate"))
    if rt_gevals < min_gevals:
        raise AssertionError(f"first-hit RT throughput {rt_gevals:.3f} < {min_gevals:.3f} G eval/s")
    if rt_max_abs > max_abs:
        raise AssertionError(f"first-hit RT parity diff {rt_max_abs:.6g} > {max_abs:.6g}")

    tel = mc.telemetry()
    rec = tel.new_record(
        worktree=mc.WORKTREE,
        dataset={
            "source": "synthetic",
            "name": "cuda_rt_partitioned_firsthit",
            "rows_paths": case,
        },
        config={
            "backend": "cuda",
            "gafime": {
                "measure": "cuda_rt_firsthit_scale",
                "min_gevals": min_gevals,
                "max_abs": max_abs,
            },
        },
    )
    rec["results"].update(
        {
            "status": "pass",
            "gpu_rt_score_ms": rt_ms,
            "gpu_rt_score_gevals": rt_gevals,
            "gpu_rt_first_call_ms": first_ms,
            "gpu_rt_warm_p50_ms": warm_p50_ms,
            "gpu_rt_warm_best_ms": warm_best_ms,
            "gpu_rt_warm_samples": warm_samples,
            "rt_groups": groups,
            "rt_paths_per_group": paths_per_group,
            "rt_rays": rays,
            "rt_ray_rate_g_per_s": ray_rate,
            "rt_hits": hits,
            "rt_hit_rate": hit_rate,
            "logical_work_denominator": "rows_x_paths_membership_equivalent",
            "reported_timing": "resident_warm_p50",
            "rt_max_abs": rt_max_abs,
            "command": cmd,
        }
    )
    tel.write_run(rec, mc.OUTDIR)
    print(f"artifact in {mc.OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
