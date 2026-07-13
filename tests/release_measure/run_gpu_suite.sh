#!/usr/bin/env bash
# GPU subset of the v1 release measurement suite. Set GAFIME_GPU=cuda,
# GAFIME_GPU=rocm, or GAFIME_GPU=metal after building the matching payload.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
export PYTHONPATH="${ROOT}/python:${HERE}"
GPU="${GAFIME_GPU:-}"
PY="${GAFIME_PY:-python3}"
case "$GPU" in
  cuda | rocm | metal) ;;
  *)
    echo "GAFIME_GPU must select cuda, rocm, or metal (received: ${GPU:-<unset>})" >&2
    exit 2
    ;;
esac
export GAFIME_GRAPH_BACKEND="$GPU"
export GAFIME_BACKEND="$GPU"
export GAFIME_BACKENDS="$GPU"

GPU_SCRIPTS=(
  contract_03_family_metric_backend_surface.py
  backend_01_availability_smoke.py
  backend_03_e2e_smoke_per_backend.py
)
if [[ "$GPU" == "cuda" || "$GPU" == "rocm" ]]; then
  GPU_SCRIPTS+=(
    graph_01_replay_parity.py
    graph_02_launch_shaping_timing.py
    backend_02_cross_backend_parity.py
    perf_01_residency_session_benefit.py
    perf_02_metric_cache_benefit.py
  )
fi
if [[ "$GPU" == "cuda" ]]; then
  GPU_SCRIPTS+=(perf_05_cuda_rt_firsthit_scale.py)
fi

echo "GPU suite on backend=$GPU using $PY"
FAILURES=()
for s in "${GPU_SCRIPTS[@]}"; do
  echo "================ $s ($GPU) ================"
  if ! "$PY" "${HERE}/$s"; then
    echo "!! $s exited non-zero (continuing)"
    FAILURES+=("$s")
  fi
  echo
done
if ((${#FAILURES[@]} != 0)); then
  printf 'GPU suite failed: %s\n' "${FAILURES[*]}" >&2
  exit 1
fi
echo "GPU suite done. Telemetry artifacts in ~/gafime_telemetry/"
