#!/usr/bin/env bash
# GPU subset of the v1 release measurement suite. Pass GAFIME_GPU=cuda,
# GAFIME_GPU=rocm, or GAFIME_GPU=metal after building the matching payload.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
export PYTHONPATH="${ROOT}/python:${HERE}"
GPU="${GAFIME_GPU:-cuda}"
PY="${GAFIME_PY:-python3}"
export GAFIME_GRAPH_BACKEND="$GPU"
export GAFIME_BACKEND="$GPU"

GPU_SCRIPTS=(
  contract_03_family_metric_backend_surface.py
  graph_01_replay_parity.py
  graph_02_launch_shaping_timing.py
  backend_02_cross_backend_parity.py
  backend_03_e2e_smoke_per_backend.py
  perf_01_residency_session_benefit.py
  perf_02_metric_cache_benefit.py
)

echo "GPU suite on backend=$GPU using $PY"
for s in "${GPU_SCRIPTS[@]}"; do
  echo "================ $s ($GPU) ================"
  "$PY" "${HERE}/$s" || echo "!! $s exited non-zero (continuing)"
  echo
done
echo "GPU suite done. Telemetry artifacts in ~/gafime_telemetry/"
