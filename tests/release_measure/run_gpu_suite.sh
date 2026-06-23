#!/usr/bin/env bash
# GPU subset of the v0.5 release measurement suite. Pass GAFIME_GPU=cuda (RTX 4060)
# or GAFIME_GPU=rocm (gfx1150). Uses the CUDA torch venv by default for CUDA.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="/home/hamza-usta/GAFIME-integration:${HERE}"
GPU="${GAFIME_GPU:-cuda}"
if [ "$GPU" = "cuda" ]; then
  PY="${GAFIME_PY:-/home/hamza-usta/.venvs/mc-torch-cu/bin/python}"
else
  PY="${GAFIME_PY:-/home/hamza-usta/.venvs/gafime-dl-py314/bin/python}"
fi
export GAFIME_GRAPH_BACKEND="$GPU"
export GAFIME_BACKEND="$GPU"

GPU_SCRIPTS=(
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
