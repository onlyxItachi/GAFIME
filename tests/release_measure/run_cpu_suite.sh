#!/usr/bin/env bash
# CPU-safe subset of the v0.5 release measurement suite. Run on the merged
# integration branch. Logged scripts write telemetry into ~/gafime_telemetry.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="/home/hamza-usta/GAFIME-integration:${HERE}"
PY="${GAFIME_PY:-/home/hamza-usta/.venvs/gafime-dl-py314/bin/python}"

CPU_SCRIPTS=(
  dp_01_parity_native_vs_reference.py
  dp_02_openml_tour_logged.py
  dp_03_method_effect_gated_soft.py
  dp_04_max_bins_sweep.py
  dp_05_dataset_structure_map.py
  dp_06_depth_rounds_sweep.py
  dp_07_boosting_residual_reduction.py
  dp_08_leakage_safety.py
  export_01_zero_copy_parity.py
  export_02_lifetime_safety.py
  export_03_overhead_vs_copy.py
  compile_01_plan_correctness.py
  compile_02_compiled_vs_eager.py
  backend_01_availability_smoke.py
  backend_03_e2e_smoke_per_backend.py
  perf_03_telemetry_e2e_spans.py
)

for s in "${CPU_SCRIPTS[@]}"; do
  echo "================ $s ================"
  "$PY" "${HERE}/$s" || echo "!! $s exited non-zero (continuing)"
  echo
done
echo "CPU suite done. Telemetry artifacts in ~/gafime_telemetry/"
