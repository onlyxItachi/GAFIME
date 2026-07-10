#!/usr/bin/env bash
# CPU-safe subset of the v1 release measurement suite. Logged scripts write
# telemetry into ~/gafime_telemetry where applicable.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
export PYTHONPATH="${ROOT}/python:${HERE}"
PY="${GAFIME_PY:-python3}"

CPU_SCRIPTS=(
  contract_00_policy_files.py
  contract_01_top_level_numpy_parity.py
  contract_02_feature_generation_reference.py
  contract_03_family_metric_backend_surface.py
  contract_04_adaptive_mi_quantization.py
  dp_02_openml_tour_logged.py
  dp_03_method_effect_gated_soft.py
  dp_05_dataset_structure_map.py
  dp_06_depth_rounds_sweep.py
  dp_07_boosting_residual_reduction.py
  dp_08_leakage_safety.py
  compile_01_plan_correctness.py
  compile_02_compiled_vs_eager.py
  backend_01_availability_smoke.py
  backend_03_e2e_smoke_per_backend.py
)

FAILURES=()
for s in "${CPU_SCRIPTS[@]}"; do
  echo "================ $s ================"
  if ! "$PY" "${HERE}/$s"; then
    echo "!! $s exited non-zero (continuing)"
    FAILURES+=("$s")
  fi
  echo
done
if ((${#FAILURES[@]} != 0)); then
  printf 'CPU suite failed: %s\n' "${FAILURES[*]}" >&2
  exit 1
fi
echo "CPU suite done. Telemetry artifacts in ~/gafime_telemetry/"
