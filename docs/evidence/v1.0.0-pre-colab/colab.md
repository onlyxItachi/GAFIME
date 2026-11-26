# Colab qualification for GAFIME v1.0.0

Use the source and official artifact coordinates in [the checkpoint](README.md). Keep the original Actions ZIP and the two installed wheel files with the session evidence. Run each backend on its actual device and keep evidence from different GPU models in separate directories.

## Stage and verify the packages

On an authenticated workstation, fetch the source at the **merged release SHA** and download the official ZIP using the artifact ID in the checkpoint. Copy the ZIP to Colab through your own storage. Keep GitHub credentials out of notebook cells and outputs.

```bash
set -euo pipefail
sha=06d82a4082e96f0821abe0e323ea91f4c7ee6744
tree=1fd446e65ff6b2303e3d8b10e34e76c75ce6d216
run=45857407051
repo=onlyxItachi/GAFIME
src="$PWD/gafime-source"
bundle="$PWD/frozen"
out="$PWD/qualification"
mkdir -p "$bundle" "$out"
git clone --no-checkout "https://github.com/$repo.git" "$src"
git -C "$src" checkout --detach "$sha"
test "$(git -C "$src" rev-parse 'HEAD^{tree}')" = "$tree"
# Verify the ZIP SHA-256 from the checkpoint before extracting it.
unzip official-release-bundle.zip -d "$bundle"
(cd "$bundle" && sha256sum --check --strict SHA256SUMS)
python "$src/.github/scripts/release_bundle.py" verify --artifacts "$bundle" \
  --source-sha "$sha" --run-id "$run" --repository "$repo"
```

Use one CPython interpreter from 3.10 through 3.14 on Linux x86_64 with glibc ≥2.28. CPython 3.12 is the documented starting choice for the legacy A/B below. Match the wheel tag to the interpreter and create a fresh environment per variant. The selected package filenames and hashes are in the checkpoint's [SHA256SUMS](SHA256SUMS). Run the commands in one shell script per allocation, or source the shared shell variables again in each Colab shell cell. Keep one output directory per actual GPU and run from session scratch.

```bash
tag=$(python -c 'import sys; assert sys.implementation.name == "cpython"; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
core="$bundle/gafime-1.0.0-$tag-$tag-manylinux_2_28_x86_64.whl"
cuda="$bundle/gafime_cuda-1.0.0-$tag-$tag-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
test -f "$core" && test -f "$cuda"
sha256sum "$core" "$cuda" > "$out/session-wheels.sha256"
python -m venv "$PWD/v1-venv"
py="$PWD/v1-venv/bin/python"
unset PYTHONPATH GAFIME_CUDA_V1_LIB GAFIME_ROCM_V1_LIB GAFIME_METAL_V1_LIB GAFIME_CUDA_RT_V1_LIB
"$py" -m pip install 'polars>=1.3,<2' numpy scikit-learn liac-arff
"$py" -m pip freeze > "$out/benchmark-deps.txt"
"$py" -m pip install --no-deps "$core" "$cuda"
"$py" -m pip check
"$py" -m pip freeze --all > "$out/environment-packages.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total --format=csv > "$out/gpu.csv"
nvidia-smi -q > "$out/nvidia-smi.txt"
{ uname -a; ldd --version; taskset -pc "$$"; free -b; "$py" -VV; } > "$out/host.txt"
```

RTX PRO 6000 Blackwell should report compute capability **12.0**, matching the wheel's `sm_120` image. The CUDA 13.3 build uses system `libcudart.so.13`. NVIDIA lists Linux driver 610.43.02 for CUDA 13.3 GA and describes conditional CUDA 13.x minor-version compatibility from driver branch 580. Record the actual driver and use execution of all profiles to establish compatibility. [NVIDIA GPU table](https://developer.nvidia.com/cuda/gpus) · [CUDA 13.3 release notes](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html) · [Driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)

If `libcudart.so.13` does not resolve, use the exact runtime archive and SHA from the release workflow's `CUDA_CUDART_LINUX_URL` and `CUDA_CUDART_LINUX_SHA256` values. Download and verify that archive in session scratch, extract it there, and add only its `lib` directory to this session's `LD_LIBRARY_PATH`. Preserve the resolved real library path and digest. A driver failure makes that allocation unsuitable for the candidate. Keep its diagnostics and repeat on a compatible allocation. [Pinned runtime inputs](https://github.com/onlyxItachi/GAFIME/blob/06d82a4082e96f0821abe0e323ea91f4c7ee6744/.github/workflows/build_wheels.yml)

```bash
# Only when the compatible system library is unavailable in the session:
runtime=cuda_cudart-linux-x86_64-13.3.29-archive
mkdir -p "$out/runtime"
curl --fail --location --output "$out/runtime/$runtime.tar.xz" \
  "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/$runtime.tar.xz"
printf '%s  %s\n' 1e59c4888267d27ba1a9bd0f3669a6439db1334a96e754cd9013c7c73e18dc9d \
  "$out/runtime/$runtime.tar.xz" | sha256sum --check --strict
tar -xJf "$out/runtime/$runtime.tar.xz" -C "$out/runtime"
export LD_LIBRARY_PATH="$out/runtime/$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Required frozen-wheel CUDA execution

```bash
suite="$src/tests/release_measure"
cuda_so=$("$py" -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("gafime_cuda").origin).parent / "libgafime_cuda.so")')
ldd "$cuda_so" | tee "$out/cuda-linkage.txt"
runtime_so=$(awk '$1 == "libcudart.so.13" && $2 == "=>" && $3 ~ /^\// { print $3 }' "$out/cuda-linkage.txt")
test -f "$runtime_so"
sha256sum "$(readlink -f "$runtime_so")" > "$out/cudart.sha256"
sha256sum "$cuda_so" > "$out/cuda-payload.sha256"
"$py" "$suite/installed_wheel_smoke.py" --source-root "$src" 2>&1 | tee "$out/core-smoke.log"
"$py" "$suite/installed_payload_smoke.py" --backend cuda --source-root "$src" \
  --execute-profiles 2>&1 | tee "$out/cuda-smoke.log"
"$py" "$suite/precision_01_end_to_end_profiles.py" --backend core --backend cuda \
  > "$out/precision-core-cuda.json" 2> "$out/precision-core-cuda.stderr.log"
```

Require a compatible runtime in `ldd`, execution on the assigned CUDA device and a passing precision report for Core and CUDA across fp32, mixed and fp64. Preserve all profile cases, 18 cross-backend comparisons, the harness tolerances, fallback/skip status and stderr.

## Stable workload and performance evidence

Use the current benchmark infrastructure and keep installed cold ABI lifecycle, public end-to-end timing, public file acquisition and resident Core throughput evidence distinct. The existing Stable protocols define correctness, profile-order controls, raw samples, precision and resource acceptance. [Release measurement contract](https://github.com/onlyxItachi/GAFIME/blob/06d82a4082e96f0821abe0e323ea91f4c7ee6744/tests/release_measure/README.md) · [Release operations](https://github.com/onlyxItachi/GAFIME/blob/06d82a4082e96f0821abe0e323ea91f4c7ee6744/docs/releases/release-operations.md)

```bash
"$py" "$suite/cold_lifecycle.py" --backend cuda --payload "$cuda_so" --wheel "$cuda" \
  --source-root "$src" --profile fp32,mixed,fp64 --repetitions 30 --output "$out/cuda-cold.json"
printf '%s\n' '{"schema":"gafime.precision-profile-native-evidence.v1","status":"not_collected","arithmetic_claims_valid":false,"artifacts":[]}' > "$out/native-evidence.json"
"$py" "$suite/perf_13_precision_profiles.py" --source-root "current=$src" \
  --wheel "current=$core" --wheel "current=$cuda" --native-evidence "$out/native-evidence.json" \
  --backend core --backend cuda --profile fp32,mixed,fp64 --order-repetitions 5 \
  --workload release --input-policy common-f64,native --output "$out/core-cuda-public.json"
```

The candidate-only manifest above supports end-to-end observations. For a formal same-ABI comparison, use the existing A/B+B/A harness with a separately bound compatible baseline, native-evidence manifests, at least 10 warmups and 30 samples, 100 ms sample regions, and the profile-order controls. Investigate a confidence lower bound above 1% regression. A lower bound above 3% is blocking. An upper bound at or below 1% is clean. Retain overlapping intervals as inconclusive. Keep input identity, CPU/wall samples, fresh-process peak RSS and cold/repeat state. For file workloads use `perf_14_public_ingest.py`, `perf_14_parser_control.py` and `perf_14_ingest_budget.py` as in the contract workflow; preserve the parser and cost verdicts and expand to the affected configured/default miss/hit paths where required.

`perf_13` writes its report at the end. A full Core plus CUDA matrix can outlast an allocation. If needed, run one complete backend per session and retain a separate report for each. Check `valid_for_e2e_performance_claims`, `valid_for_comparative_performance_claims`, `valid_for_arithmetic_claims`, and `claim_failures` before accepting a claim.

Formal resident Core Stable qualification requires an authorized open repository-owned PR targeting `main` at the exact candidate head and the dedicated `gafime-core-stable` runner. Use the existing workflow and the pinned baseline `d52199f44aa80ab8ef50c18db95dd1630961cdaf`. Require its independent validator's `stable_release_ready=true`. Generic Colab CPU figures are supplemental. [Core production workflow](https://github.com/onlyxItachi/GAFIME/blob/06d82a4082e96f0821abe0e323ea91f4c7ee6744/.github/workflows/core_precision_production_benchmark.yml)

```bash
# After authorization, an exact-head PR to main and a compatible runner exist:
gh workflow run core_precision_production_benchmark.yml --repo "$repo" \
  --ref release/v1.0.0 -f mode=stable -f expected_candidate_sha="$sha" \
  -f pull_request_number="$admission_pr"
```

## User-requested v0.4.7 comparison

Use the existing `perf_08_v047_distribution_ab.py` campaign on the same GPU allocation, Python build, Polars, NumPy and fixture-preparation dependencies. The CPython 3.12 Linux baseline files are:

| File | SHA-256 |
| --- | --- |
| `gafime-0.4.7-1-cp312-cp312-manylinux_2_28_x86_64.whl` | `8aa98d1afe6d5dbeee104cfa013997bd55d20c82556920d74c46dc527bb76ffd` |
| `gafime_cuda-0.4.7-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl` | `10b4ff60894f7bed6b01674c42d2e20154ba01119964c5ae914daca71b53b3c6` |

Verify the canonical PyPI files before installing them into a separate legacy environment. Its CUDA wheel contains `sm_120` code by static inspection. Record physical success separately for each device. [Core files](https://pypi.org/project/gafime/0.4.7/#files) · [CUDA files](https://pypi.org/project/gafime-cuda/0.4.7/#files)

```bash
test "$tag" = cp312
python -m venv "$PWD/v047-venv"
basepy="$PWD/v047-venv/bin/python"
mkdir -p "$out/legacy-wheels"
"$basepy" -m pip download --only-binary=:all: --no-deps --dest "$out/legacy-wheels" \
  gafime==0.4.7 gafime-cuda==0.4.7
legacy_core="$out/legacy-wheels/gafime-0.4.7-1-cp312-cp312-manylinux_2_28_x86_64.whl"
legacy_cuda="$out/legacy-wheels/gafime_cuda-0.4.7-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
printf '%s  %s\n' 8aa98d1afe6d5dbeee104cfa013997bd55d20c82556920d74c46dc527bb76ffd "$legacy_core" \
  | sha256sum --check --strict
printf '%s  %s\n' 10b4ff60894f7bed6b01674c42d2e20154ba01119964c5ae914daca71b53b3c6 "$legacy_cuda" \
  | sha256sum --check --strict
"$basepy" -m pip install -r "$out/benchmark-deps.txt"
"$basepy" -m pip install --no-deps "$legacy_core" "$legacy_cuda"
"$basepy" -m pip check
"$basepy" "$suite/perf_08_v047_distribution_ab.py" prepare --output-dir "$out/fixtures"
# For each prepared dataset, requested backend and fresh worker, use its own output path.
"$basepy" "$suite/perf_08_v047_distribution_ab.py" worker --dataset "$dataset" \
  --backend "$backend" --metrics pearson r2 --input-format lists --max-arity 3 \
  --repeats 5 --output "$worker_json"
```

Run `sklearn_friedman1_32768x24.npz`, `openml_cpu_act_197.npz` and `openml_house_16h_574.npz` on Core and CUDA. Repeat the worker command using `"$py"` for the candidate, with `--include-compiled`. Keep the cache-disabled candidate control (`--cache-size 0`) separate. Use three fresh workers per variant and rotate process order, retain first-call and repeat data, then `aggregate --inputs <worker JSON files> --output <aggregate.json>` and `compare --baseline <baseline.json> --candidate <candidate.json> --output <comparison.json>`. Preserve exact candidate-order evidence, numeric deltas, warnings, failures, timings and resource observations. Limit the equivalent comparison to Pearson/R². Keep Stable same-ABI throughput qualification on its own tracked protocol.

## DL Method

Use the exact v1 consumer checkout recorded in the maintainer's private DL Method receipt. In a fresh consumer environment install the verified frozen Core and CUDA wheels, the consumer's declared tested dependencies, and then the consumer itself. Record engine and consumer SHAs, package hashes, PyTorch/CUDA versions, device, and exact module origins.

```bash
test "$(git -C "$dl" rev-parse HEAD)" = "$dl_sha"
"$dlpy" -m pip install --no-deps -e "$dl"
"$dlpy" -m pip check
"$dlpy" -m pytest -q "$dl/tests/test_gafime_integration.py" 2>&1 | tee "$out/dl-engine-integration.log"
```

Run the existing DL experiment runner on the chosen CUDA allocation using its tracked dataset/racer manifests. Qualify actual native execution if reporting native DL performance. Capture matched split and seed identities, finite held-out metrics, per-phase timings, path counts, peak PyTorch allocator memory, and paired deltas with intervals. Record any framework fallback explicitly. Keep the consumer's `gafime_runs/*.json` and stdout with the engine evidence. The consumer per-release gate and the Stable engine requirements take precedence over research sweeps, extended model controls, OptiX and other exploratory workloads.

### Retain with each allocation

- Official bundle/run identity, exact engine SHA/tree, wheel and installed native-library digests, dataset/config hashes, and the complete dependency and environment receipt.
- Smoke stdout/stderr, Core/CUDA precision JSON, raw cold/public timing JSON and any requested budget/validator verdicts.
- Per-worker v0.4.7 raw outputs, aggregate/compare reports, DL test/experiment reports, and explicit skip/failure logs.
- CUDA/CPU/toolchain/device metadata, run order and resource samples with their measurement scope. Preserve raw evidence before summarizing.
