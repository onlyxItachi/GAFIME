# CUDA/HIP/Metal Kernel Hardening

This note documents the CUDA, HIP, and Metal hardening checkpoint for v1
continuous GPU scoring. It does not change the public Python API or stable C
ABI.

## Scope

The CUDA payload now emits compile-time specialized kernels for:

- continuous Pearson/R2 interaction arity `1..5`,
- mutual information arity `1..5` crossed with bins
  `2,4,8,12,16,24,32,48,64,96`,
- Spearman arity `1..5`,
- top-k rank direction with block-local partial selection plus a final merge.

The HIP/ROCm payload now emits compile-time specialized kernels for:

- continuous Pearson/R2 interaction arity `1..5`,
- mutual information arity `1..5` crossed with bins
  `2,4,8,12,16,24,32,48,64,96`,
- Spearman arity `1..5`,
- top-k rank direction with block-local partial selection, final merge, and
  selected metric-row gather.

ROCm placement detection now uses HIP's `hipDeviceProp_t.integrated` capability
directly. The v1 launcher no longer guesses integrated/discrete placement from
product-name fragments, restoring the capability-only policy documented for the
v0.4.7 ROCm backend. Architecture strings remain valid for RDNA/CDNA capability
classification and diagnostics, but not physical-memory placement inference.
Managed storage is used only when an integrated device also reports managed or
concurrent-managed access. A failed `hipMallocManaged` is returned as an
allocation error; it can no longer fall back to a device-only pointer while the
copy path still treats that pointer as host-accessible.

The Metal shader does not emit host-selected template variants in this pass.
Instead, it removes the most obvious scalar hot spots without changing the Metal
ABI:

- interaction arity `1..5` uses fixed inline switch cases instead of a runtime
  product loop,
- host-side column means use f64 accumulation and the same NaN/Inf propagation
  semantics as CPU, CUDA, and HIP before centered interactions reach the
  shader,
- continuous Pearson/R2 now uses one 64-lane threadgroup per candidate instead
  of one serial thread per candidate,
- mutual-information range discovery is reduced across the 64-lane threadgroup
  instead of scanned by lane 0,
- mutual-information final accumulation and active-bin counts are reduced across
  the same threadgroup instead of accumulated by lane 0,
- ranked execution uses block-local partial top-k selection, final merge, and
  selected metric-row gather at source level, so top-k plans do not need to copy
  the full metric table back to the host,
- the Objective-C++ launcher is compiled with ARC so deleting a matrix releases
  its retained queues, pipelines, and buffers,
- the Metal loader rejects pipelines whose maximum threadgroup size cannot
  satisfy the fixed 64-lane reduction width.

Rust still owns planning, backend selection, and scheduling. `mi_bins` remains
the v0.4.1 adaptive maximum rather than a fixed launch request. The planner
selects the largest template for which `8 * bins^2 <= n_samples`, passes that
shape through `GafimeShapeHint.vendor_hint`, and clamps Metal at its 48-bin
threadgroup-memory ceiling. Two bins is the minimum fallback when no template
satisfies the density rule. Unsupported maxima round down rather than silently
expanding to `96`.

## Build Variants

`GAFIME_CUDA_RT_BUILD_MODE=off|on|both` separates generic CUDA distribution
payloads from OptiX support. `off` emits `libgafime_cuda_v1.so` without OptiX;
`on` emits that primary name with OptiX; `both` emits the generic primary plus
`libgafime_cuda_v1_rt.so`. The legacy `GAFIME_CUDA_ENABLE_OPTIX_RT` option maps
to `on` only when the explicit mode is unset.

`GAFIME_CUDA_ARCHITECTURES` controls the cubin/PTX targets emitted into the
fatbinary. Launch geometry is selected when a matrix is allocated from the
actual device compute major and `maxThreadsPerBlock`: pre-Ampere devices use 128
threads, while the current Ampere/Ada, Hopper, and Blackwell policy classes use
256. The named modern classes are dispatch seams, not claims that each class has
an independently measured kernel shape. Packaged builds therefore do not inject
one package-wide tuning SM.

Both CUDA and HIP compile at `-O3` under the repository's IEEE policy. No
fast-math, unsafe reassociation, reciprocal approximation, or flush-to-zero flag
is enabled.

## MI Kernel Policy

The first template version fully unrolled every MI bin count. Static inspection
showed that this was not the best optimization point:

- `Bins=16` expanded to roughly twenty thousand SASS instruction lines per arity.
- `Bins=32/64/96` used a compact looped path near eleven hundred SASS
  instruction lines per arity.

The current CUDA/HIP policy keeps full unroll only for tiny bins `2` and `4`;
bins `8` and above use a compact looped template specialization. This preserves
compile-time arity/bin dispatch while avoiding L1I and fatbin pressure from large
unrolled histogram traversals.

The intermediate `12/24/48` templates preserve the same eight-samples-per-joint-
cell guard while reducing the resolution jumps in the original
`2/4/8/16/32/64/96` set. The public-API quantization contract compares each new
shape with its lower power-of-two predecessor against a 96-bin large-sample
reference over five deterministic seeds. Median average-tie rank correlation
improved from `0.724 -> 0.791`, `0.790 -> 0.940`, and `0.791 -> 0.999`; median
top-12 overlap did not regress. Each adaptive result is also required to be
bit-equal to an explicit launch of the selected template. CPU permutation and
bootstrap significance passes resolve the same adaptive shape before building
their null distributions, so observed MI is never compared with a different
histogram resolution.

The high-bin MI path parallelizes range discovery and MI accumulation on CUDA,
HIP, and Metal. CUDA uses warp-shuffle block reductions. HIP can use
target-width wave reductions for min/max and integer counts, with a compact
per-wave LDS merge. The production `gfx1150` build enables it only for the
64-bin specialization; 96 bins retain the 256-thread shared tree. HIP also
retains that tree for the floating MI sum in every mode, so its addition order
does not change. Metal uses fixed 64-lane threadgroup reductions. These remove
the old thread-0/lane-0 scalar range scans without applying fast math or
reassociating the HIP floating MI accumulation.

`GAFIME_HIP_WAVE_MI_MODE` selects exact HIP specializations at build time:
`off`, `64`, `96`, or `64-96`. The distribution default is `64`; CMake rejects
other values. Embedded provenance records a mask (`0`, `1`, `2`, or `3`), and
the option does not change the MI estimator or public ABI. Exact mode `64`
repeatedly improved 64-bin throughput while leaving 96-bin code on its neutral
shared-tree path. Mode `96` and combined mode `64-96` remain diagnostic controls
because the 96-bin wave path did not produce a repeatable positive result.

## Top-K Policy

CUDA, HIP, and Metal ranked execution now use a two-stage device selector:

- up to `4096` partial blocks scan disjoint grid-stride portions of the metric
  table and emit block-local top-k scores/indices,
- the block count is also capped by `ceil(candidate_count / top_k)`, keeping
  partial score/index storage below roughly twice the candidate row count,
- a final merge kernel selects the requested global top-k from the compact
  partial list,
- only selected metric rows are gathered back for host result materialization.

This removes the previous single-block top-k bottleneck where one block scanned
every candidate row once per requested rank. CUDA and HIP trim invalid sentinel
indices before selected-row gather; Metal guards sentinel/out-of-range indices
inside the gather kernel and trims the host-visible row list after completion.
Each stage advances from the immediately previous `(score, index)` cutoff rather
than rescanning all earlier winners, reducing selection from
`O(candidate_count * top_k^2)` to `O(candidate_count * top_k)` while preserving
the deterministic tie order.

## Static Evidence

The static report is repeatable and does not initialize either GPU runtime:

```bash
python3 tests/release_measure/gpu_static_kernel_report.py \
  --cuda-lib build/cuda-template-hardening-both/libgafime_cuda_v1.so \
  --hip-lib build/rocm-template-hardening-default/libgafime_rocm_v1.so \
  --hip-target gfx1150 \
  --require-template-matrix \
  --require-topk-split \
  --require-no-spills
```

The same source was also compiled and inspected as an experimental mode-96
wave64 `gfx90a` code object; no device runtime is needed:

```bash
cmake -S src/rocm -B build/rocm-static-inspect-gfx90a \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DGAFIME_HIP_WAVE_MI_MODE=96
cmake --build build/rocm-static-inspect-gfx90a --config Release -- -j4
python3 tests/release_measure/gpu_static_kernel_report.py \
  --hip-lib build/rocm-static-inspect-gfx90a/libgafime_rocm_v1.so \
  --hip-target gfx90a \
  --require-template-matrix \
  --require-topk-split \
  --require-no-spills
```

The required checks fail if resource metadata is incomplete, an arity/bin
specialization is absent, either rank direction is absent, the merge/gather
stages are absent, the old single-block selector remains, or generated kernels
use CUDA local/stack storage or HIP private/register spills.

The measurements below are from the final local rebuild after the launcher and
production-mode decisions. They are static artifact evidence; runtime
throughput claims remain subject to the device-state qualifications below.

Static artifact checks from that local `sm_89` snapshot:

```text
libgafime_cuda_v1.so:    2,087,024 bytes
libgafime_cuda_v1_rt.so: 2,171,816 bytes
template matrix:         continuous=5/5, Spearman=5/5, MI=50/50
```

Representative `cuobjdump --dump-resource-usage` output:

```text
MI<1,96>: REG=34, SHARED=37844
MI<1,64>: REG=34, SHARED=17108
MI<1,32>: REG=34, SHARED=4564
MI<1,24>: REG=34, SHARED=2708
MI<1,16>: REG=34, SHARED=1364
MI<1,12>: REG=34, SHARED=884
MI<1,8>:  REG=34, SHARED=532
MI<1,4>:  REG=34, SHARED=276
MI<1,2>:  REG=34, SHARED=212
```

Representative SASS instruction counts from `cuobjdump --dump-sass`:

```text
MI<1,96>: 1072
MI<1,64>: 1072
MI<1,32>: 1072
MI<1,24>: 1072
MI<1,16>: 1072
MI<1,12>: 1072
MI<1,8>:  1072
MI<1,4>:  1888
MI<1,2>:  936
```

Representative CUDA top-k resource usage:

```text
partial_topk<desc/asc>: REG=23, SHARED=2056
merge_topk<desc/asc>:   REG=17, SHARED=2056
gather:                 REG=10, SHARED=0
```

Static artifact checks from the production-default `gfx1150` HIP snapshot:

```text
libgafime_rocm_v1.so: 687,824 bytes
embedded gfx1150 code object: .text=313,896 bytes (0x4ca28)
template matrix: continuous=5/5, Spearman=5/5, MI=50/50
```

Representative HIP code-object symbol sizes:

```text
MI<1,96>: 3844 bytes, private=0, vgpr=29, sgpr=23, group=43796
MI<1,64>: 4932 bytes, private=0, vgpr=29, sgpr=23, group=18132
MI<1,32>: 3780 bytes, private=0, vgpr=29, sgpr=23, group=10516
MI<1,24>: 3844 bytes, private=0, vgpr=28, sgpr=23, group=8660
MI<1,16>: 3844 bytes, private=0, vgpr=27, sgpr=23, group=7316
MI<1,12>: 3844 bytes, private=0, vgpr=27, sgpr=23, group=6836
MI<1,8>:  3844 bytes, private=0, vgpr=27, sgpr=23, group=6484
MI<1,4>:  11000 bytes, private=0, vgpr=20, sgpr=23, group=5236
MI<1,2>:  4832 bytes, private=0, vgpr=20, sgpr=23, group=5172
MI<5,96>: private=0, vgpr=36, no VGPR/SGPR spills
```

Production mode-64 HIP wave-reduction delta on `gfx1150`:

```text
shared library:          679,632 -> 687,824 bytes (+8,192, +1.2%)
code-object .text:       307,752 -> 313,896 bytes (+6,144, +2.0%)
MI<*,64> LDS:             23,060 -> 18,132 bytes (-4,928, -21.4%)
MI<*,96> code/LDS:       unchanged at 43,796 bytes LDS
MI bins 2..48:           code size and LDS unchanged
VGPR ranges and spills:  unchanged; private=0, VGPR spills=0, SGPR spills=0
```

The wave64 `gfx90a` compile emitted all `50/50` MI specializations with
`.wavefront_size=64`, `23,060/38,868` byte LDS sizes for bins `64/96`, zero
private segments, and zero VGPR/SGPR spills. Its code-object `.text` is
`354,964` bytes (`0x56a94`). `llvm-nm -u` reports no unresolved device symbols
for either inspected HIP code object.

Representative HIP top-k code-object metadata:

```text
partial_topk<desc>: 1128 bytes, private=0, vgpr=17, sgpr=29
partial_topk<asc>:  1128 bytes, private=0, vgpr=17, sgpr=29
merge_topk<desc>:   1064 bytes, private=0, vgpr=16, sgpr=29
merge_topk<asc>:    1064 bytes, private=0, vgpr=16, sgpr=29
gather:              416 bytes, private=0, vgpr=6,  sgpr=10
```

These SGPR values are the code-object metadata `.sgpr_count` values. The lower
`numbered_sgpr` ELF symbols omit implicit system registers and therefore are not
used by the repeatable report.

## Static Performance Read

The current non-device read is favorable for binary size, local-memory pressure,
and the top-k split, but it still highlights the kernels that need runtime
throughput measurement.

CUDA `sm_89` SASS instruction counts across arities:

```text
continuous<1..5>:        336..424
MI<arity,2>:             936..1024
MI<arity,4>:             1888..1984
MI<arity,8/12/16/24/32/48/64/96>: 1072..1176
Spearman<1..5>:          416..504
partial_topk:            200
merge_topk:              184
gather:                  56
```

The CUDA resource metadata shows no stack or local-memory use in these generated
scoring/top-k kernels. The remaining static pressure is shared memory:

```text
continuous<1..5>: REG=30..40, SHARED=6152
MI<*,96>:         REG=34..41, SHARED=37844
MI<*,64>:         REG=34..41, SHARED=17108
Spearman<1..5>:   REG=40..56, SHARED=12288
top-k split:      REG=10..23, SHARED=0..2056
```

HIP `gfx1150` code-object metadata is similar: no private segment, no VGPR/SGPR
spills, compact top-k kernels, and LDS pressure dominated by high-bin MI:

```text
continuous<1..5>:        2420..3316 bytes, group=6152,  vgpr=18..32
MI<arity,4>:             11000..11944 bytes, group=5236,  vgpr=20..30
MI<arity,8/12/16/24/32>: 3780..4800 bytes, group=6484..10516, vgpr=27..36
MI<arity,48>:            3780..4736 bytes, group=15764, vgpr=29..36
MI<arity,64>:            4932..5892 bytes,  group=18132, vgpr=29..36
MI<arity,96>:            3844..4736 bytes,  group=43796, vgpr=29..36
Spearman<1..5>:          1800..2380 bytes,  group=12288, vgpr=32
top-k split:             416..1128 bytes,   group=0..2056, vgpr=6..17
```

## Idle Validation Order

Run these phases only after competing CPU/GPU workloads are closed. Any
failure blocks the commit/push; the suite wrappers collect all failures but now
return nonzero when any child script fails.

Source, CPU, and public-contract gates:

```bash
cargo fmt --all -- --check
env -u GAFIME_CUDA_V1_LIB -u GAFIME_ROCM_V1_LIB -u GAFIME_METAL_V1_LIB \
  cargo test --workspace
maturin build --release \
  --interpreter .venv-release/bin/python \
  --out dist/validation-core
.venv-release/bin/python tests/release_measure/artifact_01_release_composition.py \
  --scope core-wheel \
  --artifacts dist/validation-core
env -u GAFIME_CUDA_V1_LIB -u GAFIME_ROCM_V1_LIB -u GAFIME_METAL_V1_LIB \
  GAFIME_PY=.venv-release/bin/python tests/release_measure/run_cpu_suite.sh
```

Rebuild distinct, provenance-bearing artifacts from this working tree:

```bash
: "${GAFIME_OPTIX_INCLUDE_DIR:?set this to the OptiX SDK include directory}"
cmake -S src/cuda -B build/cuda-template-hardening-both \
  -DCMAKE_BUILD_TYPE=Release \
  -DGAFIME_CUDA_RT_BUILD_MODE=both \
  -DGAFIME_CUDA_ARCHITECTURES='89-real;89-virtual' \
  -DGAFIME_OPTIX_INCLUDE_DIR="$GAFIME_OPTIX_INCLUDE_DIR"
cmake --build build/cuda-template-hardening-both --config Release -- -j4

cmake -S src/cuda -B build/cuda-template-symbolic-all \
  -DCMAKE_BUILD_TYPE=Release \
  -DGAFIME_CUDA_RT_BUILD_MODE=off \
  -DGAFIME_CUDA_ARCHITECTURES=all
cmake --build build/cuda-template-symbolic-all --config Release -- -j4

cmake -S src/rocm -B build/rocm-mi-wave-baseline \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1150 \
  -DGAFIME_HIP_WAVE_MI_MODE=off
cmake --build build/rocm-mi-wave-baseline --config Release -- -j4

cmake -S src/rocm -B build/rocm-template-hardening-default \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1150
cmake --build build/rocm-template-hardening-default --config Release -- -j4

cmake -S src/rocm -B build/rocm-template-hardening \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1150 \
  -DGAFIME_HIP_WAVE_MI_MODE=96
cmake --build build/rocm-template-hardening --config Release -- -j4

cmake -S src/rocm -B build/rocm-template-hardening-wave64-96 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1150 \
  -DGAFIME_HIP_WAVE_MI_MODE=64-96
cmake --build build/rocm-template-hardening-wave64-96 --config Release -- -j4

cmake -S src/rocm -B build/rocm-static-inspect-gfx90a \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DGAFIME_HIP_WAVE_MI_MODE=96
cmake --build build/rocm-static-inspect-gfx90a --config Release -- -j4

c++ -std=c++20 tests/gpu/cuda_v1_abi_smoke.cpp \
  -L"$PWD/build/cuda-template-hardening-both" \
  -Wl,-rpath,"$PWD/build/cuda-template-hardening-both" \
  -lgafime_cuda_v1 -o /tmp/cuda_v1_abi_smoke
c++ -std=c++20 tests/gpu/cuda_v1_abi_smoke.cpp \
  -L"$PWD/build/cuda-template-hardening-both" \
  -Wl,-rpath,"$PWD/build/cuda-template-hardening-both" \
  -lgafime_cuda_v1_rt -o /tmp/cuda_v1_abi_smoke_rt
c++ -std=c++23 tests/gpu/rocm_v1_abi_smoke.cpp \
  -L"$PWD/build/rocm-template-hardening-default" \
  -Wl,-rpath,"$PWD/build/rocm-template-hardening-default" \
  -lgafime_rocm_v1 -o /tmp/rocm_v1_abi_smoke
```

Run static and local-device correctness gates before measuring throughput:

```bash
python3 tests/release_measure/gpu_static_kernel_report.py \
  --cuda-lib build/cuda-template-hardening-both/libgafime_cuda_v1.so \
  --hip-lib build/rocm-template-hardening-default/libgafime_rocm_v1.so \
  --hip-target gfx1150 --require-template-matrix --require-topk-split \
  --require-no-spills

export PYTHONPATH="$PWD/python:$PWD/tests/release_measure"
export GAFIME_CUDA_V1_LIB="$PWD/build/cuda-template-hardening-both/libgafime_cuda_v1.so"
export GAFIME_CUDA_RT_V1_LIB="$PWD/build/cuda-template-hardening-both/libgafime_cuda_v1_rt.so"
export GAFIME_ROCM_V1_LIB="$PWD/build/rocm-template-hardening-default/libgafime_rocm_v1.so"
export GAFIME_CUDA_ABI_SMOKE=/tmp/cuda_v1_abi_smoke
export GAFIME_CUDA_RT_ABI_SMOKE=/tmp/cuda_v1_abi_smoke_rt
export GAFIME_ROCM_ABI_SMOKE=/tmp/rocm_v1_abi_smoke
python3 tests/release_measure/v1_architecture_gate.py --include-gpu
python3 tests/release_measure/backend_02_cross_backend_parity.py
GAFIME_GRAPH_BACKEND=cuda python3 tests/release_measure/graph_01_replay_parity.py
GAFIME_GRAPH_BACKEND=rocm python3 tests/release_measure/graph_01_replay_parity.py
GAFIME_GRAPH_BACKEND=cuda python3 tests/release_measure/graph_02_launch_shaping_timing.py
GAFIME_GRAPH_BACKEND=rocm python3 tests/release_measure/graph_02_launch_shaping_timing.py
```

The architecture gate runs the generic CUDA ABI smoke against the no-RT
primary, forces RT membership through the RT sibling smoke, and points the full
Rust workspace at the RT sibling so its `REQUIRE_RT` decision-path cases are
meaningful. The exported primary remains no-RT for the subsequent continuous
parity, graph, and throughput commands.

Only after those gates pass, run the absolute MI benchmark and the repeated
ROCm baseline/candidate A/B. Keep each JSON artifact and do not pool CUDA and
ROCm rates:

```bash
python3 tests/release_measure/perf_06_gpu_mi_specializations.py \
  --backend cuda --backend rocm --bins 32,64,96 \
  --rows 73728 --features 24 --max-arity 3 --warmups 8 --repeats 50 \
  --json-out /tmp/gafime-gpu-mi-specializations.json

for run in 1 2 3; do
  python3 tests/release_measure/perf_07_rocm_mi_wave_ab.py \
    --baseline-lib build/rocm-mi-wave-baseline/libgafime_rocm_v1.so \
    --optimized-lib build/rocm-template-hardening-default/libgafime_rocm_v1.so \
    --bins 32,64,96 --rows 73728 --features 24 --max-arity 3 \
    --warmups 8 --repeats 50 \
    --json-out "/tmp/gafime-rocm-mi-wave-ab-${run}.json"
done
```

The `gfx90a` runtime gate still requires a CDNA machine; static
cross-compilation on this host does not satisfy it. The Metal gates run on the
repository's Apple-hosted macOS workflow.

## Device Performance Validation

Final throughput acceptance is per device and requires an otherwise idle
device. The AMD `gpu_busy_percent` counter read zero in five one-second samples
before the ROCm measurements, so the ROCm A/B is used for the wave-mode
decision. The NVIDIA device remained occupied by desktop clients, so its
absolute rate is intentionally excluded from release evidence.

The repeatable ROCm A/B uses mode `off` as the control and compares one exact
wave-bin mask at a time. Bins `32` remain unchanged in every artifact. The
script rejects identical paths, inodes, or SHA-256 digests; reads embedded
target-architecture and wave-mask provenance; checks candidate identity and MI
parity before timing; alternates payload order in one process; and reports raw
and control-normalized speedup:

```bash
PYTHONPATH=python:tests/release_measure \
python3 tests/release_measure/perf_07_rocm_mi_wave_ab.py \
  --baseline-lib build/rocm-mi-wave-baseline/libgafime_rocm_v1.so \
  --optimized-lib build/rocm-template-hardening-default/libgafime_rocm_v1.so \
  --bins 32,64,96 --rows 73728 --features 24 --max-arity 3 \
  --warmups 8 --repeats 50 --json-out /tmp/gafime-rocm-mi-wave-ab.json
```

The row count reaches the v0.4.1 threshold for the 96-bin shape. Lower row
counts downshift to a smaller template, and both performance scripts reject a
requested label that does not match the effective shape. Acceptance requires
zero candidate-identity differences, MI delta within `1e-6`, and repeated
positive raw and control-normalized results at every enabled bin. CUDA and ROCm
absolute throughput must be recorded separately as candidate-sample GEval/s and
must not be treated as a cross-vendor speedup.

The idle `gfx1150` mode-64 A/B produced exact MI parity and these 64-bin
results:

```text
       raw speedup   control-normalized
run 1:    1.1186x          1.1637x
run 2:    1.1237x          1.1372x
run 3:    1.1040x          1.1835x
median:   1.1186x          1.1637x
```

All three runs compared baseline SHA-256
`5a8c8ed12b2a2d187c0712f886e00b8addea6dfc08ae903693e23d5f3a4f054a`
against production mode-64 SHA-256
`4bf0511154c3c0df9f55f3700b775f7f395ce3ca49a7cf4a6a1b149d3e5c7d6c`.
The untouched 96-bin path had median control-normalized speedup `1.0043x`.

Mode `96` failed its final provenance sample at `0.9762x`
control-normalized, and combined mode `64-96` measured `0.9588x` at 96 bins.
Exact mode `64` is therefore the production default: it keeps the repeatable
64-bin gain without enabling the unhelpful 96-bin path. A separate production
run over 2,324 candidates and 73,728 rows measured `4.065`, `4.108`, and
`3.643` candidate-sample GEval/s for bins `32`, `64`, and `96` respectively.
These are same-device shape rates, not a CUDA/ROCm comparison.

The final CUDA run followed five `nvidia-smi dmon` samples at `0%` SM activity;
desktop clients still produced `6..24%` display-memory activity. It measured
`42.471`, `47.028`, and `34.565` candidate-sample GEval/s for bins `32`, `64`,
and `96`. These are compute-idle local shape rates with an explicit display
traffic caveat, not a cross-vendor comparison or a display-free benchmark.
CUDA correctness, ABI, template, and parity gates do not depend on that timing
qualification.

For wave64/CDNA execution, point `GAFIME_ROCM_V1_LIB` at the `gfx90a` payload on
real CDNA hardware and set `GAFIME_REQUIRE_ROCM_WAVE64_MI=1`; the device test
then requires wave64/CDNA and checks 96-bin MI for arities `1..5` at 73,728
samples against the CPU oracle. Static `gfx90a` inspection is not a substitute
for that runtime gate.

The macOS workflow supplies the built Metal dylib/metallib and runs both direct
Rust behavioral gates. The numerical gate compares Pearson, R2, fixed-bin MI,
and Spearman for arities `1..5` on finite high-dynamic and NaN/Inf inputs
against CPU at approved absolute tolerance `0.00005`; the rank gate covers
non-tied ascending/descending order, deterministic ties, multi-block partials,
selected-row gather, large `top_k`, and `top_k > candidate_count`. A configured
payload that cannot load is a failure rather than a skipped test. GitHub Actions
run `29112217686` compiled the shader and payload, then executed both tests on
Apple hardware under the former provisional bound. Follow-up run `30207767348`
recorded a worst absolute delta of `4.045665264e-6`; the approved `0.00005`
bound is about `12.36x` that observation. This supplies runtime correctness
evidence, and the gate is not a Metal performance measurement.

Earlier tracing supplied a correctness fix: launchers and the CPU
consumer recognized different bin subsets, and an older local Python extension
silently expanded new bins to `96`. The planner and all backends now share the
ten adaptive capacities. The full architecture gate executed every CUDA and
ROCm specialization for arities `1..5` against the fixed-bin CPU reference.

## Artifact Size Context

The final local native payloads are `2,087,024` bytes for CUDA without RT,
`2,171,816` bytes for CUDA with RT, and `687,824` bytes for the default HIP
build. Their raw `both`-mode aggregate is `4,946,664` bytes (`4.72 MiB`), not the
size of one shipped payload. Relative to the seven-bin matrix, adding fifteen
intermediate MI kernels grew the CUDA no-RT payload by about `26%` and HIP by
about `24%`; individual compact-template SASS/code sizes and register counts did
not increase, so the measured growth was distribution/fatbin breadth rather
than per-launch L1I growth.

The cp312 Linux x86_64 release wheels provide the historical comparison:

```text
release  compressed wheel   uncompressed CUDA DSO   uncompressed ROCm DSO
v0.4.0      1,452,941              2,735,368                 n/a
v0.4.1      2,450,922              9,005,648                 n/a
v0.4.5      2,048,322              6,032,384                 n/a
v0.4.6      2,070,933              6,032,384                 n/a
v0.4.7      1,333,849              6,032,384                 n/a  (CUDA wheel)
v0.4.7        478,930                    n/a           3,293,592  (ROCm wheel)
```

The template matrix therefore remains below the earlier single CUDA and ROCm
payload sizes. CMake `both` mode is a build convenience and does not require
shipping both RT variants in one distribution package.

Static conclusion:

- high-bin MI remains shared/LDS-memory limited. Production HIP mode `64`
  reduces `MI<*,64>` from roughly 23 KB to 18 KB LDS without changing floating
  accumulation order; the `gfx1150` A/B rejected the analogous 96-bin path,
  while CUDA remains near 38 KB shared memory at 96 bins,
- `MI<*,4>` is the intentional tiny-bin unrolled outlier in code size; if L1I
  pressure appears in profiling, this is the first unroll policy to revisit,
- Spearman is still an algorithmic hotspot: source inspection shows
  average-tie rank computation by pairwise comparison, so sample-count scaling is
  the likely gap even though the generated CUDA/HIP kernels do not spill,
- top-k is no longer the obvious scalar bottleneck for normal `top_k`; its
  cutoff progression removes the previous quadratic factor in `top_k`. Very
  large `top_k` may still justify a multi-level merge beyond the current partial
  block plus final merge topology,
- Metal has the same source-level removal of the top-k scalar path. This Linux
  host cannot compile it without `xcrun`/`metal`, but the macOS CI build and
  direct runtime tests now prove the published source path.

The local run covered CUDA/HIP numerical parity, every configured MI bin, MI
arity `1..5`, graph replay, target refresh, top-k, ABI smoke, the full v1
architecture gate, static payload inspection, ROCm candidate-scale throughput,
and a provenance-checked ROCm A/B. Metal now has a direct ABI test for
multi-block top-k, ascending/descending direction, ties, gathered rows, and
oversized `top_k`, plus a CPU-oracle test for every continuous metric on
high-dynamic and non-finite inputs. Both passed on Apple hardware in Actions run
`29112217686` without skipping. Maintainer approval of the provisional Metal
tolerance and Metal runtime performance evidence remain open. A display-free
CUDA rerun is required before publishing the local CUDA rates as a formal
benchmark.
