# RT decision-path paper reproduction

This note reproduces the low-level CUDA/OptiX primitive described in
`rt-gbdt-hardware-ray-tracing-paper.tex`. The public Python adapter now has a
narrow compact route for complete unary Pearson/R2 plans, but this standalone
fixture still measures the low-level ABI rather than end-to-end public planning.

Run commands from the repository root. The release-readiness review rebuilt,
tested, timed, and profiled the current safe-triangle/custom-AABB payload. The
current checkpoint is separated from both the prior custom-only checkpoint and
the superseded triangle prototype;
historical numbers remain bound to `docs/evidence/rt-firsthit-sm89-timing.txt`
and its retained NCU report.

## Reference environment

- Date: 2026-07-21
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, sm_89, 7.63 GiB reported memory
- Driver: 610.43.02
- CUDA toolkit: 13.3, nvcc 13.3.73
- Archived capture OptiX headers: 7.5-compatible SDK
- Current correctness build: OptiX SDK 9.1
- Nsight Compute: 2026.2.1
- Tectonic: 0.16.9

Set `OPTIX_INCLUDE` to the OptiX SDK include directory before building.

## RT and non-RT payloads

The paper uses build mode `both` so the RT-disabled and RT-enabled payloads are
separate artifacts. The default distribution path remains RT-disabled.

```bash
cmake -S src/cuda -B build/rt-paper \
  -DCMAKE_BUILD_TYPE=Release \
  -DGAFIME_CUDA_RT_BUILD_MODE=both \
  -DGAFIME_OPTIX_INCLUDE_DIR="$OPTIX_INCLUDE" \
  '-DGAFIME_CUDA_ARCHITECTURES=89-real;89-virtual'
cmake --build build/rt-paper --parallel
```

Expected payloads:

```text
build/rt-paper/libgafime_cuda_v1.so
build/rt-paper/libgafime_cuda_v1_rt.so
```

Build the standalone benchmark against the RT payload:

```bash
c++ -std=c++20 -O3 -march=native \
  tests/gpu/cuda_rt_membership_scale_bench.cpp \
  -Lbuild/rt-paper \
  -Wl,-rpath,"$PWD/build/rt-paper" \
  -lgafime_cuda_v1_rt \
  -o build/rt-paper/cuda_rt_membership_scale_bench
```

## Current correctness validation

The current hybrid geometry design must be validated with an RT-enabled
payload. These commands do not run a benchmark:

```bash
cmake --build build/rt-paper --target gafime_cuda_v1_rt --parallel 2
GAFIME_CUDA_V1_LIB="$PWD/build/rt-paper/libgafime_cuda_v1_rt.so" \
  cargo test -p gafime-gpu-sys -- --nocapture --test-threads=1
```

The reviewed run completed 61 crate tests plus eight numeric-domain integration
tests, including required-RT 3D parity, three-axis exact-pair grouping,
large-offset target score parity, and below-former-cutoff normal spans.

## Historical source and evidence identity

The archived performance section is bound to file-level hashes for the triangle
prototype source state against which the captured claims were reviewed. The
authoritative historical manifest is
`docs/evidence/rt-gbdt-paper-source-evidence.sha256` (manifest SHA-256
`1eb2db483b5ad2881e99dbaf711af192ad6bfbfb294db443d3c5e6745f2ed429`).
It records historical source commit
`cce2839c2aa9c4c7120b40c47f8303945369c09c` plus hashes for the ABI,
triangle CUDA implementation, Rust wrapper, benchmark,
timing transcript, and profiler report. Those source hashes describe historical
content and are not expected to match the current hybrid worktree.

Neither raw benchmark stdout nor a hash of the original benchmark executable
was retained. The manifest is therefore an exact identity for the reviewed
source files, NCU report, and preserved timing transcription, not cryptographic
proof of the timing measurement or a byte-identical capture binary. The timing
values are provisional development observations. The commands below are the
replay path; do not silently replace the preserved transcript with a later run.

Verify the manifest identity and the immutable evidence artifacts. Historical
source entries can be compared with the committed triangle source; do not run a
blanket `sha256sum --check` against the redesigned worktree:

```bash
printf '%s  %s\n' \
  1eb2db483b5ad2881e99dbaf711af192ad6bfbfb294db443d3c5e6745f2ed429 \
  docs/evidence/rt-gbdt-paper-source-evidence.sha256 \
  | sha256sum --check
printf '%s  %s\n' \
  8fe2b167ecf69597cf68d34137b354fe659d18617e2b5497737157d18955c230 \
  docs/evidence/rt-firsthit-sm89-timing.txt \
  | sha256sum --check
printf '%s  %s\n' \
  5461bf86495d9a12666891bba2f334ecea8b16b3c8cb806168a557101a52c331 \
  docs/evidence/rt-firsthit-sm89-65536x8192-final.ncu-rep \
  | sha256sum --check

historical_commit=cce2839c2aa9c4c7120b40c47f8303945369c09c
while read -r expected path; do
  case "$expected" in ''|'#'*) continue ;; esac
  case "$path" in
    docs/evidence/*) actual=$(sha256sum "$path" | cut -d' ' -f1) ;;
    *) actual=$(git show "$historical_commit:$path" | sha256sum | cut -d' ' -f1) ;;
  esac
  test "$actual" = "$expected" || {
    printf 'hash mismatch: %s\n' "$path" >&2
    exit 1
  }
done < docs/evidence/rt-gbdt-paper-source-evidence.sha256
```

The captured timing transcript has SHA-256
`8fe2b167ecf69597cf68d34137b354fe659d18617e2b5497737157d18955c230`.

## Historical correctness and timing

`gpu_rt_score` is the observed resident warm p50 sample. The following
`gpu_rt_score_timing` line reports the first call, warm p50, warm best, and
sample count separately. The first call includes OptiX initialization,
GAS/IAS construction, and cache population. It does not include matrix upload,
which is reported on its own line. Correctness uses the worst absolute error
across the cold call and every warm call, not only the timed p50 sample.
The benchmark's literal `ray_rate` value is `rows * groups` divided by the full
timed score call. The paper calls this the **end-to-end effective ray rate**; it
is not an isolated hardware traversal rate.

Run the exhaustive parity case through the release-measure wrapper:

```bash
PYTHONPATH="$PWD/python:$PWD/tests/release_measure" \
GAFIME_CUDA_RT_SCALE_BENCH="$PWD/build/rt-paper/cuda_rt_membership_scale_bench" \
GAFIME_CUDA_V1_LIB="$PWD/build/rt-paper/libgafime_cuda_v1_rt.so" \
GAFIME_CUDA_RT_FIRSTHIT_CASE=262144x8192 \
GAFIME_CUDA_RT_FIRSTHIT_REPEATS=6 \
GAFIME_CUDA_RT_FIRSTHIT_MIN_GEVALS=1000 \
GAFIME_CUDA_RT_FIRSTHIT_MAX_ABS=1e-4 \
python tests/release_measure/perf_05_cuda_rt_firsthit_scale.py
```

Selected captured fields present in the timing transcript:

```text
gpu_rt_score          0.886 ms  2423.742 G eval/s
gpu_rt_score_timing first_ms=56.109044 warm_p50_ms=0.886020 \
  warm_best_ms=0.882500 warm_samples=5
firsthit work      groups=8 paths_per_group=1024 rays=2097152 \
  ray_rate=2.367 G ray/s hits=2097152 hit_rate=1.000000
score oracle      rt_max_abs=1.19209e-07
```

The `G eval/s` column is `rows * paths / time`, an all-pairs membership-
equivalent denominator. It is not the number of comparisons executed by BVH
first-hit traversal.

Run the million-path case:

```bash
build/rt-paper/cuda_rt_membership_scale_bench \
  --score-only \
  --partitioned-grid \
  --overlap-axis-pairs=8 \
  --firsthit-score \
  --rt-only \
  --throughput-only \
  --repeats=6 \
  262144x1048576
```

Selected captured fields present in the timing transcript:

```text
gpu_rt_score         20.180 ms  13621.251 G eval/s
gpu_rt_score_timing first_ms=467.746184 warm_p50_ms=20.180078 \
  warm_best_ms=19.951857 warm_samples=5
firsthit work      groups=8 paths_per_group=131072 rays=2097152 \
  ray_rate=0.104 G ray/s hits=2085890 hit_rate=0.994630
score oracle      rt_max_abs=5.58794e-09
```

Despite the retained `--throughput-only` compatibility flag, this partitioned
first-hit shape is correctness-checked. The harness uses an exact
`O(rows * groups + paths)` partition oracle instead of an exhaustive
`O(rows * paths)` CPU scan. Other throughput-only shapes still report parity as
skipped.

## Current safe-triangle/custom-AABB checkpoint

The current source passed 61 `gafime-gpu-sys` tests and eight RT numeric-domain
integration tests with the RT payload explicitly configured. At `65,536 x
8,192`, five fresh processes each made one first call and eight warm calls for
both RT and the existing exhaustive SM fallback. The median of process warm
p50s was `0.347598 ms` for first-hit RT and `41.043185 ms` for SM, with
`4.65661e-10` maximum score error for both. The observed ratio is `118.077x`,
but the SM fallback does not exploit the partition structure and is not an
algorithmically matched baseline. Relative to the prior custom-only checkpoint
at `0.886494 ms`, current RT is `2.550x` faster end to end.

The `262,144 x 8,192` release-floor replay passed at `0.880865 ms`, `2437.926 G`
membership-equivalent evaluations/s, `2.381 G` effective rays/s, and
`4.65661e-10` maximum error. The required floor is `1000 G` evaluations/s with
at most `1e-4` error.

Run this command in five fresh processes for the matched current checkpoint;
do not add `--rt-only`, because the exhaustive SM result is part of the matched
record:

```bash
build/rt-paper/cuda_rt_membership_scale_bench \
  --score-only \
  --partitioned-grid \
  --overlap-axis-pairs=8 \
  --firsthit-score \
  --throughput-only \
  --repeats=9 \
  65536x8192
```

PerfDigest reports five surrounding CUDA kernels in the fresh current report,
but Nsight Compute 2026.2.1 does not expose an OptiX ray-generation or
acceleration-structure unit for this OptiX 9.1 triangle launch. No current
`optixLaunch` duration is inferred. Target stats, grouped point packing, and
score scatter were `161.024 us`, `31.232 us`, and `8.352 us`; their duration
deltas from the prior custom-only report were `+0.239%`, `+0.412%`, and
`-1.136%`. The compact checkpoint, source hashes, and local report hash are in
`docs/evidence/rt-firsthit-hybrid-sm89-checkpoint.txt`.

The current and prior custom-only reports were compared directly with
PerfDigest:

```text
summarize_report(
  format="ncu-rep",
  report_ref="REPO/build/evidence/rt-firsthit-hybrid-dd5e812-sm89-65536x8192.ncu-rep",
  top_n=15)
compare_metrics(
  format="ncu-rep",
  report_a="REPO/build/evidence/rt-firsthit-custom-4cab6ca-sm89-65536x8192.ncu-rep",
  report_b="REPO/build/evidence/rt-firsthit-hybrid-dd5e812-sm89-65536x8192.ncu-rep",
  kernel="pack_grouped_decision_path_points_kernel",
  metrics=["duration_us", "compute_pct_peak", "dram_pct_peak",
           "achieved_occupancy", "l1_hit_rate", "l2_hit_rate",
           "registers_per_thread", "mem_throughput_gbps"])
```

## Current Nsight Compute capture

The following is the replay command for a new capture; it profiles the current
hybrid implementation and must not overwrite or be conflated with the retained
triangle report. `--repeats=1` records a fresh-process cold call
including setup; it is not a resident-warm capture:

```bash
ncu --set full \
  --target-processes all \
  --force-overwrite \
  -o build/rt-paper/rt-firsthit-hybrid-sm89-65536x8192 \
  build/rt-paper/cuda_rt_membership_scale_bench \
  --score-only \
  --partitioned-grid \
  --overlap-axis-pairs=8 \
  --firsthit-score \
  --rt-only \
  --throughput-only \
  --repeats=1 \
  65536x8192
```

Nsight full replay materially distorts wall-clock benchmark timing. Use the
report to attribute profiled units and counters, not as the paper's end-to-end
latency source.

The current report may omit OptiX ray-generation units even though the exact
score and end-to-end timing prove that traversal executed. Treat the list of
profiled units as measured coverage, not as evidence that an absent launch cost
zero time.

## Historical Nsight Compute and PerfDigest

Digest the report through the PerfDigest MCP instead of reading the raw report
directly. Replace `REPO` with the absolute repository path. These are the exact
MCP operations used for the manuscript values:

```text
platform_capabilities()
list_kernels(
  format="ncu-rep",
  report_ref="REPO/docs/evidence/rt-firsthit-sm89-65536x8192-final.ncu-rep")
get_metrics(
  format="ncu-rep",
  report_ref="REPO/docs/evidence/rt-firsthit-sm89-65536x8192-final.ncu-rep",
  kernel="optixLaunch",
  metrics=["duration_us", "compute_pct_peak", "dram_pct_peak",
           "achieved_occupancy", "l1_hit_rate", "l2_hit_rate",
           "registers_per_thread"])
```

The manuscript's historical profiler table is tied to that exact
31,848,275-byte triangle-path capture.
Its SHA-256 is
`5461bf86495d9a12666891bba2f334ecea8b16b3c8cb806168a557101a52c331`.
Verify the local evidence artifact before digesting it:

```bash
printf '%s  %s\n' \
  5461bf86495d9a12666891bba2f334ecea8b16b3c8cb806168a557101a52c331 \
  docs/evidence/rt-firsthit-sm89-65536x8192-final.ncu-rep \
  | sha256sum --check
```

Reference digest for `optixLaunch`, index 13:

| Metric | Value |
| --- | ---: |
| duration | 196.992 us |
| compute-pipe peak | 24.932% |
| DRAM peak | 10.878% |
| achieved occupancy | 54.223% |
| L1 hit rate | 53.408% |
| L2 hit rate | 96.864% |
| registers/thread | 72 |

The report contains eight approximately 156 us group GAS builds, one 49.952 us
IAS build, 25.088 us point packing, 196.992 us `optixLaunch`, and 4.480 us score
scatter/finalization. No available counter directly measures RT-core
saturation, so the paper makes no saturation-percentage claim.

## Optional RT wheel artifact

`gafime-cuda-rt` is not claimed to be available on PyPI. It is a separately
selected Linux x86_64 artifact from the `Build and Publish Wheels` GitHub Actions
workflow. The repository must configure both `GAFIME_OPTIX_SDK_ARCHIVE_URL` and
`GAFIME_OPTIX_SDK_ARCHIVE_SHA256`; the workflow verifies the licensed SDK
archive before use. The staged payload provenance records that digest, the
digest-pinned manylinux wheel builder, the separate lifecycle-fixture image,
and every CUDA toolkit RPM filename and SHA-256 from
`.github/scripts/cuda_13_3_rpms.sha256`.
Trigger the artifact-only lane and bind the selected run to the branch head
observed immediately before dispatch. Leave every publish input at its default
`false` value. An immutable tag is preferable when one exists:

```bash
ref=codex/eager-path-release-hardening
git fetch origin "$ref"
expected_sha=$(git rev-parse "origin/$ref")
dispatch_after=$(date -u +%Y-%m-%dT%H:%M:%SZ)

gh workflow run build_wheels.yml \
  --ref "$ref" \
  -f build_cuda_rt_payload=true

run_id=
for _ in $(seq 1 30); do
  run_id=$(gh run list \
    --workflow build_wheels.yml \
    --branch "$ref" \
    --event workflow_dispatch \
    --limit 20 \
    --json databaseId,createdAt,headSha \
    --jq ".[] | select(.headSha == \"$expected_sha\" and .createdAt >= \"$dispatch_after\") | .databaseId" \
    | head -n 1)
  test -n "$run_id" && break
  sleep 2
done
test -n "$run_id"
actual_sha=$(gh run view "$run_id" --json headSha --jq .headSha)
test "$actual_sha" = "$expected_sha"
gh run watch "$run_id" --exit-status
gh run download "$run_id" \
  --name cuda-rt-linux-artifacts \
  --dir build/rt-wheel-artifacts/rt
gh run download "$run_id" \
  --name cibw-wheels-linux-x86_64 \
  --dir build/rt-wheel-artifacts/core
```

Install the matching base and RT payload wheels into a clean environment. A
clean environment avoids a simultaneous `gafime-cuda` RT-off payload, which
automatic discovery intentionally rejects unless a library override is set.

```bash
python -m venv build/rt-wheel-venv
build/rt-wheel-venv/bin/python -m pip install --upgrade pip

core_wheels=(build/rt-wheel-artifacts/core/gafime-*-cp310-abi3-manylinux_2_28_x86_64.whl)
rt_wheels=(build/rt-wheel-artifacts/rt/gafime_cuda_rt-*-cp310-abi3-manylinux_2_28_x86_64.whl)
test "${#core_wheels[@]}" -eq 1
test "${#rt_wheels[@]}" -eq 1
build/rt-wheel-venv/bin/python -m pip install "${core_wheels[0]}"
build/rt-wheel-venv/bin/python -m pip install --no-deps "${rt_wheels[0]}"

repo="$PWD"
smoke_dir=$(mktemp -d)
cp tests/release_measure/installed_payload_smoke.py "$smoke_dir/"
(
  cd "$smoke_dir"
  "$repo/build/rt-wheel-venv/bin/python" installed_payload_smoke.py \
    --source-root "$repo" \
    --backend cuda \
    --cuda-rt on
)
```

## PDF

The checked-in PDF is built from the checked-in TeX with Tectonic 0.16.9. Pin
the source epoch to 2026-07-21 00:00:00 UTC and compile into two fresh
directories:

```bash
export SOURCE_DATE_EPOCH=1784592000
pdf_a=$(mktemp -d)
pdf_b=$(mktemp -d)

tectonic -X compile \
  docs/rt-gbdt-hardware-ray-tracing-paper.tex \
  --outdir "$pdf_a" \
  --keep-logs \
  --keep-intermediates
tectonic -X compile \
  docs/rt-gbdt-hardware-ray-tracing-paper.tex \
  --outdir "$pdf_b" \
  --keep-logs \
  --keep-intermediates

sha256sum \
  "$pdf_a/rt-gbdt-hardware-ray-tracing-paper.pdf" \
  "$pdf_b/rt-gbdt-hardware-ray-tracing-paper.pdf"
cmp \
  "$pdf_a/rt-gbdt-hardware-ray-tracing-paper.pdf" \
  "$pdf_b/rt-gbdt-hardware-ray-tracing-paper.pdf"
install -m 0644 \
  "$pdf_a/rt-gbdt-hardware-ray-tracing-paper.pdf" \
  docs/rt-gbdt-hardware-ray-tracing-paper.pdf
```

Validate metadata, structure, and text extraction:

```bash
mkdir -p build/rt-paper/pdf
pdfinfo docs/rt-gbdt-hardware-ray-tracing-paper.pdf
qpdf --check docs/rt-gbdt-hardware-ray-tracing-paper.pdf
pdftotext -layout \
  docs/rt-gbdt-hardware-ray-tracing-paper.pdf \
  build/rt-paper/pdf/rt-gbdt-hardware-ray-tracing-paper.txt
rg -n 'tests/gpu/cuda_rt_membership_scale_bench.cpp' \
  build/rt-paper/pdf/rt-gbdt-hardware-ray-tracing-paper.txt
rg -n 'tests/release_measure/perf_05_cuda_rt_firsthit_scale.py' \
  build/rt-paper/pdf/rt-gbdt-hardware-ray-tracing-paper.txt
```

The final PDF must identify `LaTeX with hyperref` as creator and `xdvipdfmx` as
producer. A Chrome/Skia PDF is not an accepted publication artifact because it
previously dropped the two source paths during text rendering.

Reference SHA-256 for the checked-in PDF produced by the commands above:

```text
d9033ff2f2912783b8f341dcdaa4ca76c51734cba20ceed6e67809130fab5132
```
