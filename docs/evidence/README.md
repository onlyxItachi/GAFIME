# RT profiler evidence

`rt-firsthit-sm89-65536x8192-final.ncu-rep` is the exact Nsight Compute 2026.2.1
full-set capture used by the RT decision-path paper. It was captured from the
superseded bounded-2D triangle prototype on an NVIDIA GeForce RTX 4060 Laptop
GPU (`sm_89`). The raw capture was removed from the current source tree and
archived in immutable historical Git commit `cce2839c2aa9c4c7120b40c47f8303945369c09c`.
It is not performance evidence for the current safe-triangle/custom-AABB
dispatch.

```text
size:    31,848,275 bytes
sha256:  5461bf86495d9a12666891bba2f334ecea8b16b3c8cb806168a557101a52c331
```

Restore this historic report when needed from a full-history checkout. The
restored file is Git-ignored and must not be committed or bundled into releases:

```bash
historical_commit=cce2839c2aa9c4c7120b40c47f8303945369c09c
profiler_path=docs/evidence/rt-firsthit-sm89-65536x8192-final.ncu-rep
if [ ! -f "$profiler_path" ]; then
  git show "$historical_commit:$profiler_path" > "$profiler_path"
fi
printf '%s  %s\n' 5461bf86495d9a12666891bba2f334ecea8b16b3c8cb806168a557101a52c331 "$profiler_path" | sha256sum --check
```

[Original captured blob](https://github.com/onlyxItachi/GAFIME/blob/cce2839c2aa9c4c7120b40c47f8303945369c09c/docs/evidence/rt-firsthit-sm89-65536x8192-final.ncu-rep) is a pinned historical reference, not current hybrid-implementation evidence.

Use PerfDigest to inspect the report without expanding the raw vendor metrics
into an agent or review context. The reproduction commands and bounded digest
are in `../rt-gbdt-paper-repro.md`.

`rt-firsthit-custom-sm89-checkpoint.txt` records the correctness-hardened,
custom-AABB-only checkpoint. `rt-firsthit-hybrid-sm89-checkpoint.txt` supersedes
it for the current implementation: safe bounded 2D grouped plans use exact-
guarded triangles, while every other shape retains custom AABBs. The latter
contains five-process matched timing, the release-floor replay, source and local
report hashes, the bounded PerfDigest result, and the final single-group
duplicate-mask closure. The latter includes matched five-process timing and a
bounded Nsight Systems memory-operation summary proving that two 67.109 MB mask
clears are absent. The current raw reports remain ignored: Nsight Compute
exposes only the five surrounding CUDA kernels, not the OptiX ray-generation
unit for this launch, and PerfDigest does not register `.nsys-rep`.

`rt-firsthit-sm89-timing.txt` is a preserved manual transcription of two
development runs. Raw stdout and the original executable hash were not retained,
so its hash proves transcription identity rather than authenticating a timing
measurement. The paper labels these values provisional and historical to the
triangle prototype. No benchmark was rerun for the custom-primitive redesign.

```text
sha256:  8fe2b167ecf69597cf68d34137b354fe659d18617e2b5497737157d18955c230
```

`rt-gbdt-paper-source-evidence.sha256` binds the historical ABI, triangle CUDA
implementation, Rust wrapper, benchmark, transcript, and NCU report to exact
file hashes for the reviewed prototype state. Its source entries are historical
and are not expected to match the redesigned worktree. No original benchmark
executable hash was retained. The manifest itself has SHA-256:

```text
1eb2db483b5ad2881e99dbaf711af192ad6bfbfb294db443d3c5e6745f2ed429
```

Run all hash checks from the repository root. Use PerfDigest, not direct binary
inspection, for the `.ncu-rep` contents.
