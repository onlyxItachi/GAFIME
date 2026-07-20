# RT profiler evidence

`rt-firsthit-sm89-65536x8192-final.ncu-rep` is the exact Nsight Compute 2026.2.1
full-set capture used by the RT decision-path paper. It was captured from the
superseded bounded-2D triangle prototype on an NVIDIA GeForce RTX 4060 Laptop
GPU (`sm_89`) and is intentionally excluded from Python distribution archives.
It is not performance evidence for the current ordered-float custom-primitive
implementation.

```text
size:    31,848,275 bytes
sha256:  5461bf86495d9a12666891bba2f334ecea8b16b3c8cb806168a557101a52c331
```

Use PerfDigest to inspect the report without expanding the raw vendor metrics
into an agent or review context. The reproduction commands and bounded digest
are in `../rt-gbdt-paper-repro.md`.

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
