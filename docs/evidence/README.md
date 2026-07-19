# RT profiler evidence

`rt-firsthit-sm89-65536x8192-final.ncu-rep` is the exact Nsight Compute 2026.2.1
full-set capture used by the RT decision-path paper. It was captured on an NVIDIA
GeForce RTX 4060 Laptop GPU (`sm_89`) and is intentionally excluded from Python
distribution archives.

```text
size:    31,848,275 bytes
sha256:  5461bf86495d9a12666891bba2f334ecea8b16b3c8cb806168a557101a52c331
```

Use PerfDigest to inspect the report without expanding the raw vendor metrics
into an agent or review context. The reproduction commands and bounded digest
are in `../rt-gbdt-paper-repro.md`.

`rt-firsthit-sm89-timing.txt` is the immutable plain-text transcript for the two
paper timing claims. It was transcribed from results captured on 2026-07-19; no
benchmark was rerun during the 2026-07-20 documentation review.

```text
sha256:  47e761b45dcbbc6d5a3c658939b34a3ad02d9837638a38393e034ce598b73d28
```

`rt-gbdt-paper-source-evidence.sha256` binds the ABI, CUDA implementation, Rust
wrapper, benchmark/release gates, payload workflow, transcript, and NCU report
to exact file hashes for the reviewed reproducibility state. No original
benchmark-executable hash was retained. The manifest itself has SHA-256:

```text
fd3570bffed1d7122dfe071b6b7fa5653d3b6e71e11637c937695b1dccc106bf
```

Run all hash checks from the repository root. Use PerfDigest, not direct binary
inspection, for the `.ncu-rep` contents.
