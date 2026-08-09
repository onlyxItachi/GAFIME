# Cold lifecycle benchmark

`tests/release_measure/cold_lifecycle.py` is the isolated lifecycle layer for
precision-profile performance evidence. It is deliberately separate from the
public end-to-end and native arithmetic layers.

## Contract

The driver starts one fresh subprocess for every profile sample. With the
default three profiles and 30 repetitions it therefore launches 90 workers.
The profile order is randomized, and all six permutations are represented when
all three profiles are selected. The output keeps every raw duration and binds
each worker to the exact payload and benchmark-script SHA-256. Optional
`--wheel` and `--source-root` arguments bind the package and source commit too.

Each worker imports the installed `gafime` package, performs installed-payload
discovery, loads the requested payload with `dlopen`/`LoadLibrary`, resolves
the generic ABI 1.1 operation symbols, negotiates the requested numeric route,
allocates and uploads a typed 4x2 matrix, executes one Pearson candidate, and
frees the matrix. The worker does not use the Rust loader or a dtype-suffixed
ABI symbol.

The report contains median, MAD, p05, p95, bootstrap 95% confidence intervals,
and raw samples for every observed phase. It also records status counts and
the parent-side subprocess duration. The benchmark does not claim a phase that
cannot be separated honestly:

- CUDA and HIP runtime/context initialization is the first explicit
  `cudaFree(0)` or `hipFree(0)` after the payload is loaded; Metal has no
  equivalent separate C runtime boundary and reports that phase as
  `not_observable`;
- host code-object/fatbinary registration runs in loader constructors, so it is
  reported as `observed_combined` with the exact payload's
  `dlopen`/`LoadLibrary` duration instead of assigning invented sub-times;
- target replacement and the state-aware execution-memory forecast are timed
  as their own canonical calls;
- caller-side candidate/protocol/result-buffer planning is timed separately;
- result materialization times the first typed host read of the caller-owned
  structural and metric result buffers after execute returns;
- process exit cannot be isolated from interpreter startup and JSON/provenance
  work; the parent-minus-worker residual is retained as `observed_combined`
  and is never labeled pure teardown.

## Example

```sh
python tests/release_measure/cold_lifecycle.py \
  --backend cuda \
  --payload /absolute/path/to/libgafime_cuda.so \
  --profile fp32,mixed,fp64 \
  --source-root /absolute/path/to/source \
  --wheel /absolute/path/to/gafime_cuda.whl \
  --repetitions 30 \
  --output cold-cuda.json
```

The CUDA and ROCm commands should be run on their corresponding physical
devices with the exact candidate payload. Metal uses `--profile fp32` only.
The report is evidence for the cold lifecycle; it must not be substituted for
the public or device-event benchmark layers.
