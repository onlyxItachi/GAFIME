# Memory and compute safety

## VRAM safety policy
GAFIME treats GPU memory as a constrained resource. Workloads are partitioned
into batches sized to fit available VRAM. When VRAM is insufficient, the engine
falls back to smaller batches or CPU execution.

## OOM handling strategy
- Preflight checks estimate buffer sizes before GPU allocation.
- When an allocation fails, the engine reduces batch size and retries.
- If reduction fails repeatedly, execution falls back to CPU.

## Compute control philosophy
User-provided limits cap the number of combinations and generated features.
The engine prefers clear, bounded work over heuristic searches. These limits
are applied consistently across unary and interaction scoring.
