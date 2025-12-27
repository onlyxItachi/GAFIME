# Memory & Compute Safety

## VRAM safety policy
- VRAM usage must be checked before allocating GPU buffers.
- GPU work is batched by default, sized conservatively.
- User configuration controls memory pressure.

## OOM handling strategy
- Detect VRAM availability before starting GPU work.
- If allocation fails, automatically fall back to CPU.
- If keep_in_vram is unsafe, soft-override and continue on CPU.

## Compute control philosophy
- All combinatorial work is capped by user parameters.
- Interaction generation is controlled by max_comb_size and max_combinations_per_k.
- High-order work is limited to a small, user-defined subset.
