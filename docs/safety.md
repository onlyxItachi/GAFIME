# Safety & Compute Policies (Internal)

## VRAM Safety Policy

- GPU usage is optional and must be explicitly requested or detected.
- VRAM checks precede any large batched computation.
- If VRAM is insufficient, the engine reduces batch size or falls back to CPU.

## OOM Handling Strategy

- Avoid GPU allocation when the estimated footprint exceeds available VRAM.
- Catch and surface GPU memory errors with actionable guidance.
- Never crash the process without a structured error message.

## Compute Control Philosophy

All compute cost is user-controlled via explicit parameters (e.g. combination
limits, interaction depth, and VRAM residency). GAFIME does not silently expand
search space or perform unbounded feature generation.
