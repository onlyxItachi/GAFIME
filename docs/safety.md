# Compute & Memory Safety

## VRAM Safety Policy

GPU usage is opportunistic and guarded by explicit VRAM checks. Batch sizes are
adapted to avoid over-commitment, and GPU usage can be disabled or overridden.

## OOM Handling Strategy

When a planned workload exceeds available VRAM, the system reduces batch sizes
and can fall back to CPU execution with clear, actionable messages.

## Compute Control Philosophy

User-provided parameters cap the search space and resource usage. The engine
prefers predictable, bounded execution over aggressive optimization.
