---
name: platform-detect
description: Detect the user's hardware platform and recommend optimal GAFIME engine configuration. Use when the user asks about hardware compatibility, which backend to use, GPU detection, VRAM availability, CUDA version, Apple Silicon support, or wants to know the best EngineConfig for their system. Also use when the user says things like "what GPU do I have", "will GAFIME work on my machine", "detect my hardware", or "configure for my system".
---

# Platform Detection

Detect the user's compute hardware and generate an optimal GAFIME `EngineConfig`.

## Instructions

1. Run the detection script:

   ```bash
   python .claude/skills/platform-detect/scripts/platform_detect.py
   ```

2. Read the JSON output. It reports:
   - OS and architecture
   - CUDA availability, GPU name, VRAM, compute capability
   - Apple Silicon / Metal availability
   - CPU core count and OpenMP support
   - Installed GAFIME backends (CUDA, Metal, C++ core, NumPy)
   - Recommended `backend` and `vram_budget_mb`

3. Present findings to the user in a clear summary table.

4. Generate a ready-to-use Python snippet with the recommended `EngineConfig`:

   ```python
   from gafime import GafimeEngine, EngineConfig, ComputeBudget

   config = EngineConfig(
       budget=ComputeBudget(
           vram_budget_mb=<detected_value>,
           keep_in_vram=True,
       ),
       backend="<detected_backend>",
       device_id=<detected_device>,
   )
   engine = GafimeEngine(config)
   ```

5. If no GPU is detected, reassure the user that GAFIME works fine on CPU with the NumPy fallback, just at lower throughput.

## Troubleshooting

- If the script fails with `ModuleNotFoundError`, GAFIME is not installed. Guide the user to `pip install gafime`.
- If CUDA is detected but the GAFIME CUDA backend fails to load, suggest checking that the `gafime_cuda.dll` / `libgafime_cuda.so` is present in the gafime package directory.
- On macOS, Metal is only available on Apple Silicon (arm64). Intel Macs fall back to CPU.

## Example

**User says:** "What hardware can GAFIME use on my machine?"

**Actions:** Run `platform_detect.py`, parse output.

**Result:** "You have an NVIDIA RTX 4060 with 8GB VRAM (CUDA 12.4, SM 8.9). I recommend using the CUDA backend with `vram_budget_mb=6144` (leaving 2GB headroom). Here's your config: ..."
