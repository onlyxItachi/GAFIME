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
   - Installed GAFIME native backends (CUDA when present, C++ core)
   - Installed GAFIME vendor payload packages (`gafime-cuda`, `gafime-rocm`)
   - Rust helper alias availability (`from gafime import subfunctions`)
   - Recommended `backend` and `vram_budget_mb`
   - Recommended discrete mode for the selected backend

3. Present findings to the user in a clear summary table.

4. Generate a ready-to-use Python snippet with the recommended `EngineConfig`:

   ```python
   from gafime import GafimeEngine, EngineConfig, ComputeBudget

   config = EngineConfig(
       budget=ComputeBudget(
           vram_budget_mb=<detected_value>,
           keep_in_vram=True,
           max_discrete_candidates=100_000,
       ),
       backend="<detected_backend>",
       device_id=<detected_device>,
       enable_discrete_functions=True,
       discrete_mode="soft",
   )
   engine = GafimeEngine(config)
   ```

5. If no GPU payload is installed, recommend `backend="core"` and give the
   matching payload command if hardware is visible:
   - NVIDIA: `pip install "gafime[cuda]"`
   - AMD ROCm/HIP on Linux: `pip install "gafime[rocm]"`

## Troubleshooting

- If the script fails with `ModuleNotFoundError`, GAFIME is not installed. Guide the user to `pip install gafime`.
- If NVIDIA hardware is detected but the CUDA payload is missing, suggest `pip install "gafime[cuda]"`.
- If AMD ROCm/HIP hardware is detected on Linux but the ROCm payload is missing, suggest `pip install "gafime[rocm]"`.
- On macOS, `backend="auto"` should prefer Metal and then C++ Core. CUDA is an invalid macOS backend.
- On Linux/Windows x86_64, `backend="auto"` should prefer the installed vendor payload and then C++ Core. If both CUDA and ROCm payloads are installed, default to CUDA and initialize ROCm only when explicitly requested.
- On Linux/Windows ARM64, recommend C++ Core. Current ARM wheels do not expose the CUDA backend.
- `backend="gpu"` is deprecated; recommend `auto`, `cuda`, `metal`, or `core`.
- On GPU backends, discrete feature engineering must use `discrete_mode="soft"`. Hard mode is C++ Core only.
- If `subfunctions` is unavailable, rebuild or reinstall the native Rust helper extension.

## Example

**User says:** "What hardware can GAFIME use on my machine?"

**Actions:** Run `platform_detect.py`, parse output.

**Result:** "You have an NVIDIA RTX 4060 with 8GB VRAM (CUDA 12.4, SM 8.9). I recommend using the CUDA backend with `vram_budget_mb=6144` (leaving 2GB headroom). Here's your config: ..."
