# gafime-cuda

NVIDIA CUDA native runtime payload for GAFIME.

Install through the base package extra:

```bash
pip install "gafime[cuda]"
```

This package carries only the CUDA shared library and its lightweight Python
loader package. The public API remains in `gafime`.
