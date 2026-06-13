# gafime-rocm

AMD ROCm/HIP native runtime payload for GAFIME.

Install through the base package extra:

```bash
pip install "gafime[rocm]"
```

This package carries only the ROCm/HIP shared library source/build payload and
its lightweight Python loader package. The public API remains in `gafime`.
