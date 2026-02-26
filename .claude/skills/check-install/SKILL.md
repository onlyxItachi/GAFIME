---
name: check-install
description: Verify that GAFIME is correctly installed and all components are working. Use when the user just installed GAFIME, wants to verify their setup, asks "is GAFIME working", "test my installation", "health check", "verify install", "check if GPU is working with GAFIME", or encounters import errors after installation.
---

# Installation Health Check

Run a comprehensive verification of the GAFIME installation.

## Instructions

1. Run the health check script:

   ```bash
   python .claude/skills/check-install/scripts/health_check.py
   ```

2. The script performs these checks in order:
   - Python version compatibility (3.10+)
   - Core imports (`gafime`, `numpy`, `polars`)
   - Optional imports (`scikit-learn`)
   - Backend availability (CUDA, Metal, C++ core, NumPy)
   - CLI functionality (`gafime --init` can run)
   - Functional test: runs a tiny synthetic analysis end-to-end
   - Throughput benchmark: measures combinations per second

3. Present results as a clear checklist with pass/fail status.

4. If any check fails, provide the specific fix:
   - Python too old: "Upgrade to Python 3.10+"
   - Missing numpy/polars: "pip install numpy polars"
   - Missing sklearn: "pip install gafime[sklearn]"
   - Backend not loading: Refer to the `troubleshoot-backend` skill
   - Functional test fails: Likely a packaging issue, try reinstalling

## Example

**User says:** "I just did pip install gafime, is it working?"

**Actions:** Run `health_check.py`

**Result:**

```
GAFIME Health Check
  [PASS] Python 3.11.5 (>= 3.10 required)
  [PASS] gafime v0.2.0 imported
  [PASS] numpy 1.26.4
  [PASS] polars 0.20.31
  [PASS] scikit-learn 1.4.0
  [PASS] CUDA backend (RTX 4060, 8GB VRAM)
  [SKIP] Metal backend (not macOS arm64)
  [PASS] NumPy fallback
  [PASS] Functional test: signal detected in synthetic data
  [PASS] Throughput: 18,432 combinations/sec

All checks passed! GAFIME is ready to use.
```
