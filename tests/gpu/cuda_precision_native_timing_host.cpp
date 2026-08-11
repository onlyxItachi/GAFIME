// Canonical and host CUDA timing lanes are intentionally ordinary C++
// executables.  The included harness selects those lanes at preprocessing
// time, so no NVCC compilation, CUDA device code, fatbin, or module
// registration can enter either executable.
#include "cuda_precision_native_timing.cu"
