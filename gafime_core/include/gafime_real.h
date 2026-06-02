#pragma once

// Modern C++ precision switch for native CPU/Core code.
// Default matches the GPU-oriented GAFIME path: fp32. Define
// GAFIME_USE_DOUBLE_PRECISION at compile time for diagnostic fp64 builds.
#ifdef GAFIME_USE_DOUBLE_PRECISION
using real_t = double;
#else
using real_t = float;
#endif
