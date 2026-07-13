#ifndef GAFIME_ROCM_API_HPP
#define GAFIME_ROCM_API_HPP

// Staged ROCm payload builds still define the legacy spelling. Normalize it
// before the common ABI header selects dllexport versus dllimport on Windows.
#if defined(GAFIME_BUILDING_DLL) && !defined(GAFIME_GPU_BUILDING_DLL)
#define GAFIME_GPU_BUILDING_DLL
#endif

#include "../common/gafime_gpu_abi.hpp"

#endif /* GAFIME_ROCM_API_HPP */
