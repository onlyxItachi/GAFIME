cmake_minimum_required(VERSION 3.26)
project(gafime_rocm_v1 LANGUAGES CXX HIP)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx1150")
endif()

set(GAFIME_HIP_WAVE_MI_MODE "64" CACHE STRING
    "MI bin specializations using wave reductions (off, 64, 96, or 64-96)")
set_property(CACHE GAFIME_HIP_WAVE_MI_MODE PROPERTY STRINGS off 64 96 64-96)
if(NOT GAFIME_HIP_WAVE_MI_MODE MATCHES "^(off|64|96|64-96)$")
    message(FATAL_ERROR "GAFIME_HIP_WAVE_MI_MODE must be off, 64, 96, or 64-96")
endif()
set(GAFIME_HIP_WAVE_MI_MASK 0)
if(GAFIME_HIP_WAVE_MI_MODE STREQUAL "64")
    set(GAFIME_HIP_WAVE_MI_MASK 1)
elseif(GAFIME_HIP_WAVE_MI_MODE STREQUAL "96")
    set(GAFIME_HIP_WAVE_MI_MASK 2)
elseif(GAFIME_HIP_WAVE_MI_MODE STREQUAL "64-96")
    set(GAFIME_HIP_WAVE_MI_MASK 3)
endif()

# Precision is a runtime ABI-1.1 selection, not a build-time MI mode.  One
# payload contains all concrete fp32, mixed, and fp64 specialisations.  The
# legacy ABI-1.0 MI marker remains 0 because its float-only entry points remain
# byte-for-byte compatible; ABI-1.1 capability reporting carries the real
# three-profile contract.
set(GAFIME_HIP_PRECISION_PROFILE_MASK 7)
set(GAFIME_HIP_MI_ACCUMULATION_FP64 0)

add_library(gafime_rocm_v1 SHARED
    kernels.hip
    launcher.hip
)

target_compile_definitions(gafime_rocm_v1 PRIVATE
    GAFIME_GPU_BUILDING_DLL
    GAFIME_GPU_MI_ACCUMULATION_FP64=${GAFIME_HIP_MI_ACCUMULATION_FP64}
    GAFIME_HIP_PRECISION_PROFILE_MASK=${GAFIME_HIP_PRECISION_PROFILE_MASK}
)

# -O3 is a performance/optimization flag that does not relax IEEE semantics, so it
# is permitted by the "Compiler Ownership" contract without changing numerical
# results (amdclang++/hipcc keep default within-statement FMA contraction).
target_compile_options(gafime_rocm_v1 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--std=c++23>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
)
string(REPLACE ";" "," GAFIME_HIP_TARGET_ARCH_TAG "${CMAKE_HIP_ARCHITECTURES}")
set_source_files_properties(kernels.hip PROPERTIES COMPILE_DEFINITIONS
    "GAFIME_HIP_WAVE_MI_MASK=${GAFIME_HIP_WAVE_MI_MASK};GAFIME_HIP_TARGET_ARCH_TAG=\"${GAFIME_HIP_TARGET_ARCH_TAG}\""
)
target_include_directories(gafime_rocm_v1 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../common
)

# Link the HIP runtime directly (the packaged hip-config.cmake hardcodes a
# /opt/rocm/bin/hipcc path that is not present on this install).
find_library(GAFIME_AMDHIP64 amdhip64 HINTS $ENV{ROCM_PATH} /opt/rocm PATH_SUFFIXES lib lib64)
if(GAFIME_AMDHIP64)
    target_link_libraries(gafime_rocm_v1 PRIVATE ${GAFIME_AMDHIP64})
endif()

option(GAFIME_ROCM_BUILD_TESTS "Build ROCm v1 ABI smoke tests" OFF)
if(GAFIME_ROCM_BUILD_TESTS)
    include(CTest)
    add_executable(
        gafime_rocm_v1_abi_smoke
        "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/rocm_v1_abi_smoke.cpp"
    )
    target_compile_features(gafime_rocm_v1_abi_smoke PRIVATE cxx_std_20)
    target_compile_definitions(gafime_rocm_v1_abi_smoke PRIVATE
        GAFIME_EXPECT_MI_ACCUMULATION_FP64=${GAFIME_HIP_MI_ACCUMULATION_FP64}
    )
    target_link_libraries(gafime_rocm_v1_abi_smoke PRIVATE gafime_rocm_v1)
    add_test(NAME gafime_rocm_v1_abi_smoke COMMAND gafime_rocm_v1_abi_smoke)

    add_executable(
        gafime_rocm_precision_abi_smoke
        "${CMAKE_CURRENT_LIST_DIR}/precision_abi_smoke.cpp"
    )
    target_compile_features(gafime_rocm_precision_abi_smoke PRIVATE cxx_std_20)
    target_include_directories(gafime_rocm_precision_abi_smoke PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/../common
    )
    target_link_libraries(gafime_rocm_precision_abi_smoke PRIVATE gafime_rocm_v1)
    add_test(NAME gafime_rocm_precision_abi_smoke COMMAND gafime_rocm_precision_abi_smoke)
endif()
