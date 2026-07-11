cmake_minimum_required(VERSION 3.26)
project(gafime_rocm_v1 LANGUAGES HIP)

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

add_library(gafime_rocm_v1 SHARED
    kernels.hip
    launcher.hip
)

target_compile_definitions(gafime_rocm_v1 PRIVATE GAFIME_GPU_BUILDING_DLL)

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
