cmake_minimum_required(VERSION 3.26)
project(gafime_rocm_v1 LANGUAGES HIP)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx1150")
endif()

add_library(gafime_rocm_v1 SHARED
    kernels.hip
    launcher.hip
)

# -O3 is a performance/optimization flag that does not relax IEEE semantics, so it
# is permitted by the "Compiler Ownership" contract without changing numerical
# results (amdclang++/hipcc keep default within-statement FMA contraction).
target_compile_options(gafime_rocm_v1 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--std=c++23>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
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
