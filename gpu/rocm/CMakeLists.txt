cmake_minimum_required(VERSION 3.26)
project(gafime_rocm_v1 LANGUAGES CXX HIP)

# Codegen policy (ROADMAP §2), HIP mirror of the CUDA lib. Default to the live
# gfx1150 (RDNA3.5) but allow an override. No fast-math (golden parity).
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx1150")
endif()

add_library(gafime_rocm_v1 SHARED
    host.cpp
    tune.cpp
)

set_source_files_properties(host.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(gafime_rocm_v1 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++23>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:--std=c++23>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
)
target_include_directories(gafime_rocm_v1 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../include
    ${CMAKE_CURRENT_LIST_DIR}/../common
)

# Link the HIP runtime directly (the packaged hip-config.cmake hardcodes a
# /opt/rocm/bin/hipcc path that is not present on this install).
find_library(GAFIME_AMDHIP64 amdhip64 HINTS $ENV{ROCM_PATH} /opt/rocm PATH_SUFFIXES lib lib64)
if(GAFIME_AMDHIP64)
    target_link_libraries(gafime_rocm_v1 PRIVATE ${GAFIME_AMDHIP64})
endif()
