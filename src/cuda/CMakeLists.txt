cmake_minimum_required(VERSION 3.26)
project(gafime_cuda_v1 LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

option(GAFIME_CUDA_ENABLE_OPTIX_RT "Build CUDA decision-path RT-core OptiX path" OFF)
set(GAFIME_CUDA_OPTIX_PTX_ARCH "compute_75" CACHE STRING "Virtual architecture used for OptiX PTX")
set(GAFIME_OPTIX_INCLUDE_DIR "" CACHE PATH "OptiX SDK include directory")

add_library(gafime_cuda_v1 SHARED
    kernels.cu
    rt_kernels.cu
    launcher.cu
    rt_launcher.cu
)

if(NOT DEFINED GAFIME_CUDA_ARCHITECTURES)
    set(GAFIME_CUDA_ARCHITECTURES "89-real;89-virtual")
endif()
set_target_properties(gafime_cuda_v1 PROPERTIES
    CUDA_ARCHITECTURES "${GAFIME_CUDA_ARCHITECTURES}"
)

# -O3 is a performance/optimization flag that does not relax IEEE semantics, so it
# is permitted by the "Compiler Ownership" contract without changing numerical
# results (nvcc keeps -fmad within-statement FMA regardless of -O level).
target_compile_options(gafime_cuda_v1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--std=c++23>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_include_directories(gafime_cuda_v1 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../common
)
target_link_libraries(gafime_cuda_v1 PRIVATE CUDA::cudart)

if(GAFIME_CUDA_ENABLE_OPTIX_RT)
    if(NOT GAFIME_OPTIX_INCLUDE_DIR)
        find_path(GAFIME_OPTIX_INCLUDE_DIR
            NAMES optix.h
            HINTS
                "$ENV{OPTIX_INCLUDE_DIR}"
                "$ENV{OPTIX_ROOT}/include"
                "$ENV{OPTIX_SDK_ROOT}/include"
        )
    endif()
    if(NOT GAFIME_OPTIX_INCLUDE_DIR)
        message(FATAL_ERROR "GAFIME_CUDA_ENABLE_OPTIX_RT requires GAFIME_OPTIX_INCLUDE_DIR or OPTIX_ROOT")
    endif()

    set(GAFIME_RT_OPTIX_PTX "${CMAKE_CURRENT_BINARY_DIR}/gafime_rt_decision_path.ptx")
    set(GAFIME_RT_OPTIX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/gafime_rt_optix_ptx.hpp")
    add_custom_command(
        OUTPUT "${GAFIME_RT_OPTIX_PTX}"
        COMMAND "${CMAKE_CUDA_COMPILER}"
            --std=c++23
            -O3
            --gpu-architecture=${GAFIME_CUDA_OPTIX_PTX_ARCH}
            -I"${GAFIME_OPTIX_INCLUDE_DIR}"
            -DGAFIME_CUDA_RT_OPTIX_DEVICE
            --ptx "${CMAKE_CURRENT_LIST_DIR}/rt_kernels.cu"
            -o "${GAFIME_RT_OPTIX_PTX}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/rt_kernels.cu" "${CMAKE_CURRENT_LIST_DIR}/rt_kernels.cuh"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${GAFIME_RT_OPTIX_HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
            "-DPTX=${GAFIME_RT_OPTIX_PTX}"
            "-DOUT=${GAFIME_RT_OPTIX_HEADER}"
            -P "${CMAKE_CURRENT_LIST_DIR}/embed_ptx.cmake"
        DEPENDS "${GAFIME_RT_OPTIX_PTX}" "${CMAKE_CURRENT_LIST_DIR}/embed_ptx.cmake"
        VERBATIM
    )
    add_custom_target(gafime_cuda_rt_optix_ptx DEPENDS "${GAFIME_RT_OPTIX_HEADER}")
    add_dependencies(gafime_cuda_v1 gafime_cuda_rt_optix_ptx)
    target_compile_definitions(gafime_cuda_v1 PRIVATE GAFIME_CUDA_ENABLE_OPTIX_RT=1)
    target_include_directories(gafime_cuda_v1 PRIVATE
        "${GAFIME_OPTIX_INCLUDE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
    )
    target_sources(gafime_cuda_v1 PRIVATE "${GAFIME_RT_OPTIX_HEADER}")
    target_link_libraries(gafime_cuda_v1 PRIVATE CUDA::cuda_driver)
endif()
