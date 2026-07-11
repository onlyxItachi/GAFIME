cmake_minimum_required(VERSION 3.26)
project(gafime_cuda_v1 LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

option(GAFIME_CUDA_ENABLE_OPTIX_RT "Build primary CUDA payload with OptiX RT-core support (legacy; prefer GAFIME_CUDA_RT_BUILD_MODE=on|both)" OFF)
set(GAFIME_CUDA_RT_BUILD_MODE "" CACHE STRING "CUDA RT-core build mode: off, on, or both")
set_property(CACHE GAFIME_CUDA_RT_BUILD_MODE PROPERTY STRINGS off on both)
set(GAFIME_CUDA_OPTIX_PTX_ARCH "compute_75" CACHE STRING "Virtual architecture used for OptiX PTX")
set(GAFIME_OPTIX_INCLUDE_DIR "" CACHE PATH "OptiX SDK include directory")

set(GAFIME_CUDA_V1_SOURCES
    kernels.cu
    rt_kernels.cu
    launcher.cu
    rt_launcher.cu
)

if(NOT DEFINED GAFIME_CUDA_ARCHITECTURES)
    set(GAFIME_CUDA_ARCHITECTURES "89-real;89-virtual")
endif()
set(GAFIME_CUDA_TUNING_SM "" CACHE STRING "CUDA SM used for compile-time kernel tuning traits")
if(NOT GAFIME_CUDA_TUNING_SM)
    list(GET GAFIME_CUDA_ARCHITECTURES 0 GAFIME_CUDA_TUNING_ARCH_ENTRY)
    string(REGEX MATCH "^[0-9]+" GAFIME_CUDA_TUNING_SM "${GAFIME_CUDA_TUNING_ARCH_ENTRY}")
    if(NOT GAFIME_CUDA_TUNING_SM)
        # Symbolic CMake architecture sets such as `all` do not identify one
        # tuning target. Use the distribution baseline unless explicitly set.
        set(GAFIME_CUDA_TUNING_SM "89")
    endif()
endif()
if(NOT GAFIME_CUDA_TUNING_SM MATCHES "^[0-9]+$")
    message(FATAL_ERROR "GAFIME_CUDA_TUNING_SM must be a numeric SM value")
endif()

if(NOT GAFIME_CUDA_RT_BUILD_MODE)
    if(GAFIME_CUDA_ENABLE_OPTIX_RT)
        set(GAFIME_CUDA_RT_BUILD_MODE "on")
    else()
        set(GAFIME_CUDA_RT_BUILD_MODE "off")
    endif()
endif()
if(NOT GAFIME_CUDA_RT_BUILD_MODE MATCHES "^(off|on|both)$")
    message(FATAL_ERROR "GAFIME_CUDA_RT_BUILD_MODE must be one of: off, on, both")
endif()

function(gafime_cuda_resolve_optix)
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
        message(FATAL_ERROR "CUDA RT-core support requires GAFIME_OPTIX_INCLUDE_DIR or OPTIX_ROOT")
    endif()
    set(GAFIME_OPTIX_INCLUDE_DIR "${GAFIME_OPTIX_INCLUDE_DIR}" PARENT_SCOPE)
endfunction()

function(gafime_cuda_add_v1_target target_name output_name enable_optix_rt)
    add_library(${target_name} SHARED ${GAFIME_CUDA_V1_SOURCES})
    set_target_properties(${target_name} PROPERTIES
        OUTPUT_NAME "${output_name}"
        CUDA_ARCHITECTURES "${GAFIME_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
    )

    # -O3 is a performance/optimization flag that does not relax IEEE semantics,
    # so it is permitted by the "Compiler Ownership" contract without changing
    # numerical results (nvcc keeps -fmad within-statement FMA regardless of -O).
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--std=c++23>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
    target_compile_definitions(${target_name} PRIVATE
        GAFIME_GPU_BUILDING_DLL
        GAFIME_CUDA_TUNING_SM=${GAFIME_CUDA_TUNING_SM}
    )
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/../common
    )
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)

    if(enable_optix_rt)
        gafime_cuda_resolve_optix()

        set(rt_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/${target_name}")
        set(rt_ptx "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_decision_path.ptx")
        set(rt_header "${rt_generated_dir}/gafime_rt_optix_ptx.hpp")
        add_custom_command(
            OUTPUT "${rt_ptx}"
            COMMAND "${CMAKE_CUDA_COMPILER}"
                --std=c++23
                -O3
                --gpu-architecture=${GAFIME_CUDA_OPTIX_PTX_ARCH}
                -I"${GAFIME_OPTIX_INCLUDE_DIR}"
                -DGAFIME_CUDA_RT_OPTIX_DEVICE
                --ptx "${CMAKE_CURRENT_LIST_DIR}/rt_kernels.cu"
                -o "${rt_ptx}"
            DEPENDS "${CMAKE_CURRENT_LIST_DIR}/rt_kernels.cu" "${CMAKE_CURRENT_LIST_DIR}/rt_kernels.cuh"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${rt_header}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${rt_generated_dir}"
            COMMAND "${CMAKE_COMMAND}"
                "-DPTX=${rt_ptx}"
                "-DOUT=${rt_header}"
                -P "${CMAKE_CURRENT_LIST_DIR}/embed_ptx.cmake"
            DEPENDS "${rt_ptx}" "${CMAKE_CURRENT_LIST_DIR}/embed_ptx.cmake"
            VERBATIM
        )
        add_custom_target(${target_name}_optix_ptx DEPENDS "${rt_header}")
        add_dependencies(${target_name} ${target_name}_optix_ptx)
        target_compile_definitions(${target_name} PRIVATE GAFIME_CUDA_ENABLE_OPTIX_RT=1)
        target_include_directories(${target_name} PRIVATE
            "${GAFIME_OPTIX_INCLUDE_DIR}"
            "${rt_generated_dir}"
        )
        target_sources(${target_name} PRIVATE "${rt_header}")
        target_link_libraries(${target_name} PRIVATE CUDA::cuda_driver)
    endif()
endfunction()

if(GAFIME_CUDA_RT_BUILD_MODE STREQUAL "on")
    gafime_cuda_add_v1_target(gafime_cuda_v1 gafime_cuda_v1 ON)
else()
    gafime_cuda_add_v1_target(gafime_cuda_v1 gafime_cuda_v1 OFF)
endif()

if(GAFIME_CUDA_RT_BUILD_MODE STREQUAL "both")
    gafime_cuda_add_v1_target(gafime_cuda_v1_rt gafime_cuda_v1_rt ON)
endif()
