cmake_minimum_required(VERSION 3.26)
project(gafime_cuda_v1 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

option(GAFIME_CUDA_ENABLE_OPTIX_RT "Build primary CUDA payload with OptiX RT-core support (legacy; prefer GAFIME_CUDA_RT_BUILD_MODE=on|both)" OFF)
option(GAFIME_CUDA_BUILD_TESTS "Build CUDA payload ABI and RT lifecycle regression fixtures" OFF)
set(GAFIME_CUDA_RT_BUILD_MODE "" CACHE STRING "CUDA RT-core build mode: off, on, or both")
set_property(CACHE GAFIME_CUDA_RT_BUILD_MODE PROPERTY STRINGS off on both)
set(GAFIME_CUDA_MI_ACCUMULATION_MODE "fast" CACHE STRING
    "Mutual-information arithmetic mode: fast or fp64")
set_property(CACHE GAFIME_CUDA_MI_ACCUMULATION_MODE PROPERTY STRINGS fast fp64)
set(GAFIME_CUDA_OPTIX_PTX_ARCH "compute_75" CACHE STRING "Virtual architecture used for OptiX PTX")
set(GAFIME_OPTIX_INCLUDE_DIR "" CACHE PATH "OptiX SDK include directory")

set(GAFIME_CUDA_V1_SOURCES
    kernels.cu
    launcher.cu
)
set(GAFIME_CUDA_RT_SOURCES
    rt_kernels.cu
    rt_launcher.cu
)

if(NOT DEFINED GAFIME_CUDA_ARCHITECTURES)
    set(GAFIME_CUDA_ARCHITECTURES "89-real;89-virtual")
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
if(NOT GAFIME_CUDA_MI_ACCUMULATION_MODE MATCHES "^(fast|fp64)$")
    message(FATAL_ERROR "GAFIME_CUDA_MI_ACCUMULATION_MODE must be one of: fast, fp64")
endif()
if(GAFIME_CUDA_MI_ACCUMULATION_MODE STREQUAL "fp64")
    set(GAFIME_CUDA_MI_ACCUMULATION_FP64 1)
else()
    set(GAFIME_CUDA_MI_ACCUMULATION_FP64 0)
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
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )

    # -O3 is a performance/optimization flag that does not relax IEEE semantics,
    # so it is permitted by the "Compiler Ownership" contract without changing
    # numerical results (nvcc keeps -fmad within-statement FMA regardless of -O).
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
    target_compile_definitions(${target_name} PRIVATE
        GAFIME_GPU_BUILDING_DLL
        GAFIME_GPU_MI_ACCUMULATION_FP64=${GAFIME_CUDA_MI_ACCUMULATION_FP64}
    )
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/../common
    )
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)

    if(enable_optix_rt)
        gafime_cuda_resolve_optix()

        target_sources(${target_name} PRIVATE ${GAFIME_CUDA_RT_SOURCES})
        target_compile_definitions(${target_name} PRIVATE
            GAFIME_CUDA_LOCAL_DEVICE_FLAGS=0x100u
        )

        set(rt_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/${target_name}")
        set(rt_ptx "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_decision_path.ptx")
        set(rt_header "${rt_generated_dir}/gafime_rt_optix_ptx.hpp")
        add_custom_command(
            OUTPUT "${rt_ptx}"
            COMMAND "${CMAKE_CUDA_COMPILER}"
                --std=c++20
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

if(GAFIME_CUDA_BUILD_TESTS)
    include(CTest)
    find_package(Threads REQUIRED)

    function(gafime_cuda_add_v1_tests payload_target suffix expect_rt)
        set(abi_test "gafime_cuda_v1_abi_smoke_${suffix}")
        add_executable(
            ${abi_test}
            "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/cuda_v1_abi_smoke.cpp"
        )
        target_compile_features(${abi_test} PRIVATE cxx_std_20)
        target_compile_definitions(${abi_test} PRIVATE
            GAFIME_EXPECT_MI_ACCUMULATION_FP64=${GAFIME_CUDA_MI_ACCUMULATION_FP64}
        )
        if(expect_rt)
            target_compile_definitions(${abi_test} PRIVATE GAFIME_CUDA_TEST_LOCAL_RT=1)
        endif()
        target_link_libraries(${abi_test} PRIVATE ${payload_target} CUDA::cudart)
        add_test(NAME ${abi_test} COMMAND ${abi_test})
        set_tests_properties(${abi_test} PROPERTIES SKIP_RETURN_CODE 77)
        if(expect_rt)
            set_tests_properties(
                ${abi_test}
                PROPERTIES ENVIRONMENT "GAFIME_CUDA_REQUIRE_RT_MEMBERSHIP=1"
            )
        endif()

        set(launch_policy_test "gafime_cuda_launch_policy_${suffix}")
        add_executable(
            ${launch_policy_test}
            "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/cuda_launch_policy_test.cu"
        )
        target_compile_features(${launch_policy_test} PRIVATE cxx_std_20)
        set_target_properties(${launch_policy_test} PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
        )
        target_include_directories(${launch_policy_test} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
        target_link_libraries(${launch_policy_test} PRIVATE CUDA::cudart)
        add_test(NAME ${launch_policy_test} COMMAND ${launch_policy_test})

        if(expect_rt)
            set(policy_test "gafime_cuda_rt_state_policy_${suffix}")
            add_executable(
                ${policy_test}
                "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/cuda_rt_state_policy_test.cpp"
            )
            target_compile_features(${policy_test} PRIVATE cxx_std_20)
            target_include_directories(${policy_test} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
            target_link_libraries(
                ${policy_test}
                PRIVATE CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS}
            )
            add_test(
                NAME ${policy_test}
                COMMAND ${policy_test} $<TARGET_FILE:${payload_target}>
            )
        endif()
    endfunction()

    if(GAFIME_CUDA_RT_BUILD_MODE STREQUAL "on")
        gafime_cuda_add_v1_tests(gafime_cuda_v1 rt ON)
        set(gafime_cuda_rt_test_target gafime_cuda_v1)
    else()
        gafime_cuda_add_v1_tests(gafime_cuda_v1 no_rt OFF)
    endif()
    if(GAFIME_CUDA_RT_BUILD_MODE STREQUAL "both")
        gafime_cuda_add_v1_tests(gafime_cuda_v1_rt rt ON)
        set(gafime_cuda_rt_test_target gafime_cuda_v1_rt)
    endif()

    # Compile the reproducible timing fixture with test builds, but do not add
    # performance measurements to the ordinary correctness-oriented CTest run.
    add_executable(
        gafime_cuda_spearman_target_cache_bench
        "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/cuda_spearman_target_cache_bench.cpp"
    )
    target_compile_features(gafime_cuda_spearman_target_cache_bench PRIVATE cxx_std_20)
    target_link_libraries(
        gafime_cuda_spearman_target_cache_bench
        PRIVATE gafime_cuda_v1 CUDA::cudart
    )

    if(DEFINED gafime_cuda_rt_test_target)
        add_executable(
            gafime_cuda_rt_same_device_concurrency
            "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/cuda_rt_same_device_concurrency.cpp"
        )
        target_compile_features(gafime_cuda_rt_same_device_concurrency PRIVATE cxx_std_20)
        target_link_libraries(
            gafime_cuda_rt_same_device_concurrency
            PRIVATE ${gafime_cuda_rt_test_target} CUDA::cudart Threads::Threads
        )
        add_test(
            NAME gafime_cuda_rt_same_device_concurrency
            COMMAND gafime_cuda_rt_same_device_concurrency
        )
        set_tests_properties(
            gafime_cuda_rt_same_device_concurrency
            PROPERTIES SKIP_RETURN_CODE 77
        )
    endif()
endif()
