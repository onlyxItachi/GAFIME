cmake_minimum_required(VERSION 3.26)
project(gafime_cuda_v1 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

option(GAFIME_CUDA_ENABLE_OPTIX_RT "Build primary CUDA payload with OptiX RT-core support (legacy; prefer GAFIME_CUDA_RT_BUILD_MODE=on|both)" OFF)
option(GAFIME_CUDA_BUILD_TESTS "Build CUDA payload ABI and RT lifecycle regression fixtures" OFF)
option(GAFIME_CUDA_BUILD_BENCHMARKS "Build benchmark-only native timing helpers (never added to CTest)" OFF)
set(GAFIME_CUDA_RT_BUILD_MODE "" CACHE STRING "CUDA RT-core build mode: off, on, or both")
set_property(CACHE GAFIME_CUDA_RT_BUILD_MODE PROPERTY STRINGS off on both)
set(GAFIME_CUDA_OPTIX_PTX_ARCH "compute_75" CACHE STRING "Virtual architecture used for OptiX PTX")
set(GAFIME_OPTIX_INCLUDE_DIR "" CACHE PATH "OptiX SDK include directory")

set(GAFIME_CUDA_V1_SOURCES
    precision_kernels.cu
    precision_launcher.cu
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
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
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
    )
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/../common
    )
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)
    if(UNIX AND NOT APPLE)
        set(gafime_cuda_exports_map
            "${CMAKE_CURRENT_LIST_DIR}/../common/gafime_gpu_exports.map")
        if(enable_optix_rt)
            set(gafime_cuda_exports_map
                "${CMAKE_CURRENT_LIST_DIR}/gafime_gpu_rt_exports.map")
        endif()
        target_link_options(${target_name} PRIVATE
            "-Wl,--version-script=${gafime_cuda_exports_map}"
        )
    endif()

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

    # The v2 fixture is deliberately source-local: it validates that a single
    # CUDA payload links and physically executes each compiled precision lane
    # without introducing a precision-specific distribution or test home.
    add_executable(
        gafime_cuda_precision_abi_smoke
        "${CMAKE_CURRENT_LIST_DIR}/precision_abi_smoke.cpp"
    )
    target_compile_features(gafime_cuda_precision_abi_smoke PRIVATE cxx_std_20)
    target_include_directories(gafime_cuda_precision_abi_smoke PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    target_link_libraries(gafime_cuda_precision_abi_smoke PRIVATE gafime_cuda_v1 CUDA::cudart)
    add_test(NAME gafime_cuda_precision_abi_smoke COMMAND gafime_cuda_precision_abi_smoke)
    set_tests_properties(gafime_cuda_precision_abi_smoke PROPERTIES SKIP_RETURN_CODE 77)

    function(gafime_cuda_add_v1_tests payload_target suffix expect_rt)
        set(abi_test "gafime_cuda_v1_abi_smoke_${suffix}")
        add_executable(
            ${abi_test}
            "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/cuda_v1_abi_smoke.cpp"
        )
        target_compile_features(${abi_test} PRIVATE cxx_std_20)
        target_compile_definitions(${abi_test} PRIVATE
            # ABI 1.0 remains the historical fp32 lane.  The additive v2
            # profile dispatch carries fp32/mixed/fp64 MI specializations in
            # this same payload rather than selecting one at build time.
            GAFIME_EXPECT_MI_ACCUMULATION_FP64=0
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

    set(GAFIME_ABI_CONSUMER_PAYLOAD_TARGET gafime_cuda_v1)
    set(GAFIME_ABI_CONSUMER_BACKEND_KIND 2)
    set(GAFIME_ABI_CONSUMER_EXPECTED_ROUTE_COUNT 3)
    set(GAFIME_ABI_CONSUMER_TEST_SUFFIX cuda)
    set(GAFIME_ABI_CONSUMER_ENABLE_DYNAMIC_ARITY ON)
    add_subdirectory(
        "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/abi_consumers"
        "${CMAKE_CURRENT_BINARY_DIR}/abi_consumers"
    )
    unset(GAFIME_ABI_CONSUMER_PAYLOAD_TARGET)
    unset(GAFIME_ABI_CONSUMER_BACKEND_KIND)
    unset(GAFIME_ABI_CONSUMER_EXPECTED_ROUTE_COUNT)
    unset(GAFIME_ABI_CONSUMER_TEST_SUFFIX)
    unset(GAFIME_ABI_CONSUMER_ENABLE_DYNAMIC_ARITY)
endif()

# Benchmark-only native timing decomposition.  This helper intentionally stays
# outside the payload and outside CTest: it is a supplemental direct-kernel
# instrument whose JSON must carry separate canonical stable-ABI lifecycle
# evidence before any arithmetic claim can consume it.  Keeping the option
# independent from GAFIME_CUDA_BUILD_TESTS lets release jobs build only this
# helper without enabling the correctness fixture matrix.
if(GAFIME_CUDA_BUILD_BENCHMARKS)
    if(WIN32)
        message(FATAL_ERROR
            "CUDA native benchmark helpers are Linux-only evidence producers")
    endif()
    # Native provenance must never depend on PATH or inherited Git redirection
    # variables.  Resolve one absolute executable at configure time and bake
    # its path, version, and SHA-256 into every benchmark helper.
    set(GAFIME_CUDA_BENCHMARK_GIT_EXECUTABLE "" CACHE FILEPATH
        "Absolute trusted Git executable used for CUDA benchmark provenance")
    if(NOT GAFIME_CUDA_BENCHMARK_GIT_EXECUTABLE)
        foreach(candidate "/usr/bin/git" "/bin/git")
            if(IS_EXECUTABLE "${candidate}")
                set(GAFIME_CUDA_BENCHMARK_GIT_EXECUTABLE "${candidate}")
                break()
            endif()
        endforeach()
    endif()
    if(NOT IS_ABSOLUTE "${GAFIME_CUDA_BENCHMARK_GIT_EXECUTABLE}" OR
       NOT IS_EXECUTABLE "${GAFIME_CUDA_BENCHMARK_GIT_EXECUTABLE}")
        message(FATAL_ERROR
            "CUDA benchmark provenance requires an absolute executable Git path")
    endif()
    get_filename_component(GAFIME_CUDA_BENCHMARK_GIT_EXECUTABLE
        "${GAFIME_CUDA_BENCHMARK_GIT_EXECUTABLE}" REALPATH)
    set(GAFIME_CUDA_BENCHMARK_GIT_ALLOWED FALSE)
    foreach(candidate "/usr/bin/git" "/bin/git")
        if(IS_EXECUTABLE "${candidate}")
            get_filename_component(candidate_realpath "${candidate}" REALPATH)
            if(GAFIME_CUDA_BENCHMARK_GIT_EXECUTABLE STREQUAL candidate_realpath)
                set(GAFIME_CUDA_BENCHMARK_GIT_ALLOWED TRUE)
            endif()
        endif()
    endforeach()
    if(NOT GAFIME_CUDA_BENCHMARK_GIT_ALLOWED)
        message(FATAL_ERROR
            "CUDA benchmark Git must resolve to /usr/bin/git or /bin/git")
    endif()
    set(GAFIME_CUDA_BENCHMARK_GIT_CONFIG_NULL "/dev/null")
    execute_process(
        COMMAND "${GAFIME_CUDA_BENCHMARK_GIT_EXECUTABLE}" --version
        OUTPUT_VARIABLE GAFIME_CUDA_BENCHMARK_GIT_VERSION
        ERROR_VARIABLE GAFIME_CUDA_BENCHMARK_GIT_VERSION_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GAFIME_CUDA_BENCHMARK_GIT_VERSION_STATUS)
    file(SHA256 "${GAFIME_CUDA_BENCHMARK_GIT_EXECUTABLE}"
        GAFIME_CUDA_BENCHMARK_GIT_SHA256)
    if(NOT GAFIME_CUDA_BENCHMARK_GIT_VERSION_STATUS EQUAL 0 OR
       NOT GAFIME_CUDA_BENCHMARK_GIT_VERSION MATCHES "^git version ")
        message(FATAL_ERROR "trusted CUDA benchmark Git executable is unusable")
    endif()

    # Build the common benchmark harness against the exact product tree being
    # measured.  A release A/B configures this target twice from the final
    # harness checkout: once with the frozen baseline worktree and once with
    # the candidate worktree.  This prevents a candidate-linked helper from
    # masquerading as direct-kernel baseline evidence merely because its
    # runtime --source-root points at the baseline.
    set(GAFIME_CUDA_BENCHMARK_PRODUCT_ROOT
        "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH
        "Clean GAFIME product checkout whose CUDA kernels are linked into the native timing helper")
    get_filename_component(GAFIME_CUDA_BENCHMARK_PRODUCT_ROOT
        "${GAFIME_CUDA_BENCHMARK_PRODUCT_ROOT}" REALPATH)
    set(GAFIME_CUDA_BENCHMARK_PRODUCT_SOURCE
        "${GAFIME_CUDA_BENCHMARK_PRODUCT_ROOT}/src/cuda/precision_kernels.cu")
    set(GAFIME_CUDA_BENCHMARK_PRODUCT_HEADER
        "${GAFIME_CUDA_BENCHMARK_PRODUCT_ROOT}/src/cuda/precision_kernels.cuh")
    foreach(required_product_file
            "${GAFIME_CUDA_BENCHMARK_PRODUCT_SOURCE}"
            "${GAFIME_CUDA_BENCHMARK_PRODUCT_HEADER}")
        if(NOT EXISTS "${required_product_file}")
            message(FATAL_ERROR
                "CUDA benchmark product source is missing: ${required_product_file}")
        endif()
    endforeach()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            --unset=GIT_DIR --unset=GIT_WORK_TREE --unset=GIT_INDEX_FILE
            --unset=GIT_OBJECT_DIRECTORY --unset=GIT_ALTERNATE_OBJECT_DIRECTORIES
            --unset=GIT_COMMON_DIR --unset=GIT_CEILING_DIRECTORIES
            --unset=GIT_CONFIG --unset=GIT_CONFIG_COUNT
            --unset=GIT_CONFIG_NOSYSTEM --unset=GIT_CONFIG_GLOBAL
            --unset=GIT_CONFIG_SYSTEM --unset=GIT_CONFIG_KEY_0
            --unset=GIT_CONFIG_VALUE_0
            GIT_CONFIG_NOSYSTEM=1
            GIT_CONFIG_GLOBAL=${GAFIME_CUDA_BENCHMARK_GIT_CONFIG_NULL}
            GIT_CONFIG_SYSTEM=${GAFIME_CUDA_BENCHMARK_GIT_CONFIG_NULL}
            GIT_CONFIG_COUNT=0
            "${GAFIME_CUDA_BENCHMARK_GIT_EXECUTABLE}" -C
            "${GAFIME_CUDA_BENCHMARK_PRODUCT_ROOT}" rev-parse HEAD
        OUTPUT_VARIABLE GAFIME_CUDA_BENCHMARK_PRODUCT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GAFIME_CUDA_BENCHMARK_PRODUCT_GIT_STATUS
    )
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            --unset=GIT_DIR --unset=GIT_WORK_TREE --unset=GIT_INDEX_FILE
            --unset=GIT_OBJECT_DIRECTORY --unset=GIT_ALTERNATE_OBJECT_DIRECTORIES
            --unset=GIT_COMMON_DIR --unset=GIT_CEILING_DIRECTORIES
            --unset=GIT_CONFIG --unset=GIT_CONFIG_COUNT
            --unset=GIT_CONFIG_NOSYSTEM --unset=GIT_CONFIG_GLOBAL
            --unset=GIT_CONFIG_SYSTEM --unset=GIT_CONFIG_KEY_0
            --unset=GIT_CONFIG_VALUE_0
            GIT_CONFIG_NOSYSTEM=1
            GIT_CONFIG_GLOBAL=${GAFIME_CUDA_BENCHMARK_GIT_CONFIG_NULL}
            GIT_CONFIG_SYSTEM=${GAFIME_CUDA_BENCHMARK_GIT_CONFIG_NULL}
            GIT_CONFIG_COUNT=0
            "${GAFIME_CUDA_BENCHMARK_GIT_EXECUTABLE}" -C
            "${GAFIME_CUDA_BENCHMARK_PRODUCT_ROOT}"
            status --porcelain=v1 --untracked-files=all
        OUTPUT_VARIABLE GAFIME_CUDA_BENCHMARK_PRODUCT_TREE_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GAFIME_CUDA_BENCHMARK_PRODUCT_TREE_GIT_STATUS
    )
    string(LENGTH "${GAFIME_CUDA_BENCHMARK_PRODUCT_COMMIT}"
        GAFIME_CUDA_BENCHMARK_PRODUCT_COMMIT_LENGTH)
    if(NOT GAFIME_CUDA_BENCHMARK_PRODUCT_GIT_STATUS EQUAL 0 OR
       NOT GAFIME_CUDA_BENCHMARK_PRODUCT_COMMIT_LENGTH EQUAL 40 OR
       NOT GAFIME_CUDA_BENCHMARK_PRODUCT_COMMIT MATCHES "^[0-9a-fA-F]+$")
        message(FATAL_ERROR
            "CUDA benchmark product root must resolve to a full Git commit")
    endif()
    if(NOT GAFIME_CUDA_BENCHMARK_PRODUCT_TREE_GIT_STATUS EQUAL 0 OR
       NOT GAFIME_CUDA_BENCHMARK_PRODUCT_TREE_STATUS STREQUAL "")
        message(FATAL_ERROR
            "CUDA benchmark product root must be a clean Git worktree")
    endif()
    file(SHA256 "${GAFIME_CUDA_BENCHMARK_PRODUCT_SOURCE}"
        GAFIME_CUDA_BENCHMARK_PRODUCT_SOURCE_SHA256)
    file(SHA256 "${GAFIME_CUDA_BENCHMARK_PRODUCT_HEADER}"
        GAFIME_CUDA_BENCHMARK_PRODUCT_HEADER_SHA256)
    file(READ "${GAFIME_CUDA_BENCHMARK_PRODUCT_HEADER}"
        GAFIME_CUDA_BENCHMARK_PRODUCT_HEADER_TEXT)
    string(FIND "${GAFIME_CUDA_BENCHMARK_PRODUCT_HEADER_TEXT}"
        "continuous_unary" GAFIME_CUDA_BENCHMARK_CONTINUOUS_UNARY_OFFSET)
    if(GAFIME_CUDA_BENCHMARK_CONTINUOUS_UNARY_OFFSET LESS 0)
        set(GAFIME_CUDA_BENCHMARK_HAS_CONTINUOUS_UNARY 0)
    else()
        set(GAFIME_CUDA_BENCHMARK_HAS_CONTINUOUS_UNARY 1)
    endif()

    # Keep the benchmark source gate close to the target.  The helper is a
    # release-evidence producer, so silently regressing to a generic-only probe
    # or a direct-kernel-only executable must fail at configure time.
    set(GAFIME_CUDA_NATIVE_TIMING_SOURCE
        "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/cuda_precision_native_timing.cu")
    if(NOT EXISTS "${GAFIME_CUDA_NATIVE_TIMING_SOURCE}")
        message(FATAL_ERROR "CUDA native timing source is missing")
    endif()
    file(READ "${GAFIME_CUDA_NATIVE_TIMING_SOURCE}" GAFIME_CUDA_NATIVE_TIMING_TEXT)
    foreach(required_marker
            "precision-typed-v1.1"
            "numeric-route-v2"
            "canonical_payload_api"
            "host_synchronized_payload_api"
            "supplemental_internal_kernel"
            "|device="
            "fresh_helper_process_per_variant_trial"
            "canonical_requested"
            "--input-policy")
        string(FIND "${GAFIME_CUDA_NATIVE_TIMING_TEXT}" "${required_marker}" marker_offset)
        if(marker_offset LESS 0)
            message(FATAL_ERROR
                "CUDA native timing source gate failed: missing ${required_marker}")
        endif()
    endforeach()
    unset(GAFIME_CUDA_NATIVE_TIMING_TEXT)
    unset(GAFIME_CUDA_NATIVE_TIMING_SOURCE)

    set(GAFIME_CUDA_NATIVE_TIMING_HELPER_SOURCE
        "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/cuda_precision_native_timing.cu")
    set(GAFIME_CUDA_NATIVE_TIMING_HOST_SOURCE
        "${CMAKE_CURRENT_LIST_DIR}/../../tests/gpu/cuda_precision_native_timing_host.cpp")
    if(NOT EXISTS "${GAFIME_CUDA_NATIVE_TIMING_HOST_SOURCE}")
        message(FATAL_ERROR "CUDA host-only native timing wrapper is missing")
    endif()

    # The direct helper retains the product precision translation unit for
    # supplemental_internal_kernel evidence.  Canonical and host helpers
    # include the same harness source through a C++ wrapper and deliberately do
    # not compile that source with NVCC: CUDA fatbin/module registration occurs
    # before main(), so a runtime lane flag cannot make a CUDA-compiled helper
    # a clean canonical or host process.
    function(gafime_cuda_add_benchmark_target target_name lane_value lane_name include_product_kernel)
        if(include_product_kernel)
            set(benchmark_sources "${GAFIME_CUDA_NATIVE_TIMING_HELPER_SOURCE}")
        else()
            set(benchmark_sources "${GAFIME_CUDA_NATIVE_TIMING_HOST_SOURCE}")
        endif()
        if(include_product_kernel)
            list(APPEND benchmark_sources "${GAFIME_CUDA_BENCHMARK_PRODUCT_SOURCE}")
        endif()
        add_executable(${target_name} ${benchmark_sources})
        target_compile_features(${target_name} PRIVATE cxx_std_20)
        set_target_properties(${target_name} PROPERTIES
            EXCLUDE_FROM_ALL TRUE
        )
        if(include_product_kernel)
            set_target_properties(${target_name} PROPERTIES
                CUDA_ARCHITECTURES "${GAFIME_CUDA_ARCHITECTURES}"
                CUDA_STANDARD 20
                CUDA_STANDARD_REQUIRED ON
                CUDA_EXTENSIONS OFF
            )
        else()
            set_target_properties(${target_name} PROPERTIES
                CXX_STANDARD 20
                CXX_STANDARD_REQUIRED ON
                CXX_EXTENSIONS OFF
            )
        endif()
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
        target_compile_definitions(${target_name} PRIVATE
            GAFIME_CUDA_BENCHMARK_COMPILED_LANE=${lane_value}
            GAFIME_CUDA_BENCHMARK_COMPILED_LANE_NAME="${lane_name}"
            GAFIME_NATIVE_TRUSTED_GIT_EXECUTABLE="${GAFIME_CUDA_BENCHMARK_GIT_EXECUTABLE}"
            GAFIME_NATIVE_TRUSTED_GIT_SHA256="${GAFIME_CUDA_BENCHMARK_GIT_SHA256}"
            GAFIME_NATIVE_TRUSTED_GIT_VERSION="${GAFIME_CUDA_BENCHMARK_GIT_VERSION}"
        )
        if(include_product_kernel)
            target_compile_definitions(${target_name} PRIVATE
                GAFIME_CUDA_BENCHMARK_HAS_CONTINUOUS_UNARY=${GAFIME_CUDA_BENCHMARK_HAS_CONTINUOUS_UNARY}
                GAFIME_CUDA_BENCHMARK_LINKED_PRODUCT_ROOT="${GAFIME_CUDA_BENCHMARK_PRODUCT_ROOT}"
                GAFIME_CUDA_BENCHMARK_LINKED_PRODUCT_COMMIT="${GAFIME_CUDA_BENCHMARK_PRODUCT_COMMIT}"
                GAFIME_CUDA_BENCHMARK_LINKED_PRODUCT_SOURCE_SHA256="${GAFIME_CUDA_BENCHMARK_PRODUCT_SOURCE_SHA256}"
                GAFIME_CUDA_BENCHMARK_LINKED_PRODUCT_HEADER_SHA256="${GAFIME_CUDA_BENCHMARK_PRODUCT_HEADER_SHA256}"
            )
            target_include_directories(${target_name} PRIVATE
                ${GAFIME_CUDA_BENCHMARK_PRODUCT_ROOT}/src/cuda
                ${CMAKE_CURRENT_LIST_DIR}/../common
            )
        else()
            # The non-direct helpers include only the harness-owned launch
            # policy header.  No product checkout path or product header is a
            # compile-time input to these common-control binaries.
            target_include_directories(${target_name} PRIVATE
                ${CMAKE_CURRENT_LIST_DIR}/../common
            )
        endif()
        target_link_libraries(
            ${target_name}
            PRIVATE CUDA::cudart ${CMAKE_DL_LIBS}
        )
    endfunction()

    # Preserve the historical direct target name for existing release and
    # local benchmark invocations.
    gafime_cuda_add_benchmark_target(
        gafime_cuda_precision_native_timing 1 supplemental_internal_kernel TRUE
    )
    gafime_cuda_add_benchmark_target(
        gafime_cuda_precision_native_timing_canonical 2 canonical_payload_api FALSE
    )
    gafime_cuda_add_benchmark_target(
        gafime_cuda_precision_native_timing_host 3 supplemental_host_phase FALSE
    )

    unset(GAFIME_CUDA_NATIVE_TIMING_HELPER_SOURCE)
    unset(GAFIME_CUDA_NATIVE_TIMING_HOST_SOURCE)
    unset(GAFIME_CUDA_BENCHMARK_PRODUCT_HEADER_TEXT)
endif()
